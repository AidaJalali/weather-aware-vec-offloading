from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from algorithms import OffloadTarget
from infrastructure import (
    DEFAULT_FOG_NODES,
    ExecutionModel,
    TaskRecord,
    VehicleState,
    distance,
    dynamic_backhaul_delay,
    load_tasks,
    load_vehicle_states,
    mobile_fog_nodes_by_time,
    nearest_fog,
)
from offloading_simulator import (
    AssignmentResult,
    DeterministicChannel,
    ResourceCapacities,
    ResourceState,
    simulate_assignments,
)
from weather_scenarios import WeatherScenario, normalize_scenario


@dataclass(frozen=True)
class RewardConfig:
    loss_weight: float = 0.50
    latency_weight: float = 0.35
    energy_weight: float = 0.15
    ratio_cap: float = 2.0
    max_per_attempt_loss: float = 0.35

    def __post_init__(self) -> None:
        weights = (self.loss_weight, self.latency_weight, self.energy_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("reward weights cannot be negative")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("reward weights must sum to 1")
        if self.ratio_cap <= 0.0:
            raise ValueError("ratio_cap must be positive")
        if not 0.0 < self.max_per_attempt_loss <= 1.0:
            raise ValueError("max_per_attempt_loss must be in (0, 1]")


def _percentile_scale(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 1.0
    return max(float(np.percentile(np.asarray(values), percentile)), 1e-6)


@dataclass(frozen=True)
class ObservationScale:
    data_size: float
    cycles_per_bit: float
    execution_time: float
    deadline_budget: float
    task_power: float
    fog_distance: float
    network_load: float
    queue_wait: float
    remaining_tasks: float
    remaining_compute: float
    remaining_data: float

    def __post_init__(self) -> None:
        if any(value <= 0.0 for value in asdict(self).values()):
            raise ValueError("observation scales must be positive")

    @classmethod
    def from_data(
        cls,
        tasks: Sequence[TaskRecord],
        vehicle_states: Mapping[str | tuple[float, str], VehicleState],
        *,
        percentile: float = 99.0,
    ) -> "ObservationScale":
        return cls.from_datasets(
            ((tasks, vehicle_states),),
            percentile=percentile,
        )

    @classmethod
    def from_datasets(
        cls,
        datasets: Sequence[
            tuple[
                Sequence[TaskRecord],
                Mapping[str | tuple[float, str], VehicleState],
            ]
        ],
        *,
        percentile: float = 99.0,
    ) -> "ObservationScale":
        if not datasets or not any(tasks for tasks, _ in datasets):
            raise ValueError("cannot derive observation scales without tasks")
        if not 0.0 < percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100]")

        all_tasks: list[TaskRecord] = []
        network_load_values: list[float] = []
        compute_values: list[float] = []
        data_values: list[float] = []
        fog_distances: list[float] = []
        for tasks, vehicle_states in datasets:
            all_tasks.extend(tasks)
            network_load = Counter(task.release_time for task in tasks)
            compute_by_time: Counter[float] = Counter()
            data_by_time: Counter[float] = Counter()
            for task in tasks:
                compute_by_time[task.release_time] += task.exec_time
                data_by_time[task.release_time] += task.data_size
            network_load_values.extend(network_load.values())
            compute_values.extend(compute_by_time.values())
            data_values.extend(data_by_time.values())

            fog_nodes_by_time = mobile_fog_nodes_by_time(vehicle_states)
            for task in tasks:
                vehicle = _vehicle_for_task(task, vehicle_states)
                fog_nodes = fog_nodes_by_time.get(
                    task.release_time, DEFAULT_FOG_NODES
                )
                fog = nearest_fog(vehicle, fog_nodes)
                fog_distances.append(
                    distance(vehicle.x, vehicle.y, fog.x, fog.y)
                )

        deadline_budgets = [
            max(task.deadline - task.release_time, 1e-6)
            for task in all_tasks
        ]
        deadline_scale = _percentile_scale(deadline_budgets, percentile)
        return cls(
            data_size=_percentile_scale(
                [task.data_size for task in all_tasks], percentile
            ),
            cycles_per_bit=_percentile_scale(
                [task.cycles_per_bit for task in all_tasks], percentile
            ),
            execution_time=_percentile_scale(
                [task.exec_time for task in all_tasks], percentile
            ),
            deadline_budget=deadline_scale,
            task_power=_percentile_scale(
                [task.power for task in all_tasks], percentile
            ),
            fog_distance=_percentile_scale(fog_distances, percentile),
            network_load=_percentile_scale(
                network_load_values, percentile
            ),
            queue_wait=2.0 * deadline_scale,
            remaining_tasks=_percentile_scale(
                network_load_values, percentile
            ),
            remaining_compute=_percentile_scale(
                compute_values, percentile
            ),
            remaining_data=_percentile_scale(
                data_values, percentile
            ),
        )


def _vehicle_for_task(
    task: TaskRecord,
    states: Mapping[str | tuple[float, str], VehicleState],
) -> VehicleState:
    vehicle = states.get((task.release_time, task.creator))
    if vehicle is None:
        vehicle = states.get(task.creator)
    if vehicle is None:
        raise KeyError(
            f"No vehicle state for task {task.id!r} "
            f"(creator={task.creator!r}, release={task.release_time})"
        )
    return vehicle


class VECOffloadingEnv(gym.Env[np.ndarray, int]):
    """Single-task environment with categorical LOCAL, FOG, and CLOUD actions."""

    metadata = {"render_modes": []}
    observation_fields = (
        "weather_base",
        "weather_rain",
        "weather_snow",
        "weather_fog",
        "task_data_size",
        "cycles_per_bit",
        "local_execution_time",
        "deadline_budget",
        "task_power",
        "nearest_fog_distance",
        "fog_terminal_loss_probability",
        "cloud_terminal_loss_probability",
        "network_load",
        "local_estimated_wait",
        "fog_estimated_wait",
        "cloud_estimated_wait",
        "remaining_tasks",
        "remaining_compute_demand",
        "remaining_data_volume",
    )

    def __init__(
        self,
        tasks: Sequence[TaskRecord],
        vehicle_states: Mapping[str | tuple[float, str], VehicleState],
        *,
        execution_model: ExecutionModel | None = None,
        resource_capacities: ResourceCapacities | None = None,
        reward_config: RewardConfig | None = None,
        observation_scale: ObservationScale | None = None,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("VECOffloadingEnv requires at least one task")

        self.tasks = tuple(
            sorted(
                tasks,
                key=lambda task: (task.release_time, task.deadline, task.id),
            )
        )
        self.vehicle_states = vehicle_states
        self.execution_model = execution_model or ExecutionModel()
        self.resource_capacities = resource_capacities or ResourceCapacities()
        self.reward_config = reward_config or RewardConfig()
        self.observation_scale = observation_scale or ObservationScale.from_data(
            self.tasks,
            self.vehicle_states,
        )
        self.network_load_by_time = Counter(
            task.release_time for task in self.tasks
        )
        self.fog_nodes_by_time = mobile_fog_nodes_by_time(self.vehicle_states)
        self._remaining_tasks = np.zeros(len(self.tasks), dtype=np.float32)
        self._remaining_compute = np.zeros(len(self.tasks), dtype=np.float32)
        self._remaining_data = np.zeros(len(self.tasks), dtype=np.float32)
        self._prepare_remaining_batch_features()

        for task in self.tasks:
            _vehicle_for_task(task, self.vehicle_states)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(self.observation_fields),),
            dtype=np.float32,
        )
        self._index: int | None = None
        self._resource_state = ResourceState()
        self._channel = DeterministicChannel()

    @classmethod
    def from_xml(
        cls,
        tasks_file: str | Path,
        vehicles_file: str | Path,
        **kwargs,
    ) -> "VECOffloadingEnv":
        return cls(
            tasks=load_tasks(tasks_file),
            vehicle_states=load_vehicle_states(vehicles_file),
            **kwargs,
        )

    def _prepare_remaining_batch_features(self) -> None:
        start = 0
        while start < len(self.tasks):
            end = start + 1
            release_time = self.tasks[start].release_time
            while (
                end < len(self.tasks)
                and self.tasks[end].release_time == release_time
            ):
                end += 1
            compute = 0.0
            data = 0.0
            for index in range(end - 1, start - 1, -1):
                compute += self.tasks[index].exec_time
                data += self.tasks[index].data_size
                self._remaining_tasks[index] = end - index
                self._remaining_compute[index] = compute
                self._remaining_data[index] = data
            start = end

    @staticmethod
    def action_to_target(action: int | np.integer) -> OffloadTarget:
        try:
            index = int(action)
        except (TypeError, ValueError) as error:
            raise ValueError("action must be an integer in {0, 1, 2}") from error
        targets = tuple(OffloadTarget)
        if index < 0 or index >= len(targets):
            raise ValueError("action must be an integer in {0, 1, 2}")
        return targets[index]

    @staticmethod
    def target_to_action(target: OffloadTarget | str) -> int:
        normalized = OffloadTarget(target)
        return tuple(OffloadTarget).index(normalized)

    @staticmethod
    def _scaled(value: float, maximum: float) -> float:
        return float(np.clip(value / maximum, 0.0, 1.0))

    def _fog_for_task(self, task: TaskRecord, vehicle: VehicleState):
        fog_nodes = self.fog_nodes_by_time.get(
            task.release_time,
            DEFAULT_FOG_NODES,
        )
        return nearest_fog(vehicle, fog_nodes)

    def _estimated_waits(
        self,
        task: TaskRecord,
        vehicle: VehicleState,
    ) -> tuple[float, float, float, float]:
        model = self.execution_model
        capacities = self.resource_capacities
        fog = self._fog_for_task(task, vehicle)
        fog_distance = distance(vehicle.x, vehicle.y, fog.x, fog.y)
        fog_arrival = (
            task.release_time
            + task.data_size / model.fog_bandwidth
            + fog_distance * model.fog_distance_delay_factor
        )
        cloud_arrival = (
            task.release_time
            + task.data_size / model.cloud_bandwidth
            + dynamic_backhaul_delay(
                task.weather_scenario,
                self.network_load_by_time[task.release_time],
                model,
            )
        )
        local_wait = self._resource_state.estimated_wait(
            f"LOCAL:{task.creator}", capacities, task.release_time
        )
        fog_wait = self._resource_state.estimated_wait(
            f"FOG:{fog.id}", capacities, fog_arrival
        )
        cloud_wait = self._resource_state.estimated_wait(
            "CLOUD", capacities, cloud_arrival
        )
        return local_wait, fog_wait, cloud_wait, fog_distance

    def _observation(self, index: int) -> np.ndarray:
        task = self.tasks[index]
        vehicle = _vehicle_for_task(task, self.vehicle_states)
        scenario = normalize_scenario(task.weather_scenario)
        weather_one_hot = [
            float(scenario == candidate) for candidate in WeatherScenario
        ]
        local_wait, fog_wait, cloud_wait, fog_distance = self._estimated_waits(
            task,
            vehicle,
        )
        model = self.execution_model
        max_attempts = 1 + model.max_retransmissions
        fog_per_attempt = np.clip(
            (model.fog_base_packet_loss_percent + task.plr_increase_percent)
            / 100.0,
            0.0,
            1.0,
        )
        cloud_per_attempt = np.clip(
            (model.cloud_base_packet_loss_percent + task.plr_increase_percent)
            / 100.0,
            0.0,
            1.0,
        )
        scale = self.observation_scale
        deadline_budget = max(task.deadline - task.release_time, 1e-6)

        return np.asarray(
            [
                *weather_one_hot,
                self._scaled(task.data_size, scale.data_size),
                self._scaled(task.cycles_per_bit, scale.cycles_per_bit),
                self._scaled(task.exec_time, scale.execution_time),
                self._scaled(deadline_budget, scale.deadline_budget),
                self._scaled(task.power, scale.task_power),
                self._scaled(fog_distance, scale.fog_distance),
                float(fog_per_attempt**max_attempts),
                float(cloud_per_attempt**max_attempts),
                self._scaled(
                    self.network_load_by_time[task.release_time],
                    scale.network_load,
                ),
                self._scaled(local_wait, scale.queue_wait),
                self._scaled(fog_wait, scale.queue_wait),
                self._scaled(cloud_wait, scale.queue_wait),
                self._scaled(self._remaining_tasks[index], scale.remaining_tasks),
                self._scaled(
                    self._remaining_compute[index], scale.remaining_compute
                ),
                self._scaled(self._remaining_data[index], scale.remaining_data),
            ],
            dtype=np.float32,
        )

    def _execute(
        self,
        target: OffloadTarget,
        task: TaskRecord,
    ) -> AssignmentResult:
        return simulate_assignments(
            (task,),
            (target,),
            self._resource_state,
            self._channel,
            vehicle_states=self.vehicle_states,
            model=self.execution_model,
            capacities=self.resource_capacities,
            network_load_by_time=self.network_load_by_time,
            fog_nodes_by_time=self.fog_nodes_by_time,
        )[0]

    def _reward(
        self,
        task: TaskRecord,
        outcome: AssignmentResult,
    ) -> tuple[float, dict[str, float | bool | str]]:
        config = self.reward_config
        deadline_budget = max(task.deadline - task.release_time, 1e-6)
        local_energy_reference = max(
            task.power
            * task.exec_time
            / self.execution_model.local_speedup,
            1e-6,
        )
        max_attempts = 1 + self.execution_model.max_retransmissions
        max_terminal_loss = config.max_per_attempt_loss**max_attempts
        loss_cost = min(
            outcome.final_failure_probability / max(max_terminal_loss, 1e-6),
            1.0,
        )
        if outcome.packet_lost:
            loss_cost = 1.0
        latency_ratio = outcome.latency / deadline_budget
        latency_cost = min(latency_ratio, config.ratio_cap) / config.ratio_cap
        energy_ratio = outcome.total_system_energy / local_energy_reference
        energy_cost = min(energy_ratio, config.ratio_cap) / config.ratio_cap
        reward = -(
            config.loss_weight * loss_cost
            + config.latency_weight * latency_cost
            + config.energy_weight * energy_cost
        )
        return reward, {
            "loss_cost": loss_cost,
            "latency_ratio": latency_ratio,
            "latency_cost": latency_cost,
            "energy_ratio": energy_ratio,
            "energy_cost": energy_cost,
            "normalized_latency": latency_ratio,
            "normalized_energy": energy_ratio,
            "reliability_risk": outcome.final_failure_probability,
            "deadline_missed": outcome.deadline_missed,
            "total_system_energy": outcome.total_system_energy,
            "reward_profile": "fixed_bounded",
            "loss_weight": config.loss_weight,
            "latency_weight": config.latency_weight,
            "energy_weight": config.energy_weight,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._index = 0
        channel_seed = (
            seed
            if seed is not None
            else int(self.np_random.integers(0, np.iinfo(np.int32).max))
        )
        self._channel = DeterministicChannel(seed=channel_seed)
        self._resource_state = ResourceState()
        return self._observation(0), {"task_id": self.tasks[0].id}

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._index is None:
            raise RuntimeError("reset() must be called before step()")
        if self._index >= len(self.tasks):
            raise RuntimeError("episode has ended; call reset()")

        task = self.tasks[self._index]
        target = self.action_to_target(action)
        outcome = self._execute(target, task)
        reward, reward_info = self._reward(task, outcome)

        self._index += 1
        terminated = self._index >= len(self.tasks)
        observation = (
            np.zeros(self.observation_space.shape, dtype=np.float32)
            if terminated
            else self._observation(self._index)
        )
        info = {
            "task_id": task.id,
            "scenario": task.weather_scenario,
            "target": target.value,
            "release_time": outcome.release_time,
            "deadline": outcome.deadline,
            "latency": outcome.latency,
            "queue_delay": outcome.queue_delay,
            "finish_time": outcome.finish_time,
            "energy": outcome.total_system_energy,
            "packet_lost": outcome.packet_lost,
            "packet_loss_percent": outcome.packet_loss_percent,
            "final_failure_probability": outcome.final_failure_probability,
            "expected_deadline_failure": outcome.expected_deadline_failure,
            "network_load": outcome.network_load,
            "backhaul_delay": outcome.backhaul_delay,
            "wireless_transmission_time": outcome.wireless_transmission_time,
            "transmission_attempts": outcome.transmission_attempts,
            "retransmission_count": outcome.retransmission_count,
            "local_cpu_energy": outcome.local_cpu_energy,
            "transmission_energy": outcome.transmission_energy,
            "fog_compute_energy": outcome.fog_compute_energy,
            "cloud_compute_energy": outcome.cloud_compute_energy,
            "vehicle_energy": outcome.vehicle_energy,
            "infrastructure_energy": outcome.infrastructure_energy,
            **reward_info,
        }
        return observation, float(reward), terminated, False, info
