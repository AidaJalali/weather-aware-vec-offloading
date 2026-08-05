from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
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
    load_tasks,
    load_vehicle_states,
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
class RewardProfile:
    latency_weight: float = 0.35
    energy_weight: float = 0.25
    reliability_weight: float = 0.15
    packet_loss_penalty: float = 2.0
    deadline_miss_penalty: float = 5.0

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in self.__dict__.values()):
            raise ValueError("reward weights and penalties cannot be negative")


@dataclass(frozen=True)
class RewardConfig:
    """Fixed reward by default, optionally overridden per weather scenario."""

    default: RewardProfile = field(default_factory=RewardProfile)
    weather_profiles: Mapping[str | WeatherScenario, RewardProfile] = field(
        default_factory=dict
    )
    name: str = "fixed"

    def __post_init__(self) -> None:
        normalized: dict[WeatherScenario, RewardProfile] = {}
        for scenario, profile in self.weather_profiles.items():
            if not isinstance(profile, RewardProfile):
                raise TypeError("weather profile values must be RewardProfile")
            normalized[normalize_scenario(scenario)] = profile
        object.__setattr__(
            self,
            "weather_profiles",
            MappingProxyType(normalized),
        )
        if not self.name.strip():
            raise ValueError("reward configuration name cannot be empty")

    def for_weather(
        self,
        scenario: str | WeatherScenario,
    ) -> RewardProfile:
        return self.weather_profiles.get(
            normalize_scenario(scenario),
            self.default,
        )

    @classmethod
    def adaptive_default(cls) -> "RewardConfig":
        """Initial configurable profiles; tune these through experiments."""
        return cls(
            name="weather_adaptive",
            default=RewardProfile(),
            weather_profiles={
                WeatherScenario.BASE: RewardProfile(
                    latency_weight=0.35,
                    energy_weight=0.25,
                    reliability_weight=0.15,
                    packet_loss_penalty=2.0,
                    deadline_miss_penalty=5.0,
                ),
                WeatherScenario.RAIN: RewardProfile(
                    latency_weight=0.40,
                    energy_weight=0.20,
                    reliability_weight=0.25,
                    packet_loss_penalty=2.5,
                    deadline_miss_penalty=6.0,
                ),
                WeatherScenario.SNOW: RewardProfile(
                    latency_weight=0.45,
                    energy_weight=0.20,
                    reliability_weight=0.15,
                    packet_loss_penalty=2.0,
                    deadline_miss_penalty=7.0,
                ),
                WeatherScenario.FOG: RewardProfile(
                    latency_weight=0.30,
                    energy_weight=0.20,
                    reliability_weight=0.35,
                    packet_loss_penalty=3.0,
                    deadline_miss_penalty=5.0,
                ),
            },
        )


@dataclass(frozen=True)
class ObservationScale:
    speed: float = 50.0
    execution_time: float = 30.0
    data_size: float = 3000.0
    cycles_per_bit: float = 3000.0
    deadline_slack: float = 30.0
    path_loss_db: float = 10.0
    fog_distance: float = 1500.0
    task_power: float = 10.0
    network_load: float = 50.0

    def __post_init__(self) -> None:
        if any(value <= 0.0 for value in self.__dict__.values()):
            raise ValueError("observation scales must be positive")


class VECOffloadingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-task VEC environment compatible with continuous-action SAC.

    Each step consumes one task. SAC supplies one scalar action in [-1, 1], which
    is mapped to LOCAL, FOG, or CLOUD. The transition uses the same execution
    simulator as all baseline algorithms, including persistent resource queues,
    energy, retransmission, weather-aware backhaul, and deadline handling.
    """

    metadata = {"render_modes": []}
    observation_fields = (
        "weather_base",
        "weather_rain",
        "weather_snow",
        "weather_fog",
        "vehicle_speed",
        "task_execution_time",
        "task_data_size",
        "cycles_per_bit",
        "deadline_slack",
        "path_loss_increase",
        "fog_packet_loss_probability",
        "cloud_packet_loss_probability",
        "nearest_fog_distance",
        "task_power",
        "network_load",
        "local_queue_occupancy",
        "fog_queue_occupancy",
        "cloud_queue_occupancy",
        "local_available_capacity",
        "fog_available_capacity",
        "cloud_available_capacity",
    )

    def __init__(
        self,
        tasks: Sequence[TaskRecord],
        vehicle_states: Mapping[
            str | tuple[float, str],
            VehicleState,
        ],
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
                key=lambda task: (
                    task.release_time,
                    task.deadline,
                    task.id,
                ),
            )
        )
        self.vehicle_states = vehicle_states
        self.execution_model = execution_model or ExecutionModel()
        self.resource_capacities = resource_capacities or ResourceCapacities()
        self.reward_config = reward_config or RewardConfig()
        self.observation_scale = observation_scale or ObservationScale()
        self.network_load_by_time = Counter(
            task.release_time for task in self.tasks
        )

        for task in self.tasks:
            self._vehicle_for_task(task)

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )
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

    def _vehicle_for_task(self, task: TaskRecord) -> VehicleState:
        vehicle = self.vehicle_states.get(
            (task.release_time, task.creator)
        )
        if vehicle is None:
            vehicle = self.vehicle_states.get(task.creator)
        if vehicle is None:
            raise KeyError(
                f"No vehicle state for task {task.id!r} "
                f"(creator={task.creator!r}, release={task.release_time})"
            )
        return vehicle

    @staticmethod
    def action_to_target(action: np.ndarray | Sequence[float]) -> OffloadTarget:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.size != 1 or not np.isfinite(values[0]):
            raise ValueError("action must contain one finite scalar")
        value = float(np.clip(values[0], -1.0, 1.0))
        if value < -1.0 / 3.0:
            return OffloadTarget.LOCAL
        if value < 1.0 / 3.0:
            return OffloadTarget.FOG
        return OffloadTarget.CLOUD

    @staticmethod
    def _scaled(value: float, maximum: float) -> float:
        return float(np.clip(value / maximum, 0.0, 1.0))

    def _observation(self, index: int) -> np.ndarray:
        task = self.tasks[index]
        vehicle = self._vehicle_for_task(task)
        fog = nearest_fog(vehicle, DEFAULT_FOG_NODES)
        scenario = normalize_scenario(task.weather_scenario)
        weather_one_hot = [
            float(scenario == candidate)
            for candidate in WeatherScenario
        ]
        local_occupancy, local_available = self._resource_state.capacity_status(
            f"LOCAL:{task.creator}",
            self.resource_capacities,
            task.release_time,
        )
        fog_occupancy, fog_available = self._resource_state.capacity_status(
            f"FOG:{fog.id}",
            self.resource_capacities,
            task.release_time,
        )
        cloud_occupancy, cloud_available = self._resource_state.capacity_status(
            "CLOUD",
            self.resource_capacities,
            task.release_time,
        )
        scale = self.observation_scale
        deadline_slack = max(task.deadline - task.release_time, 0.0)
        fog_packet_loss = np.clip(
            (
                self.execution_model.fog_base_packet_loss_percent
                + task.plr_increase_percent
            )
            / 100.0,
            0.0,
            1.0,
        )
        cloud_packet_loss = np.clip(
            (
                self.execution_model.cloud_base_packet_loss_percent
                + task.plr_increase_percent
            )
            / 100.0,
            0.0,
            1.0,
        )

        return np.asarray(
            [
                *weather_one_hot,
                self._scaled(vehicle.speed, scale.speed),
                self._scaled(task.exec_time, scale.execution_time),
                self._scaled(task.data_size, scale.data_size),
                self._scaled(
                    task.cycles_per_bit,
                    scale.cycles_per_bit,
                ),
                self._scaled(deadline_slack, scale.deadline_slack),
                self._scaled(
                    task.path_loss_increase_db,
                    scale.path_loss_db,
                ),
                fog_packet_loss,
                cloud_packet_loss,
                self._scaled(
                    distance(vehicle.x, vehicle.y, fog.x, fog.y),
                    scale.fog_distance,
                ),
                self._scaled(task.power, scale.task_power),
                self._scaled(
                    self.network_load_by_time[task.release_time],
                    scale.network_load,
                ),
                local_occupancy,
                fog_occupancy,
                cloud_occupancy,
                local_available,
                fog_available,
                cloud_available,
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
        )[0]

    def _reward(
        self,
        task: TaskRecord,
        outcome: AssignmentResult,
    ) -> tuple[float, dict[str, float | bool | str]]:
        available_time = max(
            task.deadline - task.release_time,
            1e-9,
        )
        total_energy = outcome.total_system_energy
        local_energy_reference = max(
            task.power
            * task.exec_time
            / self.execution_model.local_speedup,
            1e-9,
        )
        normalized_latency = outcome.latency / available_time
        normalized_energy = total_energy / local_energy_reference
        reliability_risk = float(
            np.clip(outcome.final_failure_probability, 0.0, 1.0)
        )
        deadline_missed = outcome.deadline_missed
        profile = self.reward_config.for_weather(task.weather_scenario)
        cost = (
            profile.latency_weight * normalized_latency
            + profile.energy_weight * normalized_energy
            + profile.reliability_weight * reliability_risk
            + profile.packet_loss_penalty * float(outcome.packet_lost)
            + profile.deadline_miss_penalty * float(deadline_missed)
        )
        return -cost, {
            "normalized_latency": normalized_latency,
            "normalized_energy": normalized_energy,
            "reliability_risk": reliability_risk,
            "deadline_missed": deadline_missed,
            "total_system_energy": total_energy,
            "reward_profile": self.reward_config.name,
            "latency_weight": profile.latency_weight,
            "energy_weight": profile.energy_weight,
            "reliability_weight": profile.reliability_weight,
            "packet_loss_penalty": profile.packet_loss_penalty,
            "deadline_miss_penalty": profile.deadline_miss_penalty,
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
        return self._observation(self._index), {
            "task_id": self.tasks[self._index].id,
        }

    def step(
        self,
        action: np.ndarray,
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
