from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from infrastructure import (
    DEFAULT_FOG_NODES,
    ExecutionModel,
    TaskRecord,
    VehicleState,
    distance,
    dynamic_backhaul_delay,
    nearest_fog,
)


VehicleLookup = Mapping[str | tuple[float, str], VehicleState]


@dataclass(frozen=True)
class ResourceCapacities:
    local: int = 1
    fog: int = 4
    cloud: int = 16

    def __post_init__(self) -> None:
        if min(self.local, self.fog, self.cloud) <= 0:
            raise ValueError("all execution capacities must be positive")


@dataclass
class ResourceState:
    """Resource availability that persists across control intervals."""

    local_available_times: dict[str, list[float]] = field(default_factory=dict)
    fog_available_times: dict[str, list[float]] = field(default_factory=dict)
    cloud_available_times: list[float] = field(default_factory=list)

    def copy(self) -> "ResourceState":
        return ResourceState(
            local_available_times={
                key: list(times)
                for key, times in self.local_available_times.items()
            },
            fog_available_times={
                key: list(times)
                for key, times in self.fog_available_times.items()
            },
            cloud_available_times=list(self.cloud_available_times),
        )

    def queue_for(
        self,
        resource: str,
        capacities: ResourceCapacities,
    ) -> list[float]:
        if resource.startswith("LOCAL:"):
            key = resource.removeprefix("LOCAL:")
            queue = self.local_available_times.setdefault(
                key,
                [0.0] * capacities.local,
            )
        elif resource.startswith("FOG:"):
            key = resource.removeprefix("FOG:")
            queue = self.fog_available_times.setdefault(
                key,
                [0.0] * capacities.fog,
            )
        else:
            if not self.cloud_available_times:
                self.cloud_available_times.extend([0.0] * capacities.cloud)
            queue = self.cloud_available_times

        heapq.heapify(queue)
        return queue

    def capacity_status(
        self,
        resource: str,
        capacities: ResourceCapacities,
        current_time: float,
    ) -> tuple[float, float]:
        """Return normalized busy and immediately available capacity."""
        if resource.startswith("LOCAL:"):
            key = resource.removeprefix("LOCAL:")
            queue = self.local_available_times.get(key, ())
            capacity = capacities.local
        elif resource.startswith("FOG:"):
            key = resource.removeprefix("FOG:")
            queue = self.fog_available_times.get(key, ())
            capacity = capacities.fog
        else:
            queue = self.cloud_available_times
            capacity = capacities.cloud

        busy = sum(available_at > current_time for available_at in queue)
        busy = min(busy, capacity)
        return busy / capacity, (capacity - busy) / capacity


class DeterministicChannel:
    """Repeatable packet samples shared by every offloading algorithm."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = int(seed)

    def sample(self, task_id: str, attempt_number: int) -> float:
        key = f"{self.seed}|{task_id}|{attempt_number}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        return value / float(1 << 64)

    def transmit(
        self,
        task_id: str,
        packet_loss_percent: float,
        max_retransmissions: int,
    ) -> tuple[bool, int, int]:
        loss_probability = _loss_probability(packet_loss_percent)
        max_attempts = 1 + max(0, max_retransmissions)
        for attempt in range(1, max_attempts + 1):
            if self.sample(task_id, attempt) >= loss_probability:
                return False, attempt, attempt - 1
        return True, max_attempts, max_attempts - 1


@dataclass(frozen=True)
class ExecutionOutcome:
    resource: str
    latency: float
    execution_time: float
    success_fixed_delay: float
    vehicle_energy: float
    infrastructure_energy: float
    local_cpu_energy: float
    transmission_energy: float
    fog_compute_energy: float
    cloud_compute_energy: float
    packet_loss_percent: float
    final_failure_probability: float
    backhaul_delay: float
    wireless_transmission_time: float
    transmission_attempts: int
    retransmission_count: int
    packet_lost: bool


@dataclass(frozen=True)
class AssignmentResult:
    task_id: str
    scenario: str
    target: str
    resource: str
    release_time: float
    deadline: float
    finish_time: float
    latency: float
    queue_delay: float
    energy: float
    packet_loss_percent: float
    final_failure_probability: float
    expected_deadline_failure: float
    network_load: int
    backhaul_delay: float
    wireless_transmission_time: float
    transmission_attempts: int
    retransmission_count: int
    local_cpu_energy: float
    transmission_energy: float
    fog_compute_energy: float
    cloud_compute_energy: float
    vehicle_energy: float
    infrastructure_energy: float
    total_system_energy: float
    packet_lost: bool
    deadline_missed: bool


def _target_name(target: object) -> str:
    value = getattr(target, "value", target)
    name = str(value).upper()
    if name not in {"LOCAL", "FOG", "CLOUD"}:
        raise ValueError(f"Unknown offloading target: {target!r}")
    return name


def _vehicle_for_task(task: TaskRecord, states: VehicleLookup) -> VehicleState:
    vehicle = states.get((task.release_time, task.creator))
    if vehicle is None:
        vehicle = states.get(task.creator)
    if vehicle is None:
        raise KeyError(
            f"No vehicle state for task {task.id!r} "
            f"(creator={task.creator!r}, release={task.release_time})"
        )
    return vehicle


def _loss_probability(packet_loss_percent: float) -> float:
    return max(0.0, min(packet_loss_percent / 100.0, 1.0))


def simulate_local(task: TaskRecord, model: ExecutionModel) -> ExecutionOutcome:
    execution_time = task.exec_time / model.local_speedup
    local_cpu_energy = task.power * execution_time
    return ExecutionOutcome(
        resource=f"LOCAL:{task.creator}",
        latency=execution_time,
        execution_time=execution_time,
        success_fixed_delay=0.0,
        vehicle_energy=local_cpu_energy,
        infrastructure_energy=0.0,
        local_cpu_energy=local_cpu_energy,
        transmission_energy=0.0,
        fog_compute_energy=0.0,
        cloud_compute_energy=0.0,
        packet_loss_percent=0.0,
        final_failure_probability=0.0,
        backhaul_delay=0.0,
        wireless_transmission_time=0.0,
        transmission_attempts=0,
        retransmission_count=0,
        packet_lost=False,
    )


def simulate_fog(
    task: TaskRecord,
    vehicle: VehicleState,
    channel: DeterministicChannel,
    model: ExecutionModel,
) -> ExecutionOutcome:
    fog = nearest_fog(vehicle, DEFAULT_FOG_NODES)
    wireless_time = task.data_size / model.fog_bandwidth
    access_delay = (
        distance(vehicle.x, vehicle.y, fog.x, fog.y)
        * model.fog_distance_delay_factor
    )
    execution_time = task.exec_time / model.fog_speedup
    packet_loss_percent = (
        model.fog_base_packet_loss_percent + task.plr_increase_percent
    )
    packet_lost, attempts, retries = channel.transmit(
        task.id,
        packet_loss_percent,
        model.max_retransmissions,
    )
    airtime = attempts * wireless_time
    retry_delay = retries * model.retransmission_timeout
    transmission_energy = airtime * model.vehicle_tx_power
    fog_energy = 0.0 if packet_lost else execution_time * model.fog_active_power
    latency = airtime + retry_delay
    if not packet_lost:
        latency += access_delay + execution_time

    loss_probability = _loss_probability(packet_loss_percent)
    max_attempts = 1 + max(0, model.max_retransmissions)
    return ExecutionOutcome(
        resource=f"FOG:{fog.id}",
        latency=latency,
        execution_time=execution_time,
        success_fixed_delay=access_delay,
        vehicle_energy=transmission_energy,
        infrastructure_energy=fog_energy,
        local_cpu_energy=0.0,
        transmission_energy=transmission_energy,
        fog_compute_energy=fog_energy,
        cloud_compute_energy=0.0,
        packet_loss_percent=packet_loss_percent,
        final_failure_probability=loss_probability ** max_attempts,
        backhaul_delay=0.0,
        wireless_transmission_time=wireless_time,
        transmission_attempts=attempts,
        retransmission_count=retries,
        packet_lost=packet_lost,
    )


def simulate_cloud(
    task: TaskRecord,
    channel: DeterministicChannel,
    model: ExecutionModel,
    network_load: int,
) -> ExecutionOutcome:
    wireless_time = task.data_size / model.cloud_bandwidth
    backhaul_delay = dynamic_backhaul_delay(
        scenario=task.weather_scenario,
        network_load=network_load,
        model=model,
    )
    execution_time = task.exec_time / model.cloud_speedup
    packet_loss_percent = (
        model.cloud_base_packet_loss_percent + task.plr_increase_percent
    )
    packet_lost, attempts, retries = channel.transmit(
        task.id,
        packet_loss_percent,
        model.max_retransmissions,
    )
    airtime = attempts * wireless_time
    retry_delay = retries * model.retransmission_timeout
    transmission_energy = airtime * model.vehicle_tx_power
    cloud_energy = 0.0 if packet_lost else execution_time * model.cloud_active_power
    latency = airtime + retry_delay
    if not packet_lost:
        latency += backhaul_delay + execution_time

    loss_probability = _loss_probability(packet_loss_percent)
    max_attempts = 1 + max(0, model.max_retransmissions)
    return ExecutionOutcome(
        resource="CLOUD",
        latency=latency,
        execution_time=execution_time,
        success_fixed_delay=backhaul_delay,
        vehicle_energy=transmission_energy,
        infrastructure_energy=cloud_energy,
        local_cpu_energy=0.0,
        transmission_energy=transmission_energy,
        fog_compute_energy=0.0,
        cloud_compute_energy=cloud_energy,
        packet_loss_percent=packet_loss_percent,
        final_failure_probability=loss_probability ** max_attempts,
        backhaul_delay=backhaul_delay,
        wireless_transmission_time=wireless_time,
        transmission_attempts=attempts,
        retransmission_count=retries,
        packet_lost=packet_lost,
    )


def _simulate_target(
    target: str,
    task: TaskRecord,
    vehicle: VehicleState,
    channel: DeterministicChannel,
    model: ExecutionModel,
    network_load: int,
) -> ExecutionOutcome:
    if target == "LOCAL":
        return simulate_local(task, model)
    if target == "FOG":
        return simulate_fog(task, vehicle, channel, model)
    return simulate_cloud(task, channel, model, network_load)


def _expected_deadline_failure(
    task: TaskRecord,
    outcome: ExecutionOutcome,
    available_at: float,
    model: ExecutionModel,
) -> float:
    if outcome.packet_loss_percent <= 0.0:
        finish_time = max(task.release_time, available_at) + outcome.execution_time
        return float(finish_time > task.deadline)

    loss_probability = _loss_probability(outcome.packet_loss_percent)
    max_attempts = 1 + max(0, model.max_retransmissions)
    expected_failure = loss_probability ** max_attempts

    for attempt in range(1, max_attempts + 1):
        success_probability = (
            loss_probability ** (attempt - 1)
            * (1.0 - loss_probability)
        )
        arrival_time = (
            task.release_time
            + attempt * outcome.wireless_transmission_time
            + (attempt - 1) * model.retransmission_timeout
            + outcome.success_fixed_delay
        )
        finish_time = max(arrival_time, available_at) + outcome.execution_time
        if finish_time > task.deadline:
            expected_failure += success_probability
    return min(expected_failure, 1.0)


def simulate_assignments(
    tasks: Sequence[TaskRecord],
    assignments: Sequence[object],
    resource_state: ResourceState,
    channel_randomness: DeterministicChannel,
    *,
    vehicle_states: VehicleLookup,
    model: ExecutionModel | None = None,
    capacities: ResourceCapacities | None = None,
    network_load_by_time: Mapping[float, int] | None = None,
) -> list[AssignmentResult]:
    """Execute target assignments and update persistent resource availability."""

    if len(tasks) != len(assignments):
        raise ValueError("tasks and assignments must have the same length")

    model = model or ExecutionModel()
    capacities = capacities or ResourceCapacities()
    if network_load_by_time is None:
        network_load_by_time = Counter(task.release_time for task in tasks)

    results: list[AssignmentResult | None] = [None] * len(tasks)
    indexed = sorted(
        enumerate(zip(tasks, assignments)),
        key=lambda item: (
            item[1][0].release_time,
            item[1][0].deadline,
            item[0],
        ),
    )

    for original_index, (task, raw_target) in indexed:
        target = _target_name(raw_target)
        vehicle = _vehicle_for_task(task, vehicle_states)
        network_load = int(network_load_by_time.get(task.release_time, 0))
        outcome = _simulate_target(
            target,
            task,
            vehicle,
            channel_randomness,
            model,
            network_load,
        )

        queue = resource_state.queue_for(outcome.resource, capacities)
        available_at = queue[0]
        expected_deadline_failure = _expected_deadline_failure(
            task,
            outcome,
            available_at,
            model,
        )

        if outcome.packet_lost:
            queue_delay = 0.0
            finish_time = task.release_time + outcome.latency
        else:
            available_at = heapq.heappop(queue)
            arrival_at_resource = (
                task.release_time
                + outcome.latency
                - outcome.execution_time
            )
            execution_start = max(arrival_at_resource, available_at)
            queue_delay = execution_start - arrival_at_resource
            finish_time = execution_start + outcome.execution_time
            heapq.heappush(queue, finish_time)

        latency = finish_time - task.release_time
        total_system_energy = (
            outcome.vehicle_energy + outcome.infrastructure_energy
        )
        results[original_index] = AssignmentResult(
            task_id=task.id,
            scenario=task.weather_scenario,
            target=target,
            resource=outcome.resource,
            release_time=task.release_time,
            deadline=task.deadline,
            finish_time=finish_time,
            latency=latency,
            queue_delay=queue_delay,
            energy=total_system_energy,
            packet_loss_percent=outcome.packet_loss_percent,
            final_failure_probability=outcome.final_failure_probability,
            expected_deadline_failure=expected_deadline_failure,
            network_load=network_load,
            backhaul_delay=outcome.backhaul_delay,
            wireless_transmission_time=outcome.wireless_transmission_time,
            transmission_attempts=outcome.transmission_attempts,
            retransmission_count=outcome.retransmission_count,
            local_cpu_energy=outcome.local_cpu_energy,
            transmission_energy=outcome.transmission_energy,
            fog_compute_energy=outcome.fog_compute_energy,
            cloud_compute_energy=outcome.cloud_compute_energy,
            vehicle_energy=outcome.vehicle_energy,
            infrastructure_energy=outcome.infrastructure_energy,
            total_system_energy=total_system_energy,
            packet_lost=outcome.packet_lost,
            deadline_missed=(
                outcome.packet_lost or finish_time > task.deadline
            ),
        )

    return [result for result in results if result is not None]
