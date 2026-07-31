from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from algorithms import OffloadTarget, RandomOffloader
from infrastructure import (
    DEFAULT_FOG_NODES,
    ExecutionModel,
    TaskRecord,
    VehicleState,
    distance,
    dynamic_backhaul_delay,
    load_tasks,
    load_vehicle_states,
    nearest_fog,
)


@dataclass(frozen=True)
class SimulationResult:
    task_id: str
    scenario: str
    target: str
    release_time: float
    deadline: float
    finish_time: float
    latency: float
    energy: float
    packet_loss_percent: float
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


@dataclass(frozen=True)
class ExecutionOutcome:
    latency: float
    vehicle_energy: float
    infrastructure_energy: float
    local_cpu_energy: float
    transmission_energy: float
    fog_compute_energy: float
    cloud_compute_energy: float
    packet_loss_percent: float
    backhaul_delay: float
    wireless_transmission_time: float
    transmission_attempts: int
    retransmission_count: int
    packet_lost: bool


def simulate_local(task: TaskRecord, model: ExecutionModel) -> ExecutionOutcome:
    local_execution_time = task.exec_time / model.local_speedup
    local_cpu_energy = task.power * local_execution_time
    return ExecutionOutcome(
        latency=local_execution_time,
        vehicle_energy=local_cpu_energy,
        infrastructure_energy=0.0,
        local_cpu_energy=local_cpu_energy,
        transmission_energy=0.0,
        fog_compute_energy=0.0,
        cloud_compute_energy=0.0,
        packet_loss_percent=0.0,
        backhaul_delay=0.0,
        wireless_transmission_time=0.0,
        transmission_attempts=0,
        retransmission_count=0,
        packet_lost=False,
    )


def simulate_fog(
    task: TaskRecord,
    vehicle: VehicleState,
    offloader: RandomOffloader,
    model: ExecutionModel,
) -> ExecutionOutcome:
    fog = nearest_fog(vehicle, DEFAULT_FOG_NODES)
    dist = distance(vehicle.x, vehicle.y, fog.x, fog.y)
    wireless_transmission_time = task.data_size / model.fog_bandwidth
    fog_access_delay = dist * model.fog_distance_delay_factor
    fog_execution_time = task.exec_time / model.fog_speedup
    packet_loss_percent = (
        model.fog_base_packet_loss_percent + task.plr_increase_percent
    )
    packet_lost, transmission_attempts, retransmission_count = (
        offloader.transmit_with_retries(
            packet_loss_percent,
            model.max_retransmissions,
        )
    )

    retry_time = retransmission_count * model.retransmission_timeout
    transmission_airtime = transmission_attempts * wireless_transmission_time
    transmission_energy = transmission_airtime * model.vehicle_tx_power

    if packet_lost:
        latency = transmission_airtime + retry_time
        fog_compute_energy = 0.0
    else:
        latency = transmission_airtime + retry_time + fog_access_delay + fog_execution_time
        fog_compute_energy = fog_execution_time * model.fog_active_power

    return ExecutionOutcome(
        latency=latency,
        vehicle_energy=transmission_energy,
        infrastructure_energy=fog_compute_energy,
        local_cpu_energy=0.0,
        transmission_energy=transmission_energy,
        fog_compute_energy=fog_compute_energy,
        cloud_compute_energy=0.0,
        packet_loss_percent=packet_loss_percent,
        backhaul_delay=0.0,
        wireless_transmission_time=wireless_transmission_time,
        transmission_attempts=transmission_attempts,
        retransmission_count=retransmission_count,
        packet_lost=packet_lost,
    )


def simulate_cloud(
    task: TaskRecord,
    offloader: RandomOffloader,
    model: ExecutionModel,
    network_load: int,
) -> ExecutionOutcome:
    wireless_transmission_time = task.data_size / model.cloud_bandwidth
    backhaul_delay = dynamic_backhaul_delay(
        scenario=task.weather_scenario,
        network_load=network_load,
        model=model,
    )
    cloud_execution_time = task.exec_time / model.cloud_speedup
    packet_loss_percent = (
        model.cloud_base_packet_loss_percent + task.plr_increase_percent
    )
    packet_lost, transmission_attempts, retransmission_count = (
        offloader.transmit_with_retries(
            packet_loss_percent,
            model.max_retransmissions,
        )
    )

    retry_time = retransmission_count * model.retransmission_timeout
    transmission_airtime = transmission_attempts * wireless_transmission_time
    transmission_energy = transmission_airtime * model.vehicle_tx_power

    if packet_lost:
        latency = transmission_airtime + retry_time
        cloud_compute_energy = 0.0
    else:
        latency = (
            transmission_airtime
            + retry_time
            + backhaul_delay
            + cloud_execution_time
        )
        cloud_compute_energy = cloud_execution_time * model.cloud_active_power

    return ExecutionOutcome(
        latency=latency,
        vehicle_energy=transmission_energy,
        infrastructure_energy=cloud_compute_energy,
        local_cpu_energy=0.0,
        transmission_energy=transmission_energy,
        fog_compute_energy=0.0,
        cloud_compute_energy=cloud_compute_energy,
        packet_loss_percent=packet_loss_percent,
        backhaul_delay=backhaul_delay,
        wireless_transmission_time=wireless_transmission_time,
        transmission_attempts=transmission_attempts,
        retransmission_count=retransmission_count,
        packet_lost=packet_lost,
    )


def simulate_task(
    task: TaskRecord,
    vehicle: VehicleState,
    offloader: RandomOffloader,
    model: ExecutionModel,
    network_load: int,
) -> SimulationResult:
    target = offloader.choose_target()

    if target == OffloadTarget.LOCAL:
        outcome = simulate_local(task, model)
    elif target == OffloadTarget.FOG:
        outcome = simulate_fog(task, vehicle, offloader, model)
    else:
        outcome = simulate_cloud(task, offloader, model, network_load)

    finish_time = task.release_time + outcome.latency
    deadline_missed = outcome.packet_lost or finish_time > task.deadline
    total_system_energy = outcome.vehicle_energy + outcome.infrastructure_energy

    return SimulationResult(
        task_id=task.id,
        scenario=task.weather_scenario,
        target=target.value,
        release_time=task.release_time,
        deadline=task.deadline,
        finish_time=finish_time,
        latency=outcome.latency,
        energy=outcome.vehicle_energy,
        packet_loss_percent=outcome.packet_loss_percent,
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
        deadline_missed=deadline_missed,
    )


def run_random_baseline(
    vehicles_file: str | Path = "data/sumo/vehicles/chunk_0.xml",
    tasks_file: str | Path = "data/sumo/tasks/chunk_0.xml",
    output_file: str | Path = "data/sumo/results/random_baseline_results.csv",
    seed: int = 42,
) -> list[SimulationResult]:
    vehicle_states = load_vehicle_states(vehicles_file)
    tasks = load_tasks(tasks_file)
    offloader = RandomOffloader(seed=seed)
    model = ExecutionModel()
    network_load_by_time = Counter(task.release_time for task in tasks)

    results: list[SimulationResult] = []
    for task in tasks:
        vehicle = vehicle_states.get((task.release_time, task.creator))
        if vehicle is None:
            raise ValueError(
                f"No vehicle state for task {task.id} "
                f"at time {task.release_time} (creator={task.creator})"
            )
        network_load = network_load_by_time[task.release_time]
        results.append(simulate_task(task, vehicle, offloader, model, network_load))

    write_results(results, output_file)
    return results


def write_results(results: list[SimulationResult], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "scenario",
                "target",
                "release_time",
                "deadline",
                "finish_time",
                "latency",
                "energy",
                "packet_loss_percent",
                "network_load",
                "backhaul_delay",
                "wireless_transmission_time",
                "transmission_attempts",
                "retransmission_count",
                "local_cpu_energy",
                "transmission_energy",
                "fog_compute_energy",
                "cloud_compute_energy",
                "vehicle_energy",
                "infrastructure_energy",
                "total_system_energy",
                "packet_lost",
                "deadline_missed",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicles", default="data/sumo/vehicles/chunk_0.xml")
    parser.add_argument("--tasks", default="data/sumo/tasks/chunk_0.xml")
    parser.add_argument("--output", default="data/sumo/results/random_baseline_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = run_random_baseline(
        vehicles_file=args.vehicles,
        tasks_file=args.tasks,
        output_file=args.output,
        seed=args.seed,
    )
    print(f"Random baseline simulated {len(results)} tasks.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
