from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from algorithms import OffloadTarget, RandomOffloader
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
    packet_lost: bool
    deadline_missed: bool


def simulate_local(task: TaskRecord, model: ExecutionModel) -> tuple[float, float, float]:
    execution_time = task.exec_time / model.local_speedup
    energy = execution_time * model.local_energy_rate
    return execution_time, energy, 0.0


def simulate_fog(
    task: TaskRecord,
    vehicle: VehicleState,
    model: ExecutionModel,
) -> tuple[float, float, float]:
    fog = nearest_fog(vehicle, DEFAULT_FOG_NODES)
    dist = distance(vehicle.x, vehicle.y, fog.x, fog.y)
    transmission_delay = (task.data_size / model.fog_bandwidth) + (
        dist * model.fog_distance_delay_factor
    )
    execution_time = task.exec_time / model.fog_speedup
    latency = transmission_delay + execution_time
    energy = (
        transmission_delay * model.fog_tx_energy_rate
        + execution_time * model.fog_execution_energy_rate
    )
    packet_loss_percent = (
        model.fog_base_packet_loss_percent + task.plr_increase_percent
    )
    return latency, energy, packet_loss_percent


def simulate_cloud(task: TaskRecord, model: ExecutionModel) -> tuple[float, float, float]:
    transmission_delay = (task.data_size / model.cloud_bandwidth) + model.cloud_backhaul_delay
    execution_time = task.exec_time / model.cloud_speedup
    latency = transmission_delay + execution_time
    energy = transmission_delay * model.cloud_tx_energy_rate
    packet_loss_percent = (
        model.cloud_base_packet_loss_percent + task.plr_increase_percent
    )
    return latency, energy, packet_loss_percent


def simulate_task(
    task: TaskRecord,
    vehicle: VehicleState,
    offloader: RandomOffloader,
    model: ExecutionModel,
) -> SimulationResult:
    target = offloader.choose_target()

    if target == OffloadTarget.LOCAL:
        latency, energy, packet_loss_percent = simulate_local(task, model)
    elif target == OffloadTarget.FOG:
        latency, energy, packet_loss_percent = simulate_fog(task, vehicle, model)
    else:
        latency, energy, packet_loss_percent = simulate_cloud(task, model)

    packet_lost = offloader.packet_lost(packet_loss_percent)
    finish_time = task.release_time + latency
    deadline_missed = packet_lost or finish_time > task.deadline

    return SimulationResult(
        task_id=task.id,
        scenario=task.weather_scenario,
        target=target.value,
        release_time=task.release_time,
        deadline=task.deadline,
        finish_time=finish_time,
        latency=latency,
        energy=energy,
        packet_loss_percent=packet_loss_percent,
        packet_lost=packet_lost,
        deadline_missed=deadline_missed,
    )


def run_random_baseline(
    vehicles_file: str | Path = "data/vehicles/chunk_0.xml",
    tasks_file: str | Path = "data/tasks/chunk_0.xml",
    output_file: str | Path = "data/results/random_baseline_results.csv",
    seed: int = 11,
) -> list[SimulationResult]:
    vehicle_states = load_vehicle_states(vehicles_file)
    tasks = load_tasks(tasks_file)
    offloader = RandomOffloader(seed=seed)
    model = ExecutionModel()

    results: list[SimulationResult] = []
    for task in tasks:
        vehicle = vehicle_states.get((task.release_time, task.creator))
        if vehicle is None:
            continue
        results.append(simulate_task(task, vehicle, offloader, model))

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
                "packet_lost",
                "deadline_missed",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicles", default="data/vehicles/chunk_0.xml")
    parser.add_argument("--tasks", default="data/tasks/chunk_0.xml")
    parser.add_argument("--output", default="data/results/random_baseline_results.csv")
    parser.add_argument("--seed", type=int, default=11)
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

