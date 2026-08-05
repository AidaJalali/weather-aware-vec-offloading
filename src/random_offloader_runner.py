from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from algorithms import RandomOffloader
from genetic_offloader_runner import (
    ProgressBar,
    SCENARIO_GROUP_ORDER,
    batch_tasks,
    discover_datasets,
)
from infrastructure import ExecutionModel, load_tasks, load_vehicle_states
from offloading_simulator import (
    DeterministicChannel,
    ResourceCapacities,
    ResourceState,
    simulate_assignments,
)


SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SOURCE_DIR.parent / "data" / "datasets"


@dataclass(frozen=True)
class RandomRunRow:
    algorithm: str
    split: str
    dataset: str
    scenario_group: str
    task_id: str
    scenario: str
    target: str
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


def run_random_split(
    *,
    data_root: str | Path,
    split: str,
    output_file: str | Path,
    summary_file: str | Path,
    dataset_names: Sequence[str] | None = None,
    max_datasets: int | None = None,
    max_timesteps: int | None = None,
    batch_window_seconds: int = 1,
    capacities: ResourceCapacities | None = None,
    seed: int = 999,
    show_progress: bool = True,
    progress_every: int = 20,
) -> list[RandomRunRow]:
    datasets = discover_datasets(
        data_root=data_root,
        split=split,
        names=dataset_names,
        max_datasets=max_datasets,
    )
    model = ExecutionModel()
    capacities = capacities or ResourceCapacities()
    offloader = RandomOffloader(seed=seed)
    progress = ProgressBar(enabled=show_progress, update_every=progress_every)
    rows: list[RandomRunRow] = []

    if show_progress:
        print(
            f"Running Random on {len(datasets)} {split} dataset(s) "
            f"with batch_window={batch_window_seconds}s."
        )

    for dataset_index, dataset in enumerate(datasets, start=1):
        vehicle_states = load_vehicle_states(dataset.vehicles_file)
        tasks = load_tasks(dataset.tasks_file)
        network_load_by_time = Counter(task.release_time for task in tasks)
        resource_state = ResourceState()
        channel = DeterministicChannel(seed)
        batches = list(
            batch_tasks(
                tasks,
                batch_window_seconds=batch_window_seconds,
                max_timesteps=max_timesteps,
            )
        )
        processed_tasks = 0

        for batch_index, batch in enumerate(batches, start=1):
            assignments = tuple(
                offloader.choose_target()
                for _ in batch
            )
            simulated = simulate_assignments(
                batch,
                assignments,
                resource_state,
                channel,
                vehicle_states=vehicle_states,
                model=model,
                capacities=capacities,
                network_load_by_time=network_load_by_time,
            )
            for result in simulated:
                processed_tasks += 1
                rows.append(
                    RandomRunRow(
                        algorithm="Random",
                        split=split,
                        dataset=dataset.name,
                        scenario_group=dataset.group,
                        task_id=result.task_id,
                        scenario=result.scenario,
                        target=result.target,
                        release_time=result.release_time,
                        deadline=result.deadline,
                        finish_time=result.finish_time,
                        latency=result.latency,
                        queue_delay=result.queue_delay,
                        energy=result.total_system_energy,
                        packet_loss_percent=result.packet_loss_percent,
                        final_failure_probability=result.final_failure_probability,
                        expected_deadline_failure=result.expected_deadline_failure,
                        network_load=result.network_load,
                        backhaul_delay=result.backhaul_delay,
                        wireless_transmission_time=result.wireless_transmission_time,
                        transmission_attempts=result.transmission_attempts,
                        retransmission_count=result.retransmission_count,
                        local_cpu_energy=result.local_cpu_energy,
                        transmission_energy=result.transmission_energy,
                        fog_compute_energy=result.fog_compute_energy,
                        cloud_compute_energy=result.cloud_compute_energy,
                        vehicle_energy=result.vehicle_energy,
                        infrastructure_energy=result.infrastructure_energy,
                        total_system_energy=result.total_system_energy,
                        packet_lost=result.packet_lost,
                        deadline_missed=result.deadline_missed,
                    )
                )
            progress.update(
                split=split,
                dataset_index=dataset_index,
                dataset_count=len(datasets),
                dataset_name=dataset.name,
                batch_index=batch_index,
                batch_count=len(batches),
                task_count=processed_tasks,
                unit_name="timesteps",
                force=batch_index == len(batches),
            )
        progress.finish_dataset()

    write_rows(rows, output_file)
    write_summary(summarize_rows(rows), summary_file)
    return rows


def write_rows(rows: Sequence[RandomRunRow], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(RandomRunRow.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_rows(rows: Sequence[RandomRunRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RandomRunRow]] = {key: [] for key in SCENARIO_GROUP_ORDER}
    for row in rows:
        grouped.setdefault(row.scenario_group, []).append(row)

    summary: dict[str, dict[str, float]] = {}
    for group in SCENARIO_GROUP_ORDER:
        items = grouped.get(group, [])
        total = len(items)
        if total == 0:
            summary[group] = {
                "total_tasks": 0,
                "deadline_misses": 0,
                "packet_losses": 0,
                "avg_latency": 0.0,
                "avg_energy": 0.0,
                "avg_vehicle_energy": 0.0,
                "avg_infrastructure_energy": 0.0,
                "avg_total_system_energy": 0.0,
            }
            continue

        summary[group] = {
            "total_tasks": total,
            "deadline_misses": sum(row.deadline_missed for row in items),
            "packet_losses": sum(row.packet_lost for row in items),
            "avg_latency": sum(row.latency for row in items) / total,
            "avg_energy": sum(row.total_system_energy for row in items) / total,
            "avg_vehicle_energy": sum(row.vehicle_energy for row in items) / total,
            "avg_infrastructure_energy": (
                sum(row.infrastructure_energy for row in items) / total
            ),
            "avg_total_system_energy": (
                sum(row.total_system_energy for row in items) / total
            ),
        }
    return summary


def write_summary(summary: dict[str, dict[str, float]], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_group",
                "total_tasks",
                "deadline_misses",
                "packet_losses",
                "avg_latency",
                "avg_energy",
                "avg_vehicle_energy",
                "avg_infrastructure_energy",
                "avg_total_system_energy",
            ],
        )
        writer.writeheader()
        for group in SCENARIO_GROUP_ORDER:
            row = {"scenario_group": group}
            row.update(summary[group])
            writer.writerow(row)
