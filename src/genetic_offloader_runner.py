from __future__ import annotations

import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from algorithms import GeneticBatchOffloader, GeneticOffloaderConfig
from infrastructure import (
    ExecutionModel,
    TaskRecord,
    load_tasks,
    load_vehicle_states,
)
from offloading_simulator import (
    AssignmentResult,
    DeterministicChannel,
    ResourceState,
    simulate_assignments,
)


SCENARIO_GROUP_ORDER = (
    "BASE",
    "RAIN",
    "SNOW",
    "FOG",
    "FAST_MIXED",
    "SLOW_MIXED",
    "RANDOM_MIX_1",
    "RANDOM_MIX_2",
)


class ProgressBar:
    def __init__(
        self,
        enabled: bool = True,
        update_every: int = 20,
    ) -> None:
        self.enabled = enabled
        self.update_every = max(1, update_every)
        self.started_at = time.perf_counter()

    def update(
        self,
        *,
        split: str,
        dataset_index: int,
        dataset_count: int,
        dataset_name: str,
        batch_index: int,
        batch_count: int,
        task_count: int,
        unit_name: str = "batches",
        force: bool = False,
    ) -> None:
        if not self.enabled:
            return
        if not force and batch_index % self.update_every != 0:
            return
        elapsed = time.perf_counter() - self.started_at
        print(
            f"{split} dataset {dataset_index}/{dataset_count} {dataset_name}: "
            f"{unit_name} {batch_index}/{batch_count}, "
            f"tasks {task_count}, elapsed={elapsed:.1f}s"
        )

    def finish_dataset(self) -> None:
        return


@dataclass(frozen=True)
class DatasetFiles:
    name: str
    group: str
    vehicles_file: Path
    tasks_file: Path


@dataclass(frozen=True)
class GeneticRunRow:
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
    batch_start_time: float
    batch_end_time: float
    batch_size: int
    optimization_cost: float
    optimization_generations: int
    optimization_evaluations: int
    optimization_elapsed_seconds: float
    optimization_stopped_by_time_limit: bool


def config_to_json(config: GeneticOffloaderConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(config), indent=2) + "\n",
        encoding="utf-8",
    )


def config_from_json(path: str | Path) -> GeneticOffloaderConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw.pop("path_loss_delay_per_db", None)
    return GeneticOffloaderConfig(**raw)


def dataset_group(name: str) -> str:
    lowered = name.lower()
    if "fast_mix" in lowered:
        return "FAST_MIXED"
    if "slow_mix" in lowered:
        return "SLOW_MIXED"
    if "random_mix_1" in lowered:
        return "RANDOM_MIX_1"
    if "random_mix_2" in lowered:
        return "RANDOM_MIX_2"
    for scenario in ("base", "rain", "snow", "fog"):
        if scenario in lowered.split("_"):
            return scenario.upper()
    raise ValueError(f"Cannot determine scenario group from dataset name: {name}")


def _first_chunk(folder: Path, kind: str) -> Path:
    candidates = sorted((folder / kind).glob("chunk_*.xml"))
    candidates.extend(sorted((folder / kind).glob("chunk_*.xml.gz")))
    if not candidates:
        raise FileNotFoundError(f"No {kind} chunk files found in {folder / kind}")
    return candidates[0]


def discover_datasets(
    data_root: str | Path,
    split: str,
    names: Sequence[str] | None = None,
    max_datasets: int | None = None,
) -> list[DatasetFiles]:
    split_dir = Path(data_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Dataset category not found: {split_dir}")
    if names:
        folders = [split_dir / name for name in names]
    else:
        folders = sorted(path for path in split_dir.iterdir() if path.is_dir())

    datasets: list[DatasetFiles] = []
    for folder in folders:
        if not folder.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder}")
        datasets.append(
            DatasetFiles(
                name=folder.name,
                group=dataset_group(folder.name),
                vehicles_file=_first_chunk(folder, "vehicles"),
                tasks_file=_first_chunk(folder, "tasks"),
            )
        )
        if max_datasets is not None and max_datasets > 0:
            if len(datasets) >= max_datasets:
                break
    return datasets


def batch_tasks(
    tasks: Sequence[TaskRecord],
    batch_window_seconds: int,
    max_timesteps: int | None = None,
) -> Iterable[list[TaskRecord]]:
    if batch_window_seconds <= 0:
        raise ValueError("batch_window_seconds must be positive")

    grouped: dict[int, list[TaskRecord]] = {}
    for task in tasks:
        bucket = int(task.release_time) // batch_window_seconds
        grouped.setdefault(bucket, []).append(task)

    emitted = 0
    for bucket in sorted(grouped):
        batch = sorted(
            grouped[bucket],
            key=lambda item: (item.release_time, item.deadline, item.id),
        )
        if batch:
            yield batch
            emitted += 1
            if max_timesteps is not None and max_timesteps > 0:
                if emitted >= max_timesteps:
                    return


def _rows_from_results(
    *,
    results: Sequence[AssignmentResult],
    split: str,
    dataset: DatasetFiles,
    batch_start_time: float,
    batch_end_time: float,
    optimization_cost: float,
    optimization_generations: int,
    optimization_evaluations: int,
    optimization_elapsed_seconds: float,
    optimization_stopped_by_time_limit: bool,
) -> list[GeneticRunRow]:
    rows: list[GeneticRunRow] = []
    for result in results:
        rows.append(
            GeneticRunRow(
                algorithm="GeneticBatch",
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
                batch_start_time=batch_start_time,
                batch_end_time=batch_end_time,
                batch_size=len(results),
                optimization_cost=optimization_cost,
                optimization_generations=optimization_generations,
                optimization_evaluations=optimization_evaluations,
                optimization_elapsed_seconds=optimization_elapsed_seconds,
                optimization_stopped_by_time_limit=optimization_stopped_by_time_limit,
            )
        )
    return rows


def run_genetic_split(
    *,
    data_root: str | Path,
    split: str,
    output_file: str | Path,
    summary_file: str | Path,
    config: GeneticOffloaderConfig,
    dataset_names: Sequence[str] | None = None,
    max_datasets: int | None = None,
    max_timesteps: int | None = None,
    batch_window_seconds: int = 1,
    seed: int = 42,
    show_progress: bool = True,
    progress_every: int = 20,
) -> list[GeneticRunRow]:
    datasets = discover_datasets(
        data_root=data_root,
        split=split,
        names=dataset_names,
        max_datasets=max_datasets,
    )
    model = ExecutionModel()
    optimizer = GeneticBatchOffloader(config=config, model=model)
    progress = ProgressBar(enabled=show_progress, update_every=progress_every)
    all_rows: list[GeneticRunRow] = []

    if show_progress:
        print(
            f"Running GeneticBatch on {len(datasets)} {split} dataset(s) "
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
            result = optimizer.optimize(
                batch,
                vehicle_states,
                resource_state=resource_state,
                channel_randomness=channel,
                network_load_by_time=network_load_by_time,
            )
            simulated = simulate_assignments(
                batch,
                result.assignments,
                resource_state,
                channel,
                vehicle_states=vehicle_states,
                model=model,
                capacities=config.resource_capacities,
                network_load_by_time=network_load_by_time,
            )
            processed_tasks += len(batch)
            all_rows.extend(
                _rows_from_results(
                    results=simulated,
                    split=split,
                    dataset=dataset,
                    batch_start_time=min(task.release_time for task in batch),
                    batch_end_time=max(task.release_time for task in batch),
                    optimization_cost=result.evaluation.cost,
                    optimization_generations=result.generations,
                    optimization_evaluations=result.evaluations,
                    optimization_elapsed_seconds=result.elapsed_seconds,
                    optimization_stopped_by_time_limit=result.stopped_by_time_limit,
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

    write_rows(all_rows, output_file)
    write_summary(summarize_rows(all_rows), summary_file)
    return all_rows


def write_rows(rows: Sequence[GeneticRunRow], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(GeneticRunRow.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def summarize_rows(rows: Sequence[GeneticRunRow]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[GeneticRunRow]] = {key: [] for key in SCENARIO_GROUP_ORDER}
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
