from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from algorithms import OffloadTarget, RandomOffloader
from infrastructure import (
    ExecutionModel,
    TaskRecord,
    VehicleState,
    load_tasks,
    load_vehicle_states,
)
from offloading_simulator import (
    AssignmentResult,
    DeterministicChannel,
    ResourceCapacities,
    ResourceState,
    simulate_assignments,
)


SimulationResult = AssignmentResult


def simulate_task(
    task: TaskRecord,
    vehicle: VehicleState,
    offloader: RandomOffloader,
    model: ExecutionModel,
    network_load: int,
    *,
    target: OffloadTarget | None = None,
    resource_state: ResourceState | None = None,
    channel: DeterministicChannel | None = None,
    capacities: ResourceCapacities | None = None,
) -> SimulationResult:
    """Compatibility wrapper for simulating one task with the shared engine."""

    selected_target = target or offloader.choose_target()
    return simulate_assignments(
        (task,),
        (selected_target,),
        resource_state or ResourceState(),
        channel or DeterministicChannel(offloader.seed),
        vehicle_states={(task.release_time, task.creator): vehicle},
        model=model,
        capacities=capacities,
        network_load_by_time={task.release_time: network_load},
    )[0]


_SOURCE_DIR = Path(__file__).resolve().parent
_DEFAULT_DATA = _SOURCE_DIR.parent / "data" / "sumo"


def run_random_baseline(
    vehicles_file: str | Path | None = None,
    tasks_file: str | Path | None = None,
    output_file: str | Path | None = None,
    seed: int = 42,
) -> list[SimulationResult]:
    if vehicles_file is None:
        vehicles_file = _DEFAULT_DATA / "vehicles" / "chunk_0.xml"
    if tasks_file is None:
        tasks_file = _DEFAULT_DATA / "tasks" / "chunk_0.xml"
    if output_file is None:
        output_file = _DEFAULT_DATA / "results" / "random_baseline_results.csv"

    vehicle_states = load_vehicle_states(vehicles_file)
    tasks = load_tasks(tasks_file)
    offloader = RandomOffloader(seed=seed)
    assignments = tuple(offloader.choose_target() for _ in tasks)
    network_load_by_time = Counter(task.release_time for task in tasks)
    results = simulate_assignments(
        tasks,
        assignments,
        ResourceState(),
        DeterministicChannel(seed),
        vehicle_states=vehicle_states,
        model=ExecutionModel(),
        capacities=ResourceCapacities(),
        network_load_by_time=network_load_by_time,
    )

    write_results(results, output_file)
    return results


def write_results(
    results: list[SimulationResult],
    output_file: str | Path,
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(AssignmentResult.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vehicles",
        default=str(_DEFAULT_DATA / "vehicles" / "chunk_0.xml"),
    )
    parser.add_argument(
        "--tasks",
        default=str(_DEFAULT_DATA / "tasks" / "chunk_0.xml"),
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_DATA / "results" / "random_baseline_results.csv"),
    )
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
