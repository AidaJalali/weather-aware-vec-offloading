from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from genetic_offloader_runner import discover_datasets
from infrastructure import ExecutionModel, load_tasks, load_vehicle_states
from offloading_simulator import ResourceCapacities
from weather_scenarios import WeatherScenario


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "datasets"


@dataclass
class ScenarioLoad:
    task_count: int = 0
    simulation_steps: int = 0
    local_compute_seconds: float = 0.0
    active_users: float = 0.0
    mobile_fogs: float = 0.0


def analyze_split(data_root: str | Path, split: str) -> dict[str, dict[str, float]]:
    model = ExecutionModel()
    capacities = ResourceCapacities()
    totals = {scenario.value: ScenarioLoad() for scenario in WeatherScenario}

    for dataset in discover_datasets(data_root, split):
        tasks = load_tasks(dataset.tasks_file)
        states = load_vehicle_states(dataset.vehicles_file)
        tasks_by_time: dict[float, list] = defaultdict(list)
        for task in tasks:
            tasks_by_time[task.release_time].append(task)

        states_by_time: dict[float, list] = defaultdict(list)
        for key, state in states.items():
            if isinstance(key, tuple):
                states_by_time[float(state.time)].append(state)

        for timestep, timestep_states in states_by_time.items():
            if not timestep_states:
                continue
            scenario = timestep_states[0].weather_scenario
            timestep_tasks = tasks_by_time.get(timestep, ())
            load = totals[scenario]
            load.simulation_steps += 1
            load.task_count += len(timestep_tasks)
            load.local_compute_seconds += sum(
                task.exec_time for task in timestep_tasks
            )
            load.active_users += sum(
                state.vehicle_type == "PKW_special" for state in timestep_states
            )
            load.mobile_fogs += sum(
                state.vehicle_type == "LKW_special" for state in timestep_states
            )

    report: dict[str, dict[str, float]] = {}
    for scenario, load in totals.items():
        steps = max(load.simulation_steps, 1)
        average_users = load.active_users / steps
        average_fogs = load.mobile_fogs / steps
        compute_demand = load.local_compute_seconds / steps
        local_equivalent_capacity = average_users * model.local_speedup
        fog_equivalent_capacity = (
            average_fogs * capacities.fog * model.fog_speedup
        )
        cloud_equivalent_capacity = capacities.cloud * model.cloud_speedup
        combined_capacity = (
            local_equivalent_capacity
            + fog_equivalent_capacity
            + cloud_equivalent_capacity
        )
        report[scenario] = {
            "simulation_steps": load.simulation_steps,
            "tasks": load.task_count,
            "tasks_per_second": load.task_count / steps,
            "required_local_compute_seconds_per_second": compute_demand,
            "average_active_users": average_users,
            "average_mobile_fogs": average_fogs,
            "combined_capacity_utilization_lower_bound": (
                compute_demand / max(combined_capacity, 1e-9)
            ),
            "all_local_utilization": (
                compute_demand / max(local_equivalent_capacity, 1e-9)
            ),
            "all_fog_utilization": (
                compute_demand / max(fog_equivalent_capacity, 1e-9)
            ),
            "all_cloud_utilization": (
                compute_demand / max(cloud_equivalent_capacity, 1e-9)
            ),
        }
    return report


def queue_growth_from_results(path: str | Path) -> dict[str, float]:
    by_scenario: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_scenario[row["scenario"]].append(
                (float(row["release_time"]), float(row["queue_delay"]))
            )
    growth: dict[str, float] = {}
    for scenario, values in by_scenario.items():
        values.sort()
        window = max(1, len(values) // 10)
        first = sum(delay for _, delay in values[:window]) / window
        last = sum(delay for _, delay in values[-window:]) / window
        duration = max(values[-1][0] - values[0][0], 1.0)
        growth[scenario] = (last - first) / duration
    return growth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report offered workload and configured compute capacity."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--split", choices=("train", "finetune", "test"), default="test"
    )
    parser.add_argument("--results-file")
    args = parser.parse_args()
    report = analyze_split(args.data_root, args.split)
    growth = (
        queue_growth_from_results(args.results_file)
        if args.results_file
        else {}
    )
    print(
        "scenario tasks/s compute/s users fogs lower_util all_local "
        "all_fog all_cloud queue_growth/s"
    )
    for scenario in WeatherScenario:
        row = report[scenario.value]
        queue_growth = growth.get(scenario.value)
        queue_text = "n/a" if queue_growth is None else f"{queue_growth:.3f}"
        print(
            f"{scenario.value:8s} "
            f"{row['tasks_per_second']:7.2f} "
            f"{row['required_local_compute_seconds_per_second']:9.2f} "
            f"{row['average_active_users']:5.1f} "
            f"{row['average_mobile_fogs']:4.1f} "
            f"{row['combined_capacity_utilization_lower_bound']:10.3f} "
            f"{row['all_local_utilization']:9.3f} "
            f"{row['all_fog_utilization']:8.3f} "
            f"{row['all_cloud_utilization']:10.3f} "
            f"{queue_text:>14s}"
        )


if __name__ == "__main__":
    main()
