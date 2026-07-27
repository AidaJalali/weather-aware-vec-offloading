from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from infrastructure import TaskRecord, VehicleState, load_tasks, load_vehicle_states
from vec_offloading_env import RewardConfig, VECOffloadingEnv
from weather_scenarios import WeatherScenario


ACTION_VALUES = {
    "LOCAL": -1.0,
    "FOG": 0.0,
    "CLOUD": 1.0,
}


@dataclass(frozen=True)
class RewardComparisonRow:
    mode: str
    scenario: str
    task_count: int
    total_reward: float
    average_reward: float
    average_latency: float
    average_energy: float
    deadline_misses: int
    packet_losses: int


def evaluate_reward_config(
    tasks: Sequence[TaskRecord],
    vehicle_states: dict[
        str | tuple[float, str],
        VehicleState,
    ],
    actions: Sequence[float],
    reward_config: RewardConfig,
    *,
    seed: int,
) -> list[RewardComparisonRow]:
    if len(actions) != len(tasks):
        raise ValueError("one action is required for every task")

    env = VECOffloadingEnv(
        tasks,
        vehicle_states,
        reward_config=reward_config,
    )
    env.reset(seed=seed)
    grouped: dict[str, list[dict]] = defaultdict(list)

    for action in actions:
        _, reward, terminated, _, info = env.step(
            np.asarray([action], dtype=np.float32)
        )
        grouped[info["scenario"]].append(
            {
                "reward": reward,
                "latency": info["latency"],
                "energy": info["total_system_energy"],
                "deadline_missed": info["deadline_missed"],
                "packet_lost": info["packet_lost"],
            }
        )
        if terminated:
            break

    rows: list[RewardComparisonRow] = []
    all_items: list[dict] = []
    scenario_order = [scenario.value for scenario in WeatherScenario]
    for scenario in scenario_order:
        items = grouped.get(scenario, [])
        if not items:
            continue
        all_items.extend(items)
        rows.append(_summarize(reward_config.name, scenario, items))
    rows.append(_summarize(reward_config.name, "ALL", all_items))
    return rows


def _summarize(
    mode: str,
    scenario: str,
    items: Sequence[dict],
) -> RewardComparisonRow:
    count = len(items)
    if count == 0:
        return RewardComparisonRow(
            mode=mode,
            scenario=scenario,
            task_count=0,
            total_reward=0.0,
            average_reward=0.0,
            average_latency=0.0,
            average_energy=0.0,
            deadline_misses=0,
            packet_losses=0,
        )
    total_reward = sum(item["reward"] for item in items)
    return RewardComparisonRow(
        mode=mode,
        scenario=scenario,
        task_count=count,
        total_reward=total_reward,
        average_reward=total_reward / count,
        average_latency=sum(item["latency"] for item in items) / count,
        average_energy=sum(item["energy"] for item in items) / count,
        deadline_misses=sum(
            bool(item["deadline_missed"]) for item in items
        ),
        packet_losses=sum(bool(item["packet_lost"]) for item in items),
    )


def compare_fixed_and_adaptive(
    tasks: Sequence[TaskRecord],
    vehicle_states: dict[
        str | tuple[float, str],
        VehicleState,
    ],
    actions: Sequence[float],
    *,
    seed: int = 37,
    fixed_config: RewardConfig | None = None,
    adaptive_config: RewardConfig | None = None,
) -> list[RewardComparisonRow]:
    fixed = fixed_config or RewardConfig()
    adaptive = adaptive_config or RewardConfig.adaptive_default()
    return [
        *evaluate_reward_config(
            tasks,
            vehicle_states,
            actions,
            fixed,
            seed=seed,
        ),
        *evaluate_reward_config(
            tasks,
            vehicle_states,
            actions,
            adaptive,
            seed=seed,
        ),
    ]


def write_comparison(
    rows: Sequence[RewardComparisonRow],
    output_file: str | Path,
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(RewardComparisonRow.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replay identical offloading actions with fixed and "
            "weather-adaptive reward profiles."
        )
    )
    parser.add_argument(
        "--tasks",
        default="source/data/tasks/chunk_0.xml",
    )
    parser.add_argument(
        "--vehicles",
        default="source/data/vehicles/chunk_0.xml",
    )
    parser.add_argument(
        "--target",
        choices=tuple(ACTION_VALUES),
        default="FOG",
        help="Use the same target for every task during reward comparison.",
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--output",
        default="source/data/results/reward_profile_comparison.csv",
    )
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    vehicle_states = load_vehicle_states(args.vehicles)
    action = ACTION_VALUES[args.target]
    rows = compare_fixed_and_adaptive(
        tasks,
        vehicle_states,
        [action] * len(tasks),
        seed=args.seed,
    )
    write_comparison(rows, args.output)
    print(f"Reward comparison saved to {args.output}")
    for row in rows:
        if row.scenario == "ALL":
            print(
                f"{row.mode}: avg_reward={row.average_reward:.6f}, "
                f"avg_latency={row.average_latency:.6f}, "
                f"avg_energy={row.average_energy:.6f}"
            )


if __name__ == "__main__":
    main()
