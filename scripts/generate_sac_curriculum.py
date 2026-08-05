#!/usr/bin/env python3
"""Generate independently seeded train, finetune, and test datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import random
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as Et
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from weather_scenarios import WeatherScenario


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "src"
SUMO_PIPELINE = SOURCE_DIR / "sumo_pipeline.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "datasets"
SCENARIOS = tuple(WeatherScenario)
CATEGORIES = ("train", "finetune", "test")
STAGES = ("static", "slow", "fast", "random")


@dataclass(frozen=True)
class WeatherBlock:
    scenario: str
    duration: int
    seed: int


@dataclass(frozen=True)
class DatasetPlan:
    name: str
    category: str
    stage: str
    blocks: tuple[WeatherBlock, ...]

    @property
    def duration(self) -> int:
        return sum(block.duration for block in self.blocks)

    def output_dir(self, root: Path) -> Path:
        return root / self.category / self.name


def _unique_orders(
    values: Sequence[WeatherScenario],
    count: int,
    rng: random.Random,
) -> list[tuple[WeatherScenario, ...]]:
    orders: list[tuple[WeatherScenario, ...]] = []
    seen: set[tuple[WeatherScenario, ...]] = set()
    while len(orders) < count:
        shuffled = list(values)
        rng.shuffle(shuffled)
        order = tuple(shuffled)
        if order not in seen:
            seen.add(order)
            orders.append(order)
    return orders


def _extend_unique_orders(
    orders: list[tuple[WeatherScenario, ...]],
    values: Sequence[WeatherScenario],
    count: int,
    rng: random.Random,
) -> list[tuple[WeatherScenario, ...]]:
    seen = set(orders)
    while len(orders) < count:
        shuffled = list(values)
        rng.shuffle(shuffled)
        order = tuple(shuffled)
        if order not in seen:
            seen.add(order)
            orders.append(order)
    return orders


def build_plan(base_seed: int = 20_000) -> list[DatasetPlan]:
    rng = random.Random(base_seed)
    next_seed = base_seed

    def blocks_for(
        order: Sequence[WeatherScenario],
        duration: int,
    ) -> tuple[WeatherBlock, ...]:
        nonlocal next_seed
        blocks: list[WeatherBlock] = []
        for scenario in order:
            next_seed += 1
            blocks.append(
                WeatherBlock(
                    scenario=scenario.value,
                    duration=duration,
                    seed=next_seed,
                )
            )
        return tuple(blocks)

    def random_weather_blocks(total_duration: int) -> tuple[WeatherBlock, ...]:
        durations: list[int] = []
        remaining = total_duration
        while remaining > 0:
            if remaining <= 200:
                durations.append(remaining)
                break
            duration = rng.randint(100, min(200, remaining - 100))
            durations.append(duration)
            remaining -= duration

        scenarios = list(SCENARIOS)
        rng.shuffle(scenarios)
        while len(scenarios) < len(durations):
            choices = [scenario for scenario in SCENARIOS if scenario != scenarios[-1]]
            scenarios.append(rng.choice(choices))

        blocks: list[WeatherBlock] = []
        nonlocal next_seed
        for scenario, duration in zip(scenarios, durations):
            next_seed += 1
            blocks.append(
                WeatherBlock(
                    scenario=scenario.value,
                    duration=duration,
                    seed=next_seed,
                )
            )
        return tuple(blocks)

    train: list[DatasetPlan] = []
    for scenario in SCENARIOS:
        for index in range(1, 3):
            train.append(
                DatasetPlan(
                    name=f"train_{scenario.value.lower()}_{index}",
                    category="train",
                    stage="static",
                    blocks=blocks_for((scenario,), 1000),
                )
            )

    slow_orders = _unique_orders(SCENARIOS, 5, rng)
    for index, order in enumerate(slow_orders[:4], start=1):
        train.append(
            DatasetPlan(
                name=f"train_slow_mix_{index}",
                category="train",
                stage="slow",
                blocks=blocks_for(order, 250),
            )
        )

    doubled = tuple(scenario for scenario in SCENARIOS for _ in range(2))
    fast_orders = _unique_orders(doubled, 5, rng)
    # Preserve all later dataset seeds after removing the former fast train sets.
    next_seed += 32

    # Keep the already generated mixed-test seeds immediately after training.
    test_fast = DatasetPlan(
        name="test_fast_mix",
        category="test",
        stage="fast",
        blocks=blocks_for(fast_orders[4], 125),
    )
    test_slow = DatasetPlan(
        name="test_slow_mix",
        category="test",
        stage="slow",
        blocks=blocks_for(slow_orders[4], 250),
    )

    _extend_unique_orders(slow_orders, SCENARIOS, 7, rng)
    _extend_unique_orders(fast_orders, doubled, 7, rng)

    finetune: list[DatasetPlan] = []
    for scenario in SCENARIOS:
        finetune.append(
            DatasetPlan(
                name=f"finetune_{scenario.value.lower()}",
                category="finetune",
                stage="static",
                blocks=blocks_for((scenario,), 1000),
            )
        )
    for index, order in enumerate(slow_orders[5:7], start=1):
        finetune.append(
            DatasetPlan(
                name=f"finetune_slow_mix_{index}",
                category="finetune",
                stage="slow",
                blocks=blocks_for(order, 250),
            )
        )
    for index, order in enumerate(fast_orders[5:7], start=1):
        finetune.append(
            DatasetPlan(
                name=f"finetune_fast_mix_{index}",
                category="finetune",
                stage="fast",
                blocks=blocks_for(order, 125),
            )
        )

    test_static = [
        DatasetPlan(
            name=f"test_{scenario.value.lower()}",
            category="test",
            stage="static",
            blocks=blocks_for((scenario,), 1000),
        )
        for scenario in SCENARIOS
    ]
    test_random = [
        DatasetPlan(
            name=f"test_random_mix_{index}",
            category="test",
            stage="random",
            blocks=random_weather_blocks(1000),
        )
        for index in range(1, 3)
    ]
    plans = [
        *train,
        *finetune,
        *test_static,
        test_fast,
        test_slow,
        *test_random,
    ]
    validate_plan(plans)
    return plans


def validate_plan(plans: Sequence[DatasetPlan]) -> None:
    expected_counts = {
        "train": {"static": 8, "slow": 4, "fast": 0, "random": 0},
        "finetune": {"static": 4, "slow": 2, "fast": 2, "random": 0},
        "test": {"static": 4, "slow": 1, "fast": 1, "random": 2},
    }
    counts = {
        category: {
            stage: sum(
                plan.category == category and plan.stage == stage
                for plan in plans
            )
            for stage in STAGES
        }
        for category in CATEGORIES
    }
    if counts != expected_counts:
        raise ValueError(f"Invalid dataset plan: {counts}")
    if any(plan.duration != 1000 for plan in plans):
        raise ValueError("Every dataset must contain 1000 timesteps")
    if any(not plan.name.startswith(f"{plan.category}_") for plan in plans):
        raise ValueError("Every dataset name must start with its category")

    seeds = [block.seed for plan in plans for block in plan.blocks]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Every weather block must use a unique seed")

    for plan in plans:
        scenario_counts = {
            scenario.value: sum(
                block.scenario == scenario.value for block in plan.blocks
            )
            for scenario in SCENARIOS
        }
        if plan.stage == "slow" and set(scenario_counts.values()) != {1}:
            raise ValueError(f"Invalid slow weather order in {plan.name}")
        if plan.stage == "fast" and set(scenario_counts.values()) != {2}:
            raise ValueError(f"Invalid fast weather order in {plan.name}")
        if plan.stage == "random":
            if any(count == 0 for count in scenario_counts.values()):
                raise ValueError(f"Invalid random weather set in {plan.name}")
            if not all(100 <= block.duration <= 200 for block in plan.blocks):
                raise ValueError(f"Invalid random block duration in {plan.name}")
            if any(
                current.scenario == following.scenario
                for current, following in zip(plan.blocks, plan.blocks[1:])
            ):
                raise ValueError(f"Repeated adjacent weather in {plan.name}")


def _write_static_weather(path: Path, scenario: str, duration: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "scenario"))
        for timestep in range(1, duration + 1):
            writer.writerow((timestep, scenario))


def _run_sumo_block(
    block: WeatherBlock,
    output_dir: Path,
    users: int,
    mobile_fogs: int,
) -> None:
    weather_file = output_dir.parent / f"weather_{block.seed}.csv"
    _write_static_weather(weather_file, block.scenario, block.duration)
    command = [
        sys.executable,
        str(SUMO_PIPELINE),
        "--duration",
        str(block.duration),
        "--users",
        str(users),
        "--mobile-fogs",
        str(mobile_fogs),
        "--seed",
        str(block.seed),
        "--chunk-size",
        str(block.duration),
        "--weather-schedule",
        str(weather_file),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]
    environment = {**os.environ, "PYTHONPATH": str(SOURCE_DIR)}
    subprocess.run(command, check=True, env=environment)


def _chunk_file(block_dir: Path, kind: str) -> Path:
    candidates = sorted((block_dir / kind).glob("chunk_0.xml*"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one {kind} chunk in {block_dir}, found {len(candidates)}"
        )
    return candidates[0]


def _parse_xml(path: Path) -> Et.ElementTree:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as handle:
            return Et.ElementTree(Et.fromstring(handle.read()))
    return Et.parse(path)


def _append_shifted_chunk(
    destination: Et.Element,
    source_path: Path,
    *,
    kind: str,
    offset: int,
    block_index: int,
    block: WeatherBlock,
) -> None:
    source_root = _parse_xml(source_path).getroot()
    timesteps = source_root.findall("timestep")
    if len(timesteps) != block.duration:
        raise ValueError(
            f"Seed {block.seed} produced {len(timesteps)} timesteps; "
            f"expected {block.duration}"
        )

    for local_index, timestep in enumerate(timesteps, start=1):
        local_time = int(float(timestep.get("time", "0")))
        if local_time != local_index:
            raise ValueError(
                f"Unexpected timestep {local_time} in seed {block.seed}"
            )
        timestep.set("time", str(offset + local_time))
        if kind == "tasks":
            for task in timestep.findall("task"):
                task.set(
                    "id",
                    f"S{block.seed}_C{block_index}_{task.get('id')}",
                )
                deadline = float(task.get("deadline", "0")) + offset
                task.set("deadline", f"{deadline:.2f}")
        destination.append(timestep)


def _write_gzip_xml(root: Et.Element, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Et.indent(root, space="    ", level=0)
    with gzip.open(temporary, "wb") as handle:
        Et.ElementTree(root).write(
            handle,
            encoding="utf-8",
            xml_declaration=True,
        )
    temporary.replace(path)


def _validate_merged_dataset(
    plan: DatasetPlan,
    vehicle_root: Et.Element,
    task_root: Et.Element,
) -> None:
    vehicle_timesteps = vehicle_root.findall("timestep")
    task_timesteps = task_root.findall("timestep")
    expected_times = list(range(1, plan.duration + 1))
    vehicle_times = [int(float(item.get("time", "0"))) for item in vehicle_timesteps]
    task_times = [int(float(item.get("time", "0"))) for item in task_timesteps]
    if vehicle_times != expected_times or task_times != expected_times:
        raise ValueError(f"Merged timesteps are incomplete in {plan.name}")

    expected_scenarios: dict[int, str] = {}
    current_time = 1
    for block in plan.blocks:
        for _ in range(block.duration):
            expected_scenarios[current_time] = block.scenario
            current_time += 1

    vehicles_by_time: dict[int, set[str]] = {}
    for timestep in vehicle_timesteps:
        time = int(float(timestep.get("time", "0")))
        vehicles = timestep.findall("vehicle")
        vehicles_by_time[time] = {
            vehicle.get("id", "") for vehicle in vehicles
        }
        if any(
            vehicle.get("weather_scenario") != expected_scenarios[time]
            for vehicle in vehicles
        ):
            raise ValueError(f"Vehicle weather mismatch at time {time}")

    task_ids: set[str] = set()
    for timestep in task_timesteps:
        time = int(float(timestep.get("time", "0")))
        for task in timestep.findall("task"):
            task_id = task.get("id", "")
            if task_id in task_ids:
                raise ValueError(f"Duplicate task ID in {plan.name}: {task_id}")
            task_ids.add(task_id)
            if task.get("creator") not in vehicles_by_time[time]:
                raise ValueError(f"Missing task creator at time {time}: {task_id}")
            if task.get("weather_scenario") != expected_scenarios[time]:
                raise ValueError(f"Task weather mismatch at time {time}: {task_id}")
            if float(task.get("deadline", "0")) <= time:
                raise ValueError(f"Invalid shifted deadline for task {task_id}")


def _write_dataset_schedule(plan: DatasetPlan, dataset_dir: Path) -> None:
    with (dataset_dir / "weather_schedule.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("time", "scenario", "chunk_seed"))
        timestep = 1
        for block in plan.blocks:
            for _ in range(block.duration):
                writer.writerow((timestep, block.scenario, block.seed))
                timestep += 1


def generate_dataset(
    plan: DatasetPlan,
    output_root: Path,
    *,
    users: int,
    mobile_fogs: int,
    overwrite: bool,
) -> None:
    dataset_dir = plan.output_dir(output_root)
    expected_files = (
        dataset_dir / "vehicles" / "chunk_0.xml.gz",
        dataset_dir / "tasks" / "chunk_0.xml.gz",
        dataset_dir / "metadata.json",
    )
    if all(path.exists() for path in expected_files) and not overwrite:
        _write_metadata(plan, dataset_dir, users, mobile_fogs)
        print(f"SKIP {plan.name}: already complete at {dataset_dir}")
        return
    if dataset_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Incomplete dataset directory exists: {dataset_dir}. "
            "Inspect it or rerun with --overwrite."
        )

    vehicle_root = Et.Element("fcd-export", {"version": "1.0"})
    task_root = Et.Element("fcd-export", {"version": "1.0"})
    with tempfile.TemporaryDirectory(
        prefix=f"vec_{plan.name}_",
    ) as temporary_dir:
        staging_root = Path(temporary_dir)
        offset = 0
        for block_index, block in enumerate(plan.blocks, start=1):
            print(
                f"  block {block_index}/{len(plan.blocks)} "
                f"weather={block.scenario} duration={block.duration} "
                f"seed={block.seed}"
            )
            block_dir = staging_root / f"block_{block_index}"
            _run_sumo_block(block, block_dir, users, mobile_fogs)
            _append_shifted_chunk(
                vehicle_root,
                _chunk_file(block_dir, "vehicles"),
                kind="vehicles",
                offset=offset,
                block_index=block_index,
                block=block,
            )
            _append_shifted_chunk(
                task_root,
                _chunk_file(block_dir, "tasks"),
                kind="tasks",
                offset=offset,
                block_index=block_index,
                block=block,
            )
            offset += block.duration

    _validate_merged_dataset(plan, vehicle_root, task_root)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_gzip_xml(
        vehicle_root,
        dataset_dir / "vehicles" / "chunk_0.xml.gz",
    )
    _write_gzip_xml(
        task_root,
        dataset_dir / "tasks" / "chunk_0.xml.gz",
    )
    _write_dataset_schedule(plan, dataset_dir)
    _write_metadata(plan, dataset_dir, users, mobile_fogs)
    print(f"DONE {plan.name}: {dataset_dir}")


def _write_metadata(
    plan: DatasetPlan,
    dataset_dir: Path,
    users: int,
    mobile_fogs: int,
) -> None:
    metadata = {
        **asdict(plan),
        "duration": plan.duration,
        "users": users,
        "mobile_fogs": mobile_fogs,
        "independently_seeded_blocks": True,
    }
    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def _update_manifest(
    plans: Sequence[DatasetPlan],
    output_root: Path,
    users: int,
    mobile_fogs: int,
) -> None:
    path = output_root / "manifest.json"
    manifest = {"datasets": {}}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    datasets = manifest.setdefault("datasets", {})
    selected_categories = {plan.category for plan in plans}
    for name, dataset in list(datasets.items()):
        if dataset.get("category") in selected_categories:
            del datasets[name]
    for plan in plans:
        datasets[plan.name] = {
            **asdict(plan),
            "duration": plan.duration,
            "path": str(plan.output_dir(output_root).relative_to(PROJECT_ROOT)),
        }
    manifest.update(
        {
            "users": users,
            "mobile_fogs": mobile_fogs,
            "categories": list(CATEGORIES),
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _selected_plans(
    plans: Sequence[DatasetPlan],
    group: str,
) -> list[DatasetPlan]:
    if group == "all":
        return list(plans)
    return [plan for plan in plans if plan.category == group]


def print_plan(plans: Sequence[DatasetPlan], output_root: Path) -> None:
    print(f"Output root: {output_root}")
    for plan in plans:
        order = " -> ".join(
            f"{block.scenario}({block.duration}s,seed={block.seed})"
            for block in plan.blocks
        )
        print(f"{plan.category:8s} {plan.stage:6s} {plan.name:24s}: {order}")
    print(
        f"Datasets: {len(plans)}, "
        f"timesteps: {sum(plan.duration for plan in plans)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the independently seeded SAC curriculum datasets."
    )
    parser.add_argument(
        "--group",
        choices=(*CATEGORIES, "all"),
        default="all",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--base-seed", type=int, default=20_000)
    parser.add_argument("--users", type=int, default=12)
    parser.add_argument("--mobile-fogs", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    plans = _selected_plans(build_plan(args.base_seed), args.group)
    print_plan(plans, output_root)
    if args.dry_run:
        return

    for index, plan in enumerate(plans, start=1):
        print(f"\nDataset {index}/{len(plans)}: {plan.name}")
        generate_dataset(
            plan,
            output_root,
            users=args.users,
            mobile_fogs=args.mobile_fogs,
            overwrite=args.overwrite,
        )
    _update_manifest(plans, output_root, args.users, args.mobile_fogs)


if __name__ == "__main__":
    main()
