from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from discrete_sac import DiscreteSACAgent, DiscreteSACConfig
from genetic_offloader_runner import batch_tasks, discover_datasets
from infrastructure import load_tasks, load_vehicle_states
from vec_offloading_env import ObservationScale, RewardConfig, VECOffloadingEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "datasets" / "train"
DEFAULT_VALIDATION_ROOT = PROJECT_ROOT / "data" / "datasets"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "discrete_sac"
STAGE_ORDER = ("static", "slow")
STATIC_WEATHER_ORDER = {"BASE": 0, "RAIN": 1, "SNOW": 2, "FOG": 3}


@dataclass(frozen=True)
class CurriculumDataset:
    name: str
    stage: str
    path: Path
    task_files: tuple[Path, ...]
    vehicle_files: tuple[Path, ...]


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    stage: str
    tasks: tuple
    vehicle_states: dict


@dataclass(frozen=True)
class ValidationResult:
    average_reward: float
    deadline_miss_rate: float
    packet_loss_rate: float
    average_latency: float
    average_energy: float
    total_tasks: int


def _chunk_files(dataset_dir: Path, kind: str) -> tuple[Path, ...]:
    files = sorted((dataset_dir / kind).glob("chunk_*.xml"))
    files.extend(sorted((dataset_dir / kind).glob("chunk_*.xml.gz")))
    if not files:
        raise FileNotFoundError(f"No {kind} chunks found in {dataset_dir}")
    return tuple(files)


def _dataset_metadata(path: Path) -> dict:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _dataset_sort_key(dataset: CurriculumDataset) -> tuple[int, int, str]:
    stage_index = STAGE_ORDER.index(dataset.stage)
    scenario_index = len(STATIC_WEATHER_ORDER)
    if dataset.stage == "static":
        scenario = _dataset_metadata(dataset.path)["blocks"][0]["scenario"]
        scenario_index = STATIC_WEATHER_ORDER.get(
            scenario, len(STATIC_WEATHER_ORDER)
        )
    return stage_index, scenario_index, dataset.name


def discover_curriculum(
    data_root: str | Path,
    stages: Sequence[str] = STAGE_ORDER,
) -> list[CurriculumDataset]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Training data directory not found: {root}")
    datasets: list[CurriculumDataset] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata = _dataset_metadata(folder)
        stage = metadata.get("stage")
        if stage in stages:
            datasets.append(
                CurriculumDataset(
                    name=folder.name,
                    stage=stage,
                    path=folder,
                    task_files=_chunk_files(folder, "tasks"),
                    vehicle_files=_chunk_files(folder, "vehicles"),
                )
            )
    if not datasets:
        raise FileNotFoundError(f"No requested training datasets found in {root}")
    return sorted(datasets, key=_dataset_sort_key)


def load_curriculum_dataset(dataset: CurriculumDataset) -> LoadedDataset:
    tasks = []
    states = {}
    for path in dataset.task_files:
        tasks.extend(load_tasks(path))
    for path in dataset.vehicle_files:
        states.update(load_vehicle_states(path))
    return LoadedDataset(dataset.name, dataset.stage, tuple(tasks), states)


def derive_observation_scale(
    datasets: Sequence[CurriculumDataset],
) -> ObservationScale:
    loaded_datasets = []
    for dataset in datasets:
        loaded = load_curriculum_dataset(dataset)
        loaded_datasets.append((loaded.tasks, loaded.vehicle_states))
    return ObservationScale.from_datasets(loaded_datasets, percentile=99.0)


def _limited_validation_data(
    data_root: str | Path,
    max_timesteps: int,
) -> list[LoadedDataset]:
    datasets = discover_datasets(data_root, "finetune")
    loaded: list[LoadedDataset] = []
    for dataset in datasets:
        tasks = load_tasks(dataset.tasks_file)
        selected = tuple(
            task
            for batch in batch_tasks(
                tasks,
                batch_window_seconds=1,
                max_timesteps=max_timesteps,
            )
            for task in batch
        )
        loaded.append(
            LoadedDataset(
                name=dataset.name,
                stage=dataset.group,
                tasks=selected,
                vehicle_states=load_vehicle_states(dataset.vehicles_file),
            )
        )
    return loaded


def make_environment(
    dataset: LoadedDataset,
    observation_scale: ObservationScale,
) -> VECOffloadingEnv:
    return VECOffloadingEnv(
        dataset.tasks,
        dataset.vehicle_states,
        observation_scale=observation_scale,
        reward_config=RewardConfig(),
    )


def validate_agent(
    agent: DiscreteSACAgent,
    datasets: Sequence[LoadedDataset],
    observation_scale: ObservationScale,
    *,
    seed: int,
) -> ValidationResult:
    rewards: list[float] = []
    dataset_average_rewards: list[float] = []
    latencies: list[float] = []
    energies: list[float] = []
    deadline_misses = 0
    packet_losses = 0
    for dataset_index, dataset in enumerate(datasets):
        dataset_rewards: list[float] = []
        environment = make_environment(dataset, observation_scale)
        observation, _ = environment.reset(seed=seed + dataset_index)
        while True:
            action = agent.select_action(observation, deterministic=True)
            observation, reward, terminated, _, info = environment.step(action)
            rewards.append(reward)
            dataset_rewards.append(reward)
            latencies.append(float(info["latency"]))
            energies.append(float(info["total_system_energy"]))
            deadline_misses += int(info["deadline_missed"])
            packet_losses += int(info["packet_lost"])
            if terminated:
                break
        dataset_average_rewards.append(
            sum(dataset_rewards) / len(dataset_rewards)
        )
    count = len(rewards)
    return ValidationResult(
        average_reward=(
            sum(dataset_average_rewards) / len(dataset_average_rewards)
        ),
        deadline_miss_rate=deadline_misses / count,
        packet_loss_rate=packet_losses / count,
        average_latency=sum(latencies) / count,
        average_energy=sum(energies) / count,
        total_tasks=count,
    )


def _write_history(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_curriculum(
    *,
    data_root: str | Path,
    validation_root: str | Path,
    output_dir: str | Path,
    stages: Sequence[str],
    epochs: int,
    validation_timesteps: int,
    steps_per_dataset: int | None,
    config: DiscreteSACConfig,
    device: str,
) -> Path:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if validation_timesteps <= 0:
        raise ValueError("validation_timesteps must be positive")
    if steps_per_dataset is not None and steps_per_dataset <= 0:
        raise ValueError("steps_per_dataset must be positive when provided")
    datasets = discover_curriculum(data_root, stages)
    output = Path(output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    print("Deriving observation scales from all pretraining datasets...")
    observation_scale = derive_observation_scale(datasets)
    (output / "observation_scale.json").write_text(
        json.dumps(asdict(observation_scale), indent=2) + "\n",
        encoding="utf-8",
    )
    validation_data = _limited_validation_data(
        validation_root,
        validation_timesteps,
    )
    first = load_curriculum_dataset(datasets[0])
    first_environment = make_environment(first, observation_scale)
    agent = DiscreteSACAgent(
        first_environment.observation_space.shape[0],
        first_environment.action_space.n,
        config=config,
        device=device,
    )

    rng = random.Random(config.seed)
    history: list[dict] = []
    best_reward = float("-inf")
    best_path = output / "sac_discrete_best.pt"

    for epoch in range(1, epochs + 1):
        order = list(datasets)
        rng.shuffle(order)
        print(
            f"\nEpoch {epoch}/{epochs}: "
            + " -> ".join(dataset.name for dataset in order)
        )

        for dataset_index, descriptor in enumerate(order, start=1):
            loaded = load_curriculum_dataset(descriptor)
            tasks = loaded.tasks
            if steps_per_dataset is not None:
                tasks = tasks[:steps_per_dataset]
                loaded = LoadedDataset(
                    loaded.name,
                    loaded.stage,
                    tasks,
                    loaded.vehicle_states,
                )
            environment = make_environment(loaded, observation_scale)
            observation, _ = environment.reset(
                seed=config.seed + epoch * 100 + dataset_index
            )
            episode_reward = 0.0
            episode_updates = 0
            for _ in range(len(tasks)):
                if agent.environment_steps < config.learning_starts:
                    action = agent.random_action()
                else:
                    action = agent.select_action(observation, deterministic=False)
                next_observation, reward, terminated, _, info = environment.step(
                    action
                )
                agent.observe(
                    observation,
                    action,
                    reward,
                    next_observation,
                    terminated,
                    str(info["scenario"]),
                )
                if agent.ready_to_update():
                    agent.update()
                    episode_updates += config.gradient_steps
                observation = next_observation
                episode_reward += reward
                if terminated:
                    break

            validation = validate_agent(
                agent,
                validation_data,
                observation_scale,
                seed=config.seed,
            )
            row = {
                "epoch": epoch,
                "dataset_index": dataset_index,
                "dataset": loaded.name,
                "training_tasks": len(tasks),
                "training_average_reward": episode_reward / len(tasks),
                "gradient_updates": episode_updates,
                **{
                    f"validation_{key}": value
                    for key, value in asdict(validation).items()
                },
            }
            history.append(row)
            _write_history(history, output / "validation_history.csv")
            print(
                f"  {dataset_index:02d}/{len(order)} {loaded.name}: "
                f"train_reward={row['training_average_reward']:.4f} "
                f"val_reward={validation.average_reward:.4f} "
                f"miss={validation.deadline_miss_rate:.3f} "
                f"loss={validation.packet_loss_rate:.3f} "
                f"updates={episode_updates}"
            )

            if validation.average_reward > best_reward:
                best_reward = validation.average_reward
                agent.save(
                    best_path,
                    observation_scale=asdict(observation_scale),
                    metadata={
                        "epoch": epoch,
                        "dataset": loaded.name,
                        "validation": asdict(validation),
                        "selection_metric": "average_bounded_reward",
                    },
                )

        if epoch % 2 == 0:
            checkpoint = checkpoints / f"epoch_{epoch:02d}.pt"
            agent.save(
                checkpoint,
                observation_scale=asdict(observation_scale),
                metadata={"epoch": epoch},
            )
            print(f"Epoch checkpoint: {checkpoint}")

    final_path = output / "sac_discrete_final.pt"
    agent.save(
        final_path,
        observation_scale=asdict(observation_scale),
        metadata={"epoch": epochs, "best_validation_reward": best_reward},
    )
    (output / "training_state.json").write_text(
        json.dumps(
            {
                "epochs": epochs,
                "environment_steps": agent.environment_steps,
                "gradient_updates": agent.gradient_updates,
                "best_model": str(best_path),
                "best_validation_reward": best_reward,
                "final_model": str(final_path),
                "replay_weather_counts": agent.replay.weather_counts(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nBest validation model: {best_path}")
    print(f"Final model: {final_path}")
    return best_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train categorical SAC with shuffled weather datasets."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--validation-root", default=str(DEFAULT_VALIDATION_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=list(STAGE_ORDER),
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--validation-timesteps", type=int, default=300)
    parser.add_argument("--steps-per-dataset", type=int, default=0)
    parser.add_argument("--replay-capacity", type=int, default=500_000)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--list", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = discover_curriculum(args.data_root, args.stages)
    if args.list:
        for dataset in datasets:
            print(f"{dataset.stage:6s} {dataset.name} {dataset.path}")
        return
    config = DiscreteSACConfig(
        replay_capacity=args.replay_capacity,
        gamma=args.gamma,
        seed=args.seed,
    )
    train_curriculum(
        data_root=args.data_root,
        validation_root=args.validation_root,
        output_dir=args.output_dir,
        stages=args.stages,
        epochs=args.epochs,
        validation_timesteps=args.validation_timesteps,
        steps_per_dataset=args.steps_per_dataset or None,
        config=config,
        device=args.device,
    )


if __name__ == "__main__":
    main()
