from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vec-cache")

from stable_baselines3 import SAC

from infrastructure import load_tasks, load_vehicle_states
from vec_offloading_env import RewardConfig, VECOffloadingEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "datasets" / "train"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "sac"
STAGE_ORDER = ("static", "slow")
CHECKPOINT_INTERVAL = 4
STATIC_WEATHER_ORDER = {"BASE": 0, "RAIN": 1, "SNOW": 2, "FOG": 3}


@dataclass(frozen=True)
class CurriculumDataset:
    name: str
    stage: str
    path: Path
    task_files: tuple[Path, ...]
    vehicle_files: tuple[Path, ...]


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
        metadata = _dataset_metadata(dataset.path)
        scenario = metadata["blocks"][0]["scenario"]
        scenario_index = STATIC_WEATHER_ORDER.get(
            scenario,
            len(STATIC_WEATHER_ORDER),
        )
    return stage_index, scenario_index, dataset.name


def discover_curriculum(
    data_root: str | Path,
    stages: Sequence[str] = STAGE_ORDER,
) -> list[CurriculumDataset]:
    root = Path(data_root)
    datasets: list[CurriculumDataset] = []
    if not root.exists():
        raise FileNotFoundError(f"Training data directory not found: {root}")

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


def load_environment(dataset: CurriculumDataset) -> VECOffloadingEnv:
    tasks = []
    vehicle_states = {}
    for path in dataset.task_files:
        tasks.extend(load_tasks(path))
    for path in dataset.vehicle_files:
        vehicle_states.update(load_vehicle_states(path))
    return VECOffloadingEnv(
        tasks,
        vehicle_states,
        reward_config=RewardConfig(),
    )


def _checkpoint_path(output_dir: Path, name: str) -> Path:
    return output_dir / "checkpoints" / name


def _save_training_state(
    output_dir: Path,
    *,
    stage: str,
    dataset: str,
    checkpoint: Path,
    model: SAC,
) -> None:
    state = {
        "last_stage": stage,
        "last_dataset": dataset,
        "checkpoint": str(checkpoint.with_suffix(".zip")),
        "total_training_steps": model.num_timesteps,
    }
    (output_dir / "training_state.json").write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )


def train_curriculum(
    *,
    data_root: str | Path,
    output_dir: str | Path,
    stages: Sequence[str],
    seed: int,
    steps_per_dataset: int | None,
    resume: str | Path | None,
    device: str,
) -> Path:
    datasets = discover_curriculum(data_root, stages)
    output = Path(output_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    first_environment = load_environment(datasets[0])
    if resume is None:
        model = SAC(
            "MlpPolicy",
            first_environment,
            policy_kwargs={"net_arch": [64, 64]},
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=128,
            train_freq=4,
            gradient_steps=1,
            seed=seed,
            device=device,
            verbose=1,
        )
    else:
        model = SAC.load(
            str(resume),
            env=first_environment,
            device=device,
        )

    for dataset_index, dataset in enumerate(datasets, start=1):
        environment = (
            first_environment
            if dataset_index == 1
            else load_environment(dataset)
        )
        model.set_env(environment, force_reset=True)
        training_steps = len(environment.tasks)
        if steps_per_dataset is not None:
            training_steps = min(training_steps, steps_per_dataset)

        print(
            f"\nDataset {dataset_index}/{len(datasets)}: "
            f"stage={dataset.stage} name={dataset.name} "
            f"tasks={len(environment.tasks)} train_steps={training_steps}"
        )
        model.learn(
            total_timesteps=training_steps,
            reset_num_timesteps=False,
            progress_bar=False,
        )

        if dataset_index % CHECKPOINT_INTERVAL == 0:
            checkpoint = _checkpoint_path(
                output,
                f"checkpoint_{dataset_index:02d}_{dataset.name}",
            )
            model.save(checkpoint)
            _save_training_state(
                output,
                stage=dataset.stage,
                dataset=dataset.name,
                checkpoint=checkpoint,
                model=model,
            )
            print(f"Checkpoint: {checkpoint.with_suffix('.zip')}")

    final_path = output / "sac_pretrained_final"
    model.save(final_path)
    print(f"\nFinal SAC model: {final_path.with_suffix('.zip')}")
    return final_path.with_suffix(".zip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train one small SAC model with static and slow-mixed stages."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_ORDER,
        default=list(STAGE_ORDER),
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument(
        "--steps-per-dataset",
        type=int,
        default=0,
        help="Limit training steps per dataset; 0 uses every generated task.",
    )
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--list", action="store_true", dest="list_only")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = discover_curriculum(args.data_root, args.stages)
    if args.list_only:
        for dataset in datasets:
            print(f"{dataset.stage:6s} {dataset.name:12s} {dataset.path}")
        return

    steps = args.steps_per_dataset if args.steps_per_dataset > 0 else None
    train_curriculum(
        data_root=args.data_root,
        output_dir=args.output_dir,
        stages=args.stages,
        seed=args.seed,
        steps_per_dataset=steps,
        resume=args.resume,
        device=args.device,
    )


if __name__ == "__main__":
    main()
