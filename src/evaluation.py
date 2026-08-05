from __future__ import annotations

import argparse
from pathlib import Path

from algorithms import GeneticOffloaderConfig
from genetic_offloader_runner import (
    config_from_json,
    config_to_json,
    run_genetic_split,
)
from random_offloader_runner import run_random_split
from sac_pretrained_runner import run_sac_pretrained_split


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "datasets"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "evaluation"
DEFAULT_SAC_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "models" / "sac" / "sac_pretrained_final.zip"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate offloading algorithms on a dataset category."
    )
    parser.add_argument(
        "--algorithm",
        type=str.lower,
        choices=("random", "genetic", "sac_pretrained", "both", "all"),
        default="both",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "finetune", "test"),
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=0)
    parser.add_argument("--batch-window", type=int, default=1)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--genetic-config",
        help="Optional GA config; defaults to the selected results directory.",
    )
    parser.add_argument(
        "--sac-checkpoint",
        default=str(DEFAULT_SAC_CHECKPOINT),
    )
    parser.add_argument("--device", default="cpu")
    return parser


def load_genetic_config(config_path: Path, output_dir: Path) -> GeneticOffloaderConfig:
    if config_path.exists():
        return config_from_json(config_path)

    config = GeneticOffloaderConfig()
    saved_path = output_dir / "genetic_config.json"
    config_to_json(config, saved_path)
    print(f"Genetic config not found at {config_path}; using default GA config.")
    print(f"Default GA config saved to {saved_path}")
    return config


def main() -> None:
    args = build_parser().parse_args()
    results_root = Path(args.results_root)
    max_datasets = args.max_datasets if args.max_datasets > 0 else None
    max_timesteps = args.max_timesteps if args.max_timesteps > 0 else None
    show_progress = not args.no_progress
    config_path = (
        Path(args.genetic_config)
        if args.genetic_config
        else results_root / "genetic" / "genetic_config.json"
    )
    if args.algorithm in ("genetic", "both", "all"):
        genetic_output_dir = results_root / "genetic"
        genetic_config = load_genetic_config(config_path, genetic_output_dir)
    elif config_path.exists():
        genetic_config = config_from_json(config_path)
    else:
        genetic_config = GeneticOffloaderConfig()

    if args.algorithm in ("random", "both", "all"):
        output_dir = results_root / "random"
        rows = run_random_split(
            data_root=args.data_root,
            split=args.split,
            output_file=output_dir / f"{args.split}_random_results.csv",
            summary_file=output_dir / f"{args.split}_random_summary.csv",
            dataset_names=args.datasets,
            max_datasets=max_datasets,
            max_timesteps=max_timesteps,
            batch_window_seconds=args.batch_window,
            capacities=genetic_config.resource_capacities,
            seed=args.seed,
            show_progress=show_progress,
            progress_every=args.progress_every,
        )
        print(f"Random {args.split} run simulated {len(rows)} tasks.")
        print(f"Results saved to {output_dir / f'{args.split}_random_results.csv'}")
        print(f"Summary saved to {output_dir / f'{args.split}_random_summary.csv'}")

    if args.algorithm in ("genetic", "both", "all"):
        output_dir = results_root / "genetic"
        rows = run_genetic_split(
            data_root=args.data_root,
            split=args.split,
            output_file=output_dir / f"{args.split}_genetic_results.csv",
            summary_file=output_dir / f"{args.split}_genetic_summary.csv",
            config=genetic_config,
            dataset_names=args.datasets,
            max_datasets=max_datasets,
            max_timesteps=max_timesteps,
            batch_window_seconds=args.batch_window,
            seed=args.seed,
            show_progress=show_progress,
            progress_every=args.progress_every,
        )
        print(f"Genetic {args.split} run simulated {len(rows)} tasks.")
        print(f"Results saved to {output_dir / f'{args.split}_genetic_results.csv'}")
        print(f"Summary saved to {output_dir / f'{args.split}_genetic_summary.csv'}")

    if args.algorithm in ("sac_pretrained", "all"):
        output_dir = results_root / "sac_pretrained"
        rows = run_sac_pretrained_split(
            data_root=args.data_root,
            split=args.split,
            checkpoint=args.sac_checkpoint,
            output_file=output_dir / f"{args.split}_sac_pretrained_results.csv",
            summary_file=output_dir / f"{args.split}_sac_pretrained_summary.csv",
            dataset_names=args.datasets,
            max_datasets=max_datasets,
            max_timesteps=max_timesteps,
            batch_window_seconds=args.batch_window,
            capacities=genetic_config.resource_capacities,
            seed=args.seed,
            show_progress=show_progress,
            progress_every=args.progress_every,
            device=args.device,
        )
        print(f"SAC_Pretrained {args.split} run simulated {len(rows)} tasks.")
        print(
            "Results saved to "
            f"{output_dir / f'{args.split}_sac_pretrained_results.csv'}"
        )
        print(
            "Summary saved to "
            f"{output_dir / f'{args.split}_sac_pretrained_summary.csv'}"
        )


if __name__ == "__main__":
    main()
