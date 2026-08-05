from __future__ import annotations

import argparse
from pathlib import Path

from algorithms import GeneticOffloaderConfig
from genetic_offloader_runner import config_to_json, run_genetic_split


SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = SOURCE_DIR.parent / "data" / "datasets"
DEFAULT_OUTPUT_DIR = SOURCE_DIR.parent / "outputs" / "evaluation" / "genetic"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the genetic offloader on training datasets."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=0)
    parser.add_argument("--batch-window", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--max-generations", type=int, default=25)
    parser.add_argument("--time-limit-seconds", type=float, default=0.03)
    parser.add_argument("--mutation-rate", type=float, default=0.10)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = GeneticOffloaderConfig(
        population_size=args.population_size,
        max_generations=args.max_generations,
        time_limit_seconds=args.time_limit_seconds,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )
    config_path = output_dir / "genetic_config.json"
    config_to_json(config, config_path)

    max_datasets = args.max_datasets if args.max_datasets > 0 else None
    max_timesteps = args.max_timesteps if args.max_timesteps > 0 else None
    rows = run_genetic_split(
        data_root=args.data_root,
        split="train",
        output_file=output_dir / "train_genetic_results.csv",
        summary_file=output_dir / "train_genetic_summary.csv",
        config=config,
        dataset_names=args.datasets,
        max_datasets=max_datasets,
        max_timesteps=max_timesteps,
        batch_window_seconds=args.batch_window,
        seed=args.seed,
        show_progress=not args.no_progress,
        progress_every=args.progress_every,
    )

    print(f"Genetic train run simulated {len(rows)} tasks.")
    print(f"Config saved to {config_path}")
    print(f"Results saved to {output_dir / 'train_genetic_results.csv'}")
    print(f"Summary saved to {output_dir / 'train_genetic_summary.csv'}")


if __name__ == "__main__":
    main()
