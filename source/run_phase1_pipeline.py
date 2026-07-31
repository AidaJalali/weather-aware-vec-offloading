"""Phase 1 pipeline: SUMO mobility → task generation → random baseline → plots.

Usage::

    PYTHONPATH=source python source/run_phase1_pipeline.py --duration 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sumo_pipeline import SumoPipelineConfig, _run_simulation
from weather_scenario_generator import generate_weather_schedule
from random_baseline_simulator import run_random_baseline
from plot_results import plot_all


SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SOURCE_DIR / "data"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR / "sumo"


def run_pipeline(duration: int, seed: int, output_dir: Path, overwrite: bool = False) -> None:
    weather_csv = DEFAULT_DATA_DIR / "weather_scenarios.csv"
    generate_weather_schedule(duration=duration, output_file=weather_csv)
    sumo_config = SumoPipelineConfig(
        duration=duration,
        seed=seed,
        output_dir=output_dir,
        weather_schedule=weather_csv,
        overwrite=overwrite,
    )
    _run_simulation(sumo_config)

    vehicles_dir = output_dir / "vehicles"
    tasks_dir = output_dir / "tasks"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(vehicles_dir.glob("chunk_*.xml"))
    if not chunk_files:
        raise FileNotFoundError(
            f"No chunk files found in {vehicles_dir}. "
            f"Check that the SUMO pipeline produced output."
        )

    all_results: list[Path] = []
    for vehicles_file in chunk_files:
        chunk_name = vehicles_file.stem  # e.g. chunk_0
        tasks_file = tasks_dir / f"{chunk_name}.xml"
        if not tasks_file.exists():
            raise FileNotFoundError(
                f"Missing task file {tasks_file} for vehicle chunk {vehicles_file}"
            )
        results_file = results_dir / f"random_baseline_{chunk_name}.csv"
        run_random_baseline(
            vehicles_file=vehicles_file,
            tasks_file=tasks_file,
            output_file=results_file,
            seed=seed,
        )
        all_results.append(results_file)

    summary_file = results_dir / "random_baseline_summary.csv"
    plot_all(
        results_file=all_results[0],
        summary_file=summary_file,
        output_dir=results_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 pipeline: SUMO → tasks → baseline → plots"
    )
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated data and results.",
    )
    args = parser.parse_args()

    run_pipeline(
        duration=args.duration,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        overwrite=args.overwrite,
    )
    print("Phase 1 pipeline complete.")


if __name__ == "__main__":
    main()
