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


def run_pipeline(duration: int, seed: int, output_dir: Path) -> None:
    """Generate data with SUMO, run random baseline, produce plots."""
    weather_csv = DEFAULT_DATA_DIR / "weather_scenarios.csv"

    # 1. Generate weather schedule
    generate_weather_schedule(duration=duration, output_file=weather_csv)

    # 2. Run SUMO pipeline
    sumo_config = SumoPipelineConfig(
        duration=duration,
        seed=seed,
        output_dir=output_dir,
        weather_schedule=weather_csv,
        overwrite=True,
    )
    _run_simulation(sumo_config)

    # 3. Run random baseline on the first chunk
    vehicles_file = output_dir / "vehicles" / "chunk_0.xml"
    tasks_file = output_dir / "tasks" / "chunk_0.xml"

    if not vehicles_file.exists() or not tasks_file.exists():
        raise FileNotFoundError(
            f"Expected generated data at {vehicles_file} and {tasks_file}. "
            f"Check that the SUMO pipeline produced output."
        )

    results_file = output_dir / "results" / "random_baseline_results.csv"
    run_random_baseline(
        vehicles_file=vehicles_file,
        tasks_file=tasks_file,
        output_file=results_file,
        seed=seed,
    )

    # 4. Generate plots
    summary_file = output_dir / "results" / "random_baseline_summary.csv"
    plot_all(
        results_file=results_file,
        summary_file=summary_file,
        output_dir=output_dir / "results",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1 pipeline: SUMO → tasks → baseline → plots"
    )
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
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
    )
    print("Phase 1 pipeline complete.")


if __name__ == "__main__":
    main()
