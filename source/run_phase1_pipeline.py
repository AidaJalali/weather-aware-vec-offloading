from __future__ import annotations

import argparse
import os
from pathlib import Path

from mobility_generator import generate_mobility
from plot_results import plot_all
from random_baseline_simulator import run_random_baseline
from task_and_user_generator import Config as TaskGenConfig
from task_and_user_generator import Generator
from weather_scenario_generator import generate_weather_schedule


def run_pipeline(
    mobility_file: str,
    weather_file: str,
    duration: int,
    generate_mobility_if_missing: bool = True,
) -> None:
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    mobility_path = Path(mobility_file)
    weather_path = Path(weather_file)
    default_mobility = mobility_path == Path("data/raw_mobility.xml")
    default_weather = weather_path == Path("data/weather_scenarios.csv")

    if generate_mobility_if_missing and (default_mobility or not mobility_path.exists()):
        generate_mobility(mobility_path, duration=duration)

    if not mobility_path.exists():
        raise FileNotFoundError(
            f"Mobility file not found: {mobility_path}. "
            "Provide a SUMO-style mobility XML or enable synthetic generation."
        )

    if default_weather or not weather_path.exists():
        generate_weather_schedule(duration=duration, output_file=weather_path)

    TaskGenConfig.WeatherScenarioConfig.SCENARIO_CSV = str(weather_path)
    try:
        TaskGenConfig.HardTaskConfig.TASKS = TaskGenConfig.HardTaskConfig.load_tasks()
    except FileNotFoundError:
        TaskGenConfig.HardTaskConfig.TASKS = ()

    generator = Generator()
    generator.generate_data(str(mobility_path))
    generator.save_metrics_to_csv("./data/metrics.csv")
    generator.plot_metrics("./data/metrics_visualization.png")

    run_random_baseline(
        vehicles_file="./data/vehicles/chunk_0.xml",
        tasks_file="./data/tasks/chunk_0.xml",
        output_file="./data/results/random_baseline_results.csv",
    )
    plot_all(
        results_file="./data/results/random_baseline_results.csv",
        summary_file="./data/results/random_baseline_summary.csv",
        output_dir="./data/results",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--mobility", default="data/raw_mobility.xml")
    parser.add_argument("--weather", default="data/weather_scenarios.csv")
    parser.add_argument(
        "--no-synthetic-mobility",
        action="store_true",
        help="Fail if the mobility file is missing instead of generating a tiny demo file.",
    )
    args = parser.parse_args()

    run_pipeline(
        mobility_file=args.mobility,
        weather_file=args.weather,
        duration=args.duration,
        generate_mobility_if_missing=not args.no_synthetic_mobility,
    )


if __name__ == "__main__":
    main()
