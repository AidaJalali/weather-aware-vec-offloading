from __future__ import annotations

import argparse
import csv
from pathlib import Path

from weather_scenarios import WeatherScenario


DEFAULT_ORDER = (
    WeatherScenario.BASE,
    WeatherScenario.RAIN,
    WeatherScenario.SNOW,
    WeatherScenario.FOG,
)


def default_blocks_for_duration(duration: int):
    block_size = max(1, duration // len(DEFAULT_ORDER))
    blocks = []
    start = 0
    for index, scenario in enumerate(DEFAULT_ORDER):
        end = duration - 1 if index == len(DEFAULT_ORDER) - 1 else start + block_size - 1
        blocks.append((start, end, scenario))
        start = end + 1
    return tuple(blocks)


def generate_weather_schedule(
    duration: int,
    output_file: str | Path,
    blocks=None,
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "scenario"])
        if blocks is None:
            blocks = default_blocks_for_duration(duration)

        for t in range(duration):
            scenario = WeatherScenario.BASE
            for start, end, block_scenario in blocks:
                if start <= t <= end:
                    scenario = block_scenario
                    break
            writer.writerow([t, scenario.value])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=1200)
    parser.add_argument(
        "--output",
        default="data/weather_scenarios.csv",
        help="CSV output with columns: time,scenario",
    )
    args = parser.parse_args()
    generate_weather_schedule(args.duration, args.output)
    print(f"Weather scenario schedule saved to {args.output}")


if __name__ == "__main__":
    main()
