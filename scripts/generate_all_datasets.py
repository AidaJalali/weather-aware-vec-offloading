#!/usr/bin/env python3
"""Generate all datasets for SAC training — per-weather + mixed + density variants."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from weather_scenario_generator import generate_weather_schedule
from weather_scenarios import WeatherScenario

PROJECT = Path(__file__).resolve().parent.parent
SOURCE = PROJECT / "source"
SUMO_PIPELINE = SOURCE / "sumo_pipeline.py"
DATA = SOURCE / "data" / "sumo"
ENV = {**os.environ, "PYTHONPATH": str(SOURCE)}


def run(seed: int, users: int, fogs: int, duration: int,
        output_dir: Path, weather_csv: str | None = None) -> None:
    cmd = [
        sys.executable, str(SUMO_PIPELINE),
        "--duration", str(duration),
        "--seed", str(seed),
        "--users", str(users),
        "--mobile-fogs", str(fogs),
        "--output-dir", str(output_dir),
        "--overwrite",
    ]
    if weather_csv:
        cmd.extend(["--weather-schedule", weather_csv])
    subprocess.run(cmd, check=True, env=ENV)


def per_weather_csv(scenario: WeatherScenario, duration: int) -> Path:
    path = DATA / f"_weather_{scenario.value.lower()}_{duration}s.csv"
    if not path.exists():
        blocks = ((1, duration, scenario),)
        generate_weather_schedule(duration, path, blocks=blocks)
    return path


def main() -> None:
    DURATION = 3600
    USERS, FOGS = 12, 3

    # Regen default mixed-weather CSV for 3600s
    generate_weather_schedule(DURATION, DATA / "weather_scenarios.csv")

    # ── Per-weather training (3 seeds × 4 scenarios = 12 runs) ──
    for scenario in WeatherScenario:
        csv_path = per_weather_csv(scenario, DURATION)
        for seed in [100, 200, 300]:
            name = f"weather_{scenario.value.lower()}_s{seed}"
            out = DATA / "train" / name
            if (out / "vehicles" / "chunk_0.xml").exists():
                print(f"  SKIP {name} (exists)")
                continue
            print(f"[per-weather] {name} ...")
            run(seed, USERS, FOGS, DURATION, out, weather_csv=str(csv_path))
            print(f"  -> {out}")

    # ── Mixed-weather training (10 seeds) ──
    for seed in [42, 123, 456, 789, 111, 222, 333, 444, 555, 666]:
        name = f"s{seed}_u{USERS}_f{FOGS}"
        out = DATA / "train" / name
        if (out / "vehicles" / "chunk_0.xml").exists():
            print(f"  SKIP {name} (exists)")
            continue
        print(f"[mixed] {name} ...")
        run(seed, USERS, FOGS, DURATION, out)
        print(f"  -> {out}")

    # ── Density variants (4 runs) ──
    for seed, users, fogs, label in [
        (42, 20, 5, "h"), (123, 8, 2, "l"),
        (42, 30, 7, "vh"), (123, 5, 1, "vl"),
    ]:
        name = f"s{seed}_u{users}_f{fogs}_{label}"
        out = DATA / "train" / name
        if (out / "vehicles" / "chunk_0.xml").exists():
            print(f"  SKIP {name} (exists)")
            continue
        print(f"[density] {name} ...")
        run(seed, users, fogs, DURATION, out)
        print(f"  -> {out}")

    # ── Validation: mixed-weather + per-weather (1 + 4 = 5 runs) ──
    for name_suffix, run_kwargs in [
        ("s999_u12_f3", {"seed": 999, "users": USERS, "fogs": FOGS}),
    ]:
        out = DATA / "val" / name_suffix
        if (out / "vehicles" / "chunk_0.xml").exists():
            print(f"  SKIP {name_suffix} (exists)")
            continue
        print(f"[val] {name_suffix} ...")
        run(**run_kwargs, duration=DURATION, output_dir=out)
        print(f"  -> {out}")

    # ── Per-weather validation (1 seed × 4 scenarios) ──
    for scenario in WeatherScenario:
        csv_path = per_weather_csv(scenario, DURATION)
        name = f"weather_{scenario.value.lower()}_s999"
        out = DATA / "val" / name
        if (out / "vehicles" / "chunk_0.xml").exists():
            print(f"  SKIP {name} (exists)")
            continue
        print(f"[val] {name} ...")
        run(999, USERS, FOGS, DURATION, out, weather_csv=str(csv_path))
        print(f"  -> {out}")

    # Clean up temporary per-weather CSVs
    for f in DATA.glob("_weather_*.csv"):
        f.unlink()

    print("\n=== All datasets generated ===")


if __name__ == "__main__":
    main()
