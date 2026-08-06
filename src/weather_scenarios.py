from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping


class WeatherScenario(str, Enum):
    BASE = "BASE"
    RAIN = "RAIN"
    SNOW = "SNOW"
    FOG = "FOG"


@dataclass(frozen=True)
class WeatherEffect:
    scenario: WeatherScenario
    speed_factor: float
    deadline_label: str
    deadline_free_time_range: tuple[float, float]
    cycles_per_bit_multiplier_range: tuple[float, float]
    task_generation_rate_multiplier: float
    plr_increase_percent_range: tuple[float, float]

    def sample_cycles_multiplier(self, rng=random) -> float:
        low, high = self.cycles_per_bit_multiplier_range
        return rng.uniform(low, high)

    def sample_plr_increase_percent(self, rng=random) -> float:
        low, high = self.plr_increase_percent_range
        return rng.uniform(low, high)


WEATHER_EFFECTS: Mapping[WeatherScenario, WeatherEffect] = {
    WeatherScenario.BASE: WeatherEffect(
        scenario=WeatherScenario.BASE,
        speed_factor=1.00,
        deadline_label="Normal",
        deadline_free_time_range=(3.0, 15.0),
        cycles_per_bit_multiplier_range=(1.0, 1.0),
        task_generation_rate_multiplier=1.00,
        plr_increase_percent_range=(0.0, 0.0),
    ),
    WeatherScenario.RAIN: WeatherEffect(
        scenario=WeatherScenario.RAIN,
        speed_factor=0.85,
        deadline_label="Tight Deadline",
        deadline_free_time_range=(2.0, 10.0),
        cycles_per_bit_multiplier_range=(1.2, 1.4),
        task_generation_rate_multiplier=1.20,
        plr_increase_percent_range=(15.0, 20.0),
    ),
    WeatherScenario.SNOW: WeatherEffect(
        scenario=WeatherScenario.SNOW,
        speed_factor=0.70,
        deadline_label="The Tightest Deadline",
        deadline_free_time_range=(1.0, 6.0),
        cycles_per_bit_multiplier_range=(1.5, 1.8),
        task_generation_rate_multiplier=0.50,
        plr_increase_percent_range=(5.0, 10.0),
    ),
    WeatherScenario.FOG: WeatherEffect(
        scenario=WeatherScenario.FOG,
        speed_factor=0.40,
        deadline_label="Relaxed Deadline",
        deadline_free_time_range=(5.0, 20.0),
        cycles_per_bit_multiplier_range=(1.3, 1.6),
        task_generation_rate_multiplier=1.25,
        plr_increase_percent_range=(25.0, 30.0),
    ),
}


def normalize_scenario(value: str | WeatherScenario | None) -> WeatherScenario:
    if isinstance(value, WeatherScenario):
        return value
    if value is None:
        return WeatherScenario.BASE
    normalized = str(value).strip().upper()
    if not normalized:
        return WeatherScenario.BASE
    return WeatherScenario(normalized)


def get_weather_effect(value: str | WeatherScenario | None) -> WeatherEffect:
    return WEATHER_EFFECTS[normalize_scenario(value)]


def load_scenario_by_time(path: str | Path) -> dict[int, WeatherScenario]:
    scenario_path = Path(path)
    if not scenario_path.exists():
        return {}

    with scenario_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"time", "scenario"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{scenario_path} is missing required columns: {sorted(missing)}"
            )
        return {
            int(float(row["time"])): normalize_scenario(row["scenario"])
            for row in reader
        }


def scenario_for_time(
    scenario_by_time: Mapping[int, WeatherScenario],
    step: int,
) -> WeatherScenario:
    if not scenario_by_time:
        return WeatherScenario.BASE
    if step in scenario_by_time:
        return scenario_by_time[step]
    earlier_steps = [t for t in scenario_by_time if t <= step]
    if not earlier_steps:
        return WeatherScenario.BASE
    return scenario_by_time[max(earlier_steps)]
