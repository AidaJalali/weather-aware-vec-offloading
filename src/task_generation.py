from __future__ import annotations

import math
import random
from dataclasses import dataclass

from weather_scenarios import WeatherEffect


def _poisson_sample(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    threshold = math.exp(-lam)
    product = rng.random()
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count


def _clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


@dataclass
class TaskGenerationConfig:
    min_power: float = 1.0
    max_power: float = 3.5
    min_cycles_per_bit: float = 1.0
    max_cycles_per_bit: float = 2.0
    min_data_size_mbit: float = 0.25
    max_data_size_mbit: float = 1.5
    local_frequency_ghz: float = 0.5
    normal_arrival_rate: float = 0.8
    congested_arrival_rate: float = 3.0
    lane_traffic_threshold: int = 15
    traffic_min_speed_threshold: float = 10.5
    normal_min_tasks: int = 0
    normal_max_tasks: int = 3
    congested_min_tasks: int = 2
    congested_max_tasks: int = 5
    round_digits: int = 2


class TaskGenerator:
    def __init__(
        self,
        config: TaskGenerationConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config or TaskGenerationConfig()
        self.rng = rng or random.Random()
        self._soft_task_counters: dict[str, int] = {}

    def generate_for_vehicle(
        self,
        vehicle_id: str,
        vehicle_speed: float,
        simulation_time: int,
        weather_effect: WeatherEffect,
        lane_vehicle_count: int,
    ) -> list["Task"]:
        cfg = self.config
        traffic_high = (
            lane_vehicle_count > cfg.lane_traffic_threshold
            or vehicle_speed < cfg.traffic_min_speed_threshold
        )

        if traffic_high:
            lam = cfg.congested_arrival_rate * weather_effect.task_generation_rate_multiplier
            num_tasks = _poisson_sample(lam, self.rng)
            num_tasks = _clip_int(num_tasks, cfg.congested_min_tasks, cfg.congested_max_tasks)
        else:
            lam = cfg.normal_arrival_rate * weather_effect.task_generation_rate_multiplier
            num_tasks = _poisson_sample(lam, self.rng)
            num_tasks = _clip_int(num_tasks, cfg.normal_min_tasks, cfg.normal_max_tasks)

        tasks: list[Task] = []
        for _ in range(num_tasks):
            deadline_free = round(
                self.rng.uniform(
                    weather_effect.deadline_free_time_range[0],
                    weather_effect.deadline_free_time_range[1],
                ),
                cfg.round_digits,
            )
            power = round(
                self.rng.uniform(cfg.min_power, cfg.max_power),
                cfg.round_digits,
            )
            cycles_per_bit = round(
                self.rng.uniform(cfg.min_cycles_per_bit, cfg.max_cycles_per_bit)
                * weather_effect.sample_cycles_multiplier(self.rng),
                cfg.round_digits,
            )
            data_size = round(
                self.rng.uniform(cfg.min_data_size_mbit, cfg.max_data_size_mbit),
                cfg.round_digits,
            )

            exec_time = (data_size * cycles_per_bit) / cfg.local_frequency_ghz
            deadline = round(exec_time + deadline_free) + simulation_time

            task_index = self._soft_task_counters.setdefault(vehicle_id, 0)
            self._soft_task_counters[vehicle_id] = task_index + 1

            tasks.append(Task(
                id=f"{vehicle_id}_S_{simulation_time}_{task_index}",
                deadline=deadline,
                exec_time=exec_time,
                power=power,
                creator=vehicle_id,
                cycles_per_bit=cycles_per_bit,
                data_size=data_size,
                weather_scenario=weather_effect.scenario.value,
                deadline_type=weather_effect.deadline_label,
                path_loss_increase_db=weather_effect.sample_path_loss_increase_db(self.rng),
                plr_increase_percent=weather_effect.sample_plr_increase_percent(self.rng),
            ))

        return tasks


@dataclass
class Task:
    id: str
    deadline: float
    exec_time: float
    power: float
    creator: str
    cycles_per_bit: float
    data_size: float
    weather_scenario: str
    deadline_type: str
    path_loss_increase_db: float
    plr_increase_percent: float
