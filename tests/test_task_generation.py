"""Tests for deterministic task generation module."""

from __future__ import annotations

import random
import unittest

from task_generation import TaskGenerationConfig, TaskGenerator
from weather_scenarios import WeatherScenario, get_weather_effect


class TaskGeneratorTests(unittest.TestCase):
    def test_reproducibility_same_seed(self) -> None:
        effect = get_weather_effect(WeatherScenario.BASE)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        gen1 = TaskGenerator(rng=rng1)
        gen2 = TaskGenerator(rng=rng2)

        for t in range(10):
            t1 = gen1.generate_for_vehicle(
                "PKW_000", vehicle_speed=12.0, simulation_time=t,
                weather_effect=effect, lane_vehicle_count=2,
            )
            t2 = gen2.generate_for_vehicle(
                "PKW_000", vehicle_speed=12.0, simulation_time=t,
                weather_effect=effect, lane_vehicle_count=2,
            )
            self.assertEqual(len(t1), len(t2))
            for a, b in zip(t1, t2):
                self.assertEqual(a.id, b.id)
                self.assertEqual(a.deadline, b.deadline)
                self.assertEqual(a.exec_time, b.exec_time)
                self.assertEqual(a.data_size, b.data_size)
                self.assertEqual(a.cycles_per_bit, b.cycles_per_bit)

    def test_different_seed_produces_different_tasks(self) -> None:
        effect = get_weather_effect(WeatherScenario.RAIN)
        gen1 = TaskGenerator(rng=random.Random(42))
        gen2 = TaskGenerator(rng=random.Random(99))

        # Generate many tasks to reduce chance of accidental match
        tasks1 = []
        tasks2 = []
        for t in range(50):
            tasks1.extend(gen1.generate_for_vehicle(
                f"PKW_{t:03d}", vehicle_speed=12.0, simulation_time=t,
                weather_effect=effect, lane_vehicle_count=5,
            ))
            tasks2.extend(gen2.generate_for_vehicle(
                f"PKW_{t:03d}", vehicle_speed=12.0, simulation_time=t,
                weather_effect=effect, lane_vehicle_count=5,
            ))
        # At least one of the fields should differ
        if len(tasks1) == len(tasks2) and tasks1:
            any_diff = any(
                a.id != b.id or a.deadline != b.deadline
                or a.data_size != b.data_size
                for a, b in zip(tasks1, tasks2)
            )
            self.assertTrue(any_diff,
                            "Different seeds should produce different tasks")

    def test_weather_affects_task_parameters(self) -> None:
        rng = random.Random(42)
        gen = TaskGenerator(rng=rng)
        base_effect = get_weather_effect(WeatherScenario.BASE)
        fog_effect = get_weather_effect(WeatherScenario.FOG)

        base_tasks = gen.generate_for_vehicle(
            "PKW_000", vehicle_speed=12.0, simulation_time=0,
            weather_effect=base_effect, lane_vehicle_count=2,
        )
        rng2 = random.Random(42)
        gen2 = TaskGenerator(rng=rng2)
        fog_tasks = gen2.generate_for_vehicle(
            "PKW_000", vehicle_speed=12.0, simulation_time=0,
            weather_effect=fog_effect, lane_vehicle_count=2,
        )
        if base_tasks and fog_tasks:
            # FOG scenario should include weather-dependent task attributes.
            self.assertNotEqual(
                base_tasks[0].weather_scenario,
                fog_tasks[0].weather_scenario,
            )

    def test_weather_speed_reduction_is_not_treated_as_congestion(self) -> None:
        config = TaskGenerationConfig(
            normal_min_tasks=0,
            normal_max_tasks=0,
            congested_min_tasks=5,
            congested_max_tasks=5,
        )
        generator = TaskGenerator(config=config, rng=random.Random(4))
        fog = get_weather_effect(WeatherScenario.FOG)

        tasks = generator.generate_for_vehicle(
            "PKW_000",
            vehicle_speed=5.0,
            simulation_time=1,
            weather_effect=fog,
            lane_vehicle_count=2,
        )

        self.assertEqual(tasks, [])

    def test_congestion_increases_task_count(self) -> None:
        """Busy lanes should produce more tasks than quiet ones."""
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        gen1 = TaskGenerator(rng=rng1)
        gen2 = TaskGenerator(rng=rng2)

        effect = get_weather_effect(WeatherScenario.BASE)
        normal_tasks = [
            gen1.generate_for_vehicle(
                f"PKW_{i:03d}", vehicle_speed=15.0, simulation_time=i,
                weather_effect=effect, lane_vehicle_count=2,
            )
            for i in range(20)
        ]
        congested_tasks = [
            gen2.generate_for_vehicle(
                f"PKW_{i:03d}", vehicle_speed=5.0, simulation_time=i,
                weather_effect=effect, lane_vehicle_count=50,
            )
            for i in range(20)
        ]
        normal_total = sum(len(t) for t in normal_tasks)
        congested_total = sum(len(t) for t in congested_tasks)
        self.assertGreater(congested_total, normal_total,
                           "Congestion should yield more tasks")

    def test_task_id_format(self) -> None:
        gen = TaskGenerator(rng=random.Random(42))
        effect = get_weather_effect(WeatherScenario.BASE)
        tasks = gen.generate_for_vehicle(
            "PKW_005", vehicle_speed=12.0, simulation_time=7,
            weather_effect=effect, lane_vehicle_count=1,
        )
        for t in tasks:
            self.assertIn("PKW_005_S_7_", t.id)
            self.assertEqual(t.creator, "PKW_005")

    def test_task_deadline_after_simulation_time(self) -> None:
        gen = TaskGenerator(rng=random.Random(42))
        effect = get_weather_effect(WeatherScenario.BASE)
        for t in range(50):
            tasks = gen.generate_for_vehicle(
                "PKW_000", vehicle_speed=12.0, simulation_time=t,
                weather_effect=effect, lane_vehicle_count=1,
            )
            for task in tasks:
                self.assertGreaterEqual(
                    task.deadline,
                    t + task.exec_time,
                    f"Deadline {task.deadline} should be >= "
                    f"sim time {t} + exec time {task.exec_time}",
                )


if __name__ == "__main__":
    unittest.main()
