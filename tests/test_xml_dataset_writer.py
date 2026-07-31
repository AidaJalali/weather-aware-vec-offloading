"""Tests for XML dataset writer module."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

from infrastructure import load_tasks, load_vehicle_states
from task_generation import Task, TaskGenerationConfig, TaskGenerator
from weather_scenarios import WeatherScenario, get_weather_effect
from xml_dataset_writer import (
    DatasetWriter,
    DatasetWriterConfig,
    VehicleRecord,
)


class DatasetWriterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_vehicle(self, vid: str, x: float = 100.0) -> VehicleRecord:
        return VehicleRecord(
            id=vid, x=x, y=200.0, angle=90.0, speed=12.0,
            lane="lane_0", vehicle_type="PKW_special",
            weather_scenario="BASE",
        )

    def _make_task(self, creator: str, sim_time: int) -> Task:
        effect = get_weather_effect(WeatherScenario.BASE)
        gen = TaskGenerator(rng=random.Random(42))
        tasks = gen.generate_for_vehicle(
            creator, vehicle_speed=12.0, simulation_time=sim_time,
            weather_effect=effect, lane_vehicle_count=1,
        )
        return tasks[0] if tasks else self._make_task_fallback(creator, sim_time)

    @staticmethod
    def _make_task_fallback(creator: str, sim_time: int) -> Task:
        return Task(
            id=f"{creator}_S_{sim_time}_0",
            deadline=float(sim_time + 10),
            exec_time=5.0,
            power=2.0,
            creator=creator,
            cycles_per_bit=1.5,
            data_size=1.0,
            weather_scenario="BASE",
            deadline_type="Normal",
            path_loss_increase_db=0.0,
            plr_increase_percent=0.0,
        )

    def test_write_and_read_back(self) -> None:
        writer = DatasetWriter(DatasetWriterConfig(
            output_dir=self.tmp, overwrite=True,
        ))
        for t in range(5):
            v = self._make_vehicle("PKW_000", x=100.0 + t * 10)
            task = self._make_task_fallback("PKW_000", t)
            writer.add_timestep(t, [v], [task])
        writer.finish()

        v_states = load_vehicle_states(self.tmp / "vehicles" / "chunk_0.xml")
        tasks = load_tasks(self.tmp / "tasks" / "chunk_0.xml")
        self.assertEqual(len(tasks), 5)
        self.assertGreaterEqual(len(v_states), 5)

    def test_task_missing_vehicle_raises(self) -> None:
        writer = DatasetWriter(DatasetWriterConfig(
            output_dir=self.tmp, overwrite=True,
        ))
        task = self._make_task_fallback("PKW_999", 0)
        v = self._make_vehicle("PKW_000")  # Different vehicle
        with self.assertRaises(ValueError):
            writer.add_timestep(0, [v], [task])

    def test_chunk_splitting(self) -> None:
        writer = DatasetWriter(DatasetWriterConfig(
            output_dir=self.tmp, chunk_size=10, overwrite=True,
        ))
        for t in range(25):
            v = self._make_vehicle(f"PKW_{t % 3:03d}", x=float(t * 10))
            writer.add_timestep(t, [v], [])
        writer.finish()

        self.assertTrue((self.tmp / "vehicles" / "chunk_0.xml").exists())
        self.assertTrue((self.tmp / "vehicles" / "chunk_1.xml").exists())
        self.assertTrue((self.tmp / "vehicles" / "chunk_2.xml").exists())
        self.assertFalse((self.tmp / "vehicles" / "chunk_3.xml").exists())

        # chunk_0: timesteps 0-9, chunk_1: 10-19, chunk_2: 20-24
        v0 = load_vehicle_states(self.tmp / "vehicles" / "chunk_0.xml")
        v1 = load_vehicle_states(self.tmp / "vehicles" / "chunk_1.xml")
        v2 = load_vehicle_states(self.tmp / "vehicles" / "chunk_2.xml")
        times0 = {int(t) for (t, _) in v0}
        times1 = {int(t) for (t, _) in v1}
        times2 = {int(t) for (t, _) in v2}
        self.assertEqual(times0, set(range(10)))
        self.assertEqual(times1, set(range(10, 20)))
        self.assertEqual(times2, set(range(20, 25)))

    def test_overwrite_protection(self) -> None:
        writer = DatasetWriter(DatasetWriterConfig(
            output_dir=self.tmp, overwrite=False, chunk_size=2,
        ))
        v = self._make_vehicle("PKW_000")
        writer.add_timestep(0, [v], [])
        # Force a chunk flush by crossing the boundary
        writer.add_timestep(1, [v], [])
        writer.add_timestep(2, [v], [])
        writer.finish()

        writer2 = DatasetWriter(DatasetWriterConfig(
            output_dir=self.tmp, overwrite=False, chunk_size=2,
        ))
        with self.assertRaises(FileExistsError):
            writer2.add_timestep(0, [v], [])
            writer2.add_timestep(1, [v], [])
            writer2.add_timestep(2, [v], [])

    def test_final_partial_chunk_is_written(self) -> None:
        writer = DatasetWriter(DatasetWriterConfig(
            output_dir=self.tmp, chunk_size=3600, overwrite=True,
        ))
        v = self._make_vehicle("PKW_000")
        for t in range(5):
            writer.add_timestep(t, [v], [])
        writer.finish()

        chunk = load_vehicle_states(self.tmp / "vehicles" / "chunk_0.xml")
        times = {int(t) for (t, _) in chunk}
        self.assertEqual(times, set(range(5)))

    def test_lkw_vehicle_with_matching_task_is_allowed(self) -> None:
        """Writer allows any task whose creator matches a vehicle — filtering
        of LKW tasks is the TaskGenerator's responsibility, not the writer's."""
        writer = DatasetWriter(DatasetWriterConfig(
            output_dir=self.tmp, overwrite=True,
        ))
        lkw = VehicleRecord(
            id="LKW_000", x=100.0, y=200.0, angle=90.0, speed=10.0,
            lane="lane_0", vehicle_type="LKW_special",
            weather_scenario="BASE",
        )
        task_lkw = self._make_task_fallback("LKW_000", 0)
        # Should NOT raise — creator matches a present vehicle
        writer.add_timestep(0, [lkw], [task_lkw])
        writer.finish()
        tasks = load_tasks(self.tmp / "tasks" / "chunk_0.xml")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].creator, "LKW_000")

    def test_xml_format_matches_loader_expectations(self) -> None:
        """Ensure written attributes match what infrastructure loaders consume."""
        writer = DatasetWriter(DatasetWriterConfig(
            output_dir=self.tmp, overwrite=True,
        ))
        v = VehicleRecord(
            id="PKW_000", x=123.45, y=678.90, angle=45.0, speed=13.5,
            lane="lane_1", vehicle_type="PKW_special",
            weather_scenario="RAIN",
        )
        task = Task(
            id="PKW_000_S_5_0", deadline=20.0, exec_time=3.5, power=2.1,
            creator="PKW_000", cycles_per_bit=1.8, data_size=0.9,
            weather_scenario="RAIN", deadline_type="Tight Deadline",
            path_loss_increase_db=3.2, plr_increase_percent=17.5,
        )
        writer.add_timestep(5, [v], [task])
        writer.finish()

        v_states = load_vehicle_states(self.tmp / "vehicles" / "chunk_0.xml")
        tasks = load_tasks(self.tmp / "tasks" / "chunk_0.xml")

        vs = v_states[(5.0, "PKW_000")]
        self.assertEqual(vs.x, 123.45)
        self.assertEqual(vs.y, 678.90)
        self.assertEqual(vs.speed, 13.5)
        self.assertEqual(vs.weather_scenario, "RAIN")

        self.assertEqual(len(tasks), 1)
        t = tasks[0]
        self.assertEqual(t.id, "PKW_000_S_5_0")
        self.assertEqual(t.deadline, 20.0)
        self.assertEqual(t.exec_time, 3.5)
        self.assertEqual(t.power, 2.1)
        self.assertEqual(t.creator, "PKW_000")
        self.assertEqual(t.cycles_per_bit, 1.8)
        self.assertEqual(t.data_size, 0.9)
        self.assertEqual(t.weather_scenario, "RAIN")
        self.assertEqual(t.deadline_type, "Tight Deadline")
        self.assertEqual(t.path_loss_increase_db, 3.2)
        self.assertEqual(t.plr_increase_percent, 17.5)


if __name__ == "__main__":
    unittest.main()
