from __future__ import annotations

import xml.etree.ElementTree as Et
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from task_generation import Task


@dataclass
class VehicleRecord:
    id: str
    x: float
    y: float
    angle: float
    speed: float
    lane: str
    vehicle_type: str
    weather_scenario: str


@dataclass
class DatasetWriterConfig:
    output_dir: Path
    chunk_size: int = 3600
    overwrite: bool = False


class DatasetWriter:
    def __init__(self, config: DatasetWriterConfig) -> None:
        self.config = config
        self._vehicle_buffer: list[_TimestepVehicles] = []
        self._task_buffer: list[_TimestepTasks] = []
        self._current_chunk: int = 0

    def add_timestep(
        self,
        simulation_time: int,
        vehicles: Iterable[VehicleRecord],
        tasks: Iterable[Task],
    ) -> None:
        v_list = list(vehicles)
        t_list = list(tasks)

        vehicle_ids = {v.id for v in v_list}
        for task in t_list:
            if task.creator not in vehicle_ids:
                raise ValueError(
                    f"Task {task.id} references creator {task.creator!r} "
                    f"which is not present at time {simulation_time}"
                )

        self._vehicle_buffer.append(_TimestepVehicles(simulation_time, v_list))
        self._task_buffer.append(_TimestepTasks(simulation_time, t_list))

        next_time = simulation_time + 1
        while self._crosses_chunk_boundary(next_time):
            self._flush_chunk(self._current_chunk)
            self._current_chunk += 1

    def finish(self) -> None:
        if self._vehicle_buffer or self._task_buffer:
            self._flush_chunk(self._current_chunk)

    @staticmethod
    def _vehicles_dir(output_dir: Path) -> Path:
        return output_dir / "vehicles"

    @staticmethod
    def _tasks_dir(output_dir: Path) -> Path:
        return output_dir / "tasks"

    def _crosses_chunk_boundary(self, simulation_time: int) -> bool:
        return simulation_time // self.config.chunk_size > self._current_chunk

    def _flush_chunk(self, chunk_index: int) -> None:
        cfg = self.config
        vehicles_path = (
            self._vehicles_dir(cfg.output_dir) / f"chunk_{chunk_index}.xml"
        )
        tasks_path = (
            self._tasks_dir(cfg.output_dir) / f"chunk_{chunk_index}.xml"
        )

        if not cfg.overwrite and (vehicles_path.exists() or tasks_path.exists()):
            raise FileExistsError(
                f"Output file(s) already exist at {vehicles_path} / {tasks_path}. "
                f"Use --overwrite to replace."
            )

        vehicles_path.parent.mkdir(parents=True, exist_ok=True)
        tasks_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_vehicles_xml(
            vehicles_path,
            [ts for ts in self._vehicle_buffer
             if ts.simulation_time // cfg.chunk_size == chunk_index],
        )
        self._write_tasks_xml(
            tasks_path,
            [ts for ts in self._task_buffer
             if ts.simulation_time // cfg.chunk_size == chunk_index],
        )

        self._vehicle_buffer = [
            ts for ts in self._vehicle_buffer
            if ts.simulation_time // cfg.chunk_size > chunk_index
        ]
        self._task_buffer = [
            ts for ts in self._task_buffer
            if ts.simulation_time // cfg.chunk_size > chunk_index
        ]

    @staticmethod
    def _write_vehicles_xml(
        path: Path,
        timesteps: list[_TimestepVehicles],
    ) -> None:
        root = Et.Element("fcd-export")
        root.set("version", "1.0")

        for ts in timesteps:
            time_elem = Et.SubElement(root, "timestep")
            time_elem.set("time", str(ts.simulation_time))
            for v in ts.vehicles:
                elem = Et.SubElement(time_elem, "vehicle")
                elem.set("id", v.id)
                elem.set("x", f"{v.x:.2f}")
                elem.set("y", f"{v.y:.2f}")
                elem.set("angle", f"{v.angle:.2f}")
                elem.set("speed", f"{v.speed:.2f}")
                elem.set("lane", v.lane)
                elem.set("type", v.vehicle_type)
                elem.set("weather_scenario", v.weather_scenario)

        Et.indent(root, space="    ", level=0)
        Et.ElementTree(root).write(
            str(path), encoding="utf-8", xml_declaration=True,
        )

    @staticmethod
    def _write_tasks_xml(
        path: Path,
        timesteps: list[_TimestepTasks],
    ) -> None:
        root = Et.Element("fcd-export")
        root.set("version", "1.0")

        for ts in timesteps:
            time_elem = Et.SubElement(root, "timestep")
            time_elem.set("time", str(ts.simulation_time))
            for t in ts.tasks:
                elem = Et.SubElement(time_elem, "task")
                elem.set("id", t.id)
                elem.set("deadline", f"{t.deadline:.2f}")
                elem.set("exec_time", f"{t.exec_time:.2f}")
                elem.set("power", f"{t.power:.2f}")
                elem.set("creator", t.creator)
                elem.set("cycles_per_bit", f"{t.cycles_per_bit:.2f}")
                elem.set("dataSize", f"{t.data_size:.2f}")
                elem.set("weather_scenario", t.weather_scenario)
                elem.set("deadline_type", t.deadline_type)
                elem.set("path_loss_increase_db", f"{t.path_loss_increase_db:.2f}")
                elem.set("plr_increase_percent", f"{t.plr_increase_percent:.2f}")

        Et.indent(root, space="    ", level=0)
        Et.ElementTree(root).write(
            str(path), encoding="utf-8", xml_declaration=True,
        )


class _TimestepVehicles:
    __slots__ = ("simulation_time", "vehicles")

    def __init__(self, simulation_time: int, vehicles: list[VehicleRecord]) -> None:
        self.simulation_time = simulation_time
        self.vehicles = vehicles


class _TimestepTasks:
    __slots__ = ("simulation_time", "tasks")

    def __init__(self, simulation_time: int, tasks: list[Task]) -> None:
        self.simulation_time = simulation_time
        self.tasks = tasks
