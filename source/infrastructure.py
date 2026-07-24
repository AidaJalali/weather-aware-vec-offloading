from __future__ import annotations

import math
import xml.etree.ElementTree as Et
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VehicleState:
    id: str
    time: float
    x: float
    y: float
    speed: float
    weather_scenario: str


@dataclass(frozen=True)
class TaskRecord:
    id: str
    release_time: float
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


@dataclass(frozen=True)
class FogNode:
    id: str
    x: float
    y: float


@dataclass(frozen=True)
class ExecutionModel:
    local_energy_rate: float = 1.00
    local_speedup: float = 1.00
    fog_speedup: float = 2.50
    fog_bandwidth: float = 8.00
    fog_distance_delay_factor: float = 0.002
    fog_base_packet_loss_percent: float = 2.00
    fog_tx_energy_rate: float = 0.70
    fog_execution_energy_rate: float = 0.30
    cloud_speedup: float = 5.00
    cloud_bandwidth: float = 4.00
    cloud_backhaul_delay: float = 1.50
    cloud_base_packet_loss_percent: float = 5.00
    cloud_tx_energy_rate: float = 1.20


DEFAULT_FOG_NODES = (
    FogNode(id="FOG0", x=150.0, y=150.0),
    FogNode(id="FOG1", x=450.0, y=300.0),
    FogNode(id="FOG2", x=750.0, y=550.0),
)


def distance(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def nearest_fog(vehicle: VehicleState, fog_nodes=DEFAULT_FOG_NODES) -> FogNode:
    return min(
        fog_nodes,
        key=lambda fog: distance(vehicle.x, vehicle.y, fog.x, fog.y),
    )


def load_vehicle_states(path: str | Path) -> dict[tuple[float, str], VehicleState]:
    root = Et.parse(path).getroot()
    states: dict[tuple[float, str], VehicleState] = {}
    for timestep in root.findall(".//timestep"):
        time = float(timestep.get("time"))
        for vehicle in timestep.findall("vehicle"):
            vehicle_id = vehicle.get("id")
            states[(time, vehicle_id)] = VehicleState(
                id=vehicle_id,
                time=time,
                x=float(vehicle.get("x")),
                y=float(vehicle.get("y")),
                speed=float(vehicle.get("speed")),
                weather_scenario=vehicle.get("weather_scenario", "BASE"),
            )
    return states


def load_tasks(path: str | Path) -> list[TaskRecord]:
    root = Et.parse(path).getroot()
    tasks: list[TaskRecord] = []
    for timestep in root.findall(".//timestep"):
        release_time = float(timestep.get("time"))
        for task in timestep.findall("task"):
            tasks.append(
                TaskRecord(
                    id=task.get("id"),
                    release_time=release_time,
                    deadline=float(task.get("deadline")),
                    exec_time=float(task.get("exec_time")),
                    power=float(task.get("power")),
                    creator=task.get("creator"),
                    cycles_per_bit=float(task.get("cycles_per_bit")),
                    data_size=float(task.get("dataSize")),
                    weather_scenario=task.get("weather_scenario", "BASE"),
                    deadline_type=task.get("deadline_type", "Normal"),
                    path_loss_increase_db=float(task.get("path_loss_increase_db", 0.0)),
                    plr_increase_percent=float(task.get("plr_increase_percent", 0.0)),
                )
            )
    return tasks

