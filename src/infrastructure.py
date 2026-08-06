from __future__ import annotations

import gzip
import math
import xml.etree.ElementTree as Et
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class VehicleState:
    id: str
    time: float
    x: float
    y: float
    speed: float
    weather_scenario: str
    vehicle_type: str = "PKW_special"


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
    plr_increase_percent: float


@dataclass(frozen=True)
class FogNode:
    id: str
    x: float
    y: float


@dataclass(frozen=True)
class ExecutionModel:
    local_speedup: float = 1.00
    fog_speedup: float = 2.50
    fog_bandwidth: float = 8.00
    fog_distance_delay_factor: float = 0.002
    fog_base_packet_loss_percent: float = 2.00
    fog_active_power: float = 2.00
    cloud_speedup: float = 5.00
    cloud_bandwidth: float = 4.00
    cloud_backhaul_delay: float = 1.50
    cloud_load_delay_per_task: float = 0.03
    cloud_max_load_delay: float = 1.50
    cloud_base_packet_loss_percent: float = 5.00
    cloud_active_power: float = 4.00
    vehicle_tx_power: float = 1.20
    max_retransmissions: int = 2
    retransmission_timeout: float = 1.00


WEATHER_BACKHAUL_DELAY = {
    "BASE": 0.00,
    "RAIN": 0.20,
    "SNOW": 0.30,
    "FOG": 0.50,
}


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


def mobile_fog_nodes_by_time(
    states: dict | Mapping,
) -> dict[float, tuple[FogNode, ...]]:
    grouped: dict[float, list[FogNode]] = {}
    for key, state in states.items():
        if not isinstance(key, tuple) or state.vehicle_type != "LKW_special":
            continue
        grouped.setdefault(float(state.time), []).append(
            FogNode(id=state.id, x=state.x, y=state.y)
        )
    return {
        timestep: tuple(sorted(nodes, key=lambda node: node.id))
        for timestep, nodes in grouped.items()
    }


def dynamic_backhaul_delay(
    scenario: str,
    network_load: int,
    model: ExecutionModel,
) -> float:
    weather_delay = WEATHER_BACKHAUL_DELAY.get(scenario, 0.0)
    load_delay = min(
        max(network_load, 0) * model.cloud_load_delay_per_task,
        model.cloud_max_load_delay,
    )
    return model.cloud_backhaul_delay + weather_delay + load_delay


def _open_xml(path: str | Path) -> Et.ElementTree:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            return Et.ElementTree(Et.fromstring(fh.read()))
    return Et.parse(path)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        gz = path.with_suffix(path.suffix + ".gz")
        if gz.exists():
            return gz
    return path


def load_vehicle_states(path: str | Path) -> dict[tuple[float, str], VehicleState]:
    path = _resolve_path(path)
    root = _open_xml(path).getroot()
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
                vehicle_type=vehicle.get("type", "PKW_special"),
            )
    return states


def load_tasks(path: str | Path) -> list[TaskRecord]:
    path = _resolve_path(path)
    root = _open_xml(path).getroot()
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
                    plr_increase_percent=float(task.get("plr_increase_percent", 0.0)),
                )
            )
    return tasks
