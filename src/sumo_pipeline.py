from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import sumolib
import traci

from task_generation import Task, TaskGenerator
from weather_scenarios import (
    WEATHER_EFFECTS,
    WeatherEffect,
    WeatherScenario,
    get_weather_effect,
    load_scenario_by_time,
    scenario_for_time,
)
from xml_dataset_writer import (
    DatasetWriter,
    DatasetWriterConfig,
    VehicleRecord,
)


@dataclass
class SumoPipelineConfig:
    duration: int = 120
    num_users: int = 12
    num_mobile_fogs: int = 3
    seed: int = 42
    grid_number: int = 5
    grid_length: float = 300.0
    chunk_size: int = 3600
    output_dir: Path = Path("data/sumo")
    weather_schedule: Path | None = None
    overwrite: bool = False
    keep_sumo_files: bool = False
    pkw_base_max_speed: float = 13.9
    lkw_base_max_speed: float = 11.0
    sumo_step_length: float = 1.0


def _generate_network(output_file: Path, grid_number: int, grid_length: float) -> None:
    netgenerate_bin = sumolib.checkBinary("netgenerate")
    subprocess.run(
        [
            netgenerate_bin,
            "--grid",
            "--grid.number", str(grid_number),
            "--grid.length", str(grid_length),
            "--output-file", str(output_file),
            "--no-turnarounds", "true",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _load_net_edges(net_file: Path) -> list:
    net = sumolib.net.readNet(str(net_file))
    return [e for e in net.getEdges() if e.allows("passenger")]


def _build_connected_route(
    edges: list,
    min_length: int,
    rng: random.Random,
) -> list[str]:
    if not edges:
        raise ValueError("Network has no usable edges")

    edge_lookup = {e.getID(): e for e in edges}
    route_ids: list[str] = []
    current = rng.choice(edges)

    while len(route_ids) < min_length:
        route_ids.append(current.getID())

        outgoing = list(current.getOutgoing().keys())
        outgoing = [e for e in outgoing if e.getID() in edge_lookup]

        if not outgoing:
            target = rng.choice(edges)
            try:
                net = current._net
                path_edges, _ = net.getShortestPath(current, target)
                if path_edges and len(path_edges) > 1:
                    for e in path_edges[1:]:
                        route_ids.append(e.getID())
                    current = path_edges[-1]
                    continue
            except Exception:
                pass
            current = rng.choice(edges)
            continue

        current = rng.choice(outgoing)
        attempts = 0
        while current.getID() in route_ids and attempts < len(outgoing):
            current = rng.choice(outgoing)
            attempts += 1

    return route_ids


def _generate_routes_xml(
    routes_file: Path,
    net_file: Path,
    config: SumoPipelineConfig,
    rng: random.Random,
) -> None:
    edges = _load_net_edges(net_file)
    min_edges = max(len(edges), 4 * config.duration // 22)
    entries: list[tuple[float, list[str]]] = []

    for i in range(config.num_users):
        route = _build_connected_route(edges, min_edges, rng)
        depart = float(i) * 0.5
        vid = f"PKW_{i:03d}"
        entries.append((depart, [
            f'  <vehicle id="{vid}" type="PKW_special" depart="{depart:.1f}">',
            f'    <route edges="{" ".join(route)}"/>',
            f'  </vehicle>',
        ]))

    for i in range(config.num_mobile_fogs):
        route = _build_connected_route(edges, min_edges, rng)
        depart = float(i) * 0.5
        vid = f"LKW_{i:03d}"
        entries.append((depart, [
            f'  <vehicle id="{vid}" type="LKW_special" depart="{depart:.1f}">',
            f'    <route edges="{" ".join(route)}"/>',
            f'  </vehicle>',
        ]))

    entries.sort(key=lambda item: item[0])

    lines: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        "<routes>",
        f'  <vType id="PKW_special" vClass="passenger" accel="2.6" decel="4.5" '
        f'length="5.0" maxSpeed="{config.pkw_base_max_speed}"/>',
        f'  <vType id="LKW_special" vClass="passenger" accel="1.5" decel="3.0" '
        f'length="7.0" maxSpeed="{config.lkw_base_max_speed}"/>',
    ]
    for _, block in entries:
        lines.extend(block)
    lines.append("</routes>")
    routes_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class _WeatherState:
    scenario: WeatherScenario = WeatherScenario.BASE
    effect: WeatherEffect = WEATHER_EFFECTS[WeatherScenario.BASE]


def _weather_for_time(
    step: int,
    schedule: dict[int, WeatherScenario],
    state: _WeatherState,
) -> bool:
    scenario = scenario_for_time(schedule, step)
    if scenario != state.scenario:
        state.scenario = scenario
        state.effect = get_weather_effect(scenario)
        return True
    return False


def _apply_weather_to_sumo(weather: _WeatherState, config: SumoPipelineConfig) -> None:
    factor = weather.effect.speed_factor
    pkw_speed = config.pkw_base_max_speed * factor
    lkw_speed = config.lkw_base_max_speed * factor
    try:
        traci.vehicletype.setMaxSpeed("PKW_special", pkw_speed)
        traci.vehicletype.setMaxSpeed("LKW_special", lkw_speed)
    except traci.exceptions.TraCIException:
        pass


def _read_vehicle_records(weather_scenario: str) -> list[VehicleRecord]:
    records: list[VehicleRecord] = []
    for vid in traci.vehicle.getIDList():
        pos = traci.vehicle.getPosition(vid)
        records.append(VehicleRecord(
            id=vid,
            x=pos[0],
            y=pos[1],
            angle=traci.vehicle.getAngle(vid),
            speed=traci.vehicle.getSpeed(vid),
            lane=traci.vehicle.getLaneID(vid),
            vehicle_type=traci.vehicle.getTypeID(vid),
            weather_scenario=weather_scenario,
        ))
    return records


def _count_per_lane(records: list[VehicleRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        counts[r.lane] = counts.get(r.lane, 0) + 1
    return counts


def _run_simulation(config: SumoPipelineConfig) -> None:
    rng = random.Random(config.seed)

    keep = config.keep_sumo_files
    tmp_dir = tempfile.mkdtemp(prefix="sumo_pipeline_") if not keep else None
    work_dir = Path(tmp_dir) if tmp_dir else config.output_dir / "_sumo_work"
    if keep:
        work_dir.mkdir(parents=True, exist_ok=True)

    net_file = work_dir / "grid.net.xml"
    routes_file = work_dir / "trips.rou.xml"

    _generate_network(net_file, config.grid_number, config.grid_length)
    _generate_routes_xml(routes_file, net_file, config, rng)

    weather_schedule: dict[int, WeatherScenario] = {}
    if config.weather_schedule and config.weather_schedule.exists():
        weather_schedule = load_scenario_by_time(config.weather_schedule)

    writer = DatasetWriter(DatasetWriterConfig(
        output_dir=config.output_dir,
        chunk_size=config.chunk_size,
        overwrite=config.overwrite,
    ))
    task_gen = TaskGenerator(rng=rng)
    weather_state = _WeatherState()

    sumo_bin = sumolib.checkBinary("sumo")
    traci.start(
        [
            sumo_bin,
            "--net-file", str(net_file),
            "--route-files", str(routes_file),
            "--begin", "0",
            "--end", str(config.duration),
            "--step-length", str(config.sumo_step_length),
            "--seed", str(config.seed),
            "--no-step-log", "true",
            "--time-to-teleport", "-1",
        ],
    )

    try:
        recorded = 0
        while recorded < config.duration and traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            sim_time = int(round(traci.simulation.getTime()))

            if _weather_for_time(sim_time, weather_schedule, weather_state):
                _apply_weather_to_sumo(weather_state, config)

            all_records = _read_vehicle_records(weather_state.scenario.value)
            lane_counts = _count_per_lane(all_records)

            all_tasks: list[Task] = []
            for rec in all_records:
                if rec.vehicle_type != "PKW_special":
                    continue
                lane_count = lane_counts.get(rec.lane, 0)
                tasks = task_gen.generate_for_vehicle(
                    vehicle_id=rec.id,
                    vehicle_speed=rec.speed,
                    simulation_time=sim_time,
                    weather_effect=weather_state.effect,
                    lane_vehicle_count=lane_count,
                )
                all_tasks.extend(tasks)

            writer.add_timestep(
                simulation_time=sim_time,
                vehicles=all_records,
                tasks=all_tasks,
            )
            recorded += 1
    finally:
        traci.close()

    writer.finish()

    if not keep and tmp_dir:
        shutil.rmtree(work_dir, ignore_errors=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate VEC task/mobility datasets with SUMO and TraCI."
    )
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--users", type=int, default=12)
    parser.add_argument("--mobile-fogs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-number", type=int, default=5)
    parser.add_argument("--grid-length", type=float, default=300.0)
    parser.add_argument("--chunk-size", type=int, default=3600)
    parser.add_argument("--weather-schedule", default="data/weather_scenarios.csv")
    parser.add_argument("--output-dir", default="data/sumo")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-sumo-files", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = SumoPipelineConfig(
        duration=args.duration,
        num_users=args.users,
        num_mobile_fogs=args.mobile_fogs,
        seed=args.seed,
        grid_number=args.grid_number,
        grid_length=args.grid_length,
        chunk_size=args.chunk_size,
        output_dir=Path(args.output_dir),
        weather_schedule=Path(args.weather_schedule) if args.weather_schedule else None,
        overwrite=args.overwrite,
        keep_sumo_files=args.keep_sumo_files,
    )

    _run_simulation(config)
    print(f"SUMO pipeline complete. "
          f"Output written to {config.output_dir.resolve()}")


if __name__ == "__main__":
    main()
