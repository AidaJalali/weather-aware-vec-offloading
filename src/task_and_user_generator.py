import csv
import json
import os
import pickle
import random
import xml.etree.ElementTree as Et
from collections import defaultdict
from dataclasses import dataclass
from xml.dom import minidom
import config as CNF

from weather_scenarios import (
    WeatherScenario,
    get_weather_effect,
    load_scenario_by_time,
    scenario_for_time,
)


def _poisson_sample(lam: float) -> int:
    """Small stdlib Poisson sampler to keep Phase 1 generation dependency-light."""
    if lam <= 0:
        return 0
    threshold = np_exp(-lam)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= random.random()
    return count - 1


def np_exp(value: float) -> float:
    import math
    return math.exp(value)


def _clip_int(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


class Config:
    CHUNK_SIZE = 3600  # Chunk size in seconds

    class WeatherScenarioConfig:
        SCENARIO_CSV: str = "./data/weather_scenarios.csv"

    class TaskConfig:
        # MIN_EXEC_TIME: float = 12.5  # Slightly increased execution times
        # MAX_EXEC_TIME: float = 25.0  # Tasks take longer to complete
        MIN_POWER_CONSUMPTION: float = 1.0  # Higher power consumption than before
        MAX_POWER_CONSUMPTION: float = 3.5
        DEADLINE_MIN_FREE_TIME: float = 3.0  # Less deadline flexibility # note : next time make it a little bit more
        DEADLINE_MAX_FREE_TIME: float = 15.0
        MIN_CYCLE_PER_BIT: float = 1  # *10^3
        MAX_CYCLE_PER_BIT: float = 2
        MIN_DATASIZE: float = 0.25  # *10^6
        MAX_DATASIZE: float = 1.5

    class VehicleConfig:
        TASK_GENERATION_RATE: float = 0.35  # More frequent task generation
        FUCKED_UP_TASK_GENERATION_RATE: float = 0.55
        TRAFFIC_MIN_SPEED_THRESHOLD: float = 10.5  # Lowered speed, causing occasional congestion
        LANE_TRAFFIC_THRESHOLD: int = 15  # More vehicles per lane (moderate traffic)
        MAX_COMPUTATION_POWER: float = 6  # note : change it in feature change !
        MIN_COMPUTATION_POWER: float = 2
        COMPUTATION_POWER_ROUND_DIGIT: int = 2
        # LOW_TRANSMISSION_POWER = 5
        # MEDIUM_TRANSMISSION_POWER = 10
        HIGH_TRANSMISSION_POWER = 30
        # TRANSMISSION_LIST = [LOW_TRANSMISSION_POWER, MEDIUM_TRANSMISSION_POWER, HIGH_TRANSMISSION_POWER]

    class MobileFogConfig:
        MAX_COMPUTATION_POWER: float = 12.0  # Slightly reduced power in fog nodes
        MIN_COMPUTATION_POWER: float = 7.0
        COMPUTATION_POWER_ROUND_DIGIT: int = 2

    class HardTaskConfig:
        """Periodic hard tasks HT_1..HT_3 per vehicle (P, S KB range, C range, lambda)."""
        EXEC_TIME_DIVISOR: float = 1e6
        HARD_TASK_POWER: float = 0.0
        TASKS: tuple = ()

        # α(TL): traffic impact factor for TL in {1..5}.
        # todo: change it
        ALPHA_BY_TRAFFIC_LEVEL: dict = {
            1: 1.0,
            2: 1.05,
            3: 1.10,
            4: 1.15,
            5: 1.20,
        }
        # β(W): weather impact factor for W in {1..7}.
        BETA_BY_WEATHER: dict = {
            1: 1.0,
            2: 1.05,
            3: 1.10,
            4: 1.15,
            5: 1.20,
            6: 1.25,
            7: 1.30,
        }

        max_alpha_key = max(ALPHA_BY_TRAFFIC_LEVEL)
        max_beta_key = max(BETA_BY_WEATHER)

        MAX_TASK_SIZE = 3000
        MAX_TASK_SIZE_WITH_ALPHA_BETA = max_beta_key * max_alpha_key * MAX_TASK_SIZE

        VEHICLE_TRAFFIC_PKL: str = "./precalculated_vehicle_traffic.pkl"
        WEATHER_PKL: str = "./precalculated_weather.pkl"

        # Maps NoiseConfig traffic class name → TL ∈ {1..5}.
        TRAFFIC_NAME_TO_LEVEL: dict = {
            "GreenTraffic": 1,
            "YellowTraffic": 2,
            "OrangeTraffic": 3,
            "RedTraffic": 4,
            "BlackTraffic": 5,
        }
        DEFAULT_TRAFFIC_LEVEL: int = 1

        # Maps NoiseConfig rain class name → W ∈ {1..7}.
        WEATHER_NAME_TO_LEVEL: dict = {
            "Rain0": 1,
            "Rain13": 2,
            "Rain23": 3,
            "Rain50": 4,
            "Rain100": 5,
            "Rain150": 6,
            "Rain200": 7,
        }
        DEFAULT_WEATHER_LEVEL: int = 1

        HARD_TASK_SPECS_JSON: str = "./data/hard_task_parameters_uunifast.json"

        @classmethod
        def load_tasks(cls) -> tuple:
            """Load hard-task specs produced by task_parameter_generation_uunifast.py."""
            path = cls.HARD_TASK_SPECS_JSON
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Hard task specs not found at {path}. "
                    "Run task_parameter_generation_uunifast.py first."
                )
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            tasks = []
            for type_index, spec in enumerate(data["tasks"]):
                tasks.append({
                    "type_index": type_index,
                    "period": spec["period"],
                    "size_min": spec["size_min"],
                    "size_max": spec["size_max"],
                    "cycles_min": spec["cycles_min"],
                    "cycles_max": spec["cycles_max"],
                    "lambda": spec["lambda"],
                    "core": spec["core"],
                    "utilization": spec.get("utilization", 0.0),
                    "wcet": spec.get("wcet", 0.0)
                })
            return tuple(tasks)


@dataclass
class Vehicle:
    """Represents a mobile fog node in the network with a unique identifier and spatial coordinates."""
    id: str
    x: float
    y: float
    angle: float
    speed: float
    power: float
    type: str
    lane: str
    frequency: float
    weather: float
    weather_scenario: str = WeatherScenario.BASE.value


@dataclass
class Task:
    id: str
    deadline: float
    exec_time: float  # The amount of time that this task required to execute.
    power: float  # The amount of power unit that this tasks consumes while executing.
    creator: str  # Thd id of the node who created the task.
    cycles_per_bit: float
    dataSize: float
    core: int = 0
    type_index: int = 0
    weather_scenario: str = WeatherScenario.BASE.value
    deadline_type: str = "Normal"
    path_loss_increase_db: float = 0.0
    plr_increase_percent: float = 0.0

class Generator:
    def __init__(self):
        self.current_chunk = 0
        self.current_vehicles = []
        self.current_tasks = []
        self.current_hard_tasks = []
        self.hard_task_counters = defaultdict(int)
        self.soft_task_counters = defaultdict(int)
        # Per-vehicle next release step keyed by task type_index; cleared when vehicle leaves.
        self.hard_task_release_schedule: dict[str, dict[int, float]] = {}
        self.tasks_count_per_step = defaultdict(int)
        self.average_speed_per_step = defaultdict(float)
        self.total_task_power_per_step = defaultdict(float)

        # Create output directories.
        os.makedirs("./data/vehicles", exist_ok=True)
        os.makedirs("./data/tasks", exist_ok=True)
        os.makedirs("./data/hard_tasks", exist_ok=True)

        self._vehicle_traffic_cache = self._load_pkl(
            Config.HardTaskConfig.VEHICLE_TRAFFIC_PKL
        )
        self._weather_cache = self._load_pkl(Config.HardTaskConfig.WEATHER_PKL)
        self._scenario_by_time = load_scenario_by_time(
            Config.WeatherScenarioConfig.SCENARIO_CSV
        )

    @staticmethod
    def _load_pkl(path: str) -> dict:
        if not os.path.exists(path):
            print(f"warning: pkl not found at {path}; falling back to defaults.")
            return {}
        with open(path, "rb") as f:
            return pickle.load(f)

    def _lookup_traffic_level(self, step: int, vehicle_id: str) -> tuple:
        """Return (TL, partition_name). TL is mapped from precalculated traffic class."""
        info = self._vehicle_traffic_cache.get(step, {}).get(vehicle_id)
        if info is None:
            return Config.HardTaskConfig.DEFAULT_TRAFFIC_LEVEL, None

        partition_name = info.get("zone")
        traffic_str = str(info.get("traffic", ""))
        for name, level in Config.HardTaskConfig.TRAFFIC_NAME_TO_LEVEL.items():
            if name in traffic_str:
                return level, partition_name
        return Config.HardTaskConfig.DEFAULT_TRAFFIC_LEVEL, partition_name

    def _lookup_weather_level(self, step: int, partition_name) -> int:
        """Return W mapped from the precalculated rain class for the vehicle's zone."""
        if partition_name is None:
            return Config.HardTaskConfig.DEFAULT_WEATHER_LEVEL
        weather_name = self._weather_cache.get(step, {}).get(partition_name)
        if weather_name is None:
            return Config.HardTaskConfig.DEFAULT_WEATHER_LEVEL
        return Config.HardTaskConfig.WEATHER_NAME_TO_LEVEL.get(
            str(weather_name), Config.HardTaskConfig.DEFAULT_WEATHER_LEVEL
        )

    @staticmethod
    def get_chunk_number(step: int) -> int:
        return step // Config.CHUNK_SIZE

    def save_current_chunk(self, step: int):
        chunk_num = self.get_chunk_number(step)
        if chunk_num > self.current_chunk and (
                self.current_vehicles or self.current_tasks or self.current_hard_tasks
        ):
            self._save_vehicles_chunk()
            self._save_tasks_chunk()
            self._save_hard_tasks_chunk()
            self.current_vehicles = []
            self.current_tasks = []
            self.current_hard_tasks = []
            self.current_chunk = chunk_num

    def _save_vehicles_chunk(self):
        root = Et.Element('fcd-export')
        root.set("version", "1.0")

        for time_data in self.current_vehicles:
            time_elem = Et.SubElement(root, 'timestep')
            time_elem.set('time', f"{time_data['step']}")
            for vehicle in time_data['vehicles']:
                v_elem = Et.SubElement(time_elem, 'vehicle')
                v_elem.set('id', vehicle.id)
                v_elem.set('x', f"{vehicle.x:.2f}")
                v_elem.set('y', f"{vehicle.y:.2f}")
                v_elem.set('angle', f"{vehicle.angle:.2f}")
                v_elem.set('speed', f"{vehicle.speed:.2f}")
                v_elem.set('lane', vehicle.lane)
                v_elem.set('type', vehicle.type)
                v_elem.set('power', f"{vehicle.power:.2f}")
                v_elem.set('frequency', f"{vehicle.frequency:.2f}")
                v_elem.set('weather', f"{vehicle.weather}")
                v_elem.set('weather_scenario', vehicle.weather_scenario)

        Et.indent(root, space="    ", level=0)
        tree = Et.ElementTree(root)
        tree.write(f"./data/vehicles/chunk_{self.current_chunk}.xml", encoding='utf-8', xml_declaration=True)

    def _save_tasks_chunk(self):
        root = Et.Element('fcd-export')
        root.set("version", "1.0")

        for time_data in self.current_tasks:
            time_elem = Et.SubElement(root, 'timestep')
            time_elem.set('time', f"{time_data['step']}")
            for task in time_data['tasks']:
                t_elem = Et.SubElement(time_elem, 'task')
                t_elem.set('id', task.id)
                t_elem.set('deadline', f"{task.deadline:.2f}")
                t_elem.set('exec_time', f"{task.exec_time:.2f}")
                t_elem.set('power', f"{task.power:.2f}")
                t_elem.set('creator', task.creator)
                t_elem.set('cycles_per_bit', f"{task.cycles_per_bit:.2f}")
                t_elem.set('dataSize', f"{task.dataSize:.2f}")
                t_elem.set('weather_scenario', task.weather_scenario)
                t_elem.set('deadline_type', task.deadline_type)
                t_elem.set('path_loss_increase_db', f"{task.path_loss_increase_db:.2f}")
                t_elem.set('plr_increase_percent', f"{task.plr_increase_percent:.2f}")

        Et.indent(root, space="    ", level=0)
        tree = Et.ElementTree(root)
        tree.write(f"./data/tasks/chunk_{self.current_chunk}.xml", encoding='utf-8', xml_declaration=True)

    def _save_hard_tasks_chunk(self):
        root = Et.Element('fcd-export')
        root.set("version", "1.0")

        for time_data in self.current_hard_tasks:
            time_elem = Et.SubElement(root, 'timestep')
            time_elem.set('time', f"{time_data['step']}")
            for task in time_data['tasks']:
                t_elem = Et.SubElement(time_elem, 'task')
                t_elem.set('id', task.id)
                t_elem.set('deadline', f"{task.deadline:.2f}")
                t_elem.set('exec_time', f"{task.exec_time:.2f}")
                t_elem.set('power', f"{task.power:.2f}")
                t_elem.set('creator', task.creator)
                t_elem.set('cycles_per_bit', f"{task.cycles_per_bit:.2f}")
                t_elem.set('dataSize', f"{task.dataSize:.2f}")
                t_elem.set('core', str(task.core))
                t_elem.set('type_index', str(task.type_index))
                t_elem.set('weather_scenario', task.weather_scenario)
                t_elem.set('deadline_type', task.deadline_type)
                t_elem.set('path_loss_increase_db', f"{task.path_loss_increase_db:.2f}")
                t_elem.set('plr_increase_percent', f"{task.plr_increase_percent:.2f}")

        Et.indent(root, space="    ", level=0)
        tree = Et.ElementTree(root)
        tree.write(f"./data/hard_tasks/chunk_{self.current_chunk}.xml", encoding='utf-8', xml_declaration=True)

    @staticmethod
    def _environment_scaling(traffic_level: int, weather_level: int) -> float:
        alpha = Config.HardTaskConfig.ALPHA_BY_TRAFFIC_LEVEL[traffic_level]
        beta = Config.HardTaskConfig.BETA_BY_WEATHER[weather_level]
        return alpha * beta

    def _init_hard_task_schedule(self, vehicle_id: str, entry_step: int) -> None:
        """Start periodic releases when a vehicle enters the simulation."""
        self.hard_task_release_schedule[vehicle_id] = {
            task_spec["type_index"]: float(entry_step)
            for task_spec in Config.HardTaskConfig.TASKS
        }

    def generate_hard_tasks_for_vehicle(
            self,
            step: int,
            vehicle: Vehicle,
            weather_effect,
    ) -> list[Task]:
        """Generate periodic hard tasks while the vehicle is present in the system."""
        if vehicle.id not in self.hard_task_release_schedule:
            self._init_hard_task_schedule(vehicle.id, step)

        traffic_level, partition_name = self._lookup_traffic_level(step, vehicle.id)
        weather_level = self._lookup_weather_level(step, partition_name)
        scaling = self._environment_scaling(traffic_level, weather_level)
        schedule = self.hard_task_release_schedule[vehicle.id]
        hard_tasks = []

        for task_spec in Config.HardTaskConfig.TASKS:
            type_index = task_spec["type_index"]
            period = task_spec["period"]
            if step < schedule[type_index] - 1e-9:
                continue

            size_baseline = random.uniform(task_spec["size_min"], task_spec["size_max"])
            cycles_baseline = random.uniform(task_spec["cycles_min"], task_spec["cycles_max"])
            sensitivity = task_spec["lambda"]
            data_size = round(size_baseline * scaling * sensitivity, 2)
            cycles_per_bit = round(
                cycles_baseline
                * scaling
                * sensitivity
                * weather_effect.sample_cycles_multiplier(),
                2,
            )
            exec_time = (
                data_size * cycles_per_bit
            ) / (vehicle.frequency * Config.HardTaskConfig.EXEC_TIME_DIVISOR)
            task_index = self.hard_task_counters[vehicle.id]
            task_id = f"{vehicle.id}_H_{step}_{task_index}_{period}"

            self.hard_task_counters[vehicle.id] += 1
            schedule[type_index] += period

            hard_tasks.append(Task(
                id=task_id,
                deadline=float(step + period),
                exec_time=exec_time,
                power=Config.HardTaskConfig.HARD_TASK_POWER,
                creator=vehicle.id,
                cycles_per_bit=cycles_per_bit,
                dataSize=data_size,
                core=task_spec["core"],
                type_index=type_index,
                weather_scenario=weather_effect.scenario.value,
                deadline_type=weather_effect.deadline_label,
                path_loss_increase_db=weather_effect.sample_path_loss_increase_db(),
                plr_increase_percent=weather_effect.sample_plr_increase_percent(),
            ))

        return hard_tasks

    def _clear_departed_vehicle_schedules(self, present_vehicle_ids: set[str]) -> None:
        """Drop schedules for vehicles that left so re-entry starts a new periodic cycle."""
        departed_ids = set(self.hard_task_release_schedule) - present_vehicle_ids
        for vehicle_id in departed_ids:
            del self.hard_task_release_schedule[vehicle_id]

    def calculate_metrics(self, step: float, vehicles: list[Vehicle], tasks: list[Task]):
        """Calculate metrics for the current timestep."""
        # Count tasks for this step
        self.tasks_count_per_step[step] = len(tasks)

        # Calculate average speed of user nodes (PKW_special type)
        if vehicles:
            avg_speed = sum(v.speed for v in vehicles) / len(vehicles)
            self.average_speed_per_step[step] = round(avg_speed, 2)
        else:
            self.average_speed_per_step[step] = 0.0

        # Calculate total power of tasks for this step
        self.total_task_power_per_step[step] = round(sum(task.power for task in tasks), 2)

    @staticmethod
    def generate_one_step_tasks(self, step, vehicle, lane_counter, weather_effect):
        """Generate tasks for each mobile fog node."""

        traffic_high = (
                lane_counter > Config.VehicleConfig.LANE_TRAFFIC_THRESHOLD
                or vehicle.speed < Config.VehicleConfig.TRAFFIC_MIN_SPEED_THRESHOLD
        )
        if traffic_high:
            num_tasks = _poisson_sample(
                lam=3.0 * weather_effect.task_generation_rate_multiplier
            )
            num_tasks = _clip_int(num_tasks, 2, 5)
        else:
            num_tasks = _poisson_sample(
                lam=0.8 * weather_effect.task_generation_rate_multiplier
            )
            num_tasks = _clip_int(num_tasks, 0, 3)

        tasks = []

        for _ in range(num_tasks):

            deadline_free = round(
                random.uniform(
                    weather_effect.deadline_free_time_range[0],
                    weather_effect.deadline_free_time_range[1],
                ),
                2
            )
            power = round(
                random.uniform(
                    Config.TaskConfig.MIN_POWER_CONSUMPTION,
                    Config.TaskConfig.MAX_POWER_CONSUMPTION
                ),
                2
            )
            cycles_per_bit = round(
                random.uniform(
                    Config.TaskConfig.MIN_CYCLE_PER_BIT,
                    Config.TaskConfig.MAX_CYCLE_PER_BIT
                ) * weather_effect.sample_cycles_multiplier(),
                2
            )
            dataSize = round(random.uniform(Config.TaskConfig.MIN_DATASIZE, Config.TaskConfig.MAX_DATASIZE), 2)

            exec_time = (dataSize * cycles_per_bit) / CNF.Config.UserNodeConfig.USER_NODE_FREQUENCY
            deadline = round(exec_time + deadline_free) + step

            task_index = self.soft_task_counters[vehicle.id]
            self.soft_task_counters[vehicle.id] += 1

            # chance = random.random()
            # threshold = Config.VehicleConfig.TASK_GENERATION_RATE
            # if (
            #         lane_counter > Config.VehicleConfig.LANE_TRAFFIC_THRESHOLD or
            #         vehicle.speed < Config.VehicleConfig.TRAFFIC_MIN_SPEED_THRESHOLD
            # ):
            #     threshold = Config.VehicleConfig.FUCKED_UP_TASK_GENERATION_RATE

            # if chance > threshold:
            #     return None

            tasks.append(Task(
                id=f"{vehicle.id}_S_{step}_{task_index}",
                deadline=deadline,
                exec_time=exec_time,
                power=power,
                creator=vehicle.id,
                cycles_per_bit=cycles_per_bit,
                dataSize=dataSize,
                weather_scenario=weather_effect.scenario.value,
                deadline_type=weather_effect.deadline_label,
                path_loss_increase_db=weather_effect.sample_path_loss_increase_db(),
                plr_increase_percent=weather_effect.sample_plr_increase_percent(),
            ))

        return tasks

    def generate_one_step(self, step, time_data, seen_ids_power):
        """Generate vehicles for each mobile fog node."""
        current_vehicles = []
        current_tasks = []
        current_hard_tasks = []
        lane_counter = defaultdict(int)
        scenario = scenario_for_time(self._scenario_by_time, step)
        weather_effect = get_weather_effect(scenario)

        for vehicle in time_data.findall('vehicle'):
            v_id = vehicle.get('id')
            data = dict(
                id=v_id,
                x=float(vehicle.get('x')),
                y=float(vehicle.get('y')),
                angle=90 - float(vehicle.get('angle')),
                speed=float(vehicle.get('speed')) * weather_effect.speed_factor,
                lane=vehicle.get('lane'),
                type=vehicle.get('type'),
                weather=list(WeatherScenario).index(scenario),
                weather_scenario=scenario.value,
            )

            if v_id in seen_ids_power:
                power = seen_ids_power[v_id]
                if vehicle.get('type') == "LKW_special":
                    frequency = CNF.Config.MobileFogNodeConfig.MOBILE_NODE_FREQUENCY
                elif vehicle.get('type') == "PKW_special":
                    frequency = CNF.Config.UserNodeConfig.USER_NODE_FREQUENCY
            elif vehicle.get('type') == "LKW_special":
                power = round(
                    random.uniform(
                        Config.MobileFogConfig.MIN_COMPUTATION_POWER,
                        Config.MobileFogConfig.MAX_COMPUTATION_POWER
                    ),
                    Config.MobileFogConfig.COMPUTATION_POWER_ROUND_DIGIT
                )
                frequency = CNF.Config.MobileFogNodeConfig.MOBILE_NODE_FREQUENCY
            elif vehicle.get('type') == "PKW_special":
                power = round(
                    random.uniform(
                        Config.VehicleConfig.MIN_COMPUTATION_POWER,
                        Config.VehicleConfig.MAX_COMPUTATION_POWER
                    ),
                    Config.MobileFogConfig.COMPUTATION_POWER_ROUND_DIGIT
                )
                frequency = CNF.Config.UserNodeConfig.USER_NODE_FREQUENCY
            else:
                continue

            seen_ids_power[v_id] = power
            data["power"] = power
            data["frequency"] = frequency

            vehicle_obj = Vehicle(**data)
            current_vehicles.append(vehicle_obj)
            lane_counter[vehicle_obj.lane] += 1

            if tasks := self.generate_one_step_tasks(self, step, vehicle_obj, lane_counter[vehicle_obj.lane], weather_effect):
                for task in tasks:
                    current_tasks.append(task)

            current_hard_tasks.extend(
                self.generate_hard_tasks_for_vehicle(step, vehicle_obj, weather_effect)
            )

        self._clear_departed_vehicle_schedules({vehicle.id for vehicle in current_vehicles})

        # Calculate metrics before saving the chunk.
        self.calculate_metrics(step, current_vehicles, current_tasks)

        # Add current timestep data to the chunk.
        self.current_vehicles.append({"step": step, "vehicles": current_vehicles})
        self.current_tasks.append({"step": step, "tasks": current_tasks})
        self.current_hard_tasks.append({"step": step, "tasks": current_hard_tasks})

        self.save_current_chunk(step)

        return seen_ids_power

    def generate_data(self, path: str):
        """Parse the time data from the given content."""
        with open(path, 'rb') as f:
            root = Et.parse(f).getroot()
        seen_ids_power = {}
        timesteps = root.findall('.//timestep')
        total_steps = len(timesteps)

        print(f"Starting data generation for {total_steps} timesteps...")

        for idx, time in enumerate(timesteps):
            step = round(float(time.get('time')))
            seen_ids_power = self.generate_one_step(step, time, seen_ids_power)

            # Print progress every 100 steps or at the final step
            if step % 100 == 0 or idx == total_steps - 1:
                progress_percent = (idx + 1) / total_steps * 100
                print(f"[Progress] Successfully generated up to timestep: {step} ({progress_percent:.1f}%)")

        # Save the last chunk if there's any data left.
        if self.current_vehicles or self.current_tasks or self.current_hard_tasks:
            self._save_vehicles_chunk()
            self._save_tasks_chunk()
            self._save_hard_tasks_chunk()

    def save_metrics_to_csv(self, metrics_file: str):
        """Save the collected metrics to a CSV file."""
        all_steps = sorted(set(self.tasks_count_per_step.keys()) |
                           set(self.average_speed_per_step.keys()) |
                           set(self.total_task_power_per_step.keys()))

        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'task_count', 'average_speed', 'total_task_power'])
            for step in all_steps:
                writer.writerow([
                    f"{step:.2f}",
                    self.tasks_count_per_step[step],
                    self.average_speed_per_step[step],
                    self.total_task_power_per_step[step]
                ])

    def plot_metrics(self, output_file: str):
        """Create a visualization of the metrics."""
        os.makedirs("./.matplotlib", exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", "./.matplotlib")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("warning: matplotlib is not installed; skipping metrics plot.")
            return

        steps = sorted(set(self.tasks_count_per_step.keys()) |
                       set(self.average_speed_per_step.keys()) |
                       set(self.total_task_power_per_step.keys()))
        task_counts = [self.tasks_count_per_step[step] for step in steps]
        avg_speeds = [self.average_speed_per_step[step] for step in steps]
        total_powers = [self.total_task_power_per_step[step] for step in steps]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        ax1.plot(steps, task_counts, 'b-', label='Tasks per Step')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Number of Tasks')
        ax1.set_title('Tasks Generated per Time Step')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(steps, avg_speeds, 'r-', label='Average Speed')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Speed')
        ax2.set_title('Average Speed of User Nodes per Time Step')
        ax2.grid(True)
        ax2.legend()

        ax3.plot(steps, total_powers, 'g-', label='Total Task Power')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Power Units')
        ax3.set_title('Total Power of Tasks per Time Step')
        ax3.grid(True)
        ax3.legend()

        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


def main(path: str):
    """Main function to run the task generator."""
    try:
        Config.HardTaskConfig.TASKS = Config.HardTaskConfig.load_tasks()
    except FileNotFoundError as exc:
        print(f"warning: {exc}")
        print("Hard-task generation is disabled for this run.")
        Config.HardTaskConfig.TASKS = ()
    generator = Generator()
    generator.generate_data(path)
    generator.save_metrics_to_csv("./data/metrics.csv")
    generator.plot_metrics("./data/metrics_visualization.png")


if __name__ == '__main__':
    main("./simulation.out.xml")
