from __future__ import annotations

import heapq
import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

from infrastructure import (
    DEFAULT_FOG_NODES,
    ExecutionModel,
    TaskRecord,
    VehicleState,
    distance,
    nearest_fog,
)


class OffloadTarget(str, Enum):
    LOCAL = "LOCAL"
    FOG = "FOG"
    CLOUD = "CLOUD"


class RandomOffloader:
    name = "Random"

    def __init__(self, seed: int = 11):
        self._rng = random.Random(seed)
        self._targets = tuple(OffloadTarget)

    def choose_target(self) -> OffloadTarget:
        return self._rng.choice(self._targets)

    def packet_lost(self, packet_loss_percent: float) -> bool:
        bounded = max(0.0, min(packet_loss_percent, 100.0))
        return self._rng.random() < (bounded / 100.0)

    def transmit_with_retries(
        self,
        packet_loss_percent: float,
        max_retransmissions: int,
    ) -> tuple[bool, int, int]:
        max_attempts = 1 + max(0, max_retransmissions)
        for attempt in range(1, max_attempts + 1):
            if not self.packet_lost(packet_loss_percent):
                return False, attempt, attempt - 1
        return True, max_attempts, max_attempts - 1

@dataclass(frozen=True)
class GeneticOffloaderConfig:
    """Configuration for the time-bounded batch genetic optimizer."""

    population_size: int = 32
    max_generations: int = 25
    time_limit_seconds: float = 0.15
    tournament_size: int = 3
    crossover_rate: float = 0.80
    mutation_rate: float = 0.10
    elite_count: int = 2
    local_capacity: int = 1
    fog_capacity: int = 4
    cloud_capacity: int = 16
    latency_weight: float = 0.30
    energy_weight: float = 0.20
    reliability_weight: float = 0.20
    queue_weight: float = 0.10
    deadline_miss_penalty: float = 100.0
    lateness_penalty: float = 10.0
    path_loss_delay_per_db: float = 0.01
    seed: int = 37

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.max_generations < 0:
            raise ValueError("max_generations cannot be negative")
        if self.time_limit_seconds <= 0:
            raise ValueError("time_limit_seconds must be positive")
        if not 1 <= self.tournament_size <= self.population_size:
            raise ValueError("tournament_size must be within the population size")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if not 0 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be smaller than population_size")
        if min(self.local_capacity, self.fog_capacity, self.cloud_capacity) <= 0:
            raise ValueError("all execution capacities must be positive")


@dataclass(frozen=True)
class BatchEvaluation:
    cost: float
    total_latency: float
    total_energy: float
    expected_packet_losses: float
    deadline_misses: int
    total_lateness: float
    total_queue_delay: float


@dataclass(frozen=True)
class GeneticOffloadingResult:
    assignments: tuple[OffloadTarget, ...]
    evaluation: BatchEvaluation
    generations: int
    evaluations: int
    elapsed_seconds: float
    stopped_by_time_limit: bool

    def by_task_id(
        self,
        tasks: Sequence[TaskRecord],
    ) -> dict[str, OffloadTarget]:
        if len(tasks) != len(self.assignments):
            raise ValueError("tasks must match the optimized batch")
        return {
            task.id: target
            for task, target in zip(tasks, self.assignments)
        }


VehicleLookup = Mapping[str | tuple[float, str], VehicleState]
Chromosome = tuple[OffloadTarget, ...]


class GeneticBatchOffloader:
    """Weather-aware, deadline-constrained GA for a batch of VEC tasks.

    Weather awareness comes from each TaskRecord's already adjusted execution time,
    deadline, path-loss increase, and PLR increase. Chromosomes are evaluated with
    shared execution queues so assignments account for resource contention.
    """

    name = "GeneticBatch"

    def __init__(
        self,
        config: GeneticOffloaderConfig | None = None,
        model: ExecutionModel | None = None,
    ) -> None:
        self.config = config or GeneticOffloaderConfig()
        self.model = model or ExecutionModel()
        self._rng = random.Random(self.config.seed)
        self._targets = tuple(OffloadTarget)

    @staticmethod
    def _vehicle_for_task(
        task: TaskRecord,
        vehicle_states: VehicleLookup,
    ) -> VehicleState:
        vehicle = vehicle_states.get((task.release_time, task.creator))
        if vehicle is None:
            vehicle = vehicle_states.get(task.creator)
        if vehicle is None:
            raise KeyError(
                f"No vehicle state for task {task.id!r} "
                f"(creator={task.creator!r}, release={task.release_time})"
            )
        return vehicle

    def _execution_components(
        self,
        task: TaskRecord,
        vehicle: VehicleState,
        target: OffloadTarget,
    ) -> tuple[str, float, float, float, float, int]:
        """Return resource, transmission, execution, energy, PLR, capacity."""
        if target == OffloadTarget.LOCAL:
            execution = task.exec_time / self.model.local_speedup
            return (
                f"LOCAL:{task.creator}",
                0.0,
                execution,
                execution * task.power,
                0.0,
                self.config.local_capacity,
            )

        def retry_expectations(
            packet_loss_percent: float,
        ) -> tuple[float, float, float]:
            per_attempt_loss = max(
                0.0,
                min(packet_loss_percent / 100.0, 1.0),
            )
            attempts = 1 + max(0, self.model.max_retransmissions)
            expected_attempts = sum(
                per_attempt_loss ** attempt
                for attempt in range(attempts)
            )
            expected_retries = expected_attempts - 1.0
            final_failure_probability = per_attempt_loss ** attempts
            return (
                expected_attempts,
                expected_retries,
                final_failure_probability,
            )

        weather_delay = (
            max(0.0, task.path_loss_increase_db)
            * self.config.path_loss_delay_per_db
        )
        if target == OffloadTarget.FOG:
            fog = nearest_fog(vehicle, DEFAULT_FOG_NODES)
            wireless_time = task.data_size / self.model.fog_bandwidth
            raw_plr = (
                self.model.fog_base_packet_loss_percent
                + max(0.0, task.plr_increase_percent)
            )
            expected_attempts, expected_retries, failure_probability = (
                retry_expectations(raw_plr)
            )
            transmission = (
                expected_attempts * wireless_time
                + expected_retries * self.model.retransmission_timeout
                + distance(vehicle.x, vehicle.y, fog.x, fog.y)
                * self.model.fog_distance_delay_factor
                + weather_delay
            )
            execution = task.exec_time / self.model.fog_speedup
            energy = (
                expected_attempts
                * wireless_time
                * self.model.vehicle_tx_power
                + (1.0 - failure_probability)
                * execution
                * self.model.fog_active_power
            )
            return (
                f"FOG:{fog.id}",
                transmission,
                execution,
                energy,
                failure_probability * 100.0,
                self.config.fog_capacity,
            )

        wireless_time = task.data_size / self.model.cloud_bandwidth
        raw_plr = (
            self.model.cloud_base_packet_loss_percent
            + max(0.0, task.plr_increase_percent)
        )
        expected_attempts, expected_retries, failure_probability = (
            retry_expectations(raw_plr)
        )
        transmission = (
            expected_attempts * wireless_time
            + expected_retries * self.model.retransmission_timeout
            + self.model.cloud_backhaul_delay
            + weather_delay
        )
        execution = task.exec_time / self.model.cloud_speedup
        energy = (
            expected_attempts
            * wireless_time
            * self.model.vehicle_tx_power
            + (1.0 - failure_probability)
            * execution
            * self.model.cloud_active_power
        )
        return (
            "CLOUD",
            transmission,
            execution,
            energy,
            failure_probability * 100.0,
            self.config.cloud_capacity,
        )

    def evaluate(
        self,
        chromosome: Sequence[OffloadTarget],
        tasks: Sequence[TaskRecord],
        vehicle_states: VehicleLookup,
    ) -> BatchEvaluation:
        if len(chromosome) != len(tasks):
            raise ValueError("chromosome length must equal task count")

        queues: dict[str, list[float]] = {}
        total_latency = 0.0
        total_energy = 0.0
        expected_packet_losses = 0.0
        deadline_misses = 0
        total_lateness = 0.0
        total_queue_delay = 0.0
        normalized_latency = 0.0
        normalized_energy = 0.0
        normalized_queue = 0.0

        indexed = sorted(
            enumerate(zip(tasks, chromosome)),
            key=lambda item: (
                item[1][0].release_time,
                item[1][0].deadline,
                item[0],
            ),
        )
        for _, (task, raw_target) in indexed:
            target = OffloadTarget(raw_target)
            vehicle = self._vehicle_for_task(task, vehicle_states)
            (
                resource,
                transmission,
                execution,
                energy,
                packet_loss_percent,
                capacity,
            ) = self._execution_components(task, vehicle, target)

            if resource not in queues:
                queues[resource] = [0.0] * capacity
                heapq.heapify(queues[resource])
            available_at = heapq.heappop(queues[resource])
            arrival_at_resource = task.release_time + transmission
            execution_start = max(arrival_at_resource, available_at)
            queue_delay = execution_start - arrival_at_resource
            finish_time = execution_start + execution
            heapq.heappush(queues[resource], finish_time)

            latency = finish_time - task.release_time
            lateness = max(0.0, finish_time - task.deadline)
            missed = lateness > 0.0
            available_time = max(task.deadline - task.release_time, 1e-9)
            local_energy_reference = max(
                task.exec_time / self.model.local_speedup
                * task.power,
                1e-9,
            )

            total_latency += latency
            total_energy += energy
            expected_packet_losses += packet_loss_percent / 100.0
            deadline_misses += int(missed)
            total_lateness += lateness
            total_queue_delay += queue_delay
            normalized_latency += latency / available_time
            normalized_energy += energy / local_energy_reference
            normalized_queue += queue_delay / available_time

        count = max(len(tasks), 1)
        cost = (
            self.config.latency_weight * normalized_latency / count
            + self.config.energy_weight * normalized_energy / count
            + self.config.reliability_weight
            * expected_packet_losses
            / count
            + self.config.queue_weight * normalized_queue / count
            + self.config.deadline_miss_penalty * deadline_misses
            + self.config.lateness_penalty * total_lateness
        )
        return BatchEvaluation(
            cost=cost,
            total_latency=total_latency,
            total_energy=total_energy,
            expected_packet_losses=expected_packet_losses,
            deadline_misses=deadline_misses,
            total_lateness=total_lateness,
            total_queue_delay=total_queue_delay,
        )

    def _seed_population(
        self,
        task_count: int,
        tasks: Sequence[TaskRecord],
        vehicle_states: VehicleLookup,
    ) -> list[Chromosome]:
        population = [
            tuple([target] * task_count)
            for target in self._targets
        ]

        greedy: list[OffloadTarget] = []
        for task in tasks:
            single_costs = [
                (
                    self.evaluate(
                        (target,),
                        (task,),
                        vehicle_states,
                    ).cost,
                    target,
                )
                for target in self._targets
            ]
            greedy.append(min(single_costs, key=lambda item: item[0])[1])
        population.append(tuple(greedy))

        while len(population) < self.config.population_size:
            population.append(
                tuple(
                    self._rng.choice(self._targets)
                    for _ in range(task_count)
                )
            )
        return population

    def _tournament(
        self,
        scored: Sequence[tuple[float, Chromosome]],
    ) -> Chromosome:
        candidates = self._rng.sample(
            list(scored),
            self.config.tournament_size,
        )
        return min(candidates, key=lambda item: item[0])[1]

    def _crossover(
        self,
        first: Chromosome,
        second: Chromosome,
    ) -> tuple[Chromosome, Chromosome]:
        if len(first) < 2 or self._rng.random() >= self.config.crossover_rate:
            return first, second
        point = self._rng.randrange(1, len(first))
        return (
            first[:point] + second[point:],
            second[:point] + first[point:],
        )

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        genes = list(chromosome)
        for index, current in enumerate(genes):
            if self._rng.random() < self.config.mutation_rate:
                alternatives = [
                    target for target in self._targets
                    if target != current
                ]
                genes[index] = self._rng.choice(alternatives)
        return tuple(genes)

    def optimize(
        self,
        tasks: Sequence[TaskRecord],
        vehicle_states: VehicleLookup,
    ) -> GeneticOffloadingResult:
        start = time.perf_counter()
        if not tasks:
            empty = BatchEvaluation(0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)
            return GeneticOffloadingResult(
                assignments=(),
                evaluation=empty,
                generations=0,
                evaluations=0,
                elapsed_seconds=time.perf_counter() - start,
                stopped_by_time_limit=False,
            )

        population = self._seed_population(
            len(tasks),
            tasks,
            vehicle_states,
        )
        evaluations = 0
        generations = 0
        deadline = start + self.config.time_limit_seconds
        cache: dict[Chromosome, BatchEvaluation] = {}

        def score(chromosome: Chromosome) -> float:
            nonlocal evaluations
            if chromosome not in cache:
                cache[chromosome] = self.evaluate(
                    chromosome,
                    tasks,
                    vehicle_states,
                )
                evaluations += 1
            return cache[chromosome].cost

        scored = sorted((score(item), item) for item in population)
        for generation in range(self.config.max_generations):
            if time.perf_counter() >= deadline:
                break

            next_population = [
                chromosome
                for _, chromosome in scored[:self.config.elite_count]
            ]
            while len(next_population) < self.config.population_size:
                first = self._tournament(scored)
                second = self._tournament(scored)
                child_a, child_b = self._crossover(first, second)
                next_population.append(self._mutate(child_a))
                if len(next_population) < self.config.population_size:
                    next_population.append(self._mutate(child_b))

            population = next_population
            scored = sorted((score(item), item) for item in population)
            generations = generation + 1

        _, best = scored[0]
        elapsed = time.perf_counter() - start
        return GeneticOffloadingResult(
            assignments=best,
            evaluation=cache[best],
            generations=generations,
            evaluations=evaluations,
            elapsed_seconds=elapsed,
            stopped_by_time_limit=(
                generations < self.config.max_generations
                and elapsed >= self.config.time_limit_seconds
            ),
        )


BatchPolicy = Callable[
    [Sequence[TaskRecord], VehicleLookup],
    Sequence[OffloadTarget],
]


class RLGeneticFallbackController:
    """Switch between an RL policy and the GA using reward hysteresis.

    RL training remains external. Call ``observe_reward`` after each control
    interval and keep feeding transitions to the RL trainer while this controller
    is in fallback mode.
    """

    def __init__(
        self,
        rl_policy: BatchPolicy,
        genetic_offloader: GeneticBatchOffloader,
        fallback_reward_threshold: float,
        recovery_reward_threshold: float,
        reward_window: int = 5,
        consecutive_windows: int = 2,
    ) -> None:
        if recovery_reward_threshold <= fallback_reward_threshold:
            raise ValueError(
                "recovery_reward_threshold must exceed fallback_reward_threshold"
            )
        if reward_window <= 0 or consecutive_windows <= 0:
            raise ValueError("reward_window and consecutive_windows must be positive")
        self.rl_policy = rl_policy
        self.genetic_offloader = genetic_offloader
        self.fallback_reward_threshold = fallback_reward_threshold
        self.recovery_reward_threshold = recovery_reward_threshold
        self.consecutive_windows = consecutive_windows
        self._rewards: deque[float] = deque(maxlen=reward_window)
        self._bad_windows = 0
        self._good_windows = 0
        self.in_fallback = False
        self.last_source = "RL"
        self.last_ga_result: GeneticOffloadingResult | None = None

    @property
    def average_reward(self) -> float | None:
        if not self._rewards:
            return None
        return sum(self._rewards) / len(self._rewards)

    def observe_reward(self, reward: float) -> bool:
        self._rewards.append(float(reward))
        if len(self._rewards) < self._rewards.maxlen:
            return self.in_fallback

        average = self.average_reward
        assert average is not None
        if not self.in_fallback:
            self._bad_windows = (
                self._bad_windows + 1
                if average < self.fallback_reward_threshold
                else 0
            )
            if self._bad_windows >= self.consecutive_windows:
                self.in_fallback = True
                self._bad_windows = 0
                self._good_windows = 0
        else:
            self._good_windows = (
                self._good_windows + 1
                if average > self.recovery_reward_threshold
                else 0
            )
            if self._good_windows >= self.consecutive_windows:
                self.in_fallback = False
                self._good_windows = 0
                self._bad_windows = 0
        return self.in_fallback

    def choose_batch(
        self,
        tasks: Sequence[TaskRecord],
        vehicle_states: VehicleLookup,
    ) -> tuple[OffloadTarget, ...]:
        if self.in_fallback:
            result = self.genetic_offloader.optimize(tasks, vehicle_states)
            self.last_ga_result = result
            self.last_source = "GA"
            return result.assignments

        assignments = tuple(
            OffloadTarget(target)
            for target in self.rl_policy(tasks, vehicle_states)
        )
        if len(assignments) != len(tasks):
            raise ValueError("RL policy returned the wrong assignment count")
        self.last_ga_result = None
        self.last_source = "RL"
        return assignments


# Temporary compatibility alias for code written before SAC was selected.
PPOGeneticFallbackController = RLGeneticFallbackController
