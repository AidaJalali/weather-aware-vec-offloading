from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

from infrastructure import (
    ExecutionModel,
    TaskRecord,
    VehicleState,
)
from offloading_simulator import (
    DeterministicChannel,
    FogLookup,
    ResourceCapacities,
    ResourceState,
    simulate_assignments,
)


class OffloadTarget(str, Enum):
    LOCAL = "LOCAL"
    FOG = "FOG"
    CLOUD = "CLOUD"


class RandomOffloader:
    name = "Random"

    def __init__(self, seed: int = 11):
        self.seed = int(seed)
        self._rng = random.Random(seed)
        self._targets = tuple(OffloadTarget)

    def choose_target(self) -> OffloadTarget:
        return self._rng.choice(self._targets)

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

    @property
    def resource_capacities(self) -> ResourceCapacities:
        return ResourceCapacities(
            local=self.local_capacity,
            fog=self.fog_capacity,
            cloud=self.cloud_capacity,
        )


@dataclass(frozen=True)
class BatchEvaluation:
    cost: float
    total_latency: float
    total_energy: float
    expected_packet_losses: float
    expected_deadline_failures: float
    deadline_misses: int
    total_lateness: float
    total_queue_delay: float
    normalized_performance_cost: float
    normalized_queue_delay: float

    @property
    def expected_task_failures(self) -> float:
        """Packet failures plus successful executions expected to miss deadlines."""
        return self.expected_deadline_failures

    @property
    def ranking_key(self) -> tuple[float, float, float, float, float]:
        return (
            self.expected_task_failures,
            self.expected_packet_losses,
            self.normalized_performance_cost,
            self.normalized_queue_delay,
            self.total_lateness,
        )


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

    Weather awareness comes from each TaskRecord's adjusted execution time,
    deadline, and packet-loss-rate increase. Chromosomes are evaluated with shared
    execution queues so assignments account for resource contention.
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

    def evaluate(
        self,
        chromosome: Sequence[OffloadTarget],
        tasks: Sequence[TaskRecord],
        vehicle_states: VehicleLookup,
        *,
        resource_state: ResourceState | None = None,
        channel_randomness: DeterministicChannel | None = None,
        network_load_by_time: Mapping[float, int] | None = None,
        fog_nodes_by_time: FogLookup | None = None,
    ) -> BatchEvaluation:
        if len(chromosome) != len(tasks):
            raise ValueError("chromosome length must equal task count")

        simulated = simulate_assignments(
            tasks,
            chromosome,
            (resource_state or ResourceState()).copy(),
            channel_randomness or DeterministicChannel(self.config.seed),
            vehicle_states=vehicle_states,
            model=self.model,
            capacities=self.config.resource_capacities,
            network_load_by_time=network_load_by_time,
            fog_nodes_by_time=fog_nodes_by_time,
        )

        total_latency = 0.0
        total_energy = 0.0
        expected_packet_losses = 0.0
        expected_deadline_failures = 0.0
        deadline_misses = 0
        total_lateness = 0.0
        total_queue_delay = 0.0
        normalized_latency = 0.0
        normalized_energy = 0.0
        normalized_queue = 0.0

        for task, result in zip(tasks, simulated):
            lateness = max(0.0, result.finish_time - task.deadline)
            available_time = max(task.deadline - task.release_time, 1e-9)
            local_energy_reference = max(
                task.exec_time / self.model.local_speedup
                * task.power,
                1e-9,
            )

            total_latency += result.latency
            total_energy += result.total_system_energy
            expected_packet_losses += result.final_failure_probability
            expected_deadline_failures += result.expected_deadline_failure
            deadline_misses += int(result.deadline_missed)
            total_lateness += lateness
            total_queue_delay += result.queue_delay
            normalized_latency += result.latency / available_time
            normalized_energy += result.total_system_energy / local_energy_reference
            normalized_queue += result.queue_delay / available_time

        count = max(len(tasks), 1)
        average_normalized_latency = normalized_latency / count
        average_normalized_energy = normalized_energy / count
        average_normalized_queue = normalized_queue / count
        normalized_performance_cost = (
            self.config.latency_weight * average_normalized_latency
            + self.config.energy_weight * average_normalized_energy
        )
        cost = (
            normalized_performance_cost
            + self.config.reliability_weight
            * expected_packet_losses
            / count
            + self.config.queue_weight * average_normalized_queue
            + self.config.deadline_miss_penalty * expected_deadline_failures
            + self.config.lateness_penalty * total_lateness
        )
        return BatchEvaluation(
            cost=cost,
            total_latency=total_latency,
            total_energy=total_energy,
            expected_packet_losses=expected_packet_losses,
            expected_deadline_failures=expected_deadline_failures,
            deadline_misses=deadline_misses,
            total_lateness=total_lateness,
            total_queue_delay=total_queue_delay,
            normalized_performance_cost=normalized_performance_cost,
            normalized_queue_delay=average_normalized_queue,
        )

    def _seed_population(
        self,
        task_count: int,
        tasks: Sequence[TaskRecord],
        vehicle_states: VehicleLookup,
        resource_state: ResourceState,
        channel_randomness: DeterministicChannel,
        network_load_by_time: Mapping[float, int] | None,
        fog_nodes_by_time: FogLookup | None,
    ) -> list[Chromosome]:
        population = [
            tuple([target] * task_count)
            for target in self._targets
        ]

        greedy: list[OffloadTarget] = []
        for task in tasks:
            single_scores = [
                (
                    self.evaluate(
                        (target,),
                        (task,),
                        vehicle_states,
                        resource_state=resource_state,
                        channel_randomness=channel_randomness,
                        network_load_by_time=network_load_by_time,
                        fog_nodes_by_time=fog_nodes_by_time,
                    ).ranking_key,
                    target,
                )
                for target in self._targets
            ]
            greedy.append(min(single_scores, key=lambda item: item[0])[1])
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
        scored: Sequence[
            tuple[tuple[float, float, float, float, float], Chromosome]
        ],
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
        *,
        resource_state: ResourceState | None = None,
        channel_randomness: DeterministicChannel | None = None,
        network_load_by_time: Mapping[float, int] | None = None,
        fog_nodes_by_time: FogLookup | None = None,
    ) -> GeneticOffloadingResult:
        start = time.perf_counter()
        if not tasks:
            empty = BatchEvaluation(
                cost=0.0,
                total_latency=0.0,
                total_energy=0.0,
                expected_packet_losses=0.0,
                expected_deadline_failures=0.0,
                deadline_misses=0,
                total_lateness=0.0,
                total_queue_delay=0.0,
                normalized_performance_cost=0.0,
                normalized_queue_delay=0.0,
            )
            return GeneticOffloadingResult(
                assignments=(),
                evaluation=empty,
                generations=0,
                evaluations=0,
                elapsed_seconds=time.perf_counter() - start,
                stopped_by_time_limit=False,
            )

        current_state = resource_state or ResourceState()
        channel = channel_randomness or DeterministicChannel(self.config.seed)
        population = self._seed_population(
            len(tasks),
            tasks,
            vehicle_states,
            current_state,
            channel,
            network_load_by_time,
            fog_nodes_by_time,
        )
        evaluations = 0
        generations = 0
        deadline = start + self.config.time_limit_seconds
        cache: dict[Chromosome, BatchEvaluation] = {}

        def score(
            chromosome: Chromosome,
        ) -> tuple[float, float, float, float, float]:
            nonlocal evaluations
            if chromosome not in cache:
                cache[chromosome] = self.evaluate(
                    chromosome,
                    tasks,
                    vehicle_states,
                    resource_state=current_state,
                    channel_randomness=channel,
                    network_load_by_time=network_load_by_time,
                    fog_nodes_by_time=fog_nodes_by_time,
                )
                evaluations += 1
            return cache[chromosome].ranking_key

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
        *,
        resource_state: ResourceState | None = None,
        channel_randomness: DeterministicChannel | None = None,
        network_load_by_time: Mapping[float, int] | None = None,
        fog_nodes_by_time: FogLookup | None = None,
    ) -> tuple[OffloadTarget, ...]:
        if self.in_fallback:
            result = self.genetic_offloader.optimize(
                tasks,
                vehicle_states,
                resource_state=resource_state,
                channel_randomness=channel_randomness,
                network_load_by_time=network_load_by_time,
                fog_nodes_by_time=fog_nodes_by_time,
            )
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
