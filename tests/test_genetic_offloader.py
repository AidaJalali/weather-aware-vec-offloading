from __future__ import annotations

import time
import unittest

from algorithms import (
    GeneticBatchOffloader,
    GeneticOffloaderConfig,
    OffloadTarget,
    RLGeneticFallbackController,
)
from infrastructure import TaskRecord, VehicleState


def make_task(
    task_id: str,
    *,
    release: float = 0.0,
    deadline: float = 10.0,
    exec_time: float = 8.0,
    plr: float = 0.0,
    path_loss: float = 0.0,
    scenario: str = "BASE",
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        release_time=release,
        deadline=deadline,
        exec_time=exec_time,
        power=1.0,
        creator="vehicle-1",
        cycles_per_bit=1.0,
        data_size=0.1,
        weather_scenario=scenario,
        deadline_type="Normal",
        path_loss_increase_db=path_loss,
        plr_increase_percent=plr,
    )


def vehicle_states() -> dict[str, VehicleState]:
    return {
        "vehicle-1": VehicleState(
            id="vehicle-1",
            time=0.0,
            x=150.0,
            y=150.0,
            speed=10.0,
            weather_scenario="BASE",
        )
    }


class GeneticBatchOffloaderTests(unittest.TestCase):
    def test_empty_batch(self) -> None:
        result = GeneticBatchOffloader().optimize([], {})

        self.assertEqual(result.assignments, ())
        self.assertEqual(result.evaluation.cost, 0.0)

    def test_queue_contention_is_included(self) -> None:
        optimizer = GeneticBatchOffloader(
            GeneticOffloaderConfig(local_capacity=1)
        )
        tasks = [make_task(f"task-{index}", deadline=30.0) for index in range(3)]

        evaluation = optimizer.evaluate(
            (OffloadTarget.LOCAL,) * 3,
            tasks,
            vehicle_states(),
        )

        self.assertGreater(evaluation.total_queue_delay, 0.0)
        self.assertEqual(evaluation.deadline_misses, 0)

    def test_deadline_penalty_prefers_feasible_batch(self) -> None:
        optimizer = GeneticBatchOffloader(
            GeneticOffloaderConfig(
                population_size=24,
                max_generations=20,
                time_limit_seconds=0.2,
                seed=4,
            )
        )
        tasks = [
            make_task(f"task-{index}", deadline=7.0, exec_time=10.0)
            for index in range(4)
        ]

        all_local = optimizer.evaluate(
            (OffloadTarget.LOCAL,) * len(tasks),
            tasks,
            vehicle_states(),
        )
        result = optimizer.optimize(tasks, vehicle_states())

        self.assertGreater(all_local.deadline_misses, 0)
        self.assertEqual(result.evaluation.deadline_misses, 0)
        self.assertLess(result.evaluation.cost, all_local.cost)

    def test_adverse_weather_increases_network_assignment_cost(self) -> None:
        optimizer = GeneticBatchOffloader(
            GeneticOffloaderConfig(
                reliability_weight=1.0,
                path_loss_delay_per_db=0.05,
            )
        )
        base = make_task("base", deadline=20.0)
        fog = make_task(
            "fog",
            deadline=20.0,
            plr=30.0,
            path_loss=6.0,
            scenario="FOG",
        )

        base_cost = optimizer.evaluate(
            (OffloadTarget.FOG,),
            (base,),
            vehicle_states(),
        ).cost
        adverse_cost = optimizer.evaluate(
            (OffloadTarget.FOG,),
            (fog,),
            vehicle_states(),
        ).cost

        self.assertGreater(adverse_cost, base_cost)

    def test_optimizer_changes_decision_when_weather_risk_is_high(self) -> None:
        optimizer = GeneticBatchOffloader(
            GeneticOffloaderConfig(
                population_size=12,
                max_generations=5,
                time_limit_seconds=0.1,
                reliability_weight=1.0,
                seed=1,
            )
        )
        base = make_task("base", deadline=20.0)
        adverse = make_task(
            "adverse",
            deadline=20.0,
            plr=90.0,
            path_loss=6.0,
            scenario="FOG",
        )

        base_result = optimizer.optimize((base,), vehicle_states())
        adverse_result = optimizer.optimize((adverse,), vehicle_states())

        self.assertNotEqual(base_result.assignments, (OffloadTarget.LOCAL,))
        self.assertEqual(adverse_result.assignments, (OffloadTarget.LOCAL,))

    def test_optimizer_respects_practical_time_bound(self) -> None:
        limit = 0.03
        optimizer = GeneticBatchOffloader(
            GeneticOffloaderConfig(
                population_size=20,
                max_generations=100_000,
                time_limit_seconds=limit,
            )
        )
        tasks = [make_task(f"task-{index}") for index in range(12)]

        started = time.perf_counter()
        result = optimizer.optimize(tasks, vehicle_states())
        wall_time = time.perf_counter() - started

        self.assertTrue(result.stopped_by_time_limit)
        self.assertLess(wall_time, 0.20)
        self.assertEqual(len(result.assignments), len(tasks))


class RLFallbackControllerTests(unittest.TestCase):
    def test_switches_to_ga_and_recovers_to_rl(self) -> None:
        optimizer = GeneticBatchOffloader(
            GeneticOffloaderConfig(
                population_size=8,
                max_generations=2,
                time_limit_seconds=0.05,
            )
        )

        def rl_policy(tasks, states):
            return [OffloadTarget.CLOUD] * len(tasks)

        controller = RLGeneticFallbackController(
            rl_policy=rl_policy,
            genetic_offloader=optimizer,
            fallback_reward_threshold=-5.0,
            recovery_reward_threshold=0.0,
            reward_window=2,
            consecutive_windows=1,
        )
        tasks = [make_task("task")]

        self.assertEqual(
            controller.choose_batch(tasks, vehicle_states()),
            (OffloadTarget.CLOUD,),
        )
        controller.observe_reward(-10.0)
        controller.observe_reward(-10.0)
        self.assertTrue(controller.in_fallback)
        controller.choose_batch(tasks, vehicle_states())
        self.assertEqual(controller.last_source, "GA")

        controller.observe_reward(5.0)
        controller.observe_reward(5.0)
        self.assertFalse(controller.in_fallback)
        self.assertEqual(
            controller.choose_batch(tasks, vehicle_states()),
            (OffloadTarget.CLOUD,),
        )
        self.assertEqual(controller.last_source, "RL")


if __name__ == "__main__":
    unittest.main()
