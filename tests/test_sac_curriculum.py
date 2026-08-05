from __future__ import annotations

import unittest

from scripts.generate_sac_curriculum import build_plan
from genetic_offloader_runner import dataset_group


class SacCurriculumPlanTests(unittest.TestCase):
    def test_plan_has_train_finetune_and_test_datasets(self) -> None:
        plans = build_plan()
        by_category = {
            category: [plan for plan in plans if plan.category == category]
            for category in ("train", "finetune", "test")
        }

        self.assertEqual(len(by_category["train"]), 12)
        self.assertEqual(len(by_category["finetune"]), 8)
        self.assertEqual(len(by_category["test"]), 8)
        self.assertTrue(all(plan.duration == 1000 for plan in plans))
        self.assertEqual(
            {plan.name for plan in by_category["test"]},
            {
                "test_base",
                "test_rain",
                "test_snow",
                "test_fog",
                "test_fast_mix",
                "test_slow_mix",
                "test_random_mix_1",
                "test_random_mix_2",
            },
        )

    def test_every_weather_block_has_a_unique_seed(self) -> None:
        plans = build_plan()
        seeds = [block.seed for plan in plans for block in plan.blocks]

        self.assertEqual(len(seeds), len(set(seeds)))

    def test_finetune_and_test_mixed_orders_are_unseen(self) -> None:
        plans = build_plan()
        observed_orders: dict[str, set[tuple[str, ...]]] = {
            "slow": set(),
            "fast": set(),
        }
        for category in ("train", "finetune", "test"):
            for plan in plans:
                if plan.category != category or plan.stage not in observed_orders:
                    continue
                order = tuple(block.scenario for block in plan.blocks)
                self.assertNotIn(order, observed_orders[plan.stage])
                observed_orders[plan.stage].add(order)

    def test_dataset_names_start_with_their_category(self) -> None:
        for plan in build_plan():
            self.assertTrue(plan.name.startswith(f"{plan.category}_"))

    def test_expected_stage_counts(self) -> None:
        plans = build_plan()
        counts = {
            category: {
                stage: sum(
                    plan.category == category and plan.stage == stage
                    for plan in plans
                )
                for stage in ("static", "slow", "fast", "random")
            }
            for category in ("train", "finetune", "test")
        }
        self.assertEqual(
            counts,
            {
                "train": {"static": 8, "slow": 4, "fast": 0, "random": 0},
                "finetune": {"static": 4, "slow": 2, "fast": 2, "random": 0},
                "test": {"static": 4, "slow": 1, "fast": 1, "random": 2},
            },
        )

    def test_evaluation_dataset_groups_are_unambiguous(self) -> None:
        self.assertEqual(dataset_group("test_base"), "BASE")
        self.assertEqual(dataset_group("test_rain"), "RAIN")
        self.assertEqual(dataset_group("test_snow"), "SNOW")
        self.assertEqual(dataset_group("test_fog"), "FOG")
        self.assertEqual(dataset_group("test_fast_mix"), "FAST_MIXED")
        self.assertEqual(dataset_group("test_slow_mix"), "SLOW_MIXED")
        self.assertEqual(dataset_group("test_random_mix_1"), "RANDOM_MIX_1")
        self.assertEqual(dataset_group("test_random_mix_2"), "RANDOM_MIX_2")

    def test_random_test_blocks_are_between_100_and_200_steps(self) -> None:
        random_tests = [
            plan
            for plan in build_plan()
            if plan.category == "test" and plan.stage == "random"
        ]
        self.assertEqual(len(random_tests), 2)
        for plan in random_tests:
            self.assertEqual(plan.duration, 1000)
            self.assertTrue(
                all(100 <= block.duration <= 200 for block in plan.blocks)
            )
            self.assertEqual(
                {block.scenario for block in plan.blocks},
                {"BASE", "RAIN", "SNOW", "FOG"},
            )


if __name__ == "__main__":
    unittest.main()
