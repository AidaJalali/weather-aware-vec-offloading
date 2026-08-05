from __future__ import annotations

import unittest

from algorithms import OffloadTarget
from online_hybrid import (
    PacketLossFallbackMonitor,
    PacketLossMonitorConfig,
    TEST_ORDER,
    build_parser,
    target_to_action,
)
from vec_offloading_env import VECOffloadingEnv


class PacketLossFallbackMonitorTests(unittest.TestCase):
    def test_switches_to_ga_and_recovers_with_hysteresis(self) -> None:
        monitor = PacketLossFallbackMonitor(
            PacketLossMonitorConfig(
                window_size=4,
                check_every=2,
                fallback_loss_rate=0.5,
                recovery_loss_rate=0.25,
                consecutive_checks=2,
                minimum_fallback_tasks=4,
            )
        )

        events = [True, True, False, False, True, True]
        switches = [monitor.observe(event) for event in events]
        self.assertEqual(switches[-1], "SAC_TO_GA")
        self.assertTrue(monitor.in_fallback)

        recovery_switches = [monitor.observe(False) for _ in range(6)]
        self.assertEqual(recovery_switches[-1], "GA_TO_SAC")
        self.assertFalse(monitor.in_fallback)

    def test_thresholds_require_hysteresis_gap(self) -> None:
        with self.assertRaises(ValueError):
            PacketLossMonitorConfig(
                fallback_loss_rate=0.05,
                recovery_loss_rate=0.05,
            )

    def test_target_actions_map_back_to_the_same_target(self) -> None:
        for target in OffloadTarget:
            action = target_to_action(target)
            self.assertEqual(
                target,
                VECOffloadingEnv.action_to_target(action),
            )

    def test_online_protocol_defaults_to_ordered_test_streams(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.split, "test")
        self.assertEqual(
            TEST_ORDER,
            (
                "test_base",
                "test_rain",
                "test_snow",
                "test_fog",
                "test_fast_mix",
                "test_slow_mix",
                "test_random_mix_1",
                "test_random_mix_2",
            ),
        )


if __name__ == "__main__":
    unittest.main()
