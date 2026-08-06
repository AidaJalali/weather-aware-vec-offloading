from __future__ import annotations

import unittest

from online_hybrid import (
    HybridFallbackMonitor,
    HybridMonitorConfig,
    TEST_ORDER,
    build_parser,
)


class HybridFallbackMonitorTests(unittest.TestCase):
    def test_switches_on_deadline_degradation_and_recovers(self) -> None:
        monitor = HybridFallbackMonitor(
            HybridMonitorConfig(
                window_size=4,
                check_every=2,
                fallback_loss_rate=0.5,
                recovery_loss_rate=0.25,
                fallback_deadline_miss_rate=0.5,
                recovery_deadline_miss_rate=0.25,
                fallback_normalized_latency=1.0,
                recovery_normalized_latency=0.5,
                consecutive_checks=2,
                minimum_fallback_tasks=4,
            )
        )

        switches = []
        for _ in range(6):
            switches.append(
                monitor.observe(
                    loss_event=False,
                    deadline_missed=True,
                    normalized_latency=0.5,
                )
            )
        self.assertEqual(switches[-1], ("SAC_TO_GA", "deadline_miss"))
        self.assertTrue(monitor.in_fallback)

        recovery_switches = []
        for _ in range(8):
            recovery_switches.append(
                monitor.observe(
                    loss_event=False,
                    deadline_missed=False,
                    normalized_latency=0.2,
                )
            )
        self.assertIn(("GA_TO_SAC", "recovered"), recovery_switches)
        self.assertFalse(monitor.in_fallback)

    def test_switches_when_latency_is_bad_without_packet_loss(self) -> None:
        monitor = HybridFallbackMonitor(
            HybridMonitorConfig(
                window_size=2,
                check_every=1,
                consecutive_checks=1,
                minimum_fallback_tasks=0,
            )
        )
        monitor.observe(
            loss_event=False,
            deadline_missed=False,
            normalized_latency=2.0,
        )
        event, trigger = monitor.observe(
            loss_event=False,
            deadline_missed=False,
            normalized_latency=2.0,
        )
        self.assertEqual(event, "SAC_TO_GA")
        self.assertEqual(trigger, "latency")

    def test_thresholds_require_hysteresis_gap(self) -> None:
        with self.assertRaises(ValueError):
            HybridMonitorConfig(
                fallback_loss_rate=0.05,
                recovery_loss_rate=0.05,
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
