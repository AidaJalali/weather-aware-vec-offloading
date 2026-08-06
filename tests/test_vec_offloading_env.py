from __future__ import annotations

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from algorithms import OffloadTarget
from infrastructure import ExecutionModel, TaskRecord, VehicleState
from vec_offloading_env import RewardConfig, VECOffloadingEnv


def make_task(
    task_id: str,
    *,
    release_time: float = 0.0,
    scenario: str = "BASE",
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        release_time=release_time,
        deadline=release_time + 10.0,
        exec_time=5.0,
        power=2.0,
        creator="vehicle-1",
        cycles_per_bit=1.5,
        data_size=0.5,
        weather_scenario=scenario,
        deadline_type="Normal",
        plr_increase_percent=0.0,
    )


def make_vehicle(
    *,
    time: float = 0.0,
    scenario: str = "BASE",
) -> VehicleState:
    return VehicleState(
        id="vehicle-1",
        time=time,
        x=150.0,
        y=150.0,
        speed=12.0,
        weather_scenario=scenario,
    )


class VECOffloadingEnvTests(unittest.TestCase):
    def test_discrete_actions_map_to_targets(self) -> None:
        self.assertEqual(VECOffloadingEnv.action_to_target(0), OffloadTarget.LOCAL)
        self.assertEqual(VECOffloadingEnv.action_to_target(1), OffloadTarget.FOG)
        self.assertEqual(VECOffloadingEnv.action_to_target(2), OffloadTarget.CLOUD)
        for target in OffloadTarget:
            self.assertEqual(
                VECOffloadingEnv.action_to_target(
                    VECOffloadingEnv.target_to_action(target)
                ),
                target,
            )

    def test_observation_contains_loss_wait_and_remaining_batch_state(self) -> None:
        first = TaskRecord(
            **{
                **make_task("first", scenario="RAIN").__dict__,
                "plr_increase_percent": 15.0,
            }
        )
        second = make_task("second", scenario="RAIN")
        env = VECOffloadingEnv(
            [first, second],
            {(0.0, "vehicle-1"): make_vehicle(scenario="RAIN")},
        )

        observation, _ = env.reset(seed=7)
        fields = dict(zip(env.observation_fields, observation))
        self.assertEqual(fields["weather_base"], 0.0)
        self.assertEqual(fields["weather_rain"], 1.0)
        self.assertAlmostEqual(
            fields["fog_terminal_loss_probability"], 0.17**3
        )
        self.assertAlmostEqual(
            fields["cloud_terminal_loss_probability"], 0.20**3
        )
        self.assertEqual(fields["local_estimated_wait"], 0.0)
        self.assertEqual(fields["remaining_tasks"], 1.0)

        next_observation = env.step(0)[0]
        next_fields = dict(zip(env.observation_fields, next_observation))
        self.assertGreater(next_fields["local_estimated_wait"], 0.0)
        self.assertEqual(next_fields["remaining_tasks"], 0.5)
        self.assertLess(next_fields["remaining_compute_demand"], 1.0)
        self.assertLess(next_fields["remaining_data_volume"], 1.0)

    def test_local_step_uses_fixed_bounded_reward(self) -> None:
        task = make_task("task")
        env = VECOffloadingEnv(
            [task],
            {"vehicle-1": make_vehicle()},
            execution_model=ExecutionModel(local_speedup=1.0),
        )

        observation, reset_info = env.reset(seed=7)
        next_observation, reward, terminated, truncated, info = env.step(0)

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(reset_info["task_id"], "task")
        self.assertEqual(info["target"], "LOCAL")
        self.assertEqual(info["latency"], task.exec_time)
        self.assertEqual(info["total_system_energy"], task.power * task.exec_time)
        self.assertAlmostEqual(reward, -0.1625)
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 0.0)
        self.assertEqual(info["reward_profile"], "fixed_bounded")
        self.assertEqual(info["loss_weight"], 0.50)
        self.assertEqual(info["latency_weight"], 0.35)
        self.assertEqual(info["energy_weight"], 0.15)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        np.testing.assert_array_equal(
            next_observation,
            np.zeros(env.observation_space.shape, dtype=np.float32),
        )

    def test_cloud_step_uses_weather_aware_dynamic_backhaul(self) -> None:
        base_env = VECOffloadingEnv(
            [make_task("base", scenario="BASE")],
            {"vehicle-1": make_vehicle()},
        )
        fog_env = VECOffloadingEnv(
            [make_task("fog", scenario="FOG")],
            {"vehicle-1": make_vehicle(scenario="FOG")},
        )
        base_env.reset(seed=11)
        fog_env.reset(seed=11)
        base_info = base_env.step(2)[4]
        fog_info = fog_env.step(2)[4]

        self.assertGreater(fog_info["backhaul_delay"], base_info["backhaul_delay"])

    def test_reset_seed_reproduces_packet_outcome(self) -> None:
        task = TaskRecord(
            **{
                **make_task("lossy").__dict__,
                "plr_increase_percent": 45.0,
            }
        )
        env = VECOffloadingEnv([task], {"vehicle-1": make_vehicle()})

        env.reset(seed=123)
        first = env.step(1)[4]
        env.reset(seed=123)
        second = env.step(1)[4]

        self.assertEqual(first["packet_lost"], second["packet_lost"])
        self.assertEqual(
            first["transmission_attempts"], second["transmission_attempts"]
        )

    def test_reward_config_rejects_invalid_weights(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            RewardConfig(loss_weight=0.5, latency_weight=0.5, energy_weight=0.5)

    def test_environment_passes_gymnasium_checker(self) -> None:
        tasks = [
            make_task("task-1", release_time=0.0),
            make_task("task-2", release_time=1.0),
        ]
        states = {
            (0.0, "vehicle-1"): make_vehicle(time=0.0),
            (1.0, "vehicle-1"): make_vehicle(time=1.0),
        }
        check_env(VECOffloadingEnv(tasks, states), skip_render_check=True)

    def test_missing_vehicle_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "No vehicle state"):
            VECOffloadingEnv([make_task("task")], {})


if __name__ == "__main__":
    unittest.main()
