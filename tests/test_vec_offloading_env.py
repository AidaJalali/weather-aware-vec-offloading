from __future__ import annotations

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from algorithms import OffloadTarget
from compare_reward_profiles import compare_fixed_and_adaptive
from infrastructure import ExecutionModel, TaskRecord, VehicleState
from vec_offloading_env import (
    RewardConfig,
    RewardProfile,
    VECOffloadingEnv,
)


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
        path_loss_increase_db=0.0,
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
    def test_action_regions_map_to_offloading_targets(self) -> None:
        self.assertEqual(
            VECOffloadingEnv.action_to_target(np.array([-1.0])),
            OffloadTarget.LOCAL,
        )
        self.assertEqual(
            VECOffloadingEnv.action_to_target(np.array([0.0])),
            OffloadTarget.FOG,
        )
        self.assertEqual(
            VECOffloadingEnv.action_to_target(np.array([1.0])),
            OffloadTarget.CLOUD,
        )

    def test_observation_contains_weather_loss_queue_and_capacity_state(self) -> None:
        first = TaskRecord(
            **{
                **make_task("first", scenario="RAIN").__dict__,
                "plr_increase_percent": 15.0,
            }
        )
        second = make_task("second", release_time=1.0, scenario="RAIN")
        env = VECOffloadingEnv(
            [first, second],
            {
                (0.0, "vehicle-1"): make_vehicle(scenario="RAIN"),
                (1.0, "vehicle-1"): make_vehicle(
                    time=1.0,
                    scenario="RAIN",
                ),
            },
        )

        observation, _ = env.reset(seed=7)
        fields = {
            name: observation[index]
            for index, name in enumerate(env.observation_fields)
        }
        self.assertEqual(fields["weather_base"], 0.0)
        self.assertEqual(fields["weather_rain"], 1.0)
        self.assertAlmostEqual(fields["fog_packet_loss_probability"], 0.17)
        self.assertAlmostEqual(fields["cloud_packet_loss_probability"], 0.20)
        self.assertEqual(fields["local_queue_occupancy"], 0.0)
        self.assertEqual(fields["local_available_capacity"], 1.0)

        next_observation = env.step(
            np.array([-1.0], dtype=np.float32)
        )[0]
        next_fields = {
            name: next_observation[index]
            for index, name in enumerate(env.observation_fields)
        }
        self.assertEqual(next_fields["local_queue_occupancy"], 1.0)
        self.assertEqual(next_fields["local_available_capacity"], 0.0)

    def test_local_step_uses_existing_execution_model(self) -> None:
        task = make_task("task")
        model = ExecutionModel(local_speedup=1.0)
        env = VECOffloadingEnv(
            [task],
            {"vehicle-1": make_vehicle()},
            execution_model=model,
        )

        observation, reset_info = env.reset(seed=7)
        next_observation, reward, terminated, truncated, info = env.step(
            np.array([-1.0], dtype=np.float32)
        )

        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(reset_info["task_id"], "task")
        self.assertEqual(info["target"], "LOCAL")
        self.assertEqual(info["latency"], task.exec_time)
        self.assertEqual(
            info["total_system_energy"],
            task.power * task.exec_time,
        )
        self.assertLess(reward, 0.0)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        np.testing.assert_array_equal(
            next_observation,
            np.zeros(env.observation_space.shape, dtype=np.float32),
        )

    def test_cloud_step_uses_weather_aware_dynamic_backhaul(self) -> None:
        base_task = make_task("base", scenario="BASE")
        fog_task = make_task("fog", scenario="FOG")
        base_env = VECOffloadingEnv(
            [base_task],
            {"vehicle-1": make_vehicle()},
        )
        fog_env = VECOffloadingEnv(
            [fog_task],
            {"vehicle-1": make_vehicle(scenario="FOG")},
        )

        base_env.reset(seed=11)
        fog_env.reset(seed=11)
        _, _, _, _, base_info = base_env.step(
            np.array([1.0], dtype=np.float32)
        )
        _, _, _, _, fog_info = fog_env.step(
            np.array([1.0], dtype=np.float32)
        )

        self.assertGreater(
            fog_info["backhaul_delay"],
            base_info["backhaul_delay"],
        )

    def test_adaptive_reward_selects_weather_profile(self) -> None:
        task = make_task("rain", scenario="RAIN")
        env = VECOffloadingEnv(
            [task],
            {"vehicle-1": make_vehicle(scenario="RAIN")},
            reward_config=RewardConfig.adaptive_default(),
        )

        env.reset(seed=11)
        _, _, _, _, info = env.step(
            np.array([-1.0], dtype=np.float32)
        )

        self.assertEqual(info["reward_profile"], "weather_adaptive")
        self.assertEqual(info["latency_weight"], 0.40)
        self.assertEqual(info["energy_weight"], 0.20)
        self.assertEqual(info["reliability_weight"], 0.25)
        self.assertEqual(info["deadline_miss_penalty"], 6.0)

    def test_custom_weather_profile_overrides_default(self) -> None:
        custom = RewardProfile(
            latency_weight=0.8,
            energy_weight=0.1,
            reliability_weight=0.1,
            packet_loss_penalty=4.0,
            deadline_miss_penalty=9.0,
        )
        config = RewardConfig(
            name="custom",
            weather_profiles={"SNOW": custom},
        )

        self.assertIs(config.for_weather("SNOW"), custom)
        self.assertEqual(
            config.for_weather("BASE"),
            config.default,
        )

    def test_fixed_and_adaptive_comparison_replays_same_trajectory(self) -> None:
        tasks = [
            make_task("base", release_time=0.0, scenario="BASE"),
            make_task("rain", release_time=1.0, scenario="RAIN"),
        ]
        states = {
            (0.0, "vehicle-1"): make_vehicle(
                time=0.0,
                scenario="BASE",
            ),
            (1.0, "vehicle-1"): make_vehicle(
                time=1.0,
                scenario="RAIN",
            ),
        }

        rows = compare_fixed_and_adaptive(
            tasks,
            states,
            actions=[-1.0, -1.0],
            seed=17,
        )
        by_key = {(row.mode, row.scenario): row for row in rows}
        fixed = by_key[("fixed", "ALL")]
        adaptive = by_key[("weather_adaptive", "ALL")]

        self.assertEqual(fixed.task_count, adaptive.task_count)
        self.assertEqual(fixed.average_latency, adaptive.average_latency)
        self.assertEqual(fixed.average_energy, adaptive.average_energy)
        self.assertEqual(fixed.deadline_misses, adaptive.deadline_misses)
        self.assertEqual(fixed.packet_losses, adaptive.packet_losses)
        self.assertNotEqual(fixed.average_reward, adaptive.average_reward)

    def test_reset_seed_reproduces_packet_outcome(self) -> None:
        task = TaskRecord(
            **{
                **make_task("lossy").__dict__,
                "plr_increase_percent": 45.0,
            }
        )
        env = VECOffloadingEnv(
            [task],
            {"vehicle-1": make_vehicle()},
        )

        env.reset(seed=123)
        self.assertEqual(env._channel.seed, 123)
        first = env.step(np.array([0.0], dtype=np.float32))[4]
        env.reset(seed=123)
        second = env.step(np.array([0.0], dtype=np.float32))[4]

        self.assertEqual(first["packet_lost"], second["packet_lost"])
        self.assertEqual(
            first["transmission_attempts"],
            second["transmission_attempts"],
        )
        self.assertIn("final_failure_probability", first)
        self.assertIn("transmission_energy", first)

    def test_environment_passes_gymnasium_checker(self) -> None:
        tasks = [
            make_task("task-1", release_time=0.0),
            make_task("task-2", release_time=1.0),
        ]
        states = {
            (0.0, "vehicle-1"): make_vehicle(time=0.0),
            (1.0, "vehicle-1"): make_vehicle(time=1.0),
        }
        env = VECOffloadingEnv(
            tasks,
            states,
            reward_config=RewardConfig(),
        )

        check_env(env, skip_render_check=True)

    def test_missing_vehicle_state_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "No vehicle state"):
            VECOffloadingEnv([make_task("task")], {})


if __name__ == "__main__":
    unittest.main()
