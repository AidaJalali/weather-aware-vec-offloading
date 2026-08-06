from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from discrete_sac import (
    BalancedReplayBuffer,
    DiscreteSACAgent,
    DiscreteSACConfig,
)
from weather_scenarios import WeatherScenario


class BalancedReplayBufferTests(unittest.TestCase):
    def test_minibatch_is_balanced_across_available_weather(self) -> None:
        replay = BalancedReplayBuffer(capacity=80, observation_dim=2, seed=3)
        for scenario_index, scenario in enumerate(WeatherScenario):
            for _ in range(10):
                observation = np.full(2, scenario_index, dtype=np.float32)
                replay.add(observation, 0, 0.0, observation, False, scenario)

        observations = replay.sample(20)[0]
        values, counts = np.unique(observations[:, 0], return_counts=True)

        np.testing.assert_array_equal(values, np.arange(4))
        np.testing.assert_array_equal(counts, np.full(4, 5))


class DiscreteSACAgentTests(unittest.TestCase):
    def _agent(self) -> DiscreteSACAgent:
        return DiscreteSACAgent(
            observation_dim=5,
            config=DiscreteSACConfig(
                hidden_sizes=(8,),
                replay_capacity=64,
                batch_size=8,
                learning_starts=8,
                update_every=1,
                seed=5,
            ),
        )

    def test_update_and_checkpoint_round_trip(self) -> None:
        agent = self._agent()
        scenarios = tuple(WeatherScenario)
        for index in range(16):
            observation = np.full(5, index / 16, dtype=np.float32)
            next_observation = np.full(5, (index + 1) / 16, dtype=np.float32)
            agent.observe(
                observation,
                index % 3,
                -0.25,
                next_observation,
                False,
                scenarios[index % len(scenarios)],
            )

        metrics = agent.update()
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
        sample = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        probabilities = agent.action_probabilities(sample)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            agent.save(path, observation_scale={"example": 1.0})
            restored, checkpoint = DiscreteSACAgent.load(path)

        np.testing.assert_allclose(
            restored.action_probabilities(sample), probabilities
        )
        self.assertEqual(checkpoint["format"], "categorical_discrete_sac_v1")


if __name__ == "__main__":
    unittest.main()
