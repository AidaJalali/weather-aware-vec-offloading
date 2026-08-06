from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from weather_scenarios import WeatherScenario, normalize_scenario


@dataclass(frozen=True)
class DiscreteSACConfig:
    hidden_sizes: tuple[int, ...] = (64, 64)
    replay_capacity: int = 500_000
    batch_size: int = 128
    learning_starts: int = 1_000
    update_every: int = 4
    gradient_steps: int = 1
    learning_rate: float = 3e-4
    gamma: float = 0.995
    tau: float = 0.005
    initial_alpha: float = 0.20
    target_entropy_ratio: float = 0.98
    max_gradient_norm: float = 10.0
    seed: int = 37

    def __post_init__(self) -> None:
        if not self.hidden_sizes or min(self.hidden_sizes) <= 0:
            raise ValueError("hidden_sizes must contain positive values")
        if self.replay_capacity < 4 or self.batch_size <= 0:
            raise ValueError("replay_capacity and batch_size must be positive")
        if self.learning_starts < 0 or self.update_every <= 0:
            raise ValueError("invalid training schedule")
        if self.gradient_steps <= 0 or self.learning_rate <= 0.0:
            raise ValueError("gradient settings must be positive")
        if not 0.0 < self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma and tau must be in (0, 1]")
        if self.initial_alpha <= 0.0:
            raise ValueError("initial_alpha must be positive")


class _WeatherBuffer:
    def __init__(self, capacity: int, observation_dim: int) -> None:
        self.capacity = capacity
        self.observations = np.zeros(
            (capacity, observation_dim), dtype=np.float32
        )
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        self.observations[index] = observation
        self.next_observations[index] = next_observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.dones[index] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


class BalancedReplayBuffer:
    """Fixed-size replay with weather-balanced random sampling."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        seed: int,
    ) -> None:
        scenarios = tuple(WeatherScenario)
        per_weather = math.ceil(capacity / len(scenarios))
        self.capacity = capacity
        self._buffers = {
            scenario: _WeatherBuffer(per_weather, observation_dim)
            for scenario in scenarios
        }
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return min(
            sum(buffer.size for buffer in self._buffers.values()),
            self.capacity,
        )

    def weather_counts(self) -> dict[str, int]:
        return {
            scenario.value: buffer.size
            for scenario, buffer in self._buffers.items()
        }

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        scenario: str | WeatherScenario,
    ) -> None:
        self._buffers[normalize_scenario(scenario)].add(
            observation,
            action,
            reward,
            next_observation,
            done,
        )

    def sample(self, batch_size: int) -> tuple[np.ndarray, ...]:
        available = [
            buffer for buffer in self._buffers.values() if buffer.size > 0
        ]
        if not available:
            raise ValueError("cannot sample an empty replay buffer")

        base = batch_size // len(available)
        remainder = batch_size % len(available)
        parts: list[tuple[np.ndarray, ...]] = []
        for index, buffer in enumerate(available):
            count = base + int(index < remainder)
            selected = self._rng.integers(0, buffer.size, size=count)
            parts.append(
                (
                    buffer.observations[selected],
                    buffer.actions[selected],
                    buffer.rewards[selected],
                    buffer.next_observations[selected],
                    buffer.dones[selected],
                )
            )

        combined = tuple(
            np.concatenate([part[field] for part in parts], axis=0)
            for field in range(5)
        )
        order = self._rng.permutation(batch_size)
        return tuple(values[order] for values in combined)


def _mlp(input_dim: int, output_dim: int, hidden_sizes: Sequence[int]) -> nn.Module:
    layers: list[nn.Module] = []
    previous = input_dim
    for width in hidden_sizes:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class DiscreteSACAgent:
    """Small categorical Soft Actor-Critic agent for three offloading actions."""

    def __init__(
        self,
        observation_dim: int,
        action_count: int = 3,
        *,
        config: DiscreteSACConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = config or DiscreteSACConfig()
        self.observation_dim = int(observation_dim)
        self.action_count = int(action_count)
        self.device = torch.device(device)
        torch.manual_seed(self.config.seed)
        self._rng = np.random.default_rng(self.config.seed)

        hidden = self.config.hidden_sizes
        self.actor = _mlp(self.observation_dim, self.action_count, hidden).to(
            self.device
        )
        self.critic_one = _mlp(
            self.observation_dim, self.action_count, hidden
        ).to(self.device)
        self.critic_two = _mlp(
            self.observation_dim, self.action_count, hidden
        ).to(self.device)
        self.target_one = _mlp(
            self.observation_dim, self.action_count, hidden
        ).to(self.device)
        self.target_two = _mlp(
            self.observation_dim, self.action_count, hidden
        ).to(self.device)
        self.target_one.load_state_dict(self.critic_one.state_dict())
        self.target_two.load_state_dict(self.critic_two.state_dict())
        self.target_one.requires_grad_(False)
        self.target_two.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.config.learning_rate
        )
        critic_parameters = list(self.critic_one.parameters()) + list(
            self.critic_two.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters, lr=self.config.learning_rate
        )
        self.log_alpha = torch.tensor(
            math.log(self.config.initial_alpha),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=self.config.learning_rate
        )
        self.target_entropy = (
            self.config.target_entropy_ratio * math.log(self.action_count)
        )
        self.replay = BalancedReplayBuffer(
            self.config.replay_capacity,
            self.observation_dim,
            self.config.seed,
        )
        self.environment_steps = 0
        self.gradient_updates = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        ).reshape(1, -1)
        with torch.no_grad():
            probabilities = torch.softmax(self.actor(tensor), dim=-1)
        return probabilities.cpu().numpy()[0]

    def select_action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
    ) -> int:
        probabilities = self.action_probabilities(observation)
        if deterministic:
            return int(np.argmax(probabilities))
        return int(self._rng.choice(self.action_count, p=probabilities))

    def random_action(self) -> int:
        return int(self._rng.integers(self.action_count))

    def observe(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        scenario: str | WeatherScenario,
    ) -> None:
        self.replay.add(
            observation,
            action,
            reward,
            next_observation,
            done,
            scenario,
        )
        self.environment_steps += 1

    def ready_to_update(self) -> bool:
        return (
            len(self.replay) >= max(self.config.learning_starts, self.config.batch_size)
            and self.environment_steps % self.config.update_every == 0
        )

    def update(self, gradient_steps: int | None = None) -> dict[str, float]:
        steps = gradient_steps or self.config.gradient_steps
        metrics: dict[str, float] = {}
        for _ in range(steps):
            arrays = self.replay.sample(self.config.batch_size)
            observations = torch.as_tensor(
                arrays[0], dtype=torch.float32, device=self.device
            )
            actions = torch.as_tensor(
                arrays[1], dtype=torch.long, device=self.device
            ).unsqueeze(1)
            rewards = torch.as_tensor(
                arrays[2], dtype=torch.float32, device=self.device
            ).unsqueeze(1)
            next_observations = torch.as_tensor(
                arrays[3], dtype=torch.float32, device=self.device
            )
            dones = torch.as_tensor(
                arrays[4], dtype=torch.float32, device=self.device
            ).unsqueeze(1)

            with torch.no_grad():
                next_logits = self.actor(next_observations)
                next_log_probabilities = F.log_softmax(next_logits, dim=-1)
                next_probabilities = next_log_probabilities.exp()
                target_values = torch.minimum(
                    self.target_one(next_observations),
                    self.target_two(next_observations),
                )
                next_value = (
                    next_probabilities
                    * (target_values - self.alpha.detach() * next_log_probabilities)
                ).sum(dim=1, keepdim=True)
                target = rewards + self.config.gamma * (1.0 - dones) * next_value

            q_one = self.critic_one(observations).gather(1, actions)
            q_two = self.critic_two(observations).gather(1, actions)
            critic_loss = F.mse_loss(q_one, target) + F.mse_loss(q_two, target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.critic_one.parameters())
                + list(self.critic_two.parameters()),
                self.config.max_gradient_norm,
            )
            self.critic_optimizer.step()

            logits = self.actor(observations)
            log_probabilities = F.log_softmax(logits, dim=-1)
            probabilities = log_probabilities.exp()
            with torch.no_grad():
                minimum_q = torch.minimum(
                    self.critic_one(observations),
                    self.critic_two(observations),
                )
            actor_loss = (
                probabilities
                * (self.alpha.detach() * log_probabilities - minimum_q)
            ).sum(dim=1).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.max_gradient_norm
            )
            self.actor_optimizer.step()

            entropy = -(probabilities * log_probabilities).sum(dim=1).mean()
            alpha_loss = -(
                self.log_alpha * (self.target_entropy - entropy.detach())
            )
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self._soft_update_targets()
            self.gradient_updates += 1
            metrics = {
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "alpha_loss": float(alpha_loss.detach().cpu()),
                "alpha": float(self.alpha.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }
        return metrics

    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target, source in (
                (self.target_one, self.critic_one),
                (self.target_two, self.critic_two),
            ):
                for target_parameter, source_parameter in zip(
                    target.parameters(), source.parameters()
                ):
                    target_parameter.mul_(1.0 - tau)
                    target_parameter.add_(tau * source_parameter)

    def save(
        self,
        path: str | Path,
        *,
        observation_scale: Mapping[str, float],
        metadata: Mapping[str, object] | None = None,
    ) -> Path:
        output = Path(path)
        if output.suffix != ".pt":
            output = output.with_suffix(".pt")
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "categorical_discrete_sac_v1",
                "observation_dim": self.observation_dim,
                "action_count": self.action_count,
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic_one": self.critic_one.state_dict(),
                "critic_two": self.critic_two.state_dict(),
                "target_one": self.target_one.state_dict(),
                "target_two": self.target_two.state_dict(),
                "log_alpha": float(self.log_alpha.detach().cpu()),
                "environment_steps": self.environment_steps,
                "gradient_updates": self.gradient_updates,
                "observation_scale": dict(observation_scale),
                "metadata": dict(metadata or {}),
            },
            output,
        )
        return output

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> tuple["DiscreteSACAgent", dict]:
        checkpoint = torch.load(
            Path(path), map_location=device, weights_only=False
        )
        if checkpoint.get("format") != "categorical_discrete_sac_v1":
            raise ValueError("checkpoint is not a categorical discrete SAC model")
        raw_config = dict(checkpoint["config"])
        raw_config["hidden_sizes"] = tuple(raw_config["hidden_sizes"])
        agent = cls(
            checkpoint["observation_dim"],
            checkpoint["action_count"],
            config=DiscreteSACConfig(**raw_config),
            device=device,
        )
        for network, key in (
            (agent.actor, "actor"),
            (agent.critic_one, "critic_one"),
            (agent.critic_two, "critic_two"),
            (agent.target_one, "target_one"),
            (agent.target_two, "target_two"),
        ):
            network.load_state_dict(checkpoint[key])
        agent.log_alpha.data.fill_(float(checkpoint["log_alpha"]))
        agent.environment_steps = int(checkpoint.get("environment_steps", 0))
        agent.gradient_updates = int(checkpoint.get("gradient_updates", 0))
        return agent, checkpoint
