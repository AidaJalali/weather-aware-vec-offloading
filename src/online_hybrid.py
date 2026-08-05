from __future__ import annotations

import argparse
import csv
import os
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vec-cache")

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

from algorithms import GeneticBatchOffloader, GeneticOffloaderConfig, OffloadTarget
from genetic_offloader_runner import (
    DatasetFiles,
    ProgressBar,
    batch_tasks,
    config_from_json,
    discover_datasets,
)
from infrastructure import load_tasks, load_vehicle_states
from vec_offloading_env import RewardConfig, VECOffloadingEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "datasets"
DEFAULT_PRETRAINED_MODEL = (
    PROJECT_ROOT / "outputs" / "models" / "sac" / "sac_pretrained_final.zip"
)
DEFAULT_ADAPTED_MODEL = (
    PROJECT_ROOT / "outputs" / "models" / "sac_online" / "sac_adapted_final.zip"
)
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "online"
FINETUNE_ORDER = (
    "finetune_base",
    "finetune_rain",
    "finetune_snow",
    "finetune_fog",
    "finetune_slow_mix_1",
    "finetune_slow_mix_2",
    "finetune_fast_mix_1",
    "finetune_fast_mix_2",
)
TEST_ORDER = (
    "test_base",
    "test_rain",
    "test_snow",
    "test_fog",
    "test_fast_mix",
    "test_slow_mix",
    "test_random_mix_1",
    "test_random_mix_2",
)


@dataclass(frozen=True)
class PacketLossMonitorConfig:
    window_size: int = 100
    check_every: int = 20
    fallback_loss_rate: float = 0.05
    recovery_loss_rate: float = 0.02
    consecutive_checks: int = 2
    minimum_fallback_tasks: int = 100

    def __post_init__(self) -> None:
        if self.window_size <= 0 or self.check_every <= 0:
            raise ValueError("window_size and check_every must be positive")
        if self.consecutive_checks <= 0 or self.minimum_fallback_tasks < 0:
            raise ValueError("invalid fallback duration settings")
        if not 0.0 <= self.recovery_loss_rate < self.fallback_loss_rate <= 1.0:
            raise ValueError(
                "loss thresholds must satisfy 0 <= recovery < fallback <= 1"
            )


class PacketLossFallbackMonitor:
    """Switch using a rolling rate of observed transmission-loss events."""

    def __init__(self, config: PacketLossMonitorConfig | None = None) -> None:
        self.config = config or PacketLossMonitorConfig()
        self._events: deque[float] = deque(maxlen=self.config.window_size)
        self._observations = 0
        self._bad_checks = 0
        self._good_checks = 0
        self._fallback_tasks = 0
        self.in_fallback = False

    @property
    def source(self) -> str:
        return "GA" if self.in_fallback else "SAC"

    @property
    def rolling_loss_rate(self) -> float | None:
        if not self._events:
            return None
        return sum(self._events) / len(self._events)

    def observe(self, loss_event: bool) -> str:
        self._events.append(float(loss_event))
        self._observations += 1
        if self.in_fallback:
            self._fallback_tasks += 1

        if len(self._events) < self.config.window_size:
            return ""
        if self._observations % self.config.check_every != 0:
            return ""

        loss_rate = self.rolling_loss_rate
        assert loss_rate is not None
        if not self.in_fallback:
            self._bad_checks = (
                self._bad_checks + 1
                if loss_rate >= self.config.fallback_loss_rate
                else 0
            )
            if self._bad_checks >= self.config.consecutive_checks:
                self.in_fallback = True
                self._bad_checks = 0
                self._good_checks = 0
                self._fallback_tasks = 0
                return "SAC_TO_GA"
            return ""

        can_recover = self._fallback_tasks >= self.config.minimum_fallback_tasks
        self._good_checks = (
            self._good_checks + 1
            if can_recover and loss_rate <= self.config.recovery_loss_rate
            else 0
        )
        if self._good_checks >= self.config.consecutive_checks:
            self.in_fallback = False
            self._good_checks = 0
            self._bad_checks = 0
            return "GA_TO_SAC"
        return ""


def target_to_action(target: OffloadTarget) -> np.ndarray:
    values = {
        OffloadTarget.LOCAL: -2.0 / 3.0,
        OffloadTarget.FOG: 0.0,
        OffloadTarget.CLOUD: 2.0 / 3.0,
    }
    return np.asarray([values[target]], dtype=np.float32)


class OnlineHybridEnv(VECOffloadingEnv):
    """Execute SAC normally and GeneticBatch while the loss monitor is active."""

    def __init__(
        self,
        *args,
        genetic_offloader: GeneticBatchOffloader,
        monitor: PacketLossFallbackMonitor,
        batch_window_seconds: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if batch_window_seconds <= 0:
            raise ValueError("batch_window_seconds must be positive")
        self.genetic_offloader = genetic_offloader
        self.monitor = monitor
        self.batch_window_seconds = batch_window_seconds
        self._ga_assignments: dict[str, OffloadTarget] = {}

    def reset(self, **kwargs):
        self._ga_assignments.clear()
        return super().reset(**kwargs)

    def _remaining_batch(self) -> tuple:
        assert self._index is not None
        task = self.tasks[self._index]
        bucket = int(task.release_time) // self.batch_window_seconds
        remaining = []
        for candidate in self.tasks[self._index :]:
            candidate_bucket = (
                int(candidate.release_time) // self.batch_window_seconds
            )
            if candidate_bucket != bucket:
                break
            remaining.append(candidate)
        return tuple(remaining)

    def _genetic_target(self) -> OffloadTarget:
        assert self._index is not None
        task = self.tasks[self._index]
        if task.id not in self._ga_assignments:
            batch = self._remaining_batch()
            result = self.genetic_offloader.optimize(
                batch,
                self.vehicle_states,
                resource_state=self._resource_state,
                channel_randomness=self._channel,
                network_load_by_time=self.network_load_by_time,
            )
            self._ga_assignments = result.by_task_id(batch)
        return self._ga_assignments.pop(task.id)

    def step(self, action: np.ndarray):
        if self._index is None or self._index >= len(self.tasks):
            return super().step(action)

        policy_target = self.action_to_target(action)
        source = self.monitor.source
        target = self._genetic_target() if source == "GA" else policy_target
        executed_action = target_to_action(target)
        observation, reward, terminated, truncated, info = super().step(
            executed_action
        )

        transmission_loss_event = bool(
            info["packet_lost"] or int(info["retransmission_count"]) > 0
        )
        switch_event = self.monitor.observe(transmission_loss_event)
        if switch_event == "GA_TO_SAC":
            self._ga_assignments.clear()

        info.update(
            {
                "controller_source": source,
                "controller_source_after_step": self.monitor.source,
                "policy_target": policy_target.value,
                "executed_action": float(executed_action[0]),
                "transmission_loss_event": transmission_loss_event,
                "rolling_loss_rate": self.monitor.rolling_loss_rate,
                "switch_event": switch_event,
            }
        )
        return observation, reward, terminated, truncated, info


@dataclass(frozen=True)
class OnlineRunRow:
    algorithm: str
    split: str
    dataset: str
    scenario_group: str
    task_index: int
    task_id: str
    scenario: str
    controller_source: str
    controller_source_after_step: str
    policy_target: str
    target: str
    release_time: float
    reward: float
    rolling_loss_rate: float | None
    transmission_loss_event: bool
    switch_event: str
    latency: float
    queue_delay: float
    total_system_energy: float
    packet_loss_percent: float
    transmission_attempts: int
    retransmission_count: int
    packet_lost: bool
    deadline_missed: bool


def _row_from_step(
    *,
    info: dict,
    reward: float,
    split: str,
    dataset: DatasetFiles,
    task_index: int,
) -> OnlineRunRow:
    return OnlineRunRow(
        algorithm="SAC_GA_Online",
        split=split,
        dataset=dataset.name,
        scenario_group=dataset.group,
        task_index=task_index,
        task_id=str(info["task_id"]),
        scenario=str(info["scenario"]),
        controller_source=str(info["controller_source"]),
        controller_source_after_step=str(info["controller_source_after_step"]),
        policy_target=str(info["policy_target"]),
        target=str(info["target"]),
        release_time=float(info["release_time"]),
        reward=float(reward),
        rolling_loss_rate=(
            None
            if info["rolling_loss_rate"] is None
            else float(info["rolling_loss_rate"])
        ),
        transmission_loss_event=bool(info["transmission_loss_event"]),
        switch_event=str(info["switch_event"]),
        latency=float(info["latency"]),
        queue_delay=float(info["queue_delay"]),
        total_system_energy=float(info["total_system_energy"]),
        packet_loss_percent=float(info["packet_loss_percent"]),
        transmission_attempts=int(info["transmission_attempts"]),
        retransmission_count=int(info["retransmission_count"]),
        packet_lost=bool(info["packet_lost"]),
        deadline_missed=bool(info["deadline_missed"]),
    )


class OnlineTrainingCallback(BaseCallback):
    """Log steps and store the action actually executed by the hybrid controller."""

    def __init__(
        self,
        *,
        split: str,
        dataset: DatasetFiles,
        show_progress: bool,
        progress_every: int,
    ) -> None:
        super().__init__(verbose=0)
        self.split = split
        self.dataset = dataset
        self.show_progress = show_progress
        self.progress_every = max(1, progress_every)
        self.rows: list[OnlineRunRow] = []
        self.fallback_steps_seen = 0
        self.fallback_steps_in_rollout = 0
        self._last_progress_time: int | None = None

    def _on_rollout_start(self) -> None:
        self.fallback_steps_in_rollout = 0

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        reward = float(self.locals["rewards"][0])
        buffer_actions = self.locals["buffer_actions"]
        buffer_actions[0, 0] = float(info["executed_action"])
        if info["controller_source"] == "GA":
            self.fallback_steps_seen += 1
            self.fallback_steps_in_rollout += 1
        self.rows.append(
            _row_from_step(
                info=info,
                reward=reward,
                split=self.split,
                dataset=self.dataset,
                task_index=len(self.rows) + 1,
            )
        )
        if info["switch_event"]:
            print(
                f"  switch={info['switch_event']} time={info['release_time']:.0f}s "
                f"rolling_loss={info['rolling_loss_rate']:.3f}"
            )
        simulation_time = int(float(info["release_time"]))
        if (
            self.show_progress
            and simulation_time % self.progress_every == 0
            and simulation_time != self._last_progress_time
        ):
            self._last_progress_time = simulation_time
            rolling_loss = info["rolling_loss_rate"]
            loss_text = "warming" if rolling_loss is None else f"{rolling_loss:.3f}"
            print(
                f"  time={simulation_time}s tasks={len(self.rows)} "
                f"source={info['controller_source_after_step']} "
                f"rolling_loss={loss_text}"
            )
        return True


class FallbackOnlySAC(SAC):
    """Use deterministic SAC decisions and train only during GA fallback."""

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        assert self._last_obs is not None
        action, _ = self.predict(self._last_obs, deterministic=True)
        buffer_action = self.policy.scale_action(action)
        return self.policy.unscale_action(buffer_action), buffer_action

    def learn_during_fallback(
        self,
        *,
        total_timesteps: int,
        callback: OnlineTrainingCallback,
        online_warmup: int,
    ) -> "FallbackOnlySAC":
        target_timesteps, active_callback = self._setup_learn(
            total_timesteps,
            callback,
            False,
            "run",
            False,
        )
        active_callback.on_training_start(locals(), globals())
        assert self.env is not None
        one_step = TrainFreq(1, TrainFrequencyUnit.STEP)
        update_every = max(1, self.train_freq.frequency)

        while self.num_timesteps < target_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=one_step,
                action_noise=self.action_noise,
                callback=active_callback,
                learning_starts=0,
                replay_buffer=self.replay_buffer,
                log_interval=4,
            )
            if not rollout.continue_training:
                break
            can_update = callback.fallback_steps_seen >= online_warmup
            update_due = callback.fallback_steps_seen % update_every == 0
            if (
                callback.fallback_steps_in_rollout > 0
                and can_update
                and update_due
            ):
                gradient_steps = (
                    self.gradient_steps
                    if self.gradient_steps >= 0
                    else rollout.episode_timesteps
                )
                if gradient_steps > 0:
                    self.train(
                        batch_size=self.batch_size,
                        gradient_steps=gradient_steps,
                    )

        active_callback.on_training_end()
        return self


def _ordered_datasets(
    data_root: str | Path,
    split: str,
    names: Sequence[str] | None,
    max_datasets: int | None,
) -> list[DatasetFiles]:
    selected_names = names
    if split == "finetune" and not names:
        selected_names = FINETUNE_ORDER
    elif split == "test" and not names:
        selected_names = TEST_ORDER
    return discover_datasets(
        data_root=data_root,
        split=split,
        names=selected_names,
        max_datasets=max_datasets,
    )


def _limited_tasks(
    dataset: DatasetFiles,
    max_timesteps: int | None,
    batch_window_seconds: int,
) -> tuple:
    tasks = load_tasks(dataset.tasks_file)
    if max_timesteps is None:
        return tuple(tasks)
    batches = batch_tasks(
        tasks,
        batch_window_seconds=batch_window_seconds,
        max_timesteps=max_timesteps,
    )
    return tuple(task for batch in batches for task in batch)


def _write_rows(rows: Sequence[OnlineRunRow], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(OnlineRunRow.__dataclass_fields__),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_switches(rows: Sequence[OnlineRunRow], path: str | Path) -> None:
    switches = [row for row in rows if row.switch_event]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset",
        "task_index",
        "release_time",
        "scenario",
        "switch_event",
        "rolling_loss_rate",
        "controller_source_after_step",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in switches:
            raw = asdict(row)
            writer.writerow({field: raw[field] for field in fields})


def run_online_hybrid(
    *,
    data_root: str | Path,
    split: str,
    checkpoint: str | Path,
    adapted_model: str | Path,
    results_file: str | Path,
    switch_file: str | Path,
    monitor_config: PacketLossMonitorConfig,
    genetic_config: GeneticOffloaderConfig,
    dataset_names: Sequence[str] | None = None,
    max_datasets: int | None = None,
    max_timesteps: int | None = None,
    batch_window_seconds: int = 1,
    train_online: bool = True,
    online_warmup: int = 0,
    seed: int = 999,
    device: str = "cpu",
    show_progress: bool = True,
    progress_every: int = 20,
) -> list[OnlineRunRow]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAC checkpoint not found: {checkpoint_path}")
    datasets = _ordered_datasets(
        data_root,
        split,
        dataset_names,
        max_datasets,
    )
    all_rows: list[OnlineRunRow] = []
    model: FallbackOnlySAC | None = None
    progress = ProgressBar(enabled=show_progress, update_every=progress_every)
    model_path = Path(adapted_model)

    for dataset_index, dataset in enumerate(datasets, start=1):
        tasks = _limited_tasks(dataset, max_timesteps, batch_window_seconds)
        vehicle_states = load_vehicle_states(dataset.vehicles_file)
        monitor = PacketLossFallbackMonitor(monitor_config)
        genetic = GeneticBatchOffloader(genetic_config)
        environment = OnlineHybridEnv(
            tasks,
            vehicle_states,
            resource_capacities=genetic_config.resource_capacities,
            reward_config=RewardConfig(),
            genetic_offloader=genetic,
            monitor=monitor,
            batch_window_seconds=batch_window_seconds,
        )

        if model is None:
            model = FallbackOnlySAC.load(
                str(checkpoint_path),
                env=environment,
                device=device,
            )
        else:
            model.set_env(environment, force_reset=True)

        if show_progress:
            mode = (
                "frozen SAC; fine-tune only during GA fallback"
                if train_online
                else "fully frozen evaluation"
            )
            print(
                f"{split} dataset {dataset_index}/{len(datasets)} "
                f"{dataset.name}: {len(tasks)} tasks, {mode}"
            )

        updates_before = model._n_updates
        if train_online:
            callback = OnlineTrainingCallback(
                split=split,
                dataset=dataset,
                show_progress=show_progress,
                progress_every=progress_every,
            )
            model.learn_during_fallback(
                total_timesteps=len(tasks),
                callback=callback,
                online_warmup=max(0, online_warmup),
            )
            rows = callback.rows
        else:
            rows = []
            observation, _ = environment.reset(seed=seed)
            for task_index in range(1, len(tasks) + 1):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = environment.step(
                    action
                )
                rows.append(
                    _row_from_step(
                        info=info,
                        reward=reward,
                        split=split,
                        dataset=dataset,
                        task_index=task_index,
                    )
                )
                if truncated:
                    raise RuntimeError("online evaluation ended unexpectedly")
                if terminated:
                    break

        all_rows.extend(rows)
        update_count = model._n_updates - updates_before
        if show_progress:
            ga_tasks = sum(row.controller_source == "GA" for row in rows)
            print(
                f"  completed: tasks={len(rows)} ga_tasks={ga_tasks} "
                f"sac_gradient_updates={update_count}"
            )
        _write_rows(all_rows, results_file)
        _write_switches(all_rows, switch_file)
        if train_online:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(model_path)
        batch_count = len({int(row.release_time) for row in rows})
        progress.update(
            split=split,
            dataset_index=dataset_index,
            dataset_count=len(datasets),
            dataset_name=dataset.name,
            batch_index=batch_count,
            batch_count=batch_count,
            task_count=len(rows),
            unit_name="timesteps",
            force=True,
        )

    assert model is not None
    if train_online:
        print(f"Adapted SAC model: {model_path.with_suffix('.zip')}")

    print(f"Online results: {results_file}")
    print(f"Switch log: {switch_file}")
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run online SAC adaptation with packet-loss-triggered GA fallback."
    )
    parser.add_argument(
        "--split",
        choices=("finetune", "test"),
        default="test",
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--max-datasets", type=int, default=0)
    parser.add_argument("--max-timesteps", type=int, default=0)
    parser.add_argument("--batch-window", type=int, default=1)
    parser.add_argument("--checkpoint", default=str(DEFAULT_PRETRAINED_MODEL))
    parser.add_argument("--adapted-model", default=str(DEFAULT_ADAPTED_MODEL))
    parser.add_argument("--results-file")
    parser.add_argument("--switch-file")
    parser.add_argument("--genetic-config")
    parser.add_argument("--no-train", action="store_true")
    parser.add_argument("--online-warmup", type=int, default=0)
    parser.add_argument("--loss-window", type=int, default=100)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--fallback-loss-rate", type=float, default=0.05)
    parser.add_argument("--recovery-loss-rate", type=float, default=0.02)
    parser.add_argument("--consecutive-checks", type=int, default=2)
    parser.add_argument("--minimum-fallback-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_file = Path(args.results_file or DEFAULT_RESULTS_ROOT / f"{args.split}_hybrid_results.csv")
    switch_file = Path(args.switch_file or DEFAULT_RESULTS_ROOT / f"{args.split}_hybrid_switches.csv")
    genetic_config = (
        config_from_json(args.genetic_config)
        if args.genetic_config
        else GeneticOffloaderConfig()
    )
    monitor_config = PacketLossMonitorConfig(
        window_size=args.loss_window,
        check_every=args.check_every,
        fallback_loss_rate=args.fallback_loss_rate,
        recovery_loss_rate=args.recovery_loss_rate,
        consecutive_checks=args.consecutive_checks,
        minimum_fallback_tasks=args.minimum_fallback_tasks,
    )
    run_online_hybrid(
        data_root=args.data_root,
        split=args.split,
        checkpoint=args.checkpoint,
        adapted_model=args.adapted_model,
        results_file=results_file,
        switch_file=switch_file,
        monitor_config=monitor_config,
        genetic_config=genetic_config,
        dataset_names=args.datasets,
        max_datasets=args.max_datasets or None,
        max_timesteps=args.max_timesteps or None,
        batch_window_seconds=args.batch_window,
        train_online=not args.no_train,
        online_warmup=args.online_warmup,
        seed=args.seed,
        device=args.device,
        show_progress=not args.no_progress,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
