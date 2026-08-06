from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from algorithms import GeneticBatchOffloader, GeneticOffloaderConfig, OffloadTarget
from discrete_sac import DiscreteSACAgent
from genetic_offloader_runner import (
    DatasetFiles,
    ProgressBar,
    batch_tasks,
    config_from_json,
    discover_datasets,
)
from infrastructure import load_tasks, load_vehicle_states
from vec_offloading_env import ObservationScale, RewardConfig, VECOffloadingEnv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "datasets"
DEFAULT_PRETRAINED_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "discrete_sac"
    / "sac_discrete_best.pt"
)
DEFAULT_ADAPTED_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "discrete_sac_online"
    / "sac_adapted_final.pt"
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
class HybridMonitorConfig:
    window_size: int = 100
    check_every: int = 20
    fallback_loss_rate: float = 0.05
    recovery_loss_rate: float = 0.02
    fallback_deadline_miss_rate: float = 0.10
    recovery_deadline_miss_rate: float = 0.03
    fallback_normalized_latency: float = 1.00
    recovery_normalized_latency: float = 0.70
    consecutive_checks: int = 2
    minimum_fallback_tasks: int = 100

    def __post_init__(self) -> None:
        pairs = (
            (self.recovery_loss_rate, self.fallback_loss_rate),
            (
                self.recovery_deadline_miss_rate,
                self.fallback_deadline_miss_rate,
            ),
            (
                self.recovery_normalized_latency,
                self.fallback_normalized_latency,
            ),
        )
        if self.window_size <= 0 or self.check_every <= 0:
            raise ValueError("window_size and check_every must be positive")
        if self.consecutive_checks <= 0 or self.minimum_fallback_tasks < 0:
            raise ValueError("invalid fallback duration settings")
        if any(not 0.0 <= recovery < fallback for recovery, fallback in pairs):
            raise ValueError("every recovery threshold must be below fallback")
        if self.fallback_loss_rate > 1.0:
            raise ValueError("packet-loss threshold cannot exceed 1")
        if self.fallback_deadline_miss_rate > 1.0:
            raise ValueError("deadline-miss threshold cannot exceed 1")


class HybridFallbackMonitor:
    """Use loss, deadline, and latency windows with hysteresis."""

    def __init__(self, config: HybridMonitorConfig | None = None) -> None:
        self.config = config or HybridMonitorConfig()
        self._losses: deque[float] = deque(maxlen=self.config.window_size)
        self._misses: deque[float] = deque(maxlen=self.config.window_size)
        self._latencies: deque[float] = deque(maxlen=self.config.window_size)
        self._observations = 0
        self._bad_checks = 0
        self._good_checks = 0
        self._fallback_tasks = 0
        self.in_fallback = False
        self.last_trigger = ""

    @property
    def source(self) -> str:
        return "GA" if self.in_fallback else "SAC"

    @staticmethod
    def _average(values: deque[float]) -> float | None:
        return sum(values) / len(values) if values else None

    @property
    def rolling_loss_rate(self) -> float | None:
        return self._average(self._losses)

    @property
    def rolling_deadline_miss_rate(self) -> float | None:
        return self._average(self._misses)

    @property
    def rolling_normalized_latency(self) -> float | None:
        return self._average(self._latencies)

    def observe(
        self,
        *,
        loss_event: bool,
        deadline_missed: bool,
        normalized_latency: float,
    ) -> tuple[str, str]:
        self._losses.append(float(loss_event))
        self._misses.append(float(deadline_missed))
        self._latencies.append(float(normalized_latency))
        self._observations += 1
        if self.in_fallback:
            self._fallback_tasks += 1

        if len(self._losses) < self.config.window_size:
            return "", ""
        if self._observations % self.config.check_every != 0:
            return "", ""

        loss = self.rolling_loss_rate or 0.0
        miss = self.rolling_deadline_miss_rate or 0.0
        latency = self.rolling_normalized_latency or 0.0
        bad_reasons = []
        if loss >= self.config.fallback_loss_rate:
            bad_reasons.append("packet_loss")
        if miss >= self.config.fallback_deadline_miss_rate:
            bad_reasons.append("deadline_miss")
        if latency >= self.config.fallback_normalized_latency:
            bad_reasons.append("latency")

        if not self.in_fallback:
            self._bad_checks = self._bad_checks + 1 if bad_reasons else 0
            if self._bad_checks >= self.config.consecutive_checks:
                self.in_fallback = True
                self._bad_checks = 0
                self._good_checks = 0
                self._fallback_tasks = 0
                self.last_trigger = "+".join(bad_reasons)
                return "SAC_TO_GA", self.last_trigger
            return "", ""

        recovered = (
            self._fallback_tasks >= self.config.minimum_fallback_tasks
            and loss <= self.config.recovery_loss_rate
            and miss <= self.config.recovery_deadline_miss_rate
            and latency <= self.config.recovery_normalized_latency
        )
        self._good_checks = self._good_checks + 1 if recovered else 0
        if self._good_checks >= self.config.consecutive_checks:
            self.in_fallback = False
            self._good_checks = 0
            self._bad_checks = 0
            self.last_trigger = ""
            return "GA_TO_SAC", "recovered"
        return "", ""


class OnlineHybridEnv(VECOffloadingEnv):
    """Execute discrete SAC normally and GeneticBatch during fallback."""

    def __init__(
        self,
        *args,
        genetic_offloader: GeneticBatchOffloader,
        monitor: HybridFallbackMonitor,
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
            if int(candidate.release_time) // self.batch_window_seconds != bucket:
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
                fog_nodes_by_time=self.fog_nodes_by_time,
            )
            self._ga_assignments = result.by_task_id(batch)
        return self._ga_assignments.pop(task.id)

    def step(self, action: int):
        if self._index is None or self._index >= len(self.tasks):
            return super().step(action)
        policy_target = self.action_to_target(action)
        source = self.monitor.source
        target = self._genetic_target() if source == "GA" else policy_target
        executed_action = self.target_to_action(target)
        observation, reward, terminated, truncated, info = super().step(
            executed_action
        )
        loss_event = bool(
            info["packet_lost"] or int(info["retransmission_count"]) > 0
        )
        switch_event, trigger = self.monitor.observe(
            loss_event=loss_event,
            deadline_missed=bool(info["deadline_missed"]),
            normalized_latency=float(info["normalized_latency"]),
        )
        if switch_event == "GA_TO_SAC":
            self._ga_assignments.clear()
        info.update(
            {
                "controller_source": source,
                "controller_source_after_step": self.monitor.source,
                "policy_target": policy_target.value,
                "executed_action": executed_action,
                "transmission_loss_event": loss_event,
                "rolling_loss_rate": self.monitor.rolling_loss_rate,
                "rolling_deadline_miss_rate": (
                    self.monitor.rolling_deadline_miss_rate
                ),
                "rolling_normalized_latency": (
                    self.monitor.rolling_normalized_latency
                ),
                "switch_event": switch_event,
                "switch_trigger": trigger,
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
    rolling_deadline_miss_rate: float | None
    rolling_normalized_latency: float | None
    transmission_loss_event: bool
    switch_event: str
    switch_trigger: str
    latency: float
    normalized_latency: float
    queue_delay: float
    total_system_energy: float
    packet_loss_percent: float
    transmission_attempts: int
    retransmission_count: int
    packet_lost: bool
    deadline_missed: bool


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _row_from_step(
    *,
    info: dict,
    reward: float,
    split: str,
    dataset: DatasetFiles,
    task_index: int,
) -> OnlineRunRow:
    return OnlineRunRow(
        algorithm="DiscreteSAC_GA_Online",
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
        rolling_loss_rate=_optional_float(info["rolling_loss_rate"]),
        rolling_deadline_miss_rate=_optional_float(
            info["rolling_deadline_miss_rate"]
        ),
        rolling_normalized_latency=_optional_float(
            info["rolling_normalized_latency"]
        ),
        transmission_loss_event=bool(info["transmission_loss_event"]),
        switch_event=str(info["switch_event"]),
        switch_trigger=str(info["switch_trigger"]),
        latency=float(info["latency"]),
        normalized_latency=float(info["normalized_latency"]),
        queue_delay=float(info["queue_delay"]),
        total_system_energy=float(info["total_system_energy"]),
        packet_loss_percent=float(info["packet_loss_percent"]),
        transmission_attempts=int(info["transmission_attempts"]),
        retransmission_count=int(info["retransmission_count"]),
        packet_lost=bool(info["packet_lost"]),
        deadline_missed=bool(info["deadline_missed"]),
    )


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
    return tuple(
        task
        for batch in batch_tasks(
            tasks,
            batch_window_seconds=batch_window_seconds,
            max_timesteps=max_timesteps,
        )
        for task in batch
    )


def _write_rows(rows: Sequence[OnlineRunRow], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(OnlineRunRow.__dataclass_fields__)
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _write_switches(rows: Sequence[OnlineRunRow], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "dataset",
        "task_index",
        "release_time",
        "scenario",
        "switch_event",
        "switch_trigger",
        "rolling_loss_rate",
        "rolling_deadline_miss_rate",
        "rolling_normalized_latency",
        "controller_source_after_step",
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.switch_event:
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
    monitor_config: HybridMonitorConfig,
    genetic_config: GeneticOffloaderConfig,
    dataset_names: Sequence[str] | None = None,
    max_datasets: int | None = None,
    max_timesteps: int | None = None,
    batch_window_seconds: int = 1,
    train_during_fallback: bool = True,
    seed: int = 999,
    device: str = "cpu",
    show_progress: bool = True,
    progress_every: int = 20,
) -> list[OnlineRunRow]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAC checkpoint not found: {checkpoint_path}")
    agent, checkpoint_data = DiscreteSACAgent.load(
        checkpoint_path, device=device
    )
    observation_scale = ObservationScale(
        **checkpoint_data["observation_scale"]
    )
    datasets = _ordered_datasets(
        data_root, split, dataset_names, max_datasets
    )
    all_rows: list[OnlineRunRow] = []
    progress = ProgressBar(enabled=show_progress, update_every=progress_every)
    model_path = Path(adapted_model)

    for dataset_index, dataset in enumerate(datasets, start=1):
        tasks = _limited_tasks(dataset, max_timesteps, batch_window_seconds)
        states = load_vehicle_states(dataset.vehicles_file)
        monitor = HybridFallbackMonitor(monitor_config)
        environment = OnlineHybridEnv(
            tasks,
            states,
            observation_scale=observation_scale,
            resource_capacities=genetic_config.resource_capacities,
            reward_config=RewardConfig(),
            genetic_offloader=GeneticBatchOffloader(genetic_config),
            monitor=monitor,
            batch_window_seconds=batch_window_seconds,
        )
        observation, _ = environment.reset(seed=seed)
        rows: list[OnlineRunRow] = []
        updates_before = agent.gradient_updates
        ga_tasks = 0
        last_progress_time: int | None = None
        print(
            f"{split} dataset {dataset_index}/{len(datasets)} {dataset.name}: "
            f"{len(tasks)} tasks, frozen SAC; update only during GA fallback"
        )

        for task_index in range(1, len(tasks) + 1):
            policy_action = agent.select_action(
                observation, deterministic=True
            )
            next_observation, reward, terminated, truncated, info = (
                environment.step(policy_action)
            )
            source = str(info["controller_source"])
            if train_during_fallback and source == "GA":
                agent.observe(
                    observation,
                    int(info["executed_action"]),
                    reward,
                    next_observation,
                    terminated,
                    str(info["scenario"]),
                )
            if source == "GA":
                ga_tasks += 1
                can_update = len(agent.replay) >= agent.config.batch_size
                if (
                    train_during_fallback
                    and can_update
                    and ga_tasks % agent.config.update_every == 0
                ):
                    agent.update()
            row = _row_from_step(
                info=info,
                reward=reward,
                split=split,
                dataset=dataset,
                task_index=task_index,
            )
            rows.append(row)
            if row.switch_event:
                print(
                    f"  switch={row.switch_event} time={row.release_time:.0f}s "
                    f"trigger={row.switch_trigger}"
                )
            simulation_time = int(row.release_time)
            if (
                show_progress
                and simulation_time % progress_every == 0
                and simulation_time != last_progress_time
            ):
                last_progress_time = simulation_time
                print(
                    f"  time={simulation_time}s tasks={task_index} "
                    f"source={row.controller_source_after_step}"
                )
            observation = next_observation
            if truncated:
                raise RuntimeError("online evaluation ended unexpectedly")
            if terminated:
                break

        all_rows.extend(rows)
        update_count = agent.gradient_updates - updates_before
        print(
            f"  completed: tasks={len(rows)} ga_tasks={ga_tasks} "
            f"sac_gradient_updates={update_count}"
        )
        _write_rows(all_rows, results_file)
        _write_switches(all_rows, switch_file)
        if train_during_fallback:
            agent.save(
                model_path,
                observation_scale=asdict(observation_scale),
                metadata={
                    "source_checkpoint": str(checkpoint_path),
                    "completed_dataset": dataset.name,
                    "protocol": "update_only_during_ga_fallback",
                },
            )
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

    if train_during_fallback:
        print(f"Adapted SAC model: {model_path.with_suffix('.pt')}")
    print(f"Online results: {results_file}")
    print(f"Switch log: {switch_file}")
    return all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run discrete SAC with multi-metric GA fallback."
    )
    parser.add_argument(
        "--split", choices=("finetune", "test"), default="test"
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
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--check-every", type=int, default=20)
    parser.add_argument("--fallback-loss-rate", type=float, default=0.05)
    parser.add_argument("--recovery-loss-rate", type=float, default=0.02)
    parser.add_argument("--fallback-miss-rate", type=float, default=0.10)
    parser.add_argument("--recovery-miss-rate", type=float, default=0.03)
    parser.add_argument("--fallback-latency", type=float, default=1.00)
    parser.add_argument("--recovery-latency", type=float, default=0.70)
    parser.add_argument("--consecutive-checks", type=int, default=2)
    parser.add_argument("--minimum-fallback-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results_file = Path(
        args.results_file
        or DEFAULT_RESULTS_ROOT / f"{args.split}_hybrid_results.csv"
    )
    switch_file = Path(
        args.switch_file
        or DEFAULT_RESULTS_ROOT / f"{args.split}_hybrid_switches.csv"
    )
    genetic_config = (
        config_from_json(args.genetic_config)
        if args.genetic_config
        else GeneticOffloaderConfig()
    )
    monitor_config = HybridMonitorConfig(
        window_size=args.window_size,
        check_every=args.check_every,
        fallback_loss_rate=args.fallback_loss_rate,
        recovery_loss_rate=args.recovery_loss_rate,
        fallback_deadline_miss_rate=args.fallback_miss_rate,
        recovery_deadline_miss_rate=args.recovery_miss_rate,
        fallback_normalized_latency=args.fallback_latency,
        recovery_normalized_latency=args.recovery_latency,
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
        train_during_fallback=not args.no_train,
        seed=args.seed,
        device=args.device,
        show_progress=not args.no_progress,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
