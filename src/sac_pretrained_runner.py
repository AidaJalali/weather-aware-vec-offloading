from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-vec-cache")

from stable_baselines3 import SAC

from genetic_offloader_runner import (
    ProgressBar,
    batch_tasks,
    discover_datasets,
)
from infrastructure import load_tasks, load_vehicle_states
from offloading_simulator import ResourceCapacities
from random_offloader_runner import (
    RandomRunRow,
    summarize_rows,
    write_rows,
    write_summary,
)
from vec_offloading_env import RewardConfig, VECOffloadingEnv


def _row_from_info(
    *,
    info: dict,
    split: str,
    dataset_name: str,
    scenario_group: str,
) -> RandomRunRow:
    return RandomRunRow(
        algorithm="SAC_Pretrained",
        split=split,
        dataset=dataset_name,
        scenario_group=scenario_group,
        task_id=str(info["task_id"]),
        scenario=str(info["scenario"]),
        target=str(info["target"]),
        release_time=float(info["release_time"]),
        deadline=float(info["deadline"]),
        finish_time=float(info["finish_time"]),
        latency=float(info["latency"]),
        queue_delay=float(info["queue_delay"]),
        energy=float(info["total_system_energy"]),
        packet_loss_percent=float(info["packet_loss_percent"]),
        final_failure_probability=float(info["final_failure_probability"]),
        expected_deadline_failure=float(info["expected_deadline_failure"]),
        network_load=int(info["network_load"]),
        backhaul_delay=float(info["backhaul_delay"]),
        wireless_transmission_time=float(info["wireless_transmission_time"]),
        transmission_attempts=int(info["transmission_attempts"]),
        retransmission_count=int(info["retransmission_count"]),
        local_cpu_energy=float(info["local_cpu_energy"]),
        transmission_energy=float(info["transmission_energy"]),
        fog_compute_energy=float(info["fog_compute_energy"]),
        cloud_compute_energy=float(info["cloud_compute_energy"]),
        vehicle_energy=float(info["vehicle_energy"]),
        infrastructure_energy=float(info["infrastructure_energy"]),
        total_system_energy=float(info["total_system_energy"]),
        packet_lost=bool(info["packet_lost"]),
        deadline_missed=bool(info["deadline_missed"]),
    )


def run_sac_pretrained_split(
    *,
    data_root: str | Path,
    split: str,
    checkpoint: str | Path,
    output_file: str | Path,
    summary_file: str | Path,
    dataset_names: Sequence[str] | None = None,
    max_datasets: int | None = None,
    max_timesteps: int | None = None,
    batch_window_seconds: int = 1,
    capacities: ResourceCapacities | None = None,
    seed: int = 999,
    show_progress: bool = True,
    progress_every: int = 20,
    device: str = "cpu",
) -> list[RandomRunRow]:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SAC pretrained checkpoint not found: {checkpoint_path}"
        )

    datasets = discover_datasets(
        data_root=data_root,
        split=split,
        names=dataset_names,
        max_datasets=max_datasets,
    )
    capacities = capacities or ResourceCapacities()
    progress = ProgressBar(enabled=show_progress, update_every=progress_every)
    rows: list[RandomRunRow] = []

    if show_progress:
        print(
            f"Running SAC_Pretrained on {len(datasets)} {split} dataset(s) "
            f"with batch_window={batch_window_seconds}s."
        )

    for dataset_index, dataset in enumerate(datasets, start=1):
        vehicle_states = load_vehicle_states(dataset.vehicles_file)
        all_tasks = load_tasks(dataset.tasks_file)
        batches = list(
            batch_tasks(
                all_tasks,
                batch_window_seconds=batch_window_seconds,
                max_timesteps=max_timesteps,
            )
        )
        tasks = [task for batch in batches for task in batch]
        environment = VECOffloadingEnv(
            tasks,
            vehicle_states,
            resource_capacities=capacities,
            reward_config=RewardConfig(),
        )
        model = SAC.load(
            str(checkpoint_path),
            device=device,
        )
        if model.observation_space.shape != environment.observation_space.shape:
            raise ValueError(
                "SAC checkpoint observation shape does not match the environment"
            )
        observation, _ = environment.reset(seed=seed)
        processed_tasks = 0

        for batch_index, batch in enumerate(batches, start=1):
            for _task in batch:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = environment.step(action)
                rows.append(
                    _row_from_info(
                        info=info,
                        split=split,
                        dataset_name=dataset.name,
                        scenario_group=dataset.group,
                    )
                )
                processed_tasks += 1
                if truncated:
                    raise RuntimeError("SAC evaluation ended unexpectedly")
            progress.update(
                split=split,
                dataset_index=dataset_index,
                dataset_count=len(datasets),
                dataset_name=dataset.name,
                batch_index=batch_index,
                batch_count=len(batches),
                task_count=processed_tasks,
                unit_name="timesteps",
                force=batch_index == len(batches),
            )

        if not terminated:
            raise RuntimeError("SAC evaluation did not consume every task")
        progress.finish_dataset()

    write_rows(rows, output_file)
    write_summary(summarize_rows(rows), summary_file)
    return rows
