from __future__ import annotations

import unittest

from algorithms import GeneticBatchOffloader, GeneticOffloaderConfig, OffloadTarget
from infrastructure import ExecutionModel, TaskRecord, VehicleState, dynamic_backhaul_delay
from offloading_simulator import (
    DeterministicChannel,
    ResourceCapacities,
    ResourceState,
    simulate_assignments,
)


def make_task(
    task_id: str,
    release: float,
    *,
    deadline: float = 40.0,
    exec_time: float = 10.0,
    scenario: str = "BASE",
    path_loss: float = 0.0,
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        release_time=release,
        deadline=deadline,
        exec_time=exec_time,
        power=1.0,
        creator="vehicle-1",
        cycles_per_bit=1.0,
        data_size=0.4,
        weather_scenario=scenario,
        deadline_type="Normal",
        path_loss_increase_db=path_loss,
        plr_increase_percent=0.0,
    )


def vehicle(release: float) -> VehicleState:
    return VehicleState(
        id="vehicle-1",
        time=release,
        x=150.0,
        y=150.0,
        speed=10.0,
        weather_scenario="BASE",
    )


class SharedSimulatorTests(unittest.TestCase):
    def test_resource_state_persists_across_control_intervals(self) -> None:
        first = make_task("first", 0.0)
        second = make_task("second", 1.0)
        states = {
            (0.0, "vehicle-1"): vehicle(0.0),
            (1.0, "vehicle-1"): vehicle(1.0),
        }
        resource_state = ResourceState()
        channel = DeterministicChannel(7)

        simulate_assignments(
            (first,),
            (OffloadTarget.LOCAL,),
            resource_state,
            channel,
            vehicle_states=states,
        )
        second_result = simulate_assignments(
            (second,),
            (OffloadTarget.LOCAL,),
            resource_state,
            channel,
            vehicle_states=states,
        )[0]

        self.assertEqual(second_result.queue_delay, 9.0)
        self.assertEqual(second_result.finish_time, 20.0)

    def test_cloud_uses_dynamic_backhaul_without_path_loss_delay(self) -> None:
        model = ExecutionModel(cloud_base_packet_loss_percent=0.0)
        first = make_task("same", 0.0, scenario="FOG", path_loss=0.0)
        changed_path_loss = TaskRecord(
            **{**first.__dict__, "path_loss_increase_db": 20.0}
        )
        states = {(0.0, "vehicle-1"): vehicle(0.0)}
        expected_backhaul = dynamic_backhaul_delay("FOG", 12, model)

        first_result = simulate_assignments(
            (first,),
            (OffloadTarget.CLOUD,),
            ResourceState(),
            DeterministicChannel(5),
            vehicle_states=states,
            model=model,
            network_load_by_time={0.0: 12},
        )[0]
        changed_result = simulate_assignments(
            (changed_path_loss,),
            (OffloadTarget.CLOUD,),
            ResourceState(),
            DeterministicChannel(5),
            vehicle_states=states,
            model=model,
            network_load_by_time={0.0: 12},
        )[0]

        self.assertEqual(first_result.backhaul_delay, expected_backhaul)
        self.assertEqual(first_result.latency, changed_result.latency)

    def test_channel_samples_are_repeatable_and_order_independent(self) -> None:
        channel = DeterministicChannel(19)
        expected = channel.sample("task-1", 2)

        channel.sample("another-task", 1)

        self.assertEqual(channel.sample("task-1", 2), expected)
        self.assertEqual(DeterministicChannel(19).sample("task-1", 2), expected)

    def test_ga_evaluation_matches_shared_execution(self) -> None:
        config = GeneticOffloaderConfig(local_capacity=1)
        optimizer = GeneticBatchOffloader(config)
        first = make_task("first", 0.0)
        second = make_task("second", 1.0)
        states = {
            (0.0, "vehicle-1"): vehicle(0.0),
            (1.0, "vehicle-1"): vehicle(1.0),
        }
        resource_state = ResourceState()
        channel = DeterministicChannel(3)
        simulate_assignments(
            (first,),
            (OffloadTarget.LOCAL,),
            resource_state,
            channel,
            vehicle_states=states,
            capacities=config.resource_capacities,
        )

        evaluation = optimizer.evaluate(
            (OffloadTarget.LOCAL,),
            (second,),
            states,
            resource_state=resource_state,
            channel_randomness=channel,
        )
        executed = simulate_assignments(
            (second,),
            (OffloadTarget.LOCAL,),
            resource_state.copy(),
            channel,
            vehicle_states=states,
            capacities=ResourceCapacities(local=1),
        )[0]

        self.assertEqual(evaluation.total_latency, executed.latency)
        self.assertEqual(evaluation.total_energy, executed.total_system_energy)
        self.assertEqual(evaluation.total_queue_delay, executed.queue_delay)
        self.assertEqual(evaluation.deadline_misses, int(executed.deadline_missed))


if __name__ == "__main__":
    unittest.main()
