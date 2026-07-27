# weather-aware-vec-offloading

Weather-aware adaptive task offloading in Vehicular Edge Computing (VEC) using Soft Actor-Critic (SAC) deep reinforcement learning.

## Overview

Vehicular Edge Computing (VEC) is a three-layer distributed computing architecture designed to meet the real-time processing and communication demands of vehicular networks. Instead of sending all data to a central cloud, VEC brings computation to the edge of the network — close to the data source (the vehicles).

Existing task offloading algorithms for VEC often assume ideal conditions and ignore critical real-world dynamics such as **weather conditions**. Adverse weather (rain, snow, fog) directly affects:

- **Vehicle dynamics** (reduced speed, altered mobility patterns)
- **Task characteristics** (increased processing load, tighter deadlines)
- **Communication channel quality** (signal attenuation, higher packet loss)

A Zone Manager that is unaware of these changes makes suboptimal — and potentially unsafe — decisions, leading to increased latency, lower task success rates, and degraded reliability.

This project tackles this problem by designing an **intelligent, weather-aware Zone Manager** that transforms the system from a purely performance-optimized model into a **reliable model** capable of adapting to unpredictable real-world conditions.

## Approach

- **Algorithm:** Soft Actor-Critic (SAC) — a state-of-the-art off-policy deep reinforcement learning algorithm suitable for continuous action spaces
- **Agent:** Zone Manager making dynamic task offloading decisions
- **State Space:** Comprehensive representation including weather parameters, vehicle states, task properties, and network resource availability
- **Reward Function:** Multi-objective design balancing **latency**, **energy consumption**, and **reliability** across diverse weather conditions

## Weather Scenarios

Five scenarios are evaluated:

| Scenario | Max Speed | Deadline | Cycles/Bit | Task Rate | Signal Loss | PLR Increase |
|----------|-----------|----------|------------|-----------|-------------|--------------|
| **Base** | Vmax | Normal | C | R | PL | N |
| **Rain** | -15% | Tight | 1.2–1.4x | +20% | +2–4 dB | +15–20% |
| **Snow** | -30% | Tightest | 1.5–1.8x | -50% | +1–2 dB | +5–10% |
| **Fog** | -60% | Relaxed | 1.3–1.6x | +25% | +4–6 dB | +25–30% |
| **Mixed** | Varying | Varying | Varying | Varying | Varying | Varying |

## Expected Outputs

For each scenario, the following metrics are compared against baseline (non-weather-aware) algorithms:

1. Average latency
2. Average energy consumption
3. Number of deadline misses
4. Packet loss count

## Project Phases

### Phase 1 — System Model & Environment
- Implement the VEC simulation environment
- Model vehicles, edge nodes, communication channels, and task generation
- Implement weather-specific parameter modifiers for all five scenarios

### Phase 2 — Algorithm Implementation & Evaluation
- Implement the SAC-based Zone Manager
- Implement baseline (non-weather-aware) offloading algorithms
- Run experiments across all five scenarios
- Generate comparison graphs and analysis

## Genetic fallback for Phase 2

`GeneticBatchOffloader` implements the time-bounded, weather-aware fallback
described by Solution 3. A chromosome contains one `LOCAL`, `FOG`, or `CLOUD`
assignment per task. Its fitness accounts for:

- latency and energy;
- weather-related path-loss delay and packet-loss risk;
- per-vehicle, Fog, and Cloud queue contention;
- deadline misses and total lateness.

The optimizer always retains its best chromosome and stops after either the configured
generation count or wall-clock limit:

```python
from algorithms import GeneticBatchOffloader, GeneticOffloaderConfig

fallback = GeneticBatchOffloader(
    GeneticOffloaderConfig(
        population_size=32,
        max_generations=25,
        time_limit_seconds=0.15,
    )
)
result = fallback.optimize(tasks, vehicle_states)
assignments = result.by_task_id(tasks)
```

`RLGeneticFallbackController` adds reward-window detection and hysteresis. It uses
the RL policy normally, switches to the GA after sustained low reward, and returns
after sustained recovery. SAC experience collection and fine-tuning remain the
responsibility of the training loop and should continue while the GA is active.

Run the focused tests with:

```bash
PYTHONPATH=source .venv/bin/python -m unittest discover -s tests -v
```

## Gymnasium environment for SAC

`VECOffloadingEnv` processes one generated task per environment step and delegates
execution to the same Local, Fog, and Cloud functions used by the random baseline.
This keeps energy, retransmission, packet-loss, deadline, and dynamic backhaul
behavior consistent between baseline evaluation and reinforcement learning.

Stable-Baselines3 SAC requires a continuous action space. The environment maps its
single scalar action as follows:

| SAC action | Offloading target |
|---|---|
| `[-1.0, -1/3)` | Local |
| `[-1/3, 1/3)` | Fog |
| `[1/3, 1.0]` | Cloud |

Create the environment directly from generated XML:

```python
from vec_offloading_env import VECOffloadingEnv

env = VECOffloadingEnv.from_xml(
    tasks_file="source/data/tasks/chunk_0.xml",
    vehicles_file="source/data/vehicles/chunk_0.xml",
)
observation, info = env.reset(seed=37)
```

The observation is an 11-value normalized vector containing weather, vehicle speed,
task execution time, data size, cycles per bit, deadline slack, path loss, packet-loss
increase, nearest-Fog distance, task power, and current task-generation load.

### Weather-adaptive reward

The default `RewardConfig()` preserves one fixed reward profile for every scenario.
`RewardConfig.adaptive_default()` selects configurable weights by weather:

| Scenario | Latency | Energy | Reliability | Packet-loss penalty | Deadline penalty |
|---|---:|---:|---:|---:|---:|
| Base | 0.35 | 0.25 | 0.15 | 2.0 | 5.0 |
| Rain | 0.40 | 0.20 | 0.25 | 2.5 | 6.0 |
| Snow | 0.45 | 0.20 | 0.15 | 2.0 | 7.0 |
| Fog | 0.30 | 0.20 | 0.35 | 3.0 | 5.0 |

These are initial experimental values, not final calibrated coefficients. Custom
profiles can be passed with `RewardConfig(weather_profiles={...})`.

Compare fixed and adaptive reward shaping on the exact same actions and packet-loss
samples:

```bash
PYTHONPATH=source .venv/bin/python source/compare_reward_profiles.py \
  --tasks source/data/tasks/chunk_0.xml \
  --vehicles source/data/vehicles/chunk_0.xml \
  --target FOG \
  --output source/data/results/reward_profile_comparison.csv
```

This controlled replay verifies how reward shaping changes evaluation. It does not
prove that one policy is better: the final experiment must train separate SAC models
with fixed and adaptive rewards, then evaluate both models on common held-out seeds.

## SUMO and TraCI setup

Create the project environment and install the dependencies:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r dependencies/requirements.txt
```

Verify that Python can start SUMO and read live vehicle state through TraCI:

```bash
PATH="$PWD/.venv/bin:$PATH" \
  .venv/bin/python source/sumo_traci_smoke_test.py
```

The test creates a temporary road network and vehicle, advances SUMO one step at a
time, and prints the vehicle position and speed. It does not modify project data.

## Supervision

Teaching Assistant: **Mr. Abbas Shirang**

## License

This project is an academic course project.
