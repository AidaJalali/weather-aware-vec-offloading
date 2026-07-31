# weather-aware-vec-offloading

Weather-aware adaptive task offloading in Vehicular Edge Computing (VEC) using Soft Actor-Critic (SAC) deep reinforcement learning.

## Quick start

```bash
# Create environment and install dependencies
uv venv .venv
uv pip install --python .venv/bin/python -r dependencies/requirements.txt

# Run the full Phase 1 pipeline (120-second SUMO simulation)
PYTHONPATH=source .venv/bin/python source/run_phase1_pipeline.py --duration 120

# Run tests
PYTHONPATH=source .venv/bin/python -m unittest discover -s tests -v
```

## Data generation with SUMO

Mobility data is generated using SUMO via TraCI. The pipeline creates a grid road
network, simulates vehicle movement with realistic dynamics, applies weather effects
inside the simulation, and generates offloading tasks.

### Quick: generate the full dataset

```bash
# Generate all 31 training/validation datasets (3.9M tasks, ~25 min)
PYTHONPATH=source .venv/bin/python scripts/generate_all_datasets.py
```

This produces:

```
source/data/sumo/
├── train/
│   ├── s42_u12_f3/        ┐
│   ├── s123_u12_f3/       │ 10 mixed-weather seeds
│   ├── ...                │ (BASE → RAIN → SNOW → FOG cycle)
│   ├── s666_u12_f3/       ┘
│   ├── weather_base_s100/ ┐
│   ├── weather_rain_s100/ │ 12 per-weather datasets
│   ├── weather_snow_s100/ │ (3 seeds × 4 weathers)
│   ├── weather_fog_s100/  ┘
│   ├── s42_u20_f5_h/      ┐
│   ├── s123_u8_f2_l/      │ 4 density variants
│   ├── s42_u30_f7_vh/     │ (5–30 users, 1–7 fogs)
│   └── s123_u5_f1_vl/     ┘
└── val/
    ├── s999_u12_f3/        ← mixed-weather held-out
    ├── weather_base_s999/  ┐
    ├── weather_rain_s999/  │ per-weather held-out
    ├── weather_snow_s999/  │
    └── weather_fog_s999/   ┘
```

Each dataset contains `vehicles/chunk_0.xml.gz` and `tasks/chunk_0.xml.gz` (gzipped
for git — 1.2GB → 111MB). Loaders in `infrastructure.py` read `.xml.gz` transparently:
pass a `.xml` path and it auto-resolves to `.xml.gz` if found.

### Single dataset

```bash
# Generate a 3600-second dataset with custom parameters
PYTHONPATH=source .venv/bin/python source/sumo_pipeline.py \
  --duration 3600 \
  --users 12 \
  --mobile-fogs 3 \
  --seed 42 \
  --weather-schedule source/data/weather_scenarios.csv \
  --output-dir source/data/sumo/my_dataset \
  --overwrite
```

Key modules:

| Module | Role |
|--------|------|
| `source/sumo_pipeline.py` | SUMO network, route generation, TraCI orchestration |
| `source/task_generation.py` | Deterministic seeded task-parameter generation |
| `source/xml_dataset_writer.py` | Chunk-buffered vehicle/task XML writer |
| `source/weather_scenarios.py` | Weather scenario definitions and effects |
| `source/weather_scenario_generator.py` | Weather schedule CSV generator |
| `scripts/generate_all_datasets.py` | Batch generation of all train/val datasets |

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
    tasks_file="source/data/sumo/tasks/chunk_0.xml",
    vehicles_file="source/data/sumo/vehicles/chunk_0.xml",
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
  --tasks source/data/sumo/tasks/chunk_0.xml \
  --vehicles source/data/sumo/vehicles/chunk_0.xml \
  --target FOG \
  --output source/data/sumo/results/reward_profile_comparison.csv
```

This controlled replay verifies how reward shaping changes evaluation. It does not
prove that one policy is better: the final experiment must train separate SAC models
with fixed and adaptive rewards, then evaluate both models on common held-out seeds.

## SUMO and TraCI setup

Verify SUMO/TraCI connectivity:

```bash
.venv/bin/python source/sumo_traci_smoke_test.py
```

The test creates a temporary road network and vehicle, advances SUMO one step at a
time, and prints vehicle position and speed. It does not modify project data.

## Supervision

Teaching Assistant: **Mr. Abbas Shirang**

## License

This project is an academic course project.
