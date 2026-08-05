# weather-aware-vec-offloading

Weather-aware adaptive task offloading in Vehicular Edge Computing (VEC) using Soft Actor-Critic (SAC) deep reinforcement learning.

## Quick start

```bash
# Create environment and install dependencies
uv venv .venv
uv pip install --python .venv/bin/python -r dependencies/requirements.txt

# Run the full Phase 1 pipeline (120-second SUMO simulation)
PYTHONPATH=src .venv/bin/python src/run_phase1_pipeline.py --duration 120

# Run tests
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

## Data generation with SUMO

Mobility data is generated using SUMO via TraCI. The pipeline creates a grid road
network, simulates vehicle movement with realistic dynamics, applies weather effects
inside the simulation, and generates offloading tasks.

### Dataset structure

Every active dataset contains exactly 1,000 simulation timesteps and has a name
starting with its category:

```
data/datasets/
├── train/
│   ├── train_base_1 ... train_fog_2       # 8 static
│   └── train_slow_mix_1 ... _4            # 4 x 250-second blocks
├── finetune/
│   ├── finetune_base ... finetune_fog     # 4 static
│   ├── finetune_slow_mix_1 ... _2
│   └── finetune_fast_mix_1 ... _2
└── test/
    ├── test_base, test_rain, test_snow, test_fog
    ├── test_fast_mix
    ├── test_slow_mix
    └── test_random_mix_1, test_random_mix_2
```

The 12 train datasets are used only for initial SAC training. The 8 finetune
datasets use unseen seeds for online adaptation and selecting fallback settings.
The 8 test datasets remain untouched until final comparison. In each random-mix
test, weather blocks have reproducible random durations between 100 and 200 steps.

Inspect the complete plan without running SUMO:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_sac_curriculum.py --dry-run
```

Generate missing datasets in one category, or all categories:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_sac_curriculum.py --group train
PYTHONPATH=src .venv/bin/python scripts/generate_sac_curriculum.py --group finetune
PYTHONPATH=src .venv/bin/python scripts/generate_sac_curriculum.py --group test
PYTHONPATH=src .venv/bin/python scripts/generate_sac_curriculum.py --group all
```

Complete datasets are skipped. Replacing them requires explicit `--overwrite`.
The former 3,600-step datasets remain locally under `data/sumo`, but are ignored by
Git and are not used by the active training or evaluation code.

Key modules:

| Module | Role |
|--------|------|
| `src/sumo_pipeline.py` | SUMO network, route generation, TraCI orchestration |
| `src/task_generation.py` | Deterministic seeded task-parameter generation |
| `src/xml_dataset_writer.py` | Chunk-buffered vehicle/task XML writer |
| `src/weather_scenarios.py` | Weather scenario definitions and effects |
| `src/weather_scenario_generator.py` | Weather schedule CSV generator |
| `scripts/generate_sac_curriculum.py` | Generate all train/finetune/test datasets |

## Genetic fallback for Phase 2

`GeneticBatchOffloader` implements the time-bounded, weather-aware fallback
described by Solution 3. A chromosome contains one `LOCAL`, `FOG`, or `CLOUD`
assignment per task. Its fitness accounts for:

- dynamic latency and total system energy;
- weather-related packet-loss and final-transmission-failure risk;
- persistent per-vehicle, Fog, and Cloud queue contention;
- expected deadline failures and total lateness.

GA ranks chromosomes by expected task failures first. This combines final packet
failure probability with successful executions expected to miss their deadlines.
Expected packet failures are the next explicit criterion, followed by weighted
normalized latency and total system energy.

Random, GeneticBatch, and SAC produce only target assignments. The shared
`src/offloading_simulator.py` executes those assignments with the same capacities,
deterministic packet samples, retransmissions, energy equations, and dynamic Cloud
backhaul model.

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
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
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
    tasks_file="data/datasets/test/test_base/tasks/chunk_0.xml.gz",
    vehicles_file="data/datasets/test/test_base/vehicles/chunk_0.xml.gz",
)
observation, info = env.reset(seed=37)
```

The observation is a 21-value normalized vector containing weather, vehicle speed,
task properties, deadline slack, path loss, Fog/Cloud packet-loss probabilities,
nearest-Fog distance, current task-generation load, and Local/Fog/Cloud queue and
capacity state.

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
PYTHONPATH=src .venv/bin/python src/compare_reward_profiles.py \
  --tasks data/datasets/finetune/finetune_base/tasks/chunk_0.xml.gz \
  --vehicles data/datasets/finetune/finetune_base/vehicles/chunk_0.xml.gz \
  --target FOG \
  --output outputs/reward_profile_comparison.csv
```

This controlled replay verifies how reward shaping changes evaluation. It does not
prove that one policy is better: the final experiment must train separate SAC models
with fixed and adaptive rewards, then evaluate both models on common held-out seeds.

## Staged SAC pretraining

The SAC state uses weather one-hot values, task and network characteristics,
Fog/Cloud packet-loss probabilities, and Local/Fog/Cloud queue occupancy and
available capacity. One small SAC model is trained continuously in this order:

```text
8 static datasets -> 4 slow mixed datasets
```

Start the complete curriculum after generating the data:

```bash
PYTHONPATH=src .venv/bin/python src/train_sac.py
```

The model uses a two-layer `64 x 64` policy and a bounded replay buffer. Cumulative
checkpoints are saved after datasets 4, 8, and 12 under
`outputs/models/sac/checkpoints/`. The final model is written to
`outputs/models/sac/sac_pretrained_final.zip`. These are conservative resource
settings, not tuned final hyperparameters.

For a short pipeline check before a full training run:

```bash
PYTHONPATH=src .venv/bin/python src/train_sac.py --steps-per-dataset 100
```

The current environment makes one SAC transition per task. Tasks released in the
same SUMO timestep are handled sequentially while sharing the same persistent queue
state.

## Online SAC with GeneticBatch fallback

`src/online_hybrid.py` loads the pretrained SAC and monitors a rolling transmission-
loss event rate. A loss event means that a task needed at least one retransmission or
ultimately lost its packet. The controller switches from SAC to GeneticBatch after
sustained high loss and returns to SAC after sustained recovery. Separate enter and
exit thresholds provide hysteresis.

The eight test datasets run sequentially in this order: BASE, RAIN, SNOW, FOG,
FAST_MIXED, SLOW_MIXED, RANDOM_MIX_1, and RANDOM_MIX_2. SAC is deterministic and
frozen during normal control. During fallback, GeneticBatch controls the executed
assignments and SAC fine-tunes from those actual GA actions, rewards, and next
observations. When GA releases control, SAC becomes frozen again. The adapted model
and cumulative CSV logs are saved after each completed dataset. They use one stable
output path, so pretraining is never overwritten and extra checkpoints are not made.

Run the complete online test protocol from the pretrained model:

```bash
PYTHONPATH=src .venv/bin/python src/online_hybrid.py --split test
```

This writes:

```text
outputs/models/sac_online/sac_adapted_final.zip
outputs/online/test_hybrid_results.csv
outputs/online/test_hybrid_switches.csv
```

For a shorter run on one test stream while keeping the same conditional fine-tuning
logic:

```bash
PYTHONPATH=src .venv/bin/python src/online_hybrid.py \
  --split test \
  --dataset test_slow_mix \
  --results-file outputs/online/test_stream_results.csv \
  --switch-file outputs/online/test_stream_switches.csv
```

The monitor defaults are a 100-task window, checks every 20 tasks, 5% fallback
threshold, 2% recovery threshold, two consecutive checks, and at least 100 GA tasks
before recovery. These are initial experiment settings rather than tuned values.
Use `notebooks/online_hybrid_evaluation.ipynb` to inspect switch timing and compare
the rolling loss, reward, latency, energy, and cumulative failure curves. Passing
`--no-train` disables even fallback fine-tuning and provides a fully frozen diagnostic
run. Because the normal protocol adapts while evaluating the stream, report its
results as online or prequential evaluation.

### Frozen pretrained evaluation

Evaluate the saved pretrained policy on the eight 1,000-timestep test datasets using
the same simulator as Random and GeneticBatch:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation.py \
  --algorithm sac_pretrained
```

This loads `outputs/models/sac/sac_pretrained_final.zip`, selects deterministic SAC
actions, and does not train or modify the model. Detailed results and the eight-dataset
summary are saved under `outputs/evaluation/sac_pretrained/`.

Run all three algorithms for a complete notebook comparison with:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation.py --algorithm all
```

Use `notebooks/weather_dataset_new.ipynb` to inspect the new data and
`notebooks/evaluation.ipynb` to plot the evaluation summaries.

## SUMO and TraCI setup

Verify SUMO/TraCI connectivity:

```bash
.venv/bin/python src/sumo_traci_smoke_test.py
```

The test creates a temporary road network and vehicle, advances SUMO one step at a
time, and prints vehicle position and speed. It does not modify project data.

## Supervision

Teaching Assistant: **Mr. Abbas Shirang**

## License

This project is an academic course project.
