# weather-aware-vec-offloading

Weather-aware task offloading in Vehicular Edge Computing using Random,
GeneticBatch, and categorical Soft Actor-Critic (SAC) controllers.

## Setup

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r dependencies/requirements.txt

PYTHONPATH=src:. .venv/bin/python -m unittest discover -s tests -v
```

## Dataset pipeline

SUMO and TraCI generate vehicle mobility. The task generator then applies the
weather table for BASE, RAIN, SNOW, and FOG. Weather changes vehicle speed, task
rate, compute demand, deadline type, and packet-loss-rate increase. Path loss is
not generated or used; table-driven packet loss is the communication model.

Weather-reduced speed is compared with a weather-adjusted speed threshold, so a
normal FOG vehicle is not incorrectly classified as traffic congestion.

Every active dataset contains 1,000 simulation timesteps:

```text
data/datasets/
  train/       8 static + 4 slow-mixed datasets
  finetune/    4 static + 2 slow-mixed + 2 fast-mixed datasets
  test/        4 static + fast, slow, and 2 random-mixed datasets
```

Inspect the generation plan:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_sac_curriculum.py --dry-run
```

The congestion correction changes task counts, so regenerate every active dataset
before evaluating or training:

```bash
PYTHONPATH=src .venv/bin/python scripts/generate_sac_curriculum.py \
  --group all --overwrite
```

Check the offered load against Local, mobile-Fog, and Cloud capacity:

```bash
PYTHONPATH=src .venv/bin/python src/analyze_workload.py --split train
PYTHONPATH=src .venv/bin/python src/analyze_workload.py --split test
```

The report includes tasks/s, required local-compute seconds/s, average users and
mobile Fog nodes, and lower-bound resource utilization. After an evaluation, add
`--results-file PATH` to estimate queue-delay growth by weather.

## Shared simulator

All controllers only choose `LOCAL`, `FOG`, or `CLOUD`. The shared simulator owns:

- persistent Local, mobile-Fog, and Cloud queues;
- the same resource capacities for every algorithm;
- deterministic packet samples, two retries, and retry latency;
- vehicle transmission and infrastructure compute energy;
- dynamic Cloud backhaul from weather and network load;
- packet loss, deadline misses, latency, and total system energy.

Mobile `LKW_special` vehicles are Fog nodes at their actual position each timestep.
The three fixed Fog positions are used only when a dataset has no mobile Fog state.

## Discrete SAC environment

The action space is categorical:

| Action | Target |
|---:|---|
| 0 | Local |
| 1 | Fog |
| 2 | Cloud |

The 19-value observation contains weather one-hot values, task data and compute
demand, deadline budget, task power, nearest-Fog distance, Fog/Cloud terminal loss
risk, network load, three estimated queue waits, and the remaining task count,
compute demand, and data volume in the current timestep. Numeric scales are the
99th percentiles derived only from pretraining data and are stored in the model.

The fixed bounded reward is:

```text
reward = -(0.50 * loss_cost + 0.35 * latency_cost + 0.15 * energy_cost)
```

Each cost is clipped to `[0, 1]`; reward is therefore in `[-1, 0]`. Latency already
contains queue delay, so queue delay is visible in the state but is not counted a
second time in the reward. Energy uses `total_system_energy`.

## SAC pretraining

`src/discrete_sac.py` implements a small categorical SAC actor with twin critics.
The default training protocol uses:

- a `64 x 64` network;
- 10 epochs over the 12 pretraining datasets;
- shuffled dataset order each epoch and temporal order within each dataset;
- a 500,000-transition weather-balanced replay buffer;
- random balanced minibatches and `gamma=0.995`;
- validation after every training dataset on ordered 300-timestep finetune
  slices, which include weather changes in slow and fast mixed datasets;
- best-checkpoint selection by bounded validation reward, with every finetune
  dataset weighted equally regardless of its task count.

Train it with:

```bash
PYTHONPATH=src .venv/bin/python src/train_sac.py
```

Outputs:

```text
outputs/models/discrete_sac/sac_discrete_best.pt
outputs/models/discrete_sac/sac_discrete_final.pt
outputs/models/discrete_sac/checkpoints/epoch_02.pt
outputs/models/discrete_sac/checkpoints/epoch_04.pt
outputs/models/discrete_sac/checkpoints/epoch_06.pt
outputs/models/discrete_sac/checkpoints/epoch_08.pt
outputs/models/discrete_sac/checkpoints/epoch_10.pt
outputs/models/discrete_sac/validation_history.csv
```

For a short code-path check, not a meaningful model:

```bash
PYTHONPATH=src .venv/bin/python src/train_sac.py \
  --epochs 1 --steps-per-dataset 100 --validation-timesteps 5 \
  --output-dir /tmp/vec-sac-smoke
```

The former continuous Stable-Baselines3 SAC checkpoints are incompatible with the
new discrete action and 19-field observation spaces.

## Evaluation

Regenerate data first, then rerun every controller so all comparisons use the same
corrected workload:

```bash
PYTHONPATH=src .venv/bin/python src/evaluation.py --algorithm both
PYTHONPATH=src .venv/bin/python src/evaluation.py --algorithm sac_pretrained

# Or run all three in one command:
PYTHONPATH=src .venv/bin/python src/evaluation.py --algorithm all
```

`notebooks/evaluation.ipynb` plots deadline misses, packet losses, average latency,
and average total-system energy. `notebooks/weather_dataset_new.ipynb` inspects the
weather schedules.

## Online SAC and GA fallback

The online stream begins with frozen deterministic SAC. A rolling monitor checks:

- transmission loss events, including retries;
- deadline-miss rate;
- normalized latency (`latency / deadline budget`).

Two consecutive bad checks switch control to GeneticBatch. Only while GA controls
the tasks are its transitions collected and SAC updated. Recovery requires all
three metrics to stay below their lower recovery thresholds for two checks and at
least 100 fallback tasks. The pretrained model is never overwritten.

Run the ordered test stream:

```bash
PYTHONPATH=src .venv/bin/python src/online_hybrid.py --split test
```

Outputs:

```text
outputs/models/discrete_sac_online/sac_adapted_final.pt
outputs/online/test_hybrid_results.csv
outputs/online/test_hybrid_switches.csv
```

Run one short stream or disable adaptation for diagnosis:

```bash
PYTHONPATH=src .venv/bin/python src/online_hybrid.py \
  --split test --dataset test_slow_mix --max-timesteps 400

PYTHONPATH=src .venv/bin/python src/online_hybrid.py \
  --split test --dataset test_slow_mix --no-train
```

The default thresholds are starting values, not selected hyperparameters. Use
`notebooks/online_hybrid_evaluation.ipynb` to inspect each switch and all three
rolling monitor signals before choosing final thresholds on finetune data.

## Key modules

| Module | Role |
|---|---|
| `src/task_generation.py` | Weather-aware seeded task generation |
| `src/offloading_simulator.py` | Shared queues, communication, execution, energy |
| `src/algorithms.py` | Random and GeneticBatch target selection |
| `src/vec_offloading_env.py` | Discrete action, observation, bounded reward |
| `src/discrete_sac.py` | Categorical SAC and balanced replay |
| `src/train_sac.py` | Multi-epoch shuffled pretraining and validation |
| `src/online_hybrid.py` | Frozen SAC, monitored GA fallback, online updates |

## Supervision

Teaching Assistant: **Mr. Abbas Shirang**

## License

This project is an academic course project.
