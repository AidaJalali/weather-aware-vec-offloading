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

## Supervision

Teaching Assistant: **Mr. Abbas Shirang**

## License

This project is an academic course project.
