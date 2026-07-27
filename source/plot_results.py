from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path


SCENARIO_ORDER = ("BASE", "RAIN", "SNOW", "FOG")


def load_results(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_by_scenario(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)

    summary = {}
    for scenario in SCENARIO_ORDER:
        items = grouped.get(scenario, [])
        total = len(items)
        if total == 0:
            summary[scenario] = {
                "total_tasks": 0,
                "deadline_misses": 0,
                "packet_losses": 0,
                "avg_latency": 0.0,
                "avg_energy": 0.0,
                "avg_vehicle_energy": 0.0,
                "avg_infrastructure_energy": 0.0,
                "avg_total_system_energy": 0.0,
            }
            continue

        deadline_misses = sum(row["deadline_missed"] == "True" for row in items)
        packet_losses = sum(row["packet_lost"] == "True" for row in items)
        avg_latency = sum(float(row["latency"]) for row in items) / total
        avg_vehicle_energy = sum(float(row["vehicle_energy"]) for row in items) / total
        avg_infrastructure_energy = (
            sum(float(row["infrastructure_energy"]) for row in items) / total
        )
        avg_total_system_energy = (
            sum(float(row["total_system_energy"]) for row in items) / total
        )
        summary[scenario] = {
            "total_tasks": total,
            "deadline_misses": deadline_misses,
            "packet_losses": packet_losses,
            "avg_latency": avg_latency,
            "avg_energy": avg_vehicle_energy,
            "avg_vehicle_energy": avg_vehicle_energy,
            "avg_infrastructure_energy": avg_infrastructure_energy,
            "avg_total_system_energy": avg_total_system_energy,
        }
    return summary


def write_summary(summary: dict[str, dict[str, float]], output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "total_tasks",
                "deadline_misses",
                "packet_losses",
                "avg_latency",
                "avg_energy",
                "avg_vehicle_energy",
                "avg_infrastructure_energy",
                "avg_total_system_energy",
            ],
        )
        writer.writeheader()
        for scenario in SCENARIO_ORDER:
            row = {"scenario": scenario}
            row.update(summary[scenario])
            writer.writerow(row)


def plot_metric(
    summary: dict[str, dict[str, float]],
    metric: str,
    title: str,
    ylabel: str,
    output_file: str | Path,
) -> None:
    os.makedirs("./.matplotlib", exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "./.matplotlib")
    import matplotlib.pyplot as plt

    scenarios = list(SCENARIO_ORDER)
    values = [summary[scenario][metric] for scenario in scenarios]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(scenarios, values, color=["#4d6b8a", "#527f5f", "#9a6a3a", "#6f5d9a"])
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_file, dpi=180)
    plt.close(fig)


def plot_all(
    results_file: str | Path = "data/results/random_baseline_results.csv",
    summary_file: str | Path = "data/results/random_baseline_summary.csv",
    output_dir: str | Path = "data/results",
) -> dict[str, dict[str, float]]:
    rows = load_results(results_file)
    summary = summarize_by_scenario(rows)
    write_summary(summary, summary_file)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plot_metric(
        summary,
        "deadline_misses",
        "Deadline Misses - Random Baseline",
        "Count",
        out / "deadline_misses_random.png",
    )
    plot_metric(
        summary,
        "packet_losses",
        "Packet Losses - Random Baseline",
        "Count",
        out / "packet_losses_random.png",
    )
    plot_metric(
        summary,
        "avg_latency",
        "Average Latency - Random Baseline",
        "Latency",
        out / "avg_latency_random.png",
    )
    plot_metric(
        summary,
        "avg_energy",
        "Average Energy - Random Baseline",
        "Energy",
        out / "avg_energy_random.png",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="data/results/random_baseline_results.csv")
    parser.add_argument("--summary", default="data/results/random_baseline_summary.csv")
    parser.add_argument("--output-dir", default="data/results")
    args = parser.parse_args()
    plot_all(args.results, args.summary, args.output_dir)
    print(f"Summary and plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
