from __future__ import annotations

import argparse
import math
import random
import xml.etree.ElementTree as Et
from pathlib import Path


def generate_mobility(
    output_file: str | Path,
    duration: int = 120,
    num_user_vehicles: int = 12,
    num_mobile_fogs: int = 3,
    seed: int = 7,
) -> None:
    rng = random.Random(seed)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vehicles = []
    for i in range(num_user_vehicles):
        vehicles.append({
            "id": f"PKW{i:04d}",
            "type": "PKW_special",
            "x": rng.uniform(50, 600),
            "y": rng.uniform(50, 500),
            "speed": rng.uniform(8, 18),
            "angle": rng.choice([0, 45, 90, 135, 180, 225, 270, 315]),
            "lane": f"lane_{i % 4}",
        })

    for i in range(num_mobile_fogs):
        vehicles.append({
            "id": f"LKW{i:04d}",
            "type": "LKW_special",
            "x": rng.uniform(50, 600),
            "y": rng.uniform(50, 500),
            "speed": rng.uniform(6, 14),
            "angle": rng.choice([0, 90, 180, 270]),
            "lane": f"lane_{i % 4}",
        })

    root = Et.Element("fcd-export")
    root.set("version", "1.0")

    for step in range(duration):
        time_elem = Et.SubElement(root, "timestep")
        time_elem.set("time", str(step))

        for vehicle in vehicles:
            angle_rad = math.radians(vehicle["angle"])
            x = (vehicle["x"] + vehicle["speed"] * math.cos(angle_rad) * step) % 900
            y = (vehicle["y"] + vehicle["speed"] * math.sin(angle_rad) * step) % 700

            v_elem = Et.SubElement(time_elem, "vehicle")
            v_elem.set("id", vehicle["id"])
            v_elem.set("x", f"{x:.2f}")
            v_elem.set("y", f"{y:.2f}")
            v_elem.set("angle", f"{vehicle['angle']:.2f}")
            v_elem.set("speed", f"{vehicle['speed']:.2f}")
            v_elem.set("lane", vehicle["lane"])
            v_elem.set("type", vehicle["type"])

    Et.indent(root, space="    ", level=0)
    tree = Et.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/raw_mobility.xml")
    parser.add_argument("--duration", type=int, default=120)
    parser.add_argument("--users", type=int, default=12)
    parser.add_argument("--mobile-fogs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    generate_mobility(
        output_file=args.output,
        duration=args.duration,
        num_user_vehicles=args.users,
        num_mobile_fogs=args.mobile_fogs,
        seed=args.seed,
    )
    print(f"Synthetic mobility saved to {args.output}")


if __name__ == "__main__":
    main()

