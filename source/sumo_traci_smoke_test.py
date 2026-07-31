from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import sumolib
import traci


ROUTES_XML = """\
<routes>
    <vType id="passenger" vClass="passenger" accel="2.6" decel="4.5"
           length="5.0" maxSpeed="13.9"/>
    <route id="demo_route" edges="A0A1 A1B1"/>
    <vehicle id="demo_vehicle" type="passenger" route="demo_route"
             depart="0"/>
</routes>
"""


def run_smoke_test() -> None:
    sumo_binary = sumolib.checkBinary("sumo")
    netgenerate_binary = sumolib.checkBinary("netgenerate")

    with tempfile.TemporaryDirectory(prefix="weather-vec-sumo-") as temp_dir:
        scenario_dir = Path(temp_dir)
        network_file = scenario_dir / "demo.net.xml"
        routes_file = scenario_dir / "demo.rou.xml"

        subprocess.run(
            [
                netgenerate_binary,
                "--grid",
                "--grid.number=2",
                "--grid.length=200",
                "--output-file",
                str(network_file),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        routes_file.write_text(ROUTES_XML, encoding="utf-8")

        traci.start(
            [
                sumo_binary,
                "--net-file",
                str(network_file),
                "--route-files",
                str(routes_file),
                "--no-step-log",
                "true",
            ]
        )
        try:
            observed_steps = 0
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                if "demo_vehicle" in traci.vehicle.getIDList():
                    observed_steps += 1
                    position = traci.vehicle.getPosition("demo_vehicle")
                    speed = traci.vehicle.getSpeed("demo_vehicle")
                    print(
                        f"step={traci.simulation.getTime():.0f} "
                        f"position=({position[0]:.1f}, {position[1]:.1f}) "
                        f"speed={speed:.2f}"
                    )
        finally:
            traci.close()

    if observed_steps == 0:
        raise RuntimeError("TraCI connected, but no vehicle state was observed.")
    print(f"SUMO-TraCI smoke test passed ({observed_steps} observed steps).")


if __name__ == "__main__":
    run_smoke_test()
