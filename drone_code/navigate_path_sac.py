"""Fly the Dijkstra route with the flight model. Run with Isaac Sim's python.bat.

    /c/isaacsim/python.bat drone_code/navigate_path_sac.py

Takes no arguments: set path_sac_config.PATH_NAV first. mode "pid" flies PID only.
mode "pid_sac" stops with an error if the model is missing or incompatible; it never
falls back on its own. motor_physics.py and its modes are not involved.

Uses the same reference, observations, action scaling, and timing as training,
through path_sac_policy.PathCore. The window stays open after the flight.
"""
import json
from pathlib import Path

from isaacsim import SimulationApp

import path_sac_config


run = path_sac_config.PATH_NAV

project_dir = Path(__file__).resolve().parents[1]
run_dir = project_dir / path_sac_config.RUN_DIR / run["run_name"]
model_path = project_dir / (run["model"] or run_dir / "final_model.zip")

if run["mode"] not in ("pid", "pid_sac"):
    raise SystemExit('PATH_NAV mode must be "pid" or "pid_sac".')

if run["wind_profile"] not in path_sac_config.EVAL_WIND_PROFILES:
    raise SystemExit(f"Unknown wind profile: {run['wind_profile']}")

if run["mode"] == "pid_sac":
    if not model_path.is_file():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            'Train one, set model in PATH_NAV, or set mode "pid" to fly PID only.'
        )

    problem = path_sac_config.incompatibility(model_path)

    if problem:
        raise SystemExit(problem)

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": not run["gui"]})


def main():
    import numpy as np

    from path_navigation import (
        draw_wind,
        fly_route,
        load_navigation_sim,
        make_core,
        plan_route,
    )
    from path_sac_policy import load_path_policy

    sim = load_navigation_sim(simulation_app)

    try:
        route = plan_route(sim, simulation_app)
        core = make_core(sim)

        policy = None

        if run["mode"] == "pid_sac":
            policy = load_path_policy(model_path)

        print(f"Control mode: {run['mode']}", flush=True)

        profile = path_sac_config.EVAL_WIND_PROFILES[run["wind_profile"]]

        metrics = fly_route(
            core,
            route,
            policy,
            draw_wind(run["seed"], run["wind_profile"]),
            profile,
            run["gui"],
        )

        print(json.dumps(metrics, indent=2), flush=True)

        output = run_dir / "navigation"
        output.mkdir(parents=True, exist_ok=True)

        (output / f"{run['mode']}_{run['wind_profile']}.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        print(f"Saved results in {output}", flush=True)

        # Hold at the last point until the window closes.
        if run["gui"]:
            core.apply_action(np.zeros(4, dtype=np.float32))
            state = sim.read_drone_state()

            while sim.is_running():
                core.step(state, render=True)
                state = sim.read_drone_state()

    finally:
        sim.close()


try:
    main()
finally:
    simulation_app.close()
