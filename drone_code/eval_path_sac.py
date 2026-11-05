"""Compare PID-only with PID + flight SAC. Run with Isaac Sim's python.bat.

    /c/isaacsim/python.bat drone_code/eval_path_sac.py

Takes no arguments: set path_sac_config.PATH_EVAL first. The policy is frozen and
deterministic. Each seed fixes route, start, and wind, and both modes fly it.

The open-scene routes and the Dijkstra route need different scenes, so this script
launches itself once for each and then merges the results. Writes
runs/flight_sac/<run_name>/evaluation/<model>/: results.json, flights.csv, and
summary.csv. Failed flights stay in every number. See PATH_SAC.md for metrics.
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import path_sac_config


PART_VARIABLE = "PATH_SAC_EVAL_PART"

run = path_sac_config.eval_settings()

project_dir = Path(__file__).resolve().parents[1]
run_dir = project_dir / path_sac_config.RUN_DIR / run["run_name"]
model_path = project_dir / (run["model"] or run_dir / "final_model.zip")
output_dir = (
    run_dir
    / "evaluation"
    / ("pid_only" if run["pid_only"] else model_path.stem)
)

if not run["pid_only"]:
    if not model_path.is_file():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Set pid_only in PATH_EVAL for the baseline alone."
        )

    problem = path_sac_config.incompatibility(model_path)

    if problem:
        raise SystemExit(problem)

seeds = path_sac_config.EVAL_SEEDS

if run["episodes"] is not None:
    seeds = seeds[:run["episodes"]]


def write_csv(path, rows):
    names = []

    for row in rows:
        names += [name for name in row if name not in names]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def run_parts():
    """Launch each part in its own process; merge what they saved."""
    output_dir.mkdir(parents=True, exist_ok=True)

    parts = ["open"] + (["dijkstra"] if run["include_dijkstra"] else [])
    flights = []

    for part in parts:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve())],
            env={**os.environ, PART_VARIABLE: part},
        )

        if result.returncode != 0:
            raise SystemExit(f"Evaluation part '{part}' failed.")

        flights += json.loads(
            (output_dir / f"flights_{part}.json").read_text(encoding="utf-8")
        )

    # Isaac Sim is not needed to summarize.
    from path_sac_policy import summarize

    groups = {}

    for flight in flights:
        key = (flight["scenario"], flight["wind_profile"], flight["controller"])
        groups.setdefault(key, []).append(flight)

    summary = [
        {
            "scenario": scenario,
            "wind_profile": profile,
            "controller": controller,
            **summarize(rows),
        }
        for (scenario, profile, controller), rows in groups.items()
    ]

    (output_dir / "results.json").write_text(
        json.dumps({"summary": summary, "flights": flights}, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "flights.csv", flights)
    write_csv(output_dir / "summary.csv", summary)

    print(
        f"\n{'scenario':14}{'wind':16}{'mode':10}{'RMSE':>7}{'cross':>7}"
        f"{'max':>7}{'done':>6}{'fail':>6}{'sat':>7}{'|a|':>7}{'|da|':>7}"
    )

    for s in summary:
        print(
            f"{s['scenario']:14}{s['wind_profile']:16}{s['controller']:10}"
            f"{s['position_rmse']:7.3f}{s['cross_track_rmse']:7.3f}"
            f"{s['max_deviation']:7.2f}{s['completion_rate']:6.2f}"
            f"{s['failure_rate']:6.2f}{s['saturation_rate']:7.3f}"
            f"{s['residual_norm']:7.3f}{s['residual_change']:7.3f}"
        )

    print(f"\nSaved results in {output_dir}", flush=True)


def controllers():
    from path_sac_policy import load_path_policy

    result = {"pid_only": None}

    if not run["pid_only"]:
        result["pid_sac"] = load_path_policy(model_path)

    return result


def run_open_part(simulation_app):
    import numpy as np

    from path_sac_env import make_path_sac_env

    env = make_path_sac_env(simulation_app, render_sim=run["gui"])

    def fly(seed, scenario, route_type, profile, policy):
        observation, info = env.reset(seed=seed, options={
            "route_stage": path_sac_config.EVAL_ROUTE_STAGE,
            "route_type": route_type,
            "wind_profile": profile,
        })

        reason = None

        while True:
            action = (
                np.zeros(4, dtype=np.float32)
                if policy is None
                else policy(observation)
            )

            observation, _, terminated, truncated, step_info = env.step(action)
            reason = step_info["failure_reason"]

            if terminated or truncated:
                break

        return {
            "scenario": scenario,
            "wind_profile": profile,
            "seed": int(seed),
            "wind_speed": float(np.linalg.norm(info["wind_mean"])),
            "wind_seed": info["wind_seed"],
            "route_seconds": info["route_seconds"],
            **env.trace.metrics(
                env.core.reference.path,
                reason,
                reason == "too_low",
                env.core.completion,
            ),
        }

    try:
        policies = controllers()
        flights = []

        for scenario, route_type in path_sac_config.EVAL_SCENARIOS.items():
            for profile in path_sac_config.EVAL_WIND_PROFILES:
                baseline = {}

                for name, policy in policies.items():
                    for seed in seeds:
                        row = fly(seed, scenario, route_type, profile, policy)
                        row["controller"] = name

                        # Both modes must have flown the same realization.
                        matched = (
                            row["wind_speed"], row["wind_seed"],
                            row["route_seconds"],
                        )

                        if baseline.setdefault(seed, matched) != matched:
                            raise RuntimeError(f"Flight differs on seed {seed}.")

                        flights.append(row)

                print(f"Finished {scenario} / {profile}", flush=True)

        return flights

    finally:
        env.close()


def run_dijkstra_part(simulation_app):
    from path_navigation import (
        draw_wind,
        fly_route,
        load_navigation_sim,
        make_core,
        plan_route,
    )

    sim = load_navigation_sim(simulation_app)

    try:
        route = plan_route(sim, simulation_app)
        core = make_core(sim)
        policies = controllers()
        flights = []

        for profile_name, profile in path_sac_config.EVAL_WIND_PROFILES.items():
            for name, policy in policies.items():
                for seed in seeds:
                    wind = draw_wind(seed, profile_name)

                    flights.append({
                        "scenario": "dijkstra",
                        "wind_profile": profile_name,
                        "seed": int(seed),
                        "wind_seed": wind["seed"],
                        "route_seconds": route.reference.duration,
                        "controller": name,
                        **fly_route(
                            core, route, policy, wind, profile, run["gui"]
                        ),
                    })

            print(f"Finished dijkstra / {profile_name}", flush=True)

        return flights

    finally:
        sim.close()


part = os.environ.get(PART_VARIABLE)

if part is None:
    run_parts()
    sys.exit(0)

from isaacsim import SimulationApp

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": not run["gui"]})

try:
    flights = (
        run_open_part(simulation_app)
        if part == "open"
        else run_dijkstra_part(simulation_app)
    )

    (output_dir / f"flights_{part}.json").write_text(
        json.dumps(flights, indent=2),
        encoding="utf-8",
    )
finally:
    simulation_app.close()
