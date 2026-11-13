"""Task 2: paired wind-step evaluation using the unchanged path-SAC policy.

Run with Isaac Sim python.bat; --help and missing-model checks do not launch Sim.
Writes a manifest, per-decision trajectories, and per-flight metrics. No training.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from wind_events import SCENARIOS, WindEvent, event_metrics, trajectory_event_time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-only", action="store_true")
    parser.add_argument("--model", type=Path, default=Path("runs/flight_sac/flight1/final_model.zip"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[410000, 410001, 410002])
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--event-s", type=float, default=None,
                        help="Hover defaults to 5 s; moving routes default to 35 percent into available motion")
    parser.add_argument("--route", choices=["hover", "straight", "waypoints"], default="hover")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    import math
    if args.route == "hover" and args.event_s is None:
        args.event_s = 5.0
    if (not math.isfinite(args.duration) or (args.event_s is not None and (
                not math.isfinite(args.event_s) or args.event_s <= 0 or args.duration < args.event_s + 2))
            or args.duration < 3):
        parser.error("duration must be finite and at least event-s + 2; event-s must be positive")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be unique nonnegative integers")
    if len(set(args.scenarios)) != len(args.scenarios):
        parser.error("scenarios must be unique")
    import path_sac_config as cfg
    if args.duration > cfg.EPISODE["max_seconds"]:
        parser.error("duration exceeds the environment's max_seconds")
    model = args.model if args.model.is_absolute() else ROOT / args.model
    if not args.pid_only:
        if not model.is_file():
            parser.error(f"Missing path-SAC checkpoint: {model}. Use --pid-only to verify disturbances first.")
        problem = cfg.incompatibility(model)
        if problem:
            parser.error(problem)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        parser.error(f"Output folder already exists: {output}. Choose a new --output folder to preserve previous results.")
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(ROOT), *arguments], text=True).strip()
    manifest = {
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "seeds": args.seeds, "scenarios": args.scenarios, "event_s": args.event_s,
        "duration_s": args.duration, "wind_profile": "training_range",
        "mean_wind_scenarios": SCENARIOS, "route": args.route, "policy_family": "path_sac",
        "model": None if args.pid_only else str(model),
        "model_sha256": None if args.pid_only else hashlib.sha256(model.read_bytes()).hexdigest(),
        "contract": cfg.policy_contract(), "completed": False,
        "sampling_hz": cfg.RL_RATE_HZ,
    }
    # Snapshot experiment code as well as the commit (new edits may be uncommitted).
    sources = output / "source"
    sources.mkdir()
    for name in ("eval_wind_events.py", "wind_events.py", "path_sac_policy.py", "path_sac_env.py",
                 "path_sac_config.py", "wind.py", "sim_config.py", "hover_task.py", "path_reference.py"):
        (sources / name).write_bytes((ROOT / "drone_code" / name).read_bytes())
    write_json(output / "manifest.json", manifest)

    from isaacsim import SimulationApp
    # The backend opens its own USD. Skip the empty-stage viewport warm-up,
    # which can stall headless startup before the actual simulation is loaded.
    app = SimulationApp({"headless": not args.gui, "create_new_stage": False})
    env = None
    try:
        import numpy as np
        from path_sac_env import make_path_sac_env
        from path_sac_policy import load_path_policy
        env = make_path_sac_env(app, render_sim=args.gui)
        policies = {"pid_only": None}
        if not args.pid_only:
            policies["pid_sac"] = load_path_policy(model)
        flights = []
        for scenario in args.scenarios:
            for seed in args.seeds:
                paired_start = None
                paired_weather = None
                for mode, policy in policies.items():
                    options = {"route_stage": cfg.EVAL_ROUTE_STAGE, "route_type": args.route,
                               "wind_profile": "training_range"}
                    if args.route == "hover":
                        options["hover_seconds"] = args.duration
                    observation, info = env.reset(seed=seed, options=options)
                    reference = env.core.reference
                    event_s = args.event_s if args.route == "hover" else trajectory_event_time(
                        reference, args.duration, args.event_s)
                    event_speed = float(np.linalg.norm(reference.sample(event_s).velocity))
                    write_json(output / f"reference_{seed}.json", {
                        "path": reference.path.tolist(), "times": reference.times.tolist(),
                        "velocities": reference.velocities.tolist(),
                        "motion_end_s": float(reference.motion_end),
                        "route_duration_s": float(reference.duration), "event_s": event_s,
                        "reference_speed_at_event_m_s": event_speed,
                    })
                    env.max_decisions = int(round(args.duration * cfg.RL_RATE_HZ))
                    heading = float(np.arctan2(info["wind_mean"][1], info["wind_mean"][0]))
                    event = WindEvent(scenario, heading, event_s)
                    env.core.wind_event = event
                    # Do not reset gusts here: they retain their matched seed and state.
                    env.sim.wind.mean_wind = event.at(0)
                    start = np.concatenate((observation, reference.path.flatten(), reference.times.flatten(),
                                            [info["wind_seed"], event_s]))
                    if paired_start is None:
                        paired_start = start.copy()
                    elif not np.array_equal(start, paired_start):
                        raise RuntimeError("PID/SAC initial observations, reference or wind seed differ")
                    rows, weather = [], []
                    reason = None
                    trajectory = output / f"{scenario}_{seed}_{mode}.csv"
                    with trajectory.open("w", newline="", encoding="utf-8") as handle:
                        writer = None
                        while True:
                            action = np.zeros(4, dtype=np.float32) if policy is None else policy(observation)
                            observation, _, terminated, truncated, step_info = env.step(action)
                            time = float(env.core.time)
                            error = float(step_info["position_error"])
                            point_winds = env.sim.wind.point_winds.copy()
                            weather.append(point_winds)
                            row = {"time_s": time, "error_m": error if np.isfinite(error) else None,
                                   "residual_norm": float(step_info["residual_norm"]),
                                   "saturated_steps": int(step_info["saturated_steps"])}
                            sample = env.core.sample()
                            truth = env.sim.read_drone_state()
                            row["reference_speed_m_s"] = float(np.linalg.norm(sample.velocity))
                            for axis, target, position in zip("xyz", sample.position, truth["position"]):
                                row[f"target_{axis}_m"] = float(target)
                                row[f"position_{axis}_m"] = float(position) if np.isfinite(position) else None
                            for axis, value in zip("xyz", env.sim.wind.mean_wind):
                                row[f"mean_wind_{axis}_m_s"] = float(value)
                            for index, value in enumerate(step_info["residual"]):
                                row[f"residual_{index}"] = float(value)
                            if writer is None:
                                writer = csv.DictWriter(handle, fieldnames=list(row))
                                writer.writeheader()
                            writer.writerow(row)
                            rows.append((time, error))
                            reason = step_info["failure_reason"]
                            if terminated or truncated:
                                break
                    weather = np.asarray(weather)
                    if paired_weather is None:
                        paired_weather = weather
                    else:
                        n = min(len(weather), len(paired_weather))
                        if not np.array_equal(weather[:n], paired_weather[:n]):
                            raise RuntimeError("PID/SAC airflow histories differ on their common interval")
                    times, errors = zip(*rows)
                    metrics = event_metrics(times, errors, event_s, failed=reason is not None)
                    flights.append({"scenario": scenario, "seed": seed, "controller": mode,
                                    "wind_seed": int(info["wind_seed"]), "heading_rad": heading,
                                    "failure_reason": reason, "event_s": event_s, "route": args.route,
                                    "reference_speed_at_event_m_s": event_speed,
                                    "route_duration_s": float(reference.duration), **metrics})
                    write_json(output / "flights.json", flights)
                    print(f"{scenario} seed={seed} {mode}: {metrics}", flush=True)
        manifest["completed"] = True
        write_json(output / "manifest.json", manifest)
        print(f"Saved task-2 results: {output}", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
