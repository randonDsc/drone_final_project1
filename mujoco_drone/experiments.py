"""Fixed-horizon flight runs and paired-seed scenario suites."""
from dataclasses import replace
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from functools import cache

import numpy as np

from .env import DroneEnv
from .metrics import FlightMetrics, aggregate
from .recording import FlightRecorder
from .sensors import CameraCapture
from drone_code import path_sac_config


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def versions():
    result = dict(python=platform.python_version(), platform=platform.platform(), machine=platform.machine())
    for package in ("mujoco", "numpy", "simple-pid", "gymnasium", "stable-baselines3", "torch"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


@cache
def source_state():
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=root, text=True, capture_output=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, branch = None, None
    files = list((root/"mujoco_drone").glob("*.py")) + list((root/"drone_code").glob("*.py"))
    return dict(commit=commit, branch=branch,
                sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)})


def write_csv(path, rows):
    if not rows:
        raise ValueError("No experiment rows to save")
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config, output, model=None, viewer=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = DroneEnv(config)
    recorder = FlightRecorder(output/"flight.csv")
    metrics = FlightMetrics()
    camera = None
    window = None
    try:
        observation, initial = env.reset(seed=config.seed)
        policy = None
        if model is not None:
            from .training import load_policy
            policy = load_policy(model, env)
        if viewer:
            import mujoco.viewer
            window = mujoco.viewer.launch_passive(env.sim.model, env.sim.data)
        if config.sensors.capture_camera:
            camera = CameraCapture(env.sim, config.sensors, output/"camera")
            camera.capture()
        protocol = dict(task=config.to_dict()["task"], wind=config.to_dict()["wind"],
                        timestep_s=config.physics.timestep, max_seconds=config.max_seconds,
                        controller_settings=config.to_dict()["physics"],
                        initial_state=initial["initial_state"], wind_seed=initial["wind_seed"],
                        completion=dict(path_sac_config.COMPLETION, require_final_dwell=True),
                        failure=path_sac_config.FAILURE,
                        metric_version=1, reference="analytic_quintic_v1")
        # Seeded starts and weather are recorded; the protocol is policy-independent.
        metadata = dict(schema_version=1, simulator="mujoco", mode="pid_sac" if model else "pid",
                        model_checkpoint=str(Path(model).resolve()) if model else None,
                        created_utc=datetime.now(timezone.utc).isoformat(), config=config.to_dict(),
                        protocol=protocol, protocol_hash=digest(protocol), initial=initial,
                        model_audit=env.sim.audit(), versions=versions(), source=source_state(),
                        state_source="ground_truth_body_origin",
                        sensor_model="ideal MuJoCo IMU; accelerometer reports site specific force",
                        residual_limits=env.core.limits.tolist(),
                        residual_scale=env.core.residual_scale,
                        notes="No motor speed/RPM inference; controls and actuator forces are in newtons.")
        json_write(output/"metadata.json", metadata)
        np.savez_compressed(output/"reference.npz", path=env.core.reference.path,
                            times=np.linspace(0, env.core.reference.duration, 1001),
                            positions=np.array([env.core.reference.sample(t).position for t in np.linspace(0, env.core.reference.duration, 1001)]))

        def on_step(active, state, sample):
            recorder.record(active, state, sample)
            metrics.record(active, state, sample)
            if camera:
                camera.capture()

        env.on_step = on_step
        while True:
            action = policy.predict(observation, deterministic=True)[0] if policy else np.zeros(4, dtype=np.float32)
            observation, _, terminated, truncated, _ = env.step(action)
            if window:
                if not window.is_running():
                    raise RuntimeError("Viewer closed before the scheduled experiment ended")
                window.sync()
            if terminated or truncated:
                break
        result = dict(simulator="mujoco", mode="pid_sac" if model else "pid",
                      controller_family=config.physics.controller, seed=config.seed,
                      task=config.task.kind, wind=config.wind.scenario,
                      protocol_hash=metadata["protocol_hash"], **metrics.summarize(env))
        json_write(output/"metrics.json", result)
        return result
    except Exception as error:
        json_write(output/"error.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if window:
            window.close()
        if camera:
            camera.close()
        recorder.close()
        env.close()


def run_suite(config, output, tasks, winds, seeds, model=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    json_write(output/"suite.json", dict(config=config.to_dict(), tasks=tasks, winds=winds, seeds=seeds, model=str(model) if model else None))
    rows = []
    summaries = []
    for task in tasks:
        for wind in winds:
            group = []
            for seed in seeds:
                trial = replace(config, seed=seed, task=replace(config.task, kind=task), wind=replace(config.wind, scenario=wind))
                modes = [("pid", None)] + ([("pid_sac", model)] if model else [])
                for mode, checkpoint in modes:
                    row = run_experiment(trial, output/f"{task}_{wind}_{seed}_{mode}", model=checkpoint)
                    rows.append(row)
                    group.append(row)
                    write_csv(output/"flights.csv", rows)
                    print(f"{task}/{wind}/seed={seed}/{mode}: RMSE={row['position_rmse']:.4f} m, completed={row['completed']}, failed={row['failed']}", flush=True)
            for mode in sorted({row["mode"] for row in group}):
                summaries.append(dict(task=task, wind=wind, mode=mode,
                                      **aggregate([r for r in group if r["mode"] == mode])))
            write_csv(output/"summary.csv", summaries)
    json_write(output/"aggregate.json", aggregate(rows))
    if model:
        from .comparison import compare
        for mode in ("pid", "pid_sac"):
            write_csv(output/(mode+".csv"), [r for r in rows if r["mode"] == mode])
        compare(output/"pid.csv", output/"pid_sac.csv", output/"paired")
    return rows
