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
import time
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
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config, output, model=None, viewer=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = None
    recorder = None
    metrics = FlightMetrics()
    camera = None
    window = None
    try:
        env = DroneEnv(config)
        recorder = FlightRecorder(output/"flight.csv")
        observation, initial = env.reset(seed=config.seed)
        policy = None
        if model is not None:
            from .training import load_policy
            policy = load_policy(model, env)
        if viewer:
            import mujoco.viewer
            window = mujoco.viewer.launch_passive(env.sim.model, env.sim.data)
            window.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            window.cam.trackbodyid = env.sim.body_id
            window.cam.distance = 5 if env.navigation else 3
            window.cam.elevation = -25
            wall_start = time.monotonic()
        if config.sensors.capture_camera:
            camera = CameraCapture(env.sim, config.sensors, output/"camera")
            camera.capture()
        scene = dict(name=config.scene.name,
                     manifest_sha256=env.sim.scene_manifest["sha256"] if env.sim.scene_manifest else None)
        reference_kind = (config.navigation.reference_mode if env.navigation else
                          "legacy_isaac_timed" if config.protocol == "isaac_path" else "analytic_quintic_v1")
        limits = dict(path_sac_config.FAILURE)
        if env.navigation:
            limits["max_position_error"] = config.navigation.max_position_error
        protocol = dict(task=config.to_dict()["task"], wind=config.to_dict()["wind"],
                        timestep_s=config.physics.timestep, max_seconds=config.max_seconds,
                        controller_settings=config.to_dict()["physics"],
                        initial_state=initial["initial_state"], wind_seed=initial["wind_seed"],
                        completion=dict(path_sac_config.COMPLETION, require_final_dwell=True),
                        failure=limits, experiment_protocol=config.protocol,
                        scene=scene, navigation=config.to_dict()["navigation"] if env.navigation else None,
                        reference_path_sha256=hashlib.sha256(np.ascontiguousarray(env.core.reference.path).tobytes()).hexdigest(),
                        metric_version=1, reference=reference_kind)
        # Seeded starts and weather are recorded; the protocol is policy-independent.
        metadata = dict(schema_version=2, simulator="mujoco", mode="pid_sac" if model else "pid",
                        model_checkpoint=str(Path(model).resolve()) if model else None,
                        created_utc=datetime.now(timezone.utc).isoformat(), config=config.to_dict(),
                        protocol=protocol, protocol_hash=digest(protocol), initial=initial,
                        model_audit=env.sim.audit(), versions=versions(), source=source_state(),
                        state_source="ground_truth_body_origin",
                        sensor_model="ideal MuJoCo IMU; accelerometer reports site specific force",
                        residual_limits=env.core.limits.tolist(),
                        residual_scale=env.core.residual_scale,
                        notes="No motor speed/RPM inference; controls and actuator forces are in newtons.")
        metadata["scene"] = dict(scene, source=env.sim.scene_manifest["data"].get("source") if env.sim.scene_manifest else None)
        if env.navigation:
            metadata["navigation"] = env.navigation.info()
            env.navigation.save(output/"navigation")
        json_write(output/"metadata.json", metadata)
        reference_times = np.linspace(0, env.core.reference.duration, 1001)
        if env.navigation and reference_kind == "isaac_waypoints":
            reference_times = np.array([])  # Executed feedback targets are in flight.csv.
        np.savez_compressed(output/"reference.npz", path=env.core.reference.path, times=reference_times,
                            positions=np.array([env.core.reference.sample(t).position for t in reference_times]).reshape(-1, 3))

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
                time.sleep(max(0, env.core.time-(time.monotonic()-wall_start)))
            if terminated or truncated:
                break
        result = dict(simulator="mujoco", mode="pid_sac" if model else "pid",
                      controller_family=config.physics.controller, seed=config.seed,
                      task=config.task.kind, wind=config.wind.scenario,
                      protocol_hash=metadata["protocol_hash"], **metrics.summarize(env))
        json_write(output/"metrics.json", result)
        if env.navigation:
            json_write(output/"navigation"/"final_navigation.json", env.navigation.info())
        return result
    except Exception as error:
        json_write(output/"error.json", {"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        if window:
            window.close()
        if camera:
            camera.close()
        if recorder:
            recorder.close()
        if env:
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


def plan_experiment(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    env = None
    try:
        env = DroneEnv(config)
        _, initial = env.reset(seed=config.seed)
        if env.navigation is None:
            raise ValueError("Plan-only execution requires the dijkstra task")
        env.navigation.save(output/"navigation")
        json_write(output/"metadata.json", dict(schema_version=2, config=config.to_dict(), initial=initial,
                  model_audit=env.sim.audit(), scene=env.sim.scene_manifest,
                  source=source_state(), versions=versions(), execution="planning only; no flight results"))
        return env.navigation.info()
    except Exception as error:
        json_write(output/"error.json", dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        if env:
            env.close()
