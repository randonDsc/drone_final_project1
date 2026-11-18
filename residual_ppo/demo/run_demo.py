r"""M9 Steps 4-5: fly PID, PID + PPO residual and PPO direct on the demo tracks. DESCRIPTIVE DEMO, NOT eval-v1.

    # Directory: <repo root>\residual_ppo
    & <isaac>\python.bat demo\run_demo.py --scene indoor                     # 27 flights
    & <isaac>\python.bat demo\run_demo.py --scene forest                     # 27 flights
    & <isaac>\python.bat demo\run_demo.py --scene indoor --only I1_aisle:calm --controllers ppo_direct --tag sanity

Order: checks + import guard -> contract proof (both directions) -> torch imported first -> all models loaded ->
SimulationApp -> torch pinned -> demo sim (obstacles active) -> envs (residual: DemoResidualEnv for pid and pid_ppo;
direct: DemoDirectEnv for ppo_direct) -> geometry re-exported and compared with results/m9_demo/geometry_<scene>.json
-> flights. Per case (scene x track x wind) the three controllers fly the SAME realization: the paired-flight
assertion compares route duration, max decisions, wind (seed, mean, gust scale), the drawn start state and the
simulator state right after reset (all bit-exact), and the initial observation between pid and pid_ppo (same env
type). Any mismatch raises PairingMismatch and stops the run (RESULT: FAIL).

Outputs (results/m9_demo/<tag or "flights">/): flights_<scene>.json (meta, rows, pairing, sha256 of every series
file) and .csv, and one compressed series per flight: <case>__<controller>.npz (60 Hz recorded state + reference,
per-decision positions / errors / actions). RESULT: DONE only if every flight ran and every assertion held.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR / "evaluation"))   # ppo_setup imports run_common by name
sys.path.insert(0, str(RESIDUAL_DIR))

import numpy as np  # noqa: E402

from evaluation import run_common  # noqa: E402

MODELS = {
    "pid_ppo": "runs/ppo_path/ppo_s0_1p5m/final_model.zip",
    "ppo_direct": "runs/ppo_direct/ppo_direct_s0_1p5m/final_model.zip",
}
CONTROLLER_ORDER = ("pid", "pid_ppo", "ppo_direct")
ARM_OF = {"pid": "residual", "pid_ppo": "residual", "ppo_direct": "direct"}
WIND_ORDER = ("calm", "w3p5", "w7")
SCENE_CASE_OFFSET = {"indoor": 0, "forest": 9}
CASE_SEED_BASE = 600000


class PairingMismatch(RuntimeError):
    """The three flights of one case did not start from the same realization."""


def say(msg=""):
    print(f"[M9.fly] {msg}", flush=True)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def contract_proof():
    """Both directions, before any model is loaded: each adapter must refuse the other arm's final model."""
    from evaluation import controllers as frozen
    from demo.direct_controller import DirectPpoController

    out = {}
    for label, cls, model in (("direct adapter refuses ppo_s0_1p5m", DirectPpoController, MODELS["pid_ppo"]),
                              ("pid_ppo adapter refuses ppo_direct_s0_1p5m", frozen.PidPpoController,
                               MODELS["ppo_direct"])):
        try:
            cls(model=RESIDUAL_DIR / model)
            out[label] = {"refused": False, "message": None}
        except frozen.ControllerRefused as exc:
            out[label] = {"refused": True, "message": str(exc)}
        say(f"{'PASS' if out[label]['refused'] else 'FAIL'}  contract proof: {label}: {out[label]['message']}")
    return out


def load_controllers(names):
    from evaluation import controllers as frozen
    from demo.direct_controller import DirectPpoController

    made = {}
    for name in names:
        if name == "pid":
            made[name] = frozen.PidController()
        elif name == "pid_ppo":
            made[name] = frozen.PidPpoController(model=RESIDUAL_DIR / MODELS["pid_ppo"])
        else:
            made[name] = DirectPpoController(model=RESIDUAL_DIR / MODELS["ppo_direct"])
    # Residual model == the committed M6 hash (the direct model has none committed yet; its sha is recorded).
    if "pid_ppo" in made:
        m6 = json.loads((RESIDUAL_DIR / "results/m6/m6_model_hashes.json").read_text(encoding="utf-8"))
        want = {m["model"]: m["sha256"] for m in m6["models"]}.get(f"residual_ppo/{MODELS['pid_ppo']}")
        got = hashlib.sha256((RESIDUAL_DIR / MODELS["pid_ppo"]).read_bytes()).hexdigest()
        if want != got:
            raise SystemExit(f"ppo_s0_1p5m final_model.zip sha256 {got} != committed M6 hash {want}")
        say(f"pid_ppo model sha256 == results/m6/m6_model_hashes.json ({got[:12]})")
    # Deterministic: the same observation gives the same action twice.
    probe = np.linspace(-1, 1, 128).astype(np.float32)
    for name, c in made.items():
        a1, a2 = c.act(probe), c.act(probe)
        if not np.array_equal(a1, a2):
            raise SystemExit(f"{name}: predict(deterministic=True) not repeatable")
    return made


def state_vector(state):
    return np.concatenate([np.asarray(state[k], dtype=np.float64).reshape(-1)
                           for k in ("position", "orientation", "linear_velocity", "angular_velocity")])


def fly(env, controller, seed, options, obstacles, half_size):
    observation, info = env.reset(seed=seed, options=options)
    pairing = {
        "route_seconds": float(info["route_seconds"]), "max_decisions": int(env.max_decisions),
        "wind_seed": int(info["wind_seed"]), "wind_mean": [float(v) for v in info["wind_mean"]],
        "gust_scale": float(info["gust_scale"]), "initial_state": info["initial_state"],
        "reset_state_sha256": sha256_bytes(state_vector(env.sim.read_drone_state()).tobytes()),
        "initial_obs_sha256": sha256_bytes(np.ascontiguousarray(observation).tobytes()),
    }
    env.core.start_recording()
    actions, reason = [], None
    t0 = time.perf_counter()
    while True:
        action = controller.act(observation)
        if np.shape(action) != (4,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"{controller.name}: bad action {action}")
        actions.append(np.asarray(action, dtype=np.float32))
        observation, _, terminated, truncated, step_info = env.step(action)
        reason = step_info["failure_reason"]
        if terminated or truncated:
            break
    wall = time.perf_counter() - t0
    rec = env.core.stop_recording()
    metrics = env.trace.metrics(env.core.reference.path, reason, reason == "too_low", env.core.completion)
    pos = rec[:, 1:4]
    ok = np.all(np.isfinite(pos), axis=1)
    hit = np.full(len(pos), -1)
    hit[ok] = obstacles.contacts(pos[ok], half_size)
    clear = np.full(len(pos), np.nan)
    clear[ok] = obstacles.clearance(pos[ok])[0]
    first = int(np.argmax(hit >= 0)) if np.any(hit >= 0) else None
    row = {
        **{k: metrics[k] for k in ("decisions", "failed", "failure_reason", "completed", "completion_time",
                                   "position_rmse", "cross_track_rmse", "cross_track_max", "max_deviation",
                                   "saturation_rate")},
        "action_norm_mean": metrics["residual_norm"],
        "ground_contact_too_low": metrics["collided"],
        "obstacle_contact": first is not None,
        "contact_samples_60hz": int(np.sum(hit >= 0)),
        "first_contact_t": None if first is None else float(rec[first, 0]),
        "first_contact_obstacle": None if first is None else obstacles.paths[hit[first]],
        "min_clearance_m": float(np.nanmin(clear)) if np.any(ok) else None,
        "flown_seconds": float(rec[-1, 0]),
        "env_step_wall_s": wall,
    }
    series = {"rec60": rec, "decision_positions": np.array(env.trace.positions, dtype=np.float64),
              "decision_errors": np.array(env.trace.errors, dtype=np.float64),
              "actions": np.array(actions, dtype=np.float32), "reference_path": env.core.reference.path}
    return row, pairing, series


def assert_paired(case_id, pairings):
    """Stop the run on any mismatch between the controllers' realizations of one case."""
    names = list(pairings)
    base = pairings[names[0]]
    problems = []
    for n in names[1:]:
        for k in ("route_seconds", "max_decisions", "wind_seed", "wind_mean", "gust_scale", "initial_state",
                  "reset_state_sha256"):
            if pairings[n][k] != base[k]:
                problems.append(f"{k}: {names[0]} {base[k]!r} != {n} {pairings[n][k]!r}")
    same_env = [n for n in names if ARM_OF[n] == "residual"]
    if len(same_env) == 2 and pairings[same_env[0]]["initial_obs_sha256"] != pairings[same_env[1]]["initial_obs_sha256"]:
        problems.append("initial observation differs between pid and pid_ppo (same env type)")
    if problems:
        raise PairingMismatch(f"{case_id}: " + "; ".join(problems))
    return {"controllers": names, "checked": ["route_seconds", "max_decisions", "wind_seed", "wind_mean",
                                              "gust_scale", "initial_state", "reset_state_sha256"]
            + (["initial_obs_sha256 (pid vs pid_ppo)"] if len(same_env) == 2 else []), "pass": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=["indoor", "forest"], required=True)
    ap.add_argument("--only", default=None, help="TRACK:WIND, e.g. I1_aisle:calm")
    ap.add_argument("--controllers", default=",".join(CONTROLLER_ORDER))
    ap.add_argument("--tag", default="flights")
    args = ap.parse_args()
    names = [c for c in CONTROLLER_ORDER if c in args.controllers.split(",")]
    out_dir = RESIDUAL_DIR / "results" / "m9_demo" / args.tag
    series_dir = out_dir / args.scene
    if series_dir.exists() and any(series_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite {series_dir.relative_to(RESIDUAL_DIR).as_posix()} (not empty)")
    series_dir.mkdir(parents=True, exist_ok=True)

    say(f"scene {args.scene}, controllers {names}, only {args.only}; descriptive demo, not eval-v1")
    checks_line = run_common.preflight(say)
    proof = contract_proof()
    if not all(v["refused"] for v in proof.values()):
        raise SystemExit("contract proof FAILED")

    from training import ppo_setup
    ppo_setup.import_torch_first()
    controllers = load_controllers(names)          # BEFORE SimulationApp
    say(f"loaded {list(controllers)} before SimulationApp")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        isaac_main(app, args, names, controllers, proof, checks_line, out_dir, series_dir, ppo_setup)
    finally:
        app.close()


def isaac_main(app, args, names, controllers, proof, checks_line, out_dir, series_dir, ppo_setup):
    pin = ppo_setup.pin_torch_imports()
    origins_start = ppo_setup.torch_origin_report()

    from demo import demo_config, tracks
    from demo.demo_env import RECORD_COLUMNS, DemoDirectEnv, DemoResidualEnv, make_demo_sim
    from demo.geometry import Obstacles
    from demo.scene_geometry import export_obstacles

    demo_config.register_demo_profiles()
    sim = make_demo_sim(app, args.scene, log=say)
    if run_common.loaded_outside_residual():
        raise SystemExit(f"shared modules outside residual_ppo: {run_common.loaded_outside_residual()}")
    envs = {"residual": DemoResidualEnv(sim, log=say), "direct": DemoDirectEnv(sim, log=say)}
    for name, c in controllers.items():
        env = envs[ARM_OF[name]]
        if ARM_OF[name] == "residual" and (type(env) is not DemoResidualEnv):
            raise SystemExit(f"{name} must fly DemoResidualEnv")
        c.bind_env(env)
    say("controllers bound: pid/pid_ppo -> DemoResidualEnv, ppo_direct -> DemoDirectEnv (spaces + env type)")

    geo_path = RESIDUAL_DIR / "results" / "m9_demo" / f"geometry_{args.scene}.json"
    saved = json.loads(geo_path.read_text(encoding="utf-8"))
    live = json.loads(json.dumps(export_obstacles(sim.stage, demo_config.SCENES[args.scene]["obstacle_root"])))
    if live != saved["shapes"]:
        raise SystemExit(f"live obstacle geometry differs from {geo_path.name}")
    obstacles = Obstacles(saved["shapes"])
    checked = tracks.check_tracks(args.scene, obstacles)
    if not all(t["clearance_ok"] and not t["in_distribution_flags"] for t in checked.values()):
        raise SystemExit("a track fails its clearance or in-distribution check")
    say(f"geometry re-exported == {geo_path.name}; tracks checked")

    rows, pairing_records = [], {}
    only = tuple(args.only.split(":")) if args.only else None
    track_names = list(tracks.TRACKS[args.scene])
    for ti, track_name in enumerate(track_names):
        reference = tracks.build_reference(tracks.TRACKS[args.scene][track_name])
        wind_dir = checked[track_name]["wind_direction_xy"]
        for wi, level in enumerate(WIND_ORDER):
            if only and (track_name, level) != only:
                continue
            case_index = SCENE_CASE_OFFSET[args.scene] + 3 * ti + wi
            seed = demo_config.assert_demo_seed(CASE_SEED_BASE + case_index)
            case_id = f"{args.scene}_{track_name}_{level}"
            options = {"reference": reference, "wind": demo_config.demo_wind(level, wind_dir), "track_name": track_name}
            pairings = {}
            for name in names:
                row, pairing, series = fly(envs[ARM_OF[name]], controllers[name], seed, options, obstacles,
                                           demo_config.COLLISION_HALF_SIZE)
                pairings[name] = pairing
                fname = f"{case_id}__{name}.npz"
                fpath = series_dir / fname
                np.savez_compressed(fpath, columns=np.array(RECORD_COLUMNS), **series)
                row.update(scene=args.scene, track=track_name, wind=level,
                           wind_speed=float(demo_config.WIND_LEVELS[level]["speed"]),
                           wind_direction_xy=list(map(float, wind_dir)), case_seed=seed, controller=name,
                           route_seconds=pairing["route_seconds"],
                           series_file=f"{series_dir.relative_to(RESIDUAL_DIR).as_posix()}/{fname}",
                           series_sha256=hashlib.sha256(fpath.read_bytes()).hexdigest())
                rows.append(row)
                say(f"{case_id:30s} {name:10s} rmse {row['position_rmse']:.3f} m, cross {row['cross_track_rmse']:.3f}, "
                    f"maxdev {row['max_deviation']:.3f}, end {row['failure_reason'] or 'ok'}, "
                    f"completed {row['completed']}, contact {row['obstacle_contact']} "
                    f"(min clear {row['min_clearance_m']:.2f} m), sat {row['saturation_rate']:.3f}, "
                    f"{row['decisions']} dec in {row['env_step_wall_s']:.1f} s")
            if len(names) > 1:
                pairing_records[case_id] = {"seed": seed, **assert_paired(case_id, pairings),
                                            "values": pairings[names[0]]}
                say(f"{case_id}: paired-flight assertion PASS ({', '.join(names)})")

    origins_end = ppo_setup.torch_origin_report()
    clean = origins_end["outside_package_dir"] == 0 and origins_end["modules_from_ml_archive_pip_prebundle"] == 0
    meta = {
        "label": demo_config.LABEL, "note": demo_config.NO_PERCEPTION_NOTE,
        "wind_label": demo_config.SCENES[args.scene]["wind_label"],
        "wind_direction_rule": "crosswind toward obstacles (worst side): perpendicular to the main leg, toward the "
                               "nearer obstacle along that perpendicular",
        "scene": args.scene, "usd": demo_config.SCENES[args.scene]["usd"], "checks_line": checks_line,
        "git": run_common.git_info(), "isaac_sim": run_common.isaac_version(), "hardware": run_common.hardware(),
        "versions": ppo_setup.versions(), "torch_import_pin": pin, "torch_origins_start": origins_start,
        "torch_origins_end": origins_end, "torch_origin_clean": clean, "total_mass_kg": sim.total_mass,
        "physics_dt": sim.dt, "decimation": envs["residual"].core.decimation,
        "controllers": {n: c.record() for n, c in controllers.items()}, "contract_proof": proof,
        "track_limits": demo_config.TRACK_LIMITS, "wind_levels": demo_config.WIND_LEVELS,
        "collision_half_size": demo_config.COLLISION_HALF_SIZE.tolist(),
        "geometry_file": geo_path.name, "geometry_sha256": run_common.file_sha256(geo_path),
        "tracks": checked, "pairing": pairing_records, "record_columns": list(RECORD_COLUMNS), "record_hz": 60,
    }
    record = {"meta": meta, "rows": rows}
    jpath = out_dir / f"flights_{args.scene}.json"
    jpath.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    cols = ["scene", "track", "wind", "wind_speed", "case_seed", "controller", "decisions", "failed", "failure_reason",
            "completed", "completion_time", "position_rmse", "cross_track_rmse", "cross_track_max", "max_deviation",
            "saturation_rate", "action_norm_mean", "obstacle_contact", "contact_samples_60hz", "first_contact_t",
            "first_contact_obstacle", "min_clearance_m", "ground_contact_too_low", "route_seconds", "flown_seconds",
            "env_step_wall_s", "series_file", "series_sha256"]
    with open(out_dir / f"flights_{args.scene}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    say(f"written: {jpath.relative_to(RESIDUAL_DIR).as_posix()} (+ .csv, {len(rows)} series files)")
    say(f"torch origin clean at end: {clean}")
    if not clean:
        say("RESULT: FAIL (torch origin)")
        return
    say("RESULT: DONE")


if __name__ == "__main__":
    main()
