"""M2.2 zero residual equals PID (docs/milestone2_plan.md, M2.2 and D6). NO training.

Two controllers on identical seeds, compared at every physics step:

  A  env path:      PathSacEnv.reset(seed, options), then env.step(zeros) until it ends (the team's "PID only").
  B  pure PID path: the same env.reset (with core.observe stubbed on the instance during the reset, so the PID is
                    never read with dt=0), then our own loop that mirrors PathSacEnv.step + PathCore.step line by
                    line, WITHOUT "+ correction" and WITHOUT observe().

Recorded per physics step by wrapping env.sim.step_command on the instance (shared files untouched): the pre-step
state the PID saw (position, orientation, linear/angular velocity), the PID integrators after integrate(), the wrench
sent to the mixer, the rotor speeds the mixer returned, the saturation flag, the wind wrench and the mean wind.

Modes (run in this order; D6):
  floor --tag p1 / floor --tag p2   Isaac. A twice per cell (A,A). Two separate processes. Raw traces -> traces dir.
  tolerance                         No Isaac. Noise floor from the four A runs -> results/m2/zero_residual_tolerance.json
  compare                           Isaac. Refuses without the tolerance file. A vs B per cell (A,B), then guard tests.
  supplementary                     Isaac. Added after the pre-registered run, same tolerance: A vs B on outside_range
                                    (failure termination) + roll/yaw torque guards -> zero_residual_supplementary.json

    # Directory: C:\\Users\\tucla\\Desktop\\DSC180A\\drone_final_project1
    & "<Isaac>\\python.bat" residual_ppo\\evaluation\\zero_residual_check.py floor --tag p1 *> residual_ppo\\results\\m2\\zero_residual_floor_p1.log
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
OUT_DIR = RESIDUAL_DIR / "results" / "m2"
TRACE_DIR = OUT_DIR / "zero_residual_traces"          # raw traces, gitignored
TOLERANCE_PATH = OUT_DIR / "zero_residual_tolerance.json"
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))

EXPECTED_PROTOCOL_SHA = "18658b9ce0801677369f6ee16d5d49467965fe96888a091d6d7762fd7376d0dc"
SEEDS = (300_000, 300_001, 300_002)
PROFILES = ("still_air", "training_range")
SHARED = [
    "path_sac_env", "path_sac_policy", "path_sac_config", "path_reference",
    "hover_env", "hover_task", "hover_ppo_config", "sensor_config",
    "state_estimation", "model_contract", "controller", "sim_config",
    "drone_sim", "motor", "wind", "drone_diagnostics",
    "hover_sac_policy", "hover_sac_config",
]

# Per-physics-step row layout (float64).
LAYOUT = [
    ("position", 3), ("orientation", 4), ("linear_velocity", 3), ("angular_velocity", 3),
    ("pid_integrators", 6),   # height, xy[2], attitude[3]  (after integrate)
    ("command", 4),           # wrench sent to the mixer: thrust, roll, pitch, yaw torque
    ("rotor_speeds", 4),      # final rotor commands returned by the mixer [rad/s]
    ("saturated", 1), ("wind_force", 3), ("wind_torque", 3), ("mean_wind", 3),
]
SLICES, _c = {}, 0
for _name, _w in LAYOUT:
    SLICES[_name] = slice(_c, _c + _w)
    _c += _w
ROW_WIDTH = _c
QUANTITIES = [n for n, _ in LAYOUT] + ["final_state"]
NAMED = ("position", "linear_velocity", "orientation", "angular_velocity", "pid_integrators", "rotor_speeds")
EPISODE_FIELDS = ("physics_steps", "decisions", "terminated", "truncated", "failure_reason", "completed",
                  "completion_time", "route_seconds", "wind_seed", "wind_mean", "max_decisions")

# Guard tests: a known tiny residual on thrust at decision 0 only (normalized action units).
GUARD_MAGNITUDES = (1e-6, 1e-3)
GUARD_CELLS = (("straight", "still_air", 300_000), ("turns", "training_range", 300_000))

# Supplementary run (added after the pre-registered run, Claire 2026-11-13; same bit-exact tolerance):
# outside_range exercises the failure-termination branch; guards on roll and yaw torque.
SUPP_PROFILE = "outside_range"
SUPP_MAGNITUDE = 1e-6
SUPP_GUARDS = (
    (("straight", "still_air", 300_000), 1, "roll_torque"),
    (("turns", "training_range", 300_000), 3, "yaw_torque"),
)
SUPP_PATH = OUT_DIR / "zero_residual_supplementary.json"
SUPP_SUMMARY = OUT_DIR / "zero_residual_supplementary_summary.txt"
ISAAC_ROOT_TOKEN = "<ISAAC_ROOT>"

lines = []


def say(msg=""):
    line = f"[M2.2] {msg}"
    print(line, flush=True)
    lines.append(line)


def cell_id(scenario, profile, seed):
    return f"{scenario}__{profile}__{seed}"


def all_cells(scenarios):
    return [(s, p, seed) for s in scenarios for p in PROFILES for seed in SEEDS]


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def trace_hash(episodes, cells):
    h = hashlib.sha256()
    for c in cells:
        ep = episodes[cell_id(*c)]
        h.update(np.ascontiguousarray(ep["rows"], dtype=np.float64).tobytes())
        h.update(np.ascontiguousarray(ep["final_state"], dtype=np.float64).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- comparison (NumPy only)

def max_abs_diff(x, y):
    """Max |x - y|; positions where both are equal (incl. both NaN / same inf) count 0, one-sided NaN = inf."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0:
        return 0.0
    equal = (x == y) | (np.isnan(x) & np.isnan(y))
    with np.errstate(invalid="ignore"):
        d = np.abs(x - y)
    d[equal] = 0.0
    d[np.isnan(d)] = np.inf
    return float(d.max())


def first_diff_step(ra, rb, tol=None):
    """First row index where any quantity differs (exactly, or by more than tol[q] if given)."""
    n = min(len(ra), len(rb))
    if n == 0:
        return None
    bad = np.zeros(n, dtype=bool)
    for q, sl in SLICES.items():
        x, y = ra[:n, sl], rb[:n, sl]
        equal = (x == y) | (np.isnan(x) & np.isnan(y))
        with np.errstate(invalid="ignore"):
            d = np.where(equal, 0.0, np.abs(x - y))
        d = np.where(np.isnan(d), np.inf, d)
        limit = 0.0 if tol is None else tol[q]
        bad |= np.any(d > limit, axis=1)
    idx = np.flatnonzero(bad)
    if idx.size:
        return int(idx[0])
    return None if len(ra) == len(rb) else n


def compare_episodes(a, b, tol=None):
    """Compare two recorded episodes. tol: {quantity: max allowed |diff|} (0 = exact); None = just report."""
    ra, rb = a["rows"], b["rows"]
    n = min(len(ra), len(rb))
    diffs = {q: max_abs_diff(ra[:n, sl], rb[:n, sl]) for q, sl in SLICES.items()}
    diffs["final_state"] = max_abs_diff(a["final_state"], b["final_state"])
    fields = {f: (a[f] == b[f]) for f in EPISODE_FIELDS}
    out = {
        "physics_steps": [len(ra), len(rb)],
        "same_length": len(ra) == len(rb),
        "max_abs_diff": diffs,
        "first_differing_step": first_diff_step(ra, rb),
        "episode_fields_equal": all(fields.values()),
        "episode_field_mismatches": {f: [a[f], b[f]] for f, ok in fields.items() if not ok},
        "end": [[a["failure_reason"] or ("truncated" if a["truncated"] else "?"), a["decisions"]],
                [b["failure_reason"] or ("truncated" if b["truncated"] else "?"), b["decisions"]]],
        "bitwise_identical": len(ra) == len(rb) and ra.tobytes() == rb.tobytes()
        and np.asarray(a["final_state"]).tobytes() == np.asarray(b["final_state"]).tobytes(),
    }
    if tol is not None:
        exceeded = {q: d for q, d in diffs.items() if not d <= tol[q]}
        out["exceeded"] = exceeded
        out["first_step_exceeding_tolerance"] = first_diff_step(ra, rb, tol)
        out["pass"] = out["same_length"] and out["episode_fields_equal"] and not exceeded
    return out


# ---------------------------------------------------------------- trace files

def save_traces(path, runs, meta):
    """runs: {run_name: {cell_id: episode}}. Arrays -> npz; everything else -> json sidecar."""
    arrays, sidecar = {}, {"meta": meta, "runs": {}}
    for run, episodes in runs.items():
        sidecar["runs"][run] = {}
        for cid, ep in episodes.items():
            arrays[f"{run}__{cid}__rows"] = ep["rows"]
            arrays[f"{run}__{cid}__final"] = np.asarray(ep["final_state"], dtype=np.float64)
            sidecar["runs"][run][cid] = {k: v for k, v in ep.items() if k not in ("rows", "final_state")}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=1, default=str) + "\n", encoding="utf-8")


def load_traces(path):
    sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    data = np.load(path.with_suffix(".npz"))
    runs = {}
    for run, episodes in sidecar["runs"].items():
        runs[run] = {}
        for cid, fields in episodes.items():
            ep = dict(fields)
            ep["rows"] = data[f"{run}__{cid}__rows"]
            ep["final_state"] = data[f"{run}__{cid}__final"]
            runs[run][cid] = ep
    return sidecar["meta"], runs


# ---------------------------------------------------------------- tolerance mode (no Isaac)

def mode_tolerance():
    import eval_protocol_v1 as protocol

    say("mode tolerance: noise floor from floor_p1 / floor_p2 (A twice per cell, two processes)")
    meta1, p1 = load_traces(TRACE_DIR / "floor_p1")
    meta2, p2 = load_traces(TRACE_DIR / "floor_p2")
    cells = all_cells(protocol.SCENARIOS)
    cids = [cell_id(*c) for c in cells]
    for name, runs in (("p1", p1), ("p2", p2)):
        for run in ("r1", "r2"):
            missing = [c for c in cids if c not in runs[run]]
            if missing:
                raise SystemExit(f"floor_{name} {run} is missing cells: {missing}")
    pairs = {
        "within_process_p1": (p1["r1"], p1["r2"]),
        "within_process_p2": (p2["r1"], p2["r2"]),
        "cross_process_run1": (p1["r1"], p2["r1"]),
        "cross_process_run2": (p1["r2"], p2["r2"]),
    }
    floor = {}
    problems = []
    per_pair = {}
    for pname, (ra, rb) in pairs.items():
        pmax = {q: 0.0 for q in QUANTITIES}
        bitwise = 0
        for cid in cids:
            r = compare_episodes(ra[cid], rb[cid])
            bitwise += int(r["bitwise_identical"])
            if not (r["same_length"] and r["episode_fields_equal"]):
                problems.append({"pair": pname, "cell": cid, "lengths": r["physics_steps"],
                                 "mismatches": r["episode_field_mismatches"]})
            for q, d in r["max_abs_diff"].items():
                pmax[q] = max(pmax[q], d)
        per_pair[pname] = {"max_abs_diff": pmax, "cells_bitwise_identical": bitwise, "cells": len(cids)}
        say(f"{pname:<20} bitwise identical cells {bitwise}/{len(cids)}; max |diff| "
            + ", ".join(f"{q} {d:.3g}" for q, d in pmax.items()))
    for q in QUANTITIES:
        floor[q] = max(per_pair[p]["max_abs_diff"][q] for p in pairs)
    bit_exact = all(v == 0.0 for v in floor.values())
    usable = not problems and all(np.isfinite(v) for v in floor.values())
    hashes = {f"p1_{r}": trace_hash(p1[r], cells) for r in ("r1", "r2")}
    hashes.update({f"p2_{r}": trace_hash(p2[r], cells) for r in ("r1", "r2")})
    say(f"trace sha256 (all cells): " + ", ".join(f"{k} {v[:12]}" for k, v in hashes.items()))
    say(f"noise floor per quantity: " + ", ".join(f"{q} {v:.3g}" for q, v in floor.items()))
    if problems:
        say(f"floor runs DISAGREE on episode length/fields in {len(problems)} cases -> tolerance NOT usable")
    rule = ("D6: tolerance = measured noise floor per quantity (max |diff| over the 4 A-vs-A pairs, 36 cells); "
            "bit-exact if the floor is 0. Episode lengths, end reason/step and completion must match exactly. "
            "Never loosened afterwards.")
    tolerance = {q: floor[q] for q in QUANTITIES}
    say(f"TOLERANCE: {'BIT-EXACT (floor is 0 for every quantity)' if bit_exact else 'per-quantity floor'}; "
        f"usable={usable}")
    protocol_rec = protocol.protocol_record()
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "M2.2 run 1: noise floor of the env path (A vs A) and the pre-registered tolerance (D6)",
        "rule": rule,
        "usable": usable,
        "bit_exact": bit_exact,
        "tolerance": tolerance,
        "episode_fields_must_match": list(EPISODE_FIELDS),
        "noise_floor": floor,
        "per_pair": per_pair,
        "floor_problems": problems,
        "trace_sha256": hashes,
        "cells": [list(c) for c in cells],
        "row_layout": LAYOUT,
        "protocol": protocol_rec,
        "protocol_hash_ok": protocol_rec["protocol_file_sha256"] == EXPECTED_PROTOCOL_SHA,
        "isaac_runs": {"p1": meta1, "p2": meta2},
        "tolerance_computed_with": {"python": platform.python_version(), "numpy": np.__version__,
                                    "executable": "Isaac python.bat (no SimulationApp)"
                                    if "isaacsim" in sys.executable.lower() else "other"},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TOLERANCE_PATH.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    say("written: residual_ppo/results/m2/zero_residual_tolerance.json")
    (OUT_DIR / "zero_residual_floor_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- Isaac modes

def load_sync_shared():
    spec = importlib.util.spec_from_file_location("sync_shared", RESIDUAL_DIR / "tools" / "sync_shared.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sync_shared"] = mod
    spec.loader.exec_module(mod)
    return mod


def preflight():
    """Checks (manifest) and import-path guard, before Isaac starts."""
    sync_shared = load_sync_shared()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = sync_shared.check() if sync_shared.team_repo_configured() else sync_shared.check_local()
    check_line = buf.getvalue().strip().splitlines()[-1]
    say(f"checks: {check_line} (rc={rc})")
    if rc != 0:
        print(buf.getvalue())
        raise SystemExit("shared files do not match the manifest; not running")
    root = str(RESIDUAL_DIR.resolve()).lower()
    bad = {}
    for n in SHARED:
        spec = importlib.util.find_spec(n)
        origin = spec.origin if spec else None
        if origin is None or not str(Path(origin).resolve()).lower().startswith(root):
            bad[n] = origin
    if bad:
        raise SystemExit(f"shared modules resolve outside residual_ppo: {bad}")
    say(f"import paths: all {len(SHARED)} shared modules resolve inside residual_ppo")
    return check_line


def git_info():
    def run(*args):
        return subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True, text=True).stdout.strip()
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "residual_ppo_status": run("status", "--short", "--", "residual_ppo").splitlines()}


def isaac_root():
    for d in Path(sys.executable).resolve().parents:
        if (d / "VERSION").is_file() and (d / "python.bat").is_file():
            return d
    return None


def isaac_version():
    root = isaac_root()
    return (root / "VERSION").read_text(encoding="utf-8").strip() if root else "unknown"


def redact_isaac(path):
    """Committed files carry no machine paths: the Isaac install prefix becomes <ISAAC_ROOT>."""
    path = str(Path(path).resolve())
    root = isaac_root()
    if root is not None and path.lower().startswith(str(root).lower()):
        return ISAAC_ROOT_TOKEN + path[len(str(root)):]
    return path


def hesai_report(sim):
    from pxr import Sdf, UsdPhysics
    import sensor_config

    stage = sim.stage
    root = str(Sdf.Path(sensor_config.LIDAR["path"]).GetParentPath())
    prims = rigid = mass_api = 0
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path == root or path.startswith(root + "/"):
            prims += 1
            rigid += int(prim.HasAPI(UsdPhysics.RigidBodyAPI))
            mass_api += int(prim.HasAPI(UsdPhysics.MassAPI))
    body_paths = list(sim.vehicle_bodies.prim_paths)
    masses = [float(m) for m in np.asarray(sim.vehicle_bodies.get_masses()).reshape(-1)]
    rb_mass = sum(m for p, m in zip(body_paths, masses) if p == root or p.startswith(root + "/"))
    return {"root": root, "prims": prims, "rigid_body_prims": rigid, "mass_api_prims": mass_api,
            "hesai_mass_contribution_kg": rb_mass}


class Recorder:
    """Wraps env.sim.step_command on the instance; records one row per physics step while active."""

    def __init__(self, env):
        self.rows = None
        sim = env.sim
        original = sim.step_command
        pid = env.core.pid.pid

        def recorded(state, total_thrust, roll_torque, pitch_torque, yaw_torque, render=True):
            pre = None
            if self.rows is not None:
                pre = np.concatenate([
                    np.asarray(state["position"], dtype=np.float64).reshape(3),
                    np.asarray(state["orientation"], dtype=np.float64).reshape(4),
                    np.asarray(state["linear_velocity"], dtype=np.float64).reshape(3),
                    np.asarray(state["angular_velocity"], dtype=np.float64).reshape(3),
                    [pid.height_integral], np.asarray(pid.xy_integral, dtype=np.float64).reshape(2),
                    np.asarray(pid.attitude_integral, dtype=np.float64).reshape(3),
                    np.array([total_thrust, roll_torque, pitch_torque, yaw_torque], dtype=np.float64),
                ])
                mean_wind = np.asarray(sim.wind.mean_wind, dtype=np.float64).reshape(3).copy()
            out = original(state, total_thrust, roll_torque, pitch_torque, yaw_torque, render=render)
            if pre is not None:
                speeds, saturated, wind_force, wind_torque = out
                self.rows.append(np.concatenate([
                    pre, np.asarray(speeds, dtype=np.float64).reshape(4), [float(bool(saturated))],
                    np.asarray(wind_force, dtype=np.float64).reshape(3),
                    np.asarray(wind_torque, dtype=np.float64).reshape(3), mean_wind,
                ]))
            return out

        sim.step_command = recorded

    def start(self):
        self.rows = []

    def stop(self):
        rows = np.array(self.rows, dtype=np.float64).reshape(-1, ROW_WIDTH)
        self.rows = None
        return rows


def state_vector(state):
    return np.concatenate([np.asarray(state[k], dtype=np.float64).reshape(-1)
                           for k in ("position", "orientation", "linear_velocity", "angular_velocity")])


def episode_record(env, info, rows, decisions, terminated, truncated, reason, observe_calls):
    completion = env.core.completion
    return {
        "rows": rows,
        "final_state": state_vector(env.sim.read_drone_state()),
        "physics_steps": int(len(rows)),
        "decisions": int(decisions),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "failure_reason": reason,
        "completed": bool(completion.completed),
        "completion_time": completion.completion_time,
        "route_seconds": float(info["route_seconds"]),
        "wind_seed": int(info["wind_seed"]),
        "wind_mean": [float(v) for v in info["wind_mean"]],
        "max_decisions": int(env.max_decisions),
        "observe_calls": int(observe_calls),
    }


def run_env_path(env, rec, seed, options, inject=None):
    """A: PathSacEnv with zero action. inject=(decision_index, action4) puts one nonzero action in (guard)."""
    core = env.core
    original_observe = core.observe
    calls = {"n": 0}

    def counting(state):
        calls["n"] += 1
        return original_observe(state)

    core.observe = counting
    try:
        _, info = env.reset(seed=seed, options=options)
        rec.start()
        zero = np.zeros(4, dtype=np.float32)
        decisions, terminated, truncated, reason = 0, False, False, None
        while True:
            action = zero
            if inject is not None and decisions == inject[0]:
                action = np.asarray(inject[1], dtype=np.float32)
            _, _, terminated, truncated, step_info = env.step(action)
            decisions += 1
            reason = step_info["failure_reason"]
            if terminated or truncated:
                break
        rows = rec.stop()
        assert decisions == env.decisions and len(rows) == env.trace.physics_steps
    finally:
        del core.observe
    return episode_record(env, info, rows, decisions, terminated, truncated, reason, calls["n"])


def run_pid_path(env, rec, seed, options):
    """B: same reset (observe stubbed out), then PathSacEnv.step + PathCore.step mirrored, minus the residual."""
    import hover_task
    import path_sac_config
    from hover_env import SimulationClosed

    core, sim = env.core, env.sim
    stub_calls = {"n": 0}

    def stub(state):  # the reset's single observe() call: the PID is NOT read
        stub_calls["n"] += 1
        return None

    core.observe = stub
    try:
        _, info = env.reset(seed=seed, options=options)
    finally:
        del core.observe

    pid = core.pid                    # ReferencePID (reset by env.reset -> core.begin)
    reference = core.reference
    direction_rate = core.direction_rate
    t = 0.0
    saturated = False
    decisions, terminated, truncated, reason = 0, False, False, None
    rec.start()
    while True:
        # ---- PathSacEnv.step, without apply_action / reward / observe ----
        if not sim.is_running():
            raise SimulationClosed("Simulation closed during an episode.")
        truth = sim.read_drone_state()
        for _ in range(core.decimation):
            if not hover_task.state_is_valid(truth):
                break
            # ---- PathCore.step, without "+ self.correction" ----
            if direction_rate != 0.0:
                angle = direction_rate * sim.dt
                c, s = np.cos(angle), np.sin(angle)
                x, y, z = sim.wind.mean_wind
                sim.wind.mean_wind = np.array([c*x - s*y, s*x + c*y, z])
            sample = reference.sample(t)
            command = pid.read(truth, sample, saturated)
            desired = pid.desired_attitude(truth, command)
            pid.integrate(truth, sample, sim.dt,
                          saturated or pid.acceleration_capped(truth, command, desired))
            _, sat, _, _ = sim.step_command(truth, *command, render=False)
            saturated = bool(sat)
            t += sim.dt
            # ---- back in PathSacEnv.step ----
            truth = sim.read_drone_state()
            if hover_task.state_is_valid(truth):
                core.completion.update(truth, reference.sample(t), reference.final_position, sim.dt, t)
        decisions += 1
        sample = reference.sample(t)
        reason = hover_task.failure_reason(truth, sample.position, path_sac_config.FAILURE)
        terminated = reason is not None
        truncated = not terminated and decisions >= env.max_decisions
        if terminated or truncated:
            break
    rows = rec.stop()
    ep = episode_record(env, info, rows, decisions, terminated, truncated, reason, 0)
    ep["reset_observe_stubbed"] = stub_calls["n"]
    return ep


def run_supplementary(env, rec, protocol, protocol_rec, meta, tol, tol_file, tol_sha):
    """SUPPLEMENTARY, added after the pre-registered run, same bit-exact tolerance (Claire, 2026-11-13).

    A vs B on outside_range x 6 scenarios x seeds 300000-300002 (failure-termination branch), plus 1e-6 guards on
    roll and yaw torque. Writes zero_residual_supplementary.json + summary; never touches zero_residual.json or
    the tolerance file. Computes no tracking metric (these seeds overlap the H1 pilot).
    """
    say("SUPPLEMENTARY run (added after the pre-registered run; same bit-exact tolerance)")
    cells = [(s, SUPP_PROFILE, seed) for s in protocol.SCENARIOS for seed in SEEDS]
    runs = {"A": {}, "B": {}}
    per_cell = {}
    t1 = time.perf_counter()
    for c in cells:
        cid = cell_id(*c)
        opts = protocol.reset_options(c[0], c[1])
        a = runs["A"][cid] = run_env_path(env, rec, c[2], opts)
        b = runs["B"][cid] = run_pid_path(env, rec, c[2], opts)
        r = compare_episodes(a, b, tol)
        r["observe_calls"] = [a["observe_calls"], b["observe_calls"]]
        r["reset_observe_stubbed_B"] = b["reset_observe_stubbed"]
        r["terminated"] = [a["terminated"], b["terminated"]]
        r["failure_reason"] = [a["failure_reason"], b["failure_reason"]]
        r["same_end_reason_and_step"] = (a["failure_reason"] == b["failure_reason"]
                                         and a["terminated"] == b["terminated"]
                                         and a["truncated"] == b["truncated"]
                                         and a["decisions"] == b["decisions"])
        per_cell[cid] = r
        say(f"{cid:<40} {'PASS' if r['pass'] else 'FAIL'} steps {r['physics_steps']} end {r['end'][0]} vs "
            f"{r['end'][1]} bitwise {r['bitwise_identical']} first diff {r['first_differing_step']}")
    wall = time.perf_counter() - t1
    failed = {cid: r for cid, r in per_cell.items() if r["terminated"][0] or r["terminated"][1]}
    all_pass = all(r["pass"] for r in per_cell.values())
    same_end = all(r["same_end_reason_and_step"] for r in per_cell.values())
    failed_match = all(r["same_end_reason_and_step"] and r["pass"] for r in failed.values())
    worst = {q: max(r["max_abs_diff"][q] for r in per_cell.values()) for q in QUANTITIES}
    say(f"outside_range: {len(cells)} cells, {sum(r['physics_steps'][0] for r in per_cell.values())} physics steps; "
        f"failed flights A {sum(r['terminated'][0] for r in per_cell.values())}, "
        f"B {sum(r['terminated'][1] for r in per_cell.values())}; all failed flights same reason+step and "
        f"bit-exact: {failed_match}; equivalence {'PASS' if all_pass else 'FAIL'}")
    for cid, r in failed.items():
        say(f"    failed: {cid} A {r['end'][0]} B {r['end'][1]}")

    guards = []
    for (scenario, profile, seed), index, name in SUPP_GUARDS:
        opts = protocol.reset_options(scenario, profile)
        action = [0.0, 0.0, 0.0, 0.0]
        action[index] = SUPP_MAGNITUDE
        b = run_pid_path(env, rec, seed, opts)
        a = run_env_path(env, rec, seed, opts, inject=(0, action))
        r = compare_episodes(a, b, tol)
        g = {"cell": cell_id(scenario, profile, seed), "channel": name, "action": action, "decision": 0,
             "correction": float(np.float32(SUPP_MAGNITUDE)) * float(env.core.limits[index]),
             "correction_unit": "N" if index == 0 else "N*m",
             "comparison_pass": r["pass"], "detected": not r["pass"], "exceeded": r["exceeded"],
             "first_step_exceeding_tolerance": r["first_step_exceeding_tolerance"],
             "first_differing_step": r["first_differing_step"], "max_abs_diff": r["max_abs_diff"],
             "physics_steps": r["physics_steps"], "end": r["end"]}
        guards.append(g)
        say(f"GUARD {g['cell']} {name} action[{index}]={SUPP_MAGNITUDE:g} (+{g['correction']:.3g} "
            f"{g['correction_unit']}, decision 0): comparison "
            f"{'FAILS (detected)' if g['detected'] else 'PASSES (NOT detected)'}; exceeded "
            + ", ".join(f"{q} {d:.3g}" for q, d in r["exceeded"].items())
            + f"; first diff step {g['first_differing_step']}")
    guard_ok = all(g["detected"] for g in guards)
    verdict = all_pass and same_end and guard_ok
    say(f"RESULT M2.2 SUPPLEMENTARY: {'PASS' if verdict else 'FAIL'} (equivalence {all_pass}, same end everywhere "
        f"{same_end}, failed flights {len(failed)}, guard detected {sum(g['detected'] for g in guards)}/{len(guards)})")
    meta["wall_clock_s"] = wall
    meta["trace_sha256"] = {r: trace_hash(runs[r], cells) for r in runs}
    save_traces(TRACE_DIR / "supplementary", runs, meta)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "supplementary, added after the pre-registered run, same bit-exact tolerance",
        "purpose": "M2.2 supplementary: A vs B on outside_range (exercises failure termination); "
                   "1e-6 guards on roll and yaw torque",
        "verdict": "PASS" if verdict else "FAIL",
        "equivalence_pass": all_pass, "same_end_everywhere": same_end,
        "failed_flights": {"count_A": sum(r["terminated"][0] for r in per_cell.values()),
                           "count_B": sum(r["terminated"][1] for r in per_cell.values()),
                           "all_same_reason_step_and_bit_exact": failed_match,
                           "cells": {cid: {"A": r["end"][0], "B": r["end"][1]} for cid, r in failed.items()}},
        "guard_detected_all": guard_ok,
        "tolerance_file": {"path": "residual_ppo/results/m2/zero_residual_tolerance.json",
                           "sha256_lf_at_run": tol_sha, "created_at": tol_file["created_at"],
                           "bit_exact": tol_file["bit_exact"], "tolerance": tol},
        "script_sha256_lf": file_sha256(Path(__file__)),
        "no_tracking_metrics": "No RMSE or tracking metric is computed from these traces (seeds overlap H1 pilot).",
        "worst_abs_diff": worst,
        "physics_steps_compared": sum(r["physics_steps"][0] for r in per_cell.values()),
        "cells": per_cell, "guards": guards,
        "row_layout": LAYOUT, "protocol": protocol_rec, "run": meta,
    }
    SUPP_PATH.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    say("written: residual_ppo/results/m2/zero_residual_supplementary.json, zero_residual_supplementary_summary.txt")
    SUPP_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")


def isaac_main(args):
    check_line = preflight()

    # Two torch installs: import torch (and torch._dynamo) BEFORE SimulationApp (CLAUDE.md, M2.1 probe).
    import torch
    import torch._dynamo  # noqa: F401

    t0 = time.perf_counter()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    env = None
    try:
        import gymnasium
        import stable_baselines3

        import eval_protocol_v1 as protocol
        import path_sac_env

        protocol_rec = protocol.protocol_record()
        if protocol_rec["protocol_file_sha256"] != EXPECTED_PROTOCOL_SHA:
            raise SystemExit(f"protocol hash {protocol_rec['protocol_file_sha256']} != expected {EXPECTED_PROTOCOL_SHA}")
        registered = protocol.register_profiles()
        env = path_sac_env.make_path_sac_env(app)
        say(f"startup {time.perf_counter() - t0:.1f} s; decimation {env.core.decimation}, dt {env.sim.dt!r}")
        loaded = {n: getattr(sys.modules.get(n), "__file__", None) for n in SHARED if n in sys.modules}
        outside = {n: f for n, f in loaded.items()
                   if f and not str(Path(f).resolve()).lower().startswith(str(RESIDUAL_DIR.resolve()).lower())}
        if outside:
            raise SystemExit(f"loaded shared modules outside residual_ppo: {outside}")
        rec = Recorder(env)
        meta = {
            "mode": args.mode, "tag": args.tag,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checks": check_line, "git": git_info(),
            "protocol_file_sha256": protocol_rec["protocol_file_sha256"],
            "registered_profiles": registered,
            "python": platform.python_version(), "isaac_sim": isaac_version(),
            "packages": {m.__name__: m.__version__ for m in (np, gymnasium, stable_baselines3, torch)},
            "package_files": {m.__name__: redact_isaac(m.__file__) for m in (np, torch)},
            "torch_cuda_available": torch.cuda.is_available(),
            "total_mass_kg": float(env.sim.total_mass), "hesai": hesai_report(env.sim),
            "decimation": env.core.decimation, "physics_dt": env.sim.dt,
            "residual_limits": env.core.limits.tolist(),
        }
        say(f"total_mass_kg {meta['total_mass_kg']:.6f}; HESAI {meta['hesai']}")
        say(f"packages {meta['packages']}; isaac {meta['isaac_sim']}")
        cells = all_cells(protocol.SCENARIOS)

        if args.mode == "floor":
            runs = {"r1": {}, "r2": {}}
            t1 = time.perf_counter()
            for c in cells:
                cid = cell_id(*c)
                opts = protocol.reset_options(c[0], c[1])
                for run in ("r1", "r2"):
                    runs[run][cid] = run_env_path(env, rec, c[2], opts)
                r = compare_episodes(runs["r1"][cid], runs["r2"][cid])
                ep = runs["r1"][cid]
                say(f"{cid:<40} steps {ep['physics_steps']:>5} end {ep['failure_reason'] or 'truncated'}@"
                    f"{ep['decisions']} observe {ep['observe_calls']}; r1 vs r2 bitwise {r['bitwise_identical']}"
                    f" max|dpos| {r['max_abs_diff']['position']:.3g}")
            meta["wall_clock_s"] = time.perf_counter() - t1
            meta["trace_sha256"] = {r: trace_hash(runs[r], cells) for r in runs}
            save_traces(TRACE_DIR / f"floor_{args.tag}", runs, meta)
            say(f"trace sha256: {meta['trace_sha256']}")
            say(f"written: residual_ppo/results/m2/zero_residual_traces/floor_{args.tag}.npz/.json (gitignored)")
            (OUT_DIR / f"zero_residual_floor_{args.tag}_summary.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8")
            say("RESULT: DONE")
            return

        # ---------------- compare ----------------
        if not TOLERANCE_PATH.is_file():
            raise SystemExit("zero_residual_tolerance.json not found: measure the noise floor first (D6)")
        tol_file = json.loads(TOLERANCE_PATH.read_text(encoding="utf-8"))
        if not tol_file.get("usable"):
            raise SystemExit("tolerance file marked not usable; not comparing")
        tol = {q: float(tol_file["tolerance"][q]) for q in QUANTITIES}
        tol_sha = file_sha256(TOLERANCE_PATH)
        say(f"tolerance file sha256 {tol_sha[:16]}… created {tol_file['created_at']}; "
            f"{'BIT-EXACT' if tol_file['bit_exact'] else 'floor'} tolerance")
        if args.mode == "supplementary":
            run_supplementary(env, rec, protocol, protocol_rec, meta, tol, tol_file, tol_sha)
            return
        if [list(c) for c in cells] != tol_file["cells"]:
            raise SystemExit("cells differ from the ones the floor was measured on")

        runs = {"A": {}, "B": {}}
        per_cell = {}
        t1 = time.perf_counter()
        for c in cells:
            cid = cell_id(*c)
            opts = protocol.reset_options(c[0], c[1])
            runs["A"][cid] = run_env_path(env, rec, c[2], opts)
            runs["B"][cid] = run_pid_path(env, rec, c[2], opts)
            r = compare_episodes(runs["A"][cid], runs["B"][cid], tol)
            r["observe_calls"] = [runs["A"][cid]["observe_calls"], runs["B"][cid]["observe_calls"]]
            r["reset_observe_stubbed_B"] = runs["B"][cid]["reset_observe_stubbed"]
            r["completed"] = [runs["A"][cid]["completed"], runs["B"][cid]["completed"]]
            per_cell[cid] = r
            say(f"{cid:<40} {'PASS' if r['pass'] else 'FAIL'} steps {r['physics_steps']} end {r['end'][0]} vs "
                f"{r['end'][1]} bitwise {r['bitwise_identical']} first diff {r['first_differing_step']} "
                f"observe A {r['observe_calls'][0]} B {r['observe_calls'][1]}")
        compare_s = time.perf_counter() - t1
        all_pass = all(r["pass"] for r in per_cell.values())
        worst = {q: max(r["max_abs_diff"][q] for r in per_cell.values()) for q in QUANTITIES}
        say(f"equivalence over {len(cells)} cells: {'PASS' if all_pass else 'FAIL'}; "
            f"physics steps compared {sum(r['physics_steps'][0] for r in per_cell.values())}; worst |diff| "
            + ", ".join(f"{q} {d:.3g}" for q, d in worst.items()))

        # ---------------- guard tests ----------------
        guards = []
        for scenario, profile, seed in GUARD_CELLS:
            opts = protocol.reset_options(scenario, profile)
            for mag in GUARD_MAGNITUDES:
                b = run_pid_path(env, rec, seed, opts)
                a = run_env_path(env, rec, seed, opts, inject=(0, [mag, 0.0, 0.0, 0.0]))
                r = compare_episodes(a, b, tol)
                g = {"cell": cell_id(scenario, profile, seed), "action": [mag, 0, 0, 0], "decision": 0,
                     "thrust_correction_N": float(np.float32(mag)) * float(env.core.limits[0]),
                     "comparison_pass": r["pass"], "detected": not r["pass"], "exceeded": r["exceeded"],
                     "first_step_exceeding_tolerance": r["first_step_exceeding_tolerance"],
                     "first_differing_step": r["first_differing_step"], "max_abs_diff": r["max_abs_diff"],
                     "physics_steps": r["physics_steps"], "end": r["end"]}
                guards.append(g)
                say(f"GUARD {g['cell']} action[0]={mag:g} (thrust +{g['thrust_correction_N']:.3g} N, decision 0): "
                    f"comparison {'FAILS (detected)' if g['detected'] else 'PASSES (NOT detected)'}; exceeded "
                    + ", ".join(f"{q} {d:.3g}" for q, d in r["exceeded"].items())
                    + f"; first diff step {g['first_differing_step']}")
        guard_ok = all(g["detected"] for g in guards)

        same_end = all(r["episode_fields_equal"] and r["same_length"] for r in per_cell.values())
        verdict = all_pass and guard_ok and same_end
        say(f"RESULT M2.2: {'PASS' if verdict else 'FAIL'} (equivalence {all_pass}, same end everywhere {same_end}, "
            f"guard detected {sum(g['detected'] for g in guards)}/{len(guards)})")
        meta["wall_clock_s"] = compare_s
        meta["trace_sha256"] = {r: trace_hash(runs[r], cells) for r in runs}
        save_traces(TRACE_DIR / "compare", runs, meta)
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": "M2.2 runs 2-3: env path with zero residual (A) vs pure PID loop (B), plus guard tests",
            "verdict": "PASS" if verdict else "FAIL",
            "equivalence_pass": all_pass, "same_end_everywhere": same_end, "guard_detected_all": guard_ok,
            "tolerance_file": {"path": "residual_ppo/results/m2/zero_residual_tolerance.json",
                               "sha256_lf": tol_sha, "created_at": tol_file["created_at"],
                               "bit_exact": tol_file["bit_exact"], "tolerance": tol},
            "worst_abs_diff": worst,
            "physics_steps_compared": sum(r["physics_steps"][0] for r in per_cell.values()),
            "cells": per_cell, "guards": guards,
            "row_layout": LAYOUT, "named_quantities": NAMED,
            "protocol": protocol_rec, "run": meta,
        }
        (OUT_DIR / "zero_residual.json").write_text(json.dumps(result, indent=2, default=str) + "\n",
                                                   encoding="utf-8")
        say("written: residual_ppo/results/m2/zero_residual.json, zero_residual_summary.txt")
        (OUT_DIR / "zero_residual_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except BaseException:
        import traceback
        traceback.print_exc()
        say("RESULT: FAIL (exception)")
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("floor", "tolerance", "compare", "supplementary"))
    parser.add_argument("--tag", default=None, help="floor mode: p1 or p2 (one per process)")
    args = parser.parse_args()
    if args.mode == "floor" and args.tag not in ("p1", "p2"):
        parser.error("floor needs --tag p1 or --tag p2")
    if args.mode == "tolerance":
        mode_tolerance()
    else:
        isaac_main(args)


if __name__ == "__main__":
    main()
