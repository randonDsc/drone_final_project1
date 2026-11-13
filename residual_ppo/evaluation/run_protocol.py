"""Evaluation protocol v1 runner (docs/milestone2_plan.md, section E + M2.3 runner design). NO training.

Modes:
  fly        Isaac. One controller over scenarios x profiles x seeds; the flight loop and metric call mirror
             drone_team_main/drone_code/eval_path_sac.py fly() exactly. Optional --check0 first: one cell flown with
             diagnostics off and on must give identical metrics and per-decision trace.
  reproduce  No Isaac. M2.3 first check: team README section 2 PID-only rows vs a pilot run's flights.

    # Directory: C:\\Users\\tucla\\Desktop\\DSC180A\\drone_final_project1
    & "<Isaac>\\python.bat" residual_ppo\\evaluation\\run_protocol.py fly --controller pid --seeds 300000-300009 `
        --out residual_ppo/results/m2/pilot_pid --check0 *> residual_ppo\\results\\m2\\pilot_pid.log
    & "<Isaac>\\python.bat" residual_ppo\\evaluation\\run_protocol.py reproduce --run residual_ppo/results/m2/pilot_pid
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))

import run_common as rc  # noqa: E402

CHECK0_CELL = ("turns", "training_range", 300_000)
DIAG_MIN_REF_SPEED = 0.05
PID_PART_LABEL = {
    "pid": "PID command = final - correction (exact: correction is 0)",
    "other": "PID part = final - correction (reconstructed; floating-point rounding possible)",
}

lines = []


def say(msg=""):
    line = f"[PROTO] {msg}"
    print(line, flush=True)
    lines.append(line)


def repo_rel(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        return rc.redact_isaac(p)


def parse_seeds(text):
    if "-" in text:
        a, b = (int(x) for x in text.split("-"))
        return tuple(range(a, b + 1))
    return tuple(int(x) for x in text.split(","))


def write_csv(path, rows):
    """As the team's write_csv: union of keys in first-seen order. Dict values are JSON-encoded."""
    names = []
    for row in rows:
        names += [n for n in row if n not in names]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=names)
        writer.writeheader()
        writer.writerows([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()}
                          for r in rows])


# ---------------------------------------------------------------- diagnostics (read-only wrapper)

class Diagnostics:
    """Per-physics-step diagnostics via an instance wrapper on env.sim.step_command (shared files untouched).

    Read-only: calls only pure functions (reference.sample, ReferencePID.acceleration_capped) and reads state.
    """

    def __init__(self, env, controller_name):
        import hover_task
        import path_sac_config
        import sim_config
        from wind import quaternion_to_matrix

        self.env = env
        self.active = False
        self.label = PID_PART_LABEL["pid" if controller_name == "pid" else "other"]
        self._q2m = quaternion_to_matrix
        self._g = hover_task.GRAVITY
        self._accel_cap = float(path_sac_config.PID_ACCELERATION_CAP)
        self._tan_tilt = float(np.tan(np.deg2rad(sim_config.MAX_TILT_DEGREES)))
        sim = env.sim
        original = sim.step_command

        def wrapped(state, total_thrust, roll_torque, pitch_torque, yaw_torque, render=True):
            if self.active:
                self._observe(state, np.array([total_thrust, roll_torque, pitch_torque, yaw_torque], dtype=float))
            return original(state, total_thrust, roll_torque, pitch_torque, yaw_torque, render=render)

        sim.step_command = wrapped

    def start(self):
        self.steps = self.accel_bind = self.tilt_bind = 0
        self.lag_n = 0
        self.lag_sum = self.lag_sq = self.cross_sq = 0.0
        self.active = True

    def _observe(self, state, final):
        core = self.env.core
        pid_command = final - core.correction
        desired = np.array(core.desired, dtype=float)
        sample = core.sample()  # core.time has not advanced yet: the sample this step's PID used
        self.steps += 1
        if core.pid.acceleration_capped(state, pid_command, desired):
            R = self._q2m(state["orientation"])
            vertical = max(pid_command[0] * max(float(R[2, 2]), 0.5) / core.pid.mass, 0.5 * self._g)
            if self._accel_cap <= vertical * self._tan_tilt:
                self.accel_bind += 1
            else:
                self.tilt_bind += 1
        v_ref = np.asarray(sample.velocity, dtype=float)
        speed = float(np.linalg.norm(v_ref))
        if speed > DIAG_MIN_REF_SPEED:
            u = v_ref / speed
            e = np.asarray(sample.position, dtype=float) - np.asarray(state["position"], dtype=float)
            lag = float(e @ u)
            self.lag_n += 1
            self.lag_sum += lag
            self.lag_sq += lag * lag
            self.cross_sq += float(np.sum((e - lag * u) ** 2))

    def stop(self):
        self.active = False
        n = self.lag_n
        return {
            "diag_physics_steps": self.steps,
            "accel_cap_binding_frac": self.accel_bind / self.steps if self.steps else None,
            "tilt_cap_binding_frac": self.tilt_bind / self.steps if self.steps else None,
            "along_track_lag_mean": self.lag_sum / n if n else None,
            "along_track_lag_rms": float(np.sqrt(self.lag_sq / n)) if n else None,
            "cross_track_component_rms": float(np.sqrt(self.cross_sq / n)) if n else None,
            "diag_lag_steps": n,
            "diag_pid_part": self.label,
        }


# ---------------------------------------------------------------- flight (mirrors team fly())

def fly(env, protocol, controller, seed, scenario, profile, diag=None):
    route_type = protocol.SCENARIOS[scenario]
    observation, info = env.reset(seed=seed, options={
        "route_stage": protocol.ROUTE_STAGE,
        "route_type": route_type,
        "wind_profile": profile,
    })
    initial_obs_sha256 = hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest()
    max_decisions = int(env.max_decisions)
    if diag is not None:
        diag.start()
    reason = None
    while True:
        action = controller.act(observation)
        observation, _, terminated, truncated, step_info = env.step(action)
        reason = step_info["failure_reason"]
        if terminated or truncated:
            break
    extra = diag.stop() if diag is not None else {}
    row = {
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
    row["controller"] = controller.name
    row["route_type"] = route_type
    row["initial_obs_sha256"] = initial_obs_sha256
    row["max_decisions"] = max_decisions
    row["wind_mean"] = [float(v) for v in info["wind_mean"]]
    row.update(extra)
    trace = {"positions": np.array(env.trace.positions, dtype=float), "errors": np.array(env.trace.errors),
             "final_state": np.concatenate([np.asarray(v, dtype=float).reshape(-1)
                                            for v in env.sim.read_drone_state().values()])}
    return row, trace


METRIC_KEYS = ("decisions", "failed", "failure_reason", "collided", "completed", "completion_time", "position_rmse",
               "cross_track_rmse", "cross_track_max", "max_deviation", "saturation_rate", "residual_norm",
               "residual_change", "wind_speed", "wind_seed", "route_seconds", "initial_obs_sha256", "max_decisions")


def check0(env, protocol, controller, make_diag):
    """Same cell with diagnostics off (no wrapper installed yet) and on: metrics and trace must be identical."""
    scenario, profile, seed = CHECK0_CELL
    off_row, off_trace = fly(env, protocol, controller, seed, scenario, profile, diag=None)
    diag = make_diag()
    on_row, on_trace = fly(env, protocol, controller, seed, scenario, profile, diag=diag)
    metric_diff = {k: [off_row[k], on_row[k]] for k in METRIC_KEYS if off_row[k] != on_row[k]}
    trace_equal = {k: bool(off_trace[k].shape == on_trace[k].shape
                           and off_trace[k].tobytes() == on_trace[k].tobytes()) for k in off_trace}
    ok = not metric_diff and all(trace_equal.values())
    result = {"cell": list(CHECK0_CELL), "pass": ok, "metric_differences": metric_diff,
              "trace_bitwise_equal": trace_equal, "decisions": off_row["decisions"],
              "diagnostics_on": {k: v for k, v in on_row.items() if k.startswith(("diag_", "accel_", "tilt_",
                                                                                 "along_", "cross_track_comp"))}}
    say(f"CHECK 0 {'PASS' if ok else 'FAIL'}: {CHECK0_CELL} diagnostics off vs on; metric differences "
        f"{metric_diff or 'none'}; trace bitwise equal {trace_equal}")
    return result, diag


def mode_fly(args):
    check_line = rc.preflight(say)
    import torch  # before SimulationApp (two torch installs; CLAUDE.md)
    import torch._dynamo  # noqa: F401

    t0 = time.perf_counter()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    env = None
    try:
        import gymnasium
        import stable_baselines3

        import controllers
        import eval_protocol_v1 as protocol
        import path_sac_env
        import protocol_stats

        protocol_rec = protocol.protocol_record()
        if protocol_rec["protocol_file_sha256"] != rc.EXPECTED_PROTOCOL_SHA:
            raise SystemExit(f"protocol hash {protocol_rec['protocol_file_sha256']} != {rc.EXPECTED_PROTOCOL_SHA}")
        registered = protocol.register_profiles()
        seeds = parse_seeds(args.seeds)
        allowed = set(protocol.EVAL_SEEDS) | set(protocol.VALIDATION_SEEDS)
        if not set(seeds) <= allowed:
            raise SystemExit(f"seeds outside eval/validation sets: {sorted(set(seeds) - allowed)}")
        profiles = tuple(args.profiles.split(",")) if args.profiles else protocol.PROFILES
        scenarios = tuple(protocol.SCENARIOS)
        controller = controllers.make_controller(args.controller)

        env = path_sac_env.make_path_sac_env(app)
        outside = rc.loaded_outside_residual()
        if outside:
            raise SystemExit(f"loaded shared modules outside residual_ppo: {outside}")
        startup_s = time.perf_counter() - t0
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "controller": controller.record(),
            "seeds": [seeds[0], seeds[-1], len(seeds)], "profiles": list(profiles), "scenarios": scenarios,
            "checks": check_line, "git": rc.git_info(), "protocol": protocol_rec,
            "registered_profiles": registered, "still_air_definition": registered.get(protocol.STILL_AIR_NAME),
            "python": platform.python_version(), "isaac_sim": rc.isaac_version(),
            "packages": {m.__name__: m.__version__ for m in (np, gymnasium, stable_baselines3, torch)},
            "package_files": {m.__name__: rc.redact_isaac(m.__file__) for m in (np, torch)},
            "torch_cuda_available": torch.cuda.is_available(), "device": "n/a (pid: no network)",
            "total_mass_kg": float(env.sim.total_mass), "hesai": rc.hesai_report(env.sim),
            "decimation": env.core.decimation, "physics_dt": env.sim.dt,
            "hardware": rc.hardware(), "startup_s": startup_s,
            "diagnostics": {"pid_part": PID_PART_LABEL["pid" if controller.name == "pid" else "other"],
                            "min_reference_speed_for_lag": DIAG_MIN_REF_SPEED},
            "realizations_note": "pilot copy, NOT canonical; the canonical realizations.json is the M2.4 "
                                 "30-seed PID run (plan, M2.3 design (d))" if len(seeds) != 30 else
                                 "candidate canonical (30 eval seeds)",
        }
        say(f"controller {controller.name}; seeds {seeds[0]}..{seeds[-1]} ({len(seeds)}); profiles {profiles}; "
            f"total_mass_kg {meta['total_mass_kg']:.6f}; HESAI {meta['hesai']['hesai_mass_contribution_kg']} kg; "
            f"packages {meta['packages']}; isaac {meta['isaac_sim']}")

        def make_diag():
            return Diagnostics(env, controller.name)

        check0_result = None
        if args.check0:
            check0_result, diag = check0(env, protocol, controller, make_diag)
            if not check0_result["pass"]:
                raise SystemExit("check 0 failed: diagnostics change the flight; not flying the protocol")
        else:
            diag = make_diag()

        flights = []
        t1 = time.perf_counter()
        for scenario in scenarios:
            for profile in profiles:
                for seed in seeds:
                    row, _ = fly(env, protocol, controller, seed, scenario, profile, diag=diag)
                    flights.append(row)
                say(f"finished {scenario} / {profile} ({len(seeds)} flights)")
        meta["wall_clock_s"] = time.perf_counter() - t1
        say(f"{len(flights)} flights in {meta['wall_clock_s']/60:.1f} min")

        out = Path(args.out)
        out = out if out.is_absolute() else (REPO_DIR / out)
        out.mkdir(parents=True, exist_ok=True)
        summary = protocol_stats.summaries(flights)
        realizations = {
            "note": meta["realizations_note"],
            "key": ["wind_seed", "wind_speed", "route_seconds", "initial_obs_sha256", "max_decisions"],
            "flights": {f"{f['scenario']}|{f['wind_profile']}|{f['seed']}":
                        {k: f[k] for k in ("wind_seed", "wind_speed", "route_seconds", "initial_obs_sha256",
                                           "max_decisions")} for f in flights},
        }
        (out / "results.json").write_text(json.dumps({"meta": meta, "check0": check0_result, "summary": summary,
                                                      "flights": flights}, indent=2, default=str) + "\n",
                                          encoding="utf-8")
        (out / "realizations.json").write_text(json.dumps(realizations, indent=2, default=str) + "\n",
                                               encoding="utf-8")
        write_csv(out / "flights.csv", flights)
        write_csv(out / "summary.csv", summary)
        say(f"written: {repo_rel(out)}/results.json, flights.csv, summary.csv, summary.txt, realizations.json")
        say("RESULT: DONE (flights only; no consistency check or H1 statistic computed here)")
        (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except BaseException:
        import traceback
        traceback.print_exc()
        say("RESULT: FAIL (exception)")
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


def mode_reproduce(args):
    import eval_protocol_v1 as protocol
    import protocol_stats

    run = Path(args.run)
    run = run if run.is_absolute() else (REPO_DIR / run)
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    flights = [f for f in data["flights"] if f["seed"] in protocol.TEAM_EVAL_SEEDS
               and f["wind_profile"] in protocol_stats.TEAM_README_PID and f["controller"] == "pid"]
    rep = protocol_stats.reproduction_check(flights, tuple(protocol.SCENARIOS))
    text = protocol_stats.reproduction_text(rep)
    print(text)
    out = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "M2.3 first check: reproduce team README section 2 open-scene PID-only rows (seeds 300000-300009)",
        "source_run": repo_rel(run / "results.json"),
        "source_run_sha256_lf": rc.file_sha256(run / "results.json"),
        "protocol_file_sha256": data["meta"]["protocol"]["protocol_file_sha256"],
        "packages": data["meta"]["packages"], "isaac_sim": data["meta"]["isaac_sim"],
        "total_mass_kg": data["meta"]["total_mass_kg"],
        "aggregation": "per profile: mean over the 6 scenarios of path_sac_policy.summarize() values (README 2)",
        "readme_source": "drone_team_main README.md section 2 at 61d2722",
        **rep,
    }
    dest = REPO_DIR / "residual_ppo" / "results" / "m2"
    (dest / "team_reproduction.json").write_text(json.dumps(out, indent=2, default=str) + "\n", encoding="utf-8")
    (dest / "team_reproduction.txt").write_text(text + "\n", encoding="utf-8")
    print("written: residual_ppo/results/m2/team_reproduction.json, team_reproduction.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fly")
    f.add_argument("--controller", default="pid")
    f.add_argument("--seeds", required=True, help="e.g. 300000-300009 or 300000,300001")
    f.add_argument("--profiles", default=None, help="comma list; default: all protocol profiles")
    f.add_argument("--out", required=True)
    f.add_argument("--check0", action="store_true")
    r = sub.add_parser("reproduce")
    r.add_argument("--run", required=True)
    args = parser.parse_args()
    if args.mode == "fly":
        mode_fly(args)
    else:
        mode_reproduce(args)


if __name__ == "__main__":
    main()
