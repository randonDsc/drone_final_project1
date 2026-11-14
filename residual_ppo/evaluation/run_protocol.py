"""Evaluation protocol v1 runner (docs/milestone2_plan.md, section E + M2.3 runner design). NO training.

Modes:
  fly        Isaac. One controller over scenarios x profiles x seeds; the flight loop and metric call mirror
             drone_team_main/drone_code/eval_path_sac.py fly() exactly. Optional --check0 first: one cell flown with
             diagnostics off and on must give identical metrics and per-decision trace.
  reproduce  No Isaac. M2.3 first check: team README section 2 PID-only rows vs a pilot run's flights.
  still-vs-wind  No Isaac. M2.3 consistency check (stops if violated), still-vs-wind table, H1 statistics
             (pilot on 10 seeds; the reported verdict on the M2.4 30 seeds).
  baseline-table  No Isaac. M2.4b: matched-realization check across all profiles of a run + per-profile table
             (endpoint status, failures and reasons per scenario, wind speeds, cap binding).
  compare    No Isaac. M2.4: every flight of a reference run (pilot, or a 1-cell re-run) must equal the same
             (scenario, profile, seed) flight of a candidate run bit-for-bit (all row fields + realizations entries).

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
import subprocess
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


def code_blobs():
    """git blob IDs (autocrlf-normalized, as `git hash-object`) of the evaluation code actually run: identifies the
    code exactly even when the run happens before the code is committed."""
    files = sorted((RESIDUAL_DIR / "evaluation").glob("*.py"))
    out = subprocess.run(["git", "-C", str(REPO_DIR), "hash-object", *map(str, files)], capture_output=True,
                         text=True).stdout.split()
    return dict(zip((repo_rel(f) for f in files), out))


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
        scenarios = tuple(args.scenarios.split(",")) if args.scenarios else tuple(protocol.SCENARIOS)
        if not set(scenarios) <= set(protocol.SCENARIOS) or not set(profiles) <= set(protocol.PROFILES):
            raise SystemExit(f"unknown scenario/profile: {scenarios} {profiles}")
        controller = controllers.make_controller(args.controller)
        canonical = (controller.name == "pid" and seeds == tuple(protocol.EVAL_SEEDS)
                     and profiles == tuple(protocol.PROFILES) and scenarios == tuple(protocol.SCENARIOS))

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
            "code_blobs": code_blobs(),
            "realizations_note": ("CANONICAL: M2.4 30-seed PID run (plan, M2.3 design (d)); the cross-controller "
                                  "reference, valid once the M2.4 checks pass (results/m2/m2_4_*)") if canonical else
                                 ("NOT canonical (subset run); the canonical realizations.json is the M2.4 "
                                  "30-seed PID run (plan, M2.3 design (d))"),
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


def _fmt(x, nd=3):
    return "  n/a" if x is None else f"{x:.{nd}f}"


def mode_still_vs_wind(args):
    """M2.3: consistency check (fails the step if violated), still-vs-wind table, H1 pilot statistics. No Isaac."""
    import eval_protocol_v1 as protocol
    import protocol_stats as ps

    run = Path(args.run)
    run = run if run.is_absolute() else (REPO_DIR / run)
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    flights = [f for f in data["flights"] if f["controller"] == "pid"]
    scenarios = tuple(protocol.SCENARIOS)
    profiles = tuple(data["meta"]["profiles"])
    first, last, n = data["meta"]["seeds"]
    seeds = tuple(range(first, last + 1))
    assert len(seeds) == n
    reported = seeds == tuple(protocol.EVAL_SEEDS)
    tag = "M2.4, 30 eval seeds: REPORTED" if reported else f"PILOT, {n} seeds"
    out_lines = []

    def emit(msg=""):
        print(msg)
        out_lines.append(msg)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": (f"{tag}: consistency check (matched realizations), still air vs wind table, H1 statistics "
                    "(pre-registered)"),
        "source_run": repo_rel(run / "results.json"), "source_run_sha256_lf": rc.file_sha256(run / "results.json"),
        "protocol_file_sha256": data["meta"]["protocol"]["protocol_file_sha256"],
        "packages": data["meta"]["packages"], "isaac_sim": data["meta"]["isaac_sim"],
        "total_mass_kg": data["meta"]["total_mass_kg"], "still_air_definition": data["meta"]["still_air_definition"],
        "seeds": [first, last, n], "profiles": list(profiles), "scenarios": scenarios,
        "bootstrap": {"resamples": ps.BOOT_RESAMPLES, "rng_seed": ps.BOOT_SEED, "ci": "95% percentile",
                      "unit": "seed (keeps all scenarios / both profiles together)",
                      "rng": "np.random.default_rng(0) started afresh for each statistic"},
        "note": ("M2.4 30-seed result: the reported H1 verdict." if reported else
                 "PILOT. The M2.4 30-seed result is the one reported for H1."),
    }
    dest = REPO_DIR / "residual_ppo" / "results" / "m2"
    cons = ps.consistency_check(flights, scenarios, profiles, seeds, protocol.STILL_AIR_NAME)
    result["consistency_check"] = cons
    emit(f"[{tag}] consistency check: {'PASS' if cons['pass'] else 'FAIL'} ({cons['flights']}/{cons['expected_flights']} "
         f"flights; {cons['groups_checked']} (scenario, seed) groups; fields {cons['fields']})")
    if not cons["pass"]:
        emit(f"problems: {cons['problems'][:10]}")
        (dest / f"{args.name}.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        (dest / f"{args.name}_summary.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        raise SystemExit("consistency check failed: step fails; no table or H1 statistic computed")

    table = ps.still_vs_wind_table(flights, scenarios, profiles, seeds, protocol.STILL_AIR_NAME)
    result["table"] = table
    emit("")
    emit(f"Still air vs wind (PID only; means over {n} seeds, failed flights included; diff = wind - still_air, "
         "paired per seed, mean [95% CI])")
    emit(f"{'scenario':<13}{'profile':<15}{'n':>3}{'fail':>5}{'RMSE':>7}{'cross':>7}{'maxdev':>7}  "
         f"{'dRMSE [CI]':<24}{'dcross [CI]':<24}{'dmaxdev [CI]':<24}{'acc%':>6}{'tilt%':>6}{'lag':>7}{'lagrms':>7}"
         f"{'xtrk':>7}")
    for r in table:
        def d(m):
            if f"{m}_diff_vs_still_mean" not in r:
                return f"{'':<24}"
            return (f"{r[f'{m}_diff_vs_still_mean']:+.3f} [{r[f'{m}_diff_vs_still_ci95_lo']:+.3f},"
                    f"{r[f'{m}_diff_vs_still_ci95_hi']:+.3f}]").ljust(24)
        acc, tilt = r["accel_cap_binding_frac_mean"], r["tilt_cap_binding_frac_mean"]
        emit(f"{r['scenario']:<13}{r['wind_profile']:<15}{r['flights']:>3}{r['failed_flights']:>5}"
             f"{r['position_rmse']:>7.3f}{r['cross_track_rmse']:>7.3f}{r['max_deviation']:>7.2f}  "
             f"{d('position_rmse')}{d('cross_track_rmse')}{d('max_deviation')}"
             f"{_fmt(None if acc is None else 100 * acc, 1):>6}{_fmt(None if tilt is None else 100 * tilt, 1):>6}"
             f"{_fmt(r['along_track_lag_mean_mean']):>7}{_fmt(r['along_track_lag_rms_mean']):>7}"
             f"{_fmt(r['cross_track_component_rms_mean']):>7}")
    emit(f"(maxdev column = summarize() max over the {n} flights; acc%/tilt% = accel-cap / tilt-cap binding share of "
         "physics steps; lag = along-track lag mean, + = behind; xtrk = cross-track component RMS; diagnostics over "
         "steps with |v_ref| > 0.05)")

    h1 = ps.h1_analysis(flights, scenarios, seeds, protocol.STILL_AIR_NAME, "training_range")
    result["h1_reported" if reported else "h1_pilot"] = h1
    p = h1["primary_wind_share_mse"]
    emit("")
    emit(f"H1 [{tag}] (pre-registered; seeds {first}-{last})")
    emit(f"PRIMARY wind_share_MSE = {p['value']:.3f}  95% CI [{p['ci95'][0]:.3f}, {p['ci95'][1]:.3f}]  "
         f"(mean MSE training_range {p['mean_wind']:.4f} m^2, still_air {p['mean_still']:.4f} m^2) -> {p['verdict']}")
    hv = h1["hover_sanity"]
    emit(f"hover sanity: still-air MSE {hv['still_air_mse_mean']:.5f} m^2, training_range MSE "
         f"{hv['training_range_mse_mean']:.5f} m^2, share {hv['mse_wind_share']:.3f} "
         f"[{hv['ci95'][0]:.3f}, {hv['ci95'][1]:.3f}] (threshold {hv['threshold']}) -> "
         f"{'SUSPECT: H1 not interpreted' if hv['suspect'] else 'OK'}")
    emit(f"H1 VERDICT ({'reported, M2.4' if reported else 'pilot'}): {h1['verdict']}")
    emit(f"failed flights in still_air/training_range: {len(h1['failed_flights'])}"
         + "".join(f"\n    {f}" for f in h1["failed_flights"]))
    s = h1["sensitivity_excluding_failed_seeds"]
    emit(f"sensitivity (excluding seeds {s['excluded_seeds'] or 'none'}): share {_fmt(s['value'])} CI "
         f"[{_fmt(s['ci95'][0])}, {_fmt(s['ci95'][1])}] n_seeds {s['n_seeds']} -> {s['verdict']}"
         + ("" if h1["primary_and_sensitivity_agree"] else "  ** DIFFERS FROM PRIMARY **")
         + (f" ({s['note']})" if "note" in s else ""))
    sec = h1["secondary"]
    r_ = sec["wind_share_rmse"]
    emit(f"secondary: wind_share_RMSE {r_['value']:.3f} [{r_['ci95'][0]:.3f}, {r_['ci95'][1]:.3f}]; paired RMSE "
         f"difference {sec['paired_rmse_difference_m']['value']:+.4f} m [{sec['paired_rmse_difference_m']['ci95'][0]:+.4f},"
         f" {sec['paired_rmse_difference_m']['ci95'][1]:+.4f}]")
    for sc, v in sec["per_scenario_wind_share_mse"].items():
        emit(f"    per-scenario MSE wind share {sc:<13} {v['value']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  "
             f"(MSE tr {v['mean_wind']:.4f}, still {v['mean_still']:.4f})")
    emit("(secondary results cannot change the H1 verdict)")

    (dest / f"{args.name}.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    write_csv(dest / f"{args.name}.csv", table)
    (dest / f"{args.name}_summary.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"written: residual_ppo/results/m2/{args.name}.json, .csv, _summary.txt")


def mode_baseline_table(args):
    """M2.4b: consistency (matched realizations) across ALL profiles of a run, then the per-profile table. No Isaac."""
    import eval_protocol_v1 as protocol
    import protocol_stats as ps

    run = Path(args.run)
    run = run if run.is_absolute() else (REPO_DIR / run)
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    meta = data["meta"]
    flights = data["flights"]
    scenarios = tuple(meta["scenarios"])
    profiles = tuple(meta["profiles"])
    first, last, n = meta["seeds"]
    seeds = tuple(range(first, last + 1))
    assert len(seeds) == n
    prot = meta["protocol"]
    status = {**{p: "PRIMARY" for p in prot.get("primary_profiles", [])},
              **{p: "secondary (D4)" for p in prot.get("secondary_profiles", [])},
              **{p: "reported, not primary" for p in prot.get("reported_not_primary", [])}}
    out_lines = []

    def emit(msg=""):
        print(msg)
        out_lines.append(msg)

    cons = ps.consistency_check(flights, scenarios, profiles, seeds, protocol.STILL_AIR_NAME)
    emit(f"M2.4b matched-realization (consistency) check across {len(profiles)} profiles: "
         f"{'PASS' if cons['pass'] else 'FAIL'} ({cons['flights']}/{cons['expected_flights']} flights; "
         f"{cons['groups_checked']} (scenario, seed) groups; fields {cons['fields']})")
    if not cons["pass"]:
        emit(f"problems: {cons['problems'][:10]}")
    table = []
    for prof in profiles:
        rows = [f for f in flights if f["wind_profile"] == prof]
        agg = ps.profile_aggregate(rows, prof, scenarios)
        reasons, per_scen = {}, {}
        for f in rows:
            if f["failed"]:
                reasons[f["failure_reason"]] = reasons.get(f["failure_reason"], 0) + 1
                per_scen[f["scenario"]] = per_scen.get(f["scenario"], 0) + 1
        speeds = [f["wind_speed"] for f in rows]
        acc = [f["accel_cap_binding_frac"] for f in rows if f.get("accel_cap_binding_frac") is not None]
        dec = [f["decisions"] for f in rows]
        table.append({
            **{k: v for k, v in agg.items() if not isinstance(v, (dict, list))},
            "endpoint_status": status.get(prof, "?"),
            "definition": (prot["team_profiles"].get(prof) or prot["our_profiles"].get(prof)),
            "failure_reasons": reasons,
            "failed_per_scenario": {sc: per_scen.get(sc, 0) for sc in scenarios},
            "wind_speed_min": min(speeds), "wind_speed_max": max(speeds),
            "decisions_mean": float(np.mean(dec)),
            "accel_cap_binding_frac_mean": float(np.mean(acc)) if acc else None,
        })
    emit("")
    emit(f"PID-only baseline per profile (protocol {prot['protocol_file_sha256'][:8]}; mean over the "
         f"{len(scenarios)} scenarios of summarize(); {n} seeds; failed flights included)")
    emit(f"{'profile':<16}{'status':<23}{'n':>4}{'RMSE':>7}{'cross':>7}{'maxdev':>7}{'(all)':>7}{'compl':>6}"
         f"{'fail':>5}{'fail%':>6}{'wind m/s':>12}{'dec':>6}{'acc%':>6}  failure reasons / per scenario")
    for r in table:
        emit(f"{r['profile']:<16}{r['endpoint_status']:<23}{r['flights']:>4}{r['position_rmse']:>7.3f}"
             f"{r['cross_track_rmse']:>7.3f}{r['max_deviation_mean_of_scenario_max']:>7.3f}"
             f"{r['max_deviation_overall_max']:>7.3f}{r['completed_flights']:>6}{r['failed_flights']:>5}"
             f"{100 * r['failure_rate']:>5.0f}%{r['wind_speed_min']:>6.2f}-{r['wind_speed_max']:<5.2f}"
             f"{r['decisions_mean']:>6.0f}{_fmt(None if r['accel_cap_binding_frac_mean'] is None else 100 * r['accel_cap_binding_frac_mean'], 1):>6}"
             f"  {r['failure_reasons'] or '-'}"
             + (f" {r['failed_per_scenario']}" if r['failed_flights'] else ""))
    emit("(maxdev = mean of per-scenario maxima; (all) = max over all flights; dec = mean decisions flown; "
         "acc% = accel-cap binding share of physics steps)")
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "M2.4b: matched realizations across all profiles + per-profile PID baseline table",
        "source_run": repo_rel(run / "results.json"), "source_run_sha256_lf": rc.file_sha256(run / "results.json"),
        "protocol_file_sha256": prot["protocol_file_sha256"], "packages": meta["packages"],
        "isaac_sim": meta["isaac_sim"], "total_mass_kg": meta["total_mass_kg"],
        "our_profiles": prot["our_profiles"], "seeds": [first, last, n], "profiles": list(profiles),
        "consistency_check": cons, "table": table,
    }
    dest = REPO_DIR / "residual_ppo" / "results" / "m2"
    (dest / f"{args.name}.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    (dest / f"{args.name}.txt").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"written: residual_ppo/results/m2/{args.name}.json, .txt")
    if not cons["pass"]:
        raise SystemExit("consistency check failed")


def mode_compare(args):
    """M2.4: reference flights (and their realizations entries) must equal the candidate's bit-for-bit. No Isaac."""
    import protocol_stats as ps

    def load(d):
        d = Path(d)
        d = d if d.is_absolute() else (REPO_DIR / d)
        res = json.loads((d / "results.json").read_text(encoding="utf-8"))
        real = json.loads((d / "realizations.json").read_text(encoding="utf-8"))
        return d, res, real

    ref_dir, ref, ref_real = load(args.reference)
    cand_dir, cand, cand_real = load(args.candidate)
    rows = ps.subset_equality(ref["flights"], cand["flights"])
    real_missing = [k for k in ref_real["flights"] if k not in cand_real["flights"]]
    real_diff = {k: d for k, v in ref_real["flights"].items() if k in cand_real["flights"]
                 for d in [ps.row_differences(v, cand_real["flights"][k])] if d}
    real_ok = not real_missing and not real_diff and ref_real["key"] == cand_real["key"]
    ok = rows["pass"] and real_ok
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": args.purpose,
        "reference_run": repo_rel(ref_dir / "results.json"),
        "reference_sha256_lf": rc.file_sha256(ref_dir / "results.json"),
        "candidate_run": repo_rel(cand_dir / "results.json"),
        "candidate_sha256_lf": rc.file_sha256(cand_dir / "results.json"),
        "tolerance": "bit-exact (JSON values identical; NaN == NaN; types must match)",
        "protocol_file_sha256": [ref["meta"]["protocol"]["protocol_file_sha256"],
                                 cand["meta"]["protocol"]["protocol_file_sha256"]],
        "packages": [ref["meta"]["packages"], cand["meta"]["packages"]],
        "isaac_sim": [ref["meta"]["isaac_sim"], cand["meta"]["isaac_sim"]],
        "total_mass_kg": [ref["meta"]["total_mass_kg"], cand["meta"]["total_mass_kg"]],
        "pass": ok, "flights": rows,
        "realizations": {"pass": real_ok, "compared": len(ref_real["flights"]) - len(real_missing),
                         "missing_in_candidate": real_missing, "differences": real_diff,
                         "key": ref_real["key"]},
    }
    text = (f"{args.purpose}\n"
            f"flights: {rows['identical']}/{rows['reference_flights']} identical on all {len(rows['fields_compared'])} "
            f"fields; missing {len(rows['missing_in_candidate'])}; differing {len(rows['differences'])}\n"
            f"realizations entries: {result['realizations']['compared']} compared; missing {len(real_missing)}; "
            f"differing {len(real_diff)}\n"
            f"RESULT: {'PASS (bit-for-bit)' if ok else 'FAIL'}")
    if not ok:
        text += "\n" + json.dumps({"flights": rows["differences"], "realizations": real_diff,
                                    "missing": rows["missing_in_candidate"] + real_missing}, default=str)[:4000]
    print(text)
    dest = REPO_DIR / "residual_ppo" / "results" / "m2"
    (dest / f"{args.name}.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    (dest / f"{args.name}.txt").write_text(text + "\n", encoding="utf-8")
    print(f"written: residual_ppo/results/m2/{args.name}.json, .txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("fly")
    f.add_argument("--controller", default="pid")
    f.add_argument("--seeds", required=True, help="e.g. 300000-300009 or 300000,300001")
    f.add_argument("--profiles", default=None, help="comma list; default: all protocol profiles")
    f.add_argument("--scenarios", default=None, help="comma list; default: all protocol scenarios")
    f.add_argument("--out", required=True)
    f.add_argument("--check0", action="store_true")
    r = sub.add_parser("reproduce")
    r.add_argument("--run", required=True)
    v = sub.add_parser("still-vs-wind")
    v.add_argument("--run", required=True)
    v.add_argument("--name", default="still_vs_wind", help="output basename in results/m2/")
    b = sub.add_parser("baseline-table")
    b.add_argument("--run", required=True)
    b.add_argument("--name", required=True, help="output basename in results/m2/")
    c = sub.add_parser("compare")
    c.add_argument("--reference", required=True, help="run dir whose flights must all appear in --candidate")
    c.add_argument("--candidate", required=True)
    c.add_argument("--name", required=True, help="output basename in results/m2/")
    c.add_argument("--purpose", required=True)
    args = parser.parse_args()
    if args.mode == "fly":
        mode_fly(args)
    elif args.mode == "reproduce":
        mode_reproduce(args)
    elif args.mode == "compare":
        mode_compare(args)
    elif args.mode == "baseline-table":
        mode_baseline_table(args)
    else:
        mode_still_vs_wind(args)


if __name__ == "__main__":
    main()
