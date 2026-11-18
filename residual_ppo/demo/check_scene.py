r"""M9 Step 1: in-scene checks for one demo scene. DESCRIPTIVE DEMO, NOT eval-v1.

Run with Isaac python.bat (short, headless, no model):
    # Directory: <repo root>\residual_ppo
    & <isaac>\python.bat demo\check_scene.py --scene indoor
    & <isaac>\python.bat demo\check_scene.py --scene forest

Checks (all must pass for RESULT: PASS):
  S1 total mass == 1.618614 kg (M1.5 value, 1e-6 kg), HESAI contribution 0
  S2 physics dt == 1/120 and decimation == 12 (10 Hz decisions)
  S3 the obstacle root is active and only /World/Cube was deactivated; exactly one PhysicsScene
  S4 the short check track keeps >= 2 m clearance (exact shapes)
  S5/S6 zero correction == pure PID loop, BIT-EXACT, on that track in calm air (S5) and at 7 m/s with gusts (S6):
     A = DemoResidualEnv with action 0 and 60 Hz recording on; B = the same demo reset (observe stubbed), then
     PathSacEnv.step + PathCore.step mirrored without the residual (M2.2 method, zero_residual_check.run_pid_path).
     Every physics step's pre-state, PID integrators, wrench, rotor speeds, saturation, wind force/torque and mean
     wind are compared with np.array_equal, plus decisions, end reason, completion and the final state.
Also written: the obstacle shapes (results/m9_demo/geometry_<scene>.json) and the decision rate of the A flights.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))

import numpy as np  # noqa: E402

from evaluation import run_common  # noqa: E402  (frozen helper; imported, not edited)

EXPECTED_MASS = 1.618614
CHECK_SEED = {"calm": 600900, "w7": 600901}
CHECK_TRACKS = {   # short, straight, inside the >= 2 m free space (verified as S4)
    "indoor": [(-4.75, -6.0, 2.0), (-4.75, -2.0, 2.0)],
    "forest": [(-6.0, -8.0, 2.0), (-6.0, -4.0, 2.0)],
}
ROW_WIDTH = 13 + 1 + 2 + 3 + 4 + 4 + 1 + 3 + 3 + 3


def say(msg=""):
    print(f"[M9.check] {msg}", flush=True)


class StepCommandRecorder:
    """Wraps sim.step_command on the instance (as zero_residual_check.Recorder); one row per physics step."""

    def __init__(self, sim, pid):
        self.rows = None
        original = sim.step_command

        def recorded(state, total_thrust, roll_torque, pitch_torque, yaw_torque, render=True):
            pre = None
            if self.rows is not None:
                pre = np.concatenate([
                    np.asarray(state["position"], dtype=np.float64).reshape(3),
                    np.asarray(state["orientation"], dtype=np.float64).reshape(4),
                    np.asarray(state["linear_velocity"], dtype=np.float64).reshape(3),
                    np.asarray(state["angular_velocity"], dtype=np.float64).reshape(3),
                    [pid().height_integral], np.asarray(pid().xy_integral, dtype=np.float64).reshape(2),
                    np.asarray(pid().attitude_integral, dtype=np.float64).reshape(3),
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


def run_env_path(env, rec, seed, options):
    """A: the demo env, action 0, recording on. Returns the episode record and env.step wall time."""
    _, info = env.reset(seed=seed, options=options)
    env.core.start_recording()
    rec.start()
    zero = np.zeros(4, dtype=np.float32)
    decisions, terminated, truncated, reason = 0, False, False, None
    t0 = time.perf_counter()
    while True:
        _, _, terminated, truncated, step_info = env.step(zero)
        decisions += 1
        reason = step_info["failure_reason"]
        if terminated or truncated:
            break
    wall = time.perf_counter() - t0
    rows = rec.stop()
    rec60 = env.core.stop_recording()
    return {"rows": rows, "rec60": rec60, "decisions": decisions, "terminated": terminated, "truncated": truncated,
            "reason": reason, "completed": bool(env.core.completion.completed),
            "final": state_vector(env.sim.read_drone_state()), "wall_s": wall,
            "max_decisions": int(env.max_decisions), "info": info}


def run_pid_path(env, rec, seed, options):
    """B: the same demo reset (observe stubbed: PID not read), then the PID loop with no residual (M2.2 method)."""
    import hover_task
    import path_sac_config
    from hover_env import SimulationClosed

    core, sim = env.core, env.sim
    core.observe = lambda state: None
    try:
        _, info = env.reset(seed=seed, options=options)
    finally:
        del core.observe
    pid, reference, direction_rate = core.pid, core.reference, core.direction_rate
    t, saturated = 0.0, False
    decisions, terminated, truncated, reason = 0, False, False, None
    rec.start()
    while True:
        if not sim.is_running():
            raise SimulationClosed("Simulation closed during an episode.")
        truth = sim.read_drone_state()
        for _ in range(core.decimation):
            if not hover_task.state_is_valid(truth):
                break
            if direction_rate != 0.0:
                angle = direction_rate * sim.dt
                c, s = np.cos(angle), np.sin(angle)
                x, y, z = sim.wind.mean_wind
                sim.wind.mean_wind = np.array([c*x - s*y, s*x + c*y, z])
            sample = reference.sample(t)
            command = pid.read(truth, sample, saturated)
            desired = pid.desired_attitude(truth, command)
            pid.integrate(truth, sample, sim.dt, saturated or pid.acceleration_capped(truth, command, desired))
            _, sat, _, _ = sim.step_command(truth, *command, render=False)
            saturated = bool(sat)
            t += sim.dt
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
    return {"rows": rows, "decisions": decisions, "terminated": terminated, "truncated": truncated,
            "reason": reason, "completed": bool(core.completion.completed),
            "final": state_vector(sim.read_drone_state()), "info": info}


def compare(a, b):
    same_rows = a["rows"].shape == b["rows"].shape and np.array_equal(a["rows"], b["rows"], equal_nan=True)
    first = None
    if not same_rows:
        n = min(len(a["rows"]), len(b["rows"]))
        diff = np.where(np.any(a["rows"][:n] != b["rows"][:n], axis=1))[0]
        first = int(diff[0]) if diff.size else n
    fields = {k: a[k] == b[k] for k in ("decisions", "terminated", "truncated", "reason", "completed")}
    same_final = np.array_equal(a["final"], b["final"])
    return {"physics_steps": [int(len(a["rows"])), int(len(b["rows"]))], "rows_bit_identical": bool(same_rows),
            "first_differing_step": first, "fields_equal": {k: bool(v) for k, v in fields.items()},
            "final_state_bit_identical": bool(same_final),
            "pass": bool(same_rows and all(fields.values()) and same_final)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", choices=["indoor", "forest"], required=True)
    ap.add_argument("--out-dir", default="results/m9_demo")
    args = ap.parse_args()
    out_dir = RESIDUAL_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    say(f"scene {args.scene}; descriptive demo, not eval-v1")
    checks_line = run_common.preflight(say)

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        record = isaac_main(app, args, out_dir, checks_line)
    finally:
        app.close()
    return record


def isaac_main(app, args, out_dir, checks_line):
    import path_sac_config
    from path_reference import TimedReference
    from pxr import UsdPhysics

    from demo import demo_config
    from demo.demo_env import DemoResidualEnv, make_demo_sim
    from demo.geometry import Obstacles
    from demo.scene_geometry import export_obstacles

    demo_config.register_demo_profiles()
    spec = demo_config.SCENES[args.scene]
    sim = make_demo_sim(app, args.scene, log=say)
    outside = run_common.loaded_outside_residual()
    if outside:
        raise SystemExit(f"shared modules loaded from outside residual_ppo: {outside}")

    checks = []

    def check(name, ok, detail):
        checks.append({"id": name, "pass": bool(ok), "detail": detail})
        say(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    # ---- S1-S3: build facts ----
    hesai = run_common.hesai_report(sim)
    check("S1 mass", abs(sim.total_mass - EXPECTED_MASS) < 1e-6 and hesai["hesai_mass_contribution_kg"] == 0,
          f"total_mass_kg {sim.total_mass:.7f} (expected {EXPECTED_MASS}); HESAI {hesai['hesai_mass_contribution_kg']} kg")
    env = DemoResidualEnv(sim, log=say)
    decimation = env.core.decimation
    check("S2 dt/decimation", abs(sim.dt - 1 / 120) < 1e-12 and decimation == 12,
          f"dt {sim.dt!r} s (1/120 = {1/120!r}), decimation {decimation}, decision rate {1/(sim.dt*decimation):.3f} Hz")
    phys = [str(p.GetPath()) for p in sim.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    cube = sim.stage.GetPrimAtPath("/World/Cube")
    root = sim.stage.GetPrimAtPath(spec["obstacle_root"])
    check("S3 scene", root.IsActive() and not cube.IsActive() and len(phys) == 1,
          f"{spec['obstacle_root']} active={root.IsActive()}, /World/Cube active={cube.IsActive()}, physics scenes {phys}")

    # ---- geometry export ----
    shapes = export_obstacles(sim.stage, spec["obstacle_root"])
    geometry = {"label": demo_config.LABEL, "scene": args.scene, "usd": spec["usd"],
                "obstacle_root": spec["obstacle_root"], "count": len(shapes),
                "kinds": {k: sum(s["kind"] == k for s in shapes) for k in ("box", "cylinder", "sphere")},
                "rigid_bodies": sum(s["rigid_body"] for s in shapes), "shapes": shapes}
    geo_path = out_dir / f"geometry_{args.scene}.json"
    geo_path.write_text(json.dumps(geometry, separators=(",", ":")), encoding="utf-8")
    say(f"geometry: {len(shapes)} shapes {geometry['kinds']}, rigid bodies {geometry['rigid_bodies']} -> {geo_path.name}")
    obstacles = Obstacles(shapes)

    # ---- S4: the short check track ----
    L = demo_config.TRACK_LIMITS
    reference = TimedReference(CHECK_TRACKS[args.scene], L["speed"], L["acceleration"], L["lateral_acceleration"],
                               L["vertical_speed"], start_hold=L["start_hold"], end_hold=L["end_hold"],
                               spacing=L["spacing"])
    clear, _ = obstacles.clearance(reference.path)
    check("S4 check-track clearance", clear.min() >= 2.0,
          f"min clearance {clear.min():.3f} m over {len(reference.path)} points; duration {reference.duration:.2f} s")

    # ---- S5/S6: zero correction == PID, bit-exact ----
    rec = StepCommandRecorder(sim, lambda: env.core.pid.pid)
    direction = np.array([1.0, 0.0])                     # crosswind to the check track (which runs along +y)
    flights = {}
    for sid, level in (("S5", "calm"), ("S6", "w7")):
        seed = demo_config.assert_demo_seed(CHECK_SEED[level])
        options = {"reference": reference, "wind": demo_config.demo_wind(level, direction), "track_name": "check"}
        a = run_env_path(env, rec, seed, options)
        b = run_pid_path(env, rec, seed, options)
        cmp = compare(a, b)
        max_wind = float(np.max(np.linalg.norm(a["rows"][:, -3:], axis=1)))
        rate = a["decisions"] / a["wall_s"]
        check(f"{sid} zero correction == PID ({level})", cmp["pass"],
              f"{cmp['physics_steps'][0]} physics steps, rows bit-identical {cmp['rows_bit_identical']}, "
              f"final state {cmp['final_state_bit_identical']}, fields {cmp['fields_equal']}; "
              f"{a['decisions']} decisions, end {a['reason'] or 'truncated'}, mean wind {max_wind:.3f} m/s; "
              f"60 Hz rows {len(a['rec60'])}; rate {rate:.1f} decisions/s")
        flights[level] = {"seed": seed, "comparison": cmp, "decisions": a["decisions"], "end": a["reason"],
                          "completed": a["completed"], "env_step_wall_s": a["wall_s"],
                          "decisions_per_s": rate, "rows_60hz": int(len(a["rec60"])), "mean_wind_mps": max_wind,
                          "rows_sha256_A": hashlib.sha256(a["rows"].tobytes()).hexdigest(),
                          "rows_sha256_B": hashlib.sha256(b["rows"].tobytes()).hexdigest()}

    import gymnasium  # torch / SB3 deliberately NOT imported here: after SimulationApp they could resolve to the stale
    #                   ml_archive torch (ENV-1); this check loads no model.
    record = {
        "label": demo_config.LABEL, "scene": args.scene, "usd": spec["usd"],
        "deactivated": list(spec["deactivate"]), "checks_line": checks_line, "git": run_common.git_info(),
        "isaac_sim": run_common.isaac_version(), "hardware": run_common.hardware(),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "gymnasium": gymnasium.__version__,
                     "numpy_file": run_common.redact_isaac(np.__file__)},
        "total_mass_kg": sim.total_mass, "hesai": hesai, "physics_dt": sim.dt, "decimation": decimation,
        "physics_scenes": phys, "check_track": {"points": CHECK_TRACKS[args.scene],
                                                "duration_s": reference.duration,
                                                "min_clearance_m": float(clear.min())},
        "geometry_file": geo_path.name, "geometry_sha256": run_common.file_sha256(geo_path),
        "flights": flights, "checks": checks,
        "passed": sum(c["pass"] for c in checks), "checks_total": len(checks),
    }
    record["result"] = "PASS" if record["passed"] == record["checks_total"] else "FAIL"
    out = out_dir / f"step1_{args.scene}.json"
    out.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    say(f"written: {out.relative_to(RESIDUAL_DIR).as_posix()}")
    say(f"{record['passed']}/{record['checks_total']} checks passed")
    say(f"RESULT: {record['result']}")
    return record


if __name__ == "__main__":
    main()
