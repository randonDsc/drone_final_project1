"""M8 hold diagnostic for the DIRECT arm (Isaac; no model, no training). Read-only with respect to everything else:
it changes no env, config, threshold or test, and writes only results/m8/hold_diagnostic*.

    # Directory: <repo root>
    & "<Isaac Sim>\\python.bat" residual_ppo\\tools\\hold_diagnostic.py *> residual_ppo\\results\\m8\\hold_diagnostic.log

Why: check_direct_env.py H2/H3 failed (zero command, hover, still air: drift after 3 s up to 0.2118 m, final speed up
to 0.1387 m/s), but those episodes lasted only 5.4-8.0 s. This runs 30 s per flight to see what the velocity does.

Setup, both tests: DirectEnv, validation seeds 400000-400009, scenario hover, profile still_air (no wind: mean 0,
gust scale 0), reset exactly as the env does (same initial state as check_direct_env Part H). Then 300 decisions
(30.0 s) with a CONSTANT action. Route truncation and the failure rules are bypassed: both live only in
PathSacEnv.step, so this script drives the env's own DirectCore directly, exactly as PathSacEnv.step does
(core.apply_action, then core.step for each of the 12 physics substeps; path_sac_env.py:156-176), without reward,
completion or termination. The inner loop, physics, mixer and wind are the env's own. A flight is stopped early only
if the state is invalid, height < 0.3 m or tilt > 60 deg (recorded as stopped_reason).
  (a) zero:  action 0 (velocity command 0, yaw 0) throughout.
  (b) vx1:   action (1/3, 0, 0, 0) -> velocity command vx = 1.0 m/s (float32 1/3 x 3 = 1.0000000298 m/s; errors are
             computed against the actual command), vy = vz = 0, yaw 0, throughout. This is the function the arm has
             to perform. Expected (Inference, before the run): a proportional-only velocity loop settles with a small
             error from drag, about -0.018 N / (1.6186 kg x 1.2 1/s) = -0.009 m/s in vx.
Logged every decision (plus t = 0): per-axis position, velocity, roll, pitch, yaw, body angular rate, commanded tilt
(the inner loop's desired roll / pitch / yaw, core.desired), the inner-loop integrators (xy, height, attitude) and
saturated physics steps. CSV per test, summary JSON.

Reported per seed: speed |v| and velocity error |v - v_cmd| at 5, 10, 20, 30 s; the trend of |v - v_cmd| over the
last 10 s; peak-to-peak roll and pitch over the last 10 s; for (b) the steady-state velocity error (mean of v - v_cmd
over the last 10 s, per axis).

DECISION RULE (Claire, 2026-11-16, fixed BEFORE this diagnostic was run), for the zero-command test (a):
  1. zero-command speed decays below 0.02 m/s and is still falling -> the H2/H3 thresholds were wrong; replace them
     with "speed at 30 s < 0.02 m/s" and launch.
  2. speed settles roughly constant between 0.02 and 0.15 m/s -> steady bias; investigate before training.
  3. speed grows, or attitude diverges -> real defect; stop.
Operationalization (Claude, Proposed, fixed with the rule before the run; Claire may overrule it, not the numbers):
  - trend over the last 10 s (t = 20-30 s) = least-squares slope of the speed x 10 s: "falling" if < -0.005 m/s,
    "growing" if > +0.005 m/s, otherwise "flat".
  - rule 1: speed(30 s) < 0.02 m/s AND trend falling.
  - rule 2: trend flat AND mean speed over the last 10 s in [0.02, 0.15] m/s.
  - rule 3: trend growing, OR attitude diverges = stopped early, OR max(|roll|, |pitch|) over t = 20-30 s exceeds
    1.5 x its value over t = 10-20 s and exceeds 2 deg. Rule 3 is also applied to test (b) (velocity error trend).
  - anything else (e.g. below 0.02 m/s but flat) is reported as UNCLASSIFIED for Claire; no automatic verdict.
Prints per-seed lines, the rule counts and `RESULT: DONE` (this is a diagnostic: it decides nothing by itself).

---------------------------------------------------------------------------------------------------------------------
SUITE m8b (added 2026-11-16, after the v1 run; `--suite m8b`). The default `--suite v1` is the run above, unchanged.

    & "<Isaac Sim>\\python.bat" residual_ppo\\tools\\hold_diagnostic.py --suite m8b *> residual_ppo\\results\\m8\\hold_m8b.log

Question: is the 2.8 s mode a marginally stable linear mode (decays when nothing excites it) or self-sustained
(holds or grows with no excitation)? All tests: hover, still_air reset, constant action, route truncation and
failure rules bypassed as above.
  T1  DIRECT, zero command, wind OFF, 120 s, seeds 400000-400009.          <- the decisive test (Claire)
  T2  DIRECT, zero command, gusts ON, 120 s, seeds 400000, 400001, 400006.
  T3  RESIDUAL arm (PathSacEnv on the same sim, residual action exactly 0, so the PID flies its own hover), wind
      OFF, 30 s, seeds 400000-400009.
  T4  as T3 with gusts ON.
Wind definitions (Confirmed from the code before the run):
  - OFF = the still_air profile itself. Its reset draws calm wind (hover_task.py:223-224: speed, gust_scale,
    direction rate = 0), and GustingWind.reset_episode sets gust_std = 0 x base and local_gust_std = 0 x base
    (wind.py:196-197), so every point wind is exactly the zero mean for the whole flight. The "gust std=[1. 1. 0.35]"
    line in the logs is printed ONCE when the sim is built (drone_sim.py:301-314), from the base config, before any
    episode: it is NOT the episode's wind. So the v1 runs (and check_direct_env Part H) WERE in still air. The script
    also logs the actual point wind every decision (column wind_max) to prove it.
  - ON = after the still_air reset, sim.wind.reset_episode(seed=<the episode's wind seed>, mean_wind=0,
    gust_scale=1.0): the wind field's own runtime API, no code edit. That gives the base gusts Claire quoted: gust
    std (1, 1, 0.35) m/s with a 2.0 s timescale, local std (0.3, 0.3, 0.15) m/s with 0.3 s, mean wind 0.
Analysis (fixed before the run):
  - oscillation amplitude in a window = RMS of the horizontal velocity error (vx, vy minus the command), each axis
    linearly detrended inside the window; windows are 10 s long.
  - T1/T2 envelope: least-squares fit of log(amplitude) against window centre, windows from 10 s on; time constant
    tau = -1 / slope (tau > 0: decays; tau < 0: grows, |tau| = growth time).
  - classification by R = amplitude(110-120 s) / amplitude(20-30 s): decays if R < 0.8, holds if 0.8 <= R <= 1.25,
    grows if R > 1.25. T3/T4 (30 s): R = amplitude(20-30 s) / amplitude(10-20 s), same bands.
  - also |v| (instantaneous) and the window amplitude at 30 / 60 / 90 / 120 s, and the mean velocity over the last
    10 s (the steady bias, which does not decay with the oscillation).
  - tilt response at 1/2.8 Hz: least-squares sin/cos fit of actual and commanded pitch and roll over t >= 10 s;
    gain = |actual| / |commanded|, phase = angle(actual / commanded). Plus the dominant period of pitch from a
    zero-padded FFT over t >= 10 s.
Also printed once: the geometry the sim uses (rotor application positions, vehicle COM from PhysX, body COM, masses,
inertia if the API offers it), for the steady-bias question.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))
sys.path.insert(2, str(RESIDUAL_DIR / "training"))

import run_common as rc  # noqa: E402

OUT_DIR = RESIDUAL_DIR / "results" / "m8"
DECISIONS = 300                     # 30.0 s at 10 Hz
REPORT_TIMES = (5.0, 10.0, 20.0, 30.0)
TREND_EPS = 0.005                   # m/s change over the last 10 s
RULE1_SPEED = 0.02
RULE2_RANGE = (0.02, 0.15)
DIVERGE_FACTOR, DIVERGE_MIN_DEG = 1.5, 2.0
MIN_HEIGHT, MAX_TILT_DEG = 0.3, 60.0
TESTS = {"zero": np.zeros(4, dtype=np.float32),
         "vx1": np.array([1.0 / 3.0, 0.0, 0.0, 0.0], dtype=np.float32)}


def say(msg=""):
    print(f"[M8.hold] {msg}", flush=True)


def fly(env, sim, action, seed, protocol, decisions=DECISIONS, gusts=False):
    import hover_task
    from direct_control.direct_env import action_to_command
    from path_sac_policy import euler_angles
    from wind import quaternion_to_matrix

    _, info = env.reset(seed=seed, options=protocol.reset_options("hover", protocol.STILL_AIR_NAME))
    if gusts:   # m8b "gusts ON": the wind field's own runtime API (no code edit); mean wind stays 0
        sim.wind.reset_episode(seed=info["wind_seed"], mean_wind=np.zeros(3), gust_scale=1.0)
    core = env.core
    direct = hasattr(core.pid, "velocity_command")
    v_cmd, yaw_cmd = action_to_command(action) if direct else (np.zeros(3), 0.0)
    rows, stopped = [], None

    def record(k, saturated):
        st = sim.read_drone_state()
        R = quaternion_to_matrix(st["orientation"])
        att = np.degrees(euler_angles(R))
        rate = R.T @ np.asarray(st["angular_velocity"], dtype=float)
        pid = core.pid.pid
        v = np.asarray(st["linear_velocity"], dtype=float)
        rows.append({
            "seed": seed, "decision": k, "t": round(k * 0.1, 10),
            **{f"p{a}": float(x) for a, x in zip("xyz", np.asarray(st["position"], dtype=float))},
            **{f"v{a}": float(x) for a, x in zip("xyz", v)},
            **{f"cmd_v{a}": float(x) for a, x in zip("xyz", v_cmd)},
            "roll_deg": float(att[0]), "pitch_deg": float(att[1]), "yaw_deg": float(att[2]),
            **{f"rate_{a}": float(x) for a, x in zip("xyz", rate)},
            "cmd_roll_deg": float(np.degrees(core.desired[0])), "cmd_pitch_deg": float(np.degrees(core.desired[1])),
            "cmd_yaw_deg": float(np.degrees(core.desired[2])),
            "int_x": float(pid.xy_integral[0]), "int_y": float(pid.xy_integral[1]),
            "int_height": float(pid.height_integral),
            **{f"int_att_{a}": float(x) for a, x in zip(("roll", "pitch", "yaw"), pid.attitude_integral)},
            "saturated_steps": int(saturated),
            "speed": float(np.linalg.norm(v)), "vel_error": float(np.linalg.norm(v - v_cmd)),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))),
            "wind_max": float(np.max(np.linalg.norm(np.asarray(sim.wind.point_winds, dtype=float).reshape(-1, 3),
                                                    axis=1))),
        })

    record(0, 0)
    truth = sim.read_drone_state()
    for k in range(1, decisions + 1):
        if not sim.is_running():
            stopped = "simulation not running"
            break
        core.apply_action(action)
        saturated = 0
        for _ in range(core.decimation):
            if not hover_task.state_is_valid(truth):
                break
            saturated += int(core.step(truth))
            truth = sim.read_drone_state()
        record(k, saturated)
        last = rows[-1]
        if not hover_task.state_is_valid(truth):
            stopped = "invalid state"
        elif last["pz"] < MIN_HEIGHT:
            stopped = f"height {last['pz']:.3f} m < {MIN_HEIGHT}"
        elif last["tilt_deg"] > MAX_TILT_DEG:
            stopped = f"tilt {last['tilt_deg']:.1f} deg > {MAX_TILT_DEG}"
        if stopped:
            break
    return rows, stopped, (v_cmd, yaw_cmd)


def trend(t, y):
    if len(t) < 2:
        return float("nan"), "n/a"
    change = float(np.polyfit(t, y, 1)[0]) * 10.0
    return change, ("falling" if change < -TREND_EPS else "growing" if change > TREND_EPS else "flat")


def summarize(test, rows, stopped, command):
    t = np.array([r["t"] for r in rows])
    err = np.array([r["vel_error"] for r in rows])
    speed = np.array([r["speed"] for r in rows])
    roll, pitch = np.array([r["roll_deg"] for r in rows]), np.array([r["pitch_deg"] for r in rows])
    last = t >= 20.0 - 1e-9
    mid = (t >= 10.0 - 1e-9) & (t < 20.0 - 1e-9)
    at = {f"{s:g}s": None for s in REPORT_TIMES}
    for s in REPORT_TIMES:
        idx = np.where(np.isclose(t, s))[0]
        if idx.size:
            at[f"{s:g}s"] = {"speed": float(speed[idx[0]]), "vel_error": float(err[idx[0]])}
    change, label = trend(t[last], err[last])
    tilt_last = float(np.max(np.maximum(np.abs(roll[last]), np.abs(pitch[last])))) if last.any() else float("nan")
    tilt_mid = float(np.max(np.maximum(np.abs(roll[mid]), np.abs(pitch[mid])))) if mid.any() else float("nan")
    diverges = bool(stopped) or (np.isfinite(tilt_last) and np.isfinite(tilt_mid)
                                 and tilt_last > DIVERGE_FACTOR * tilt_mid and tilt_last > DIVERGE_MIN_DEG)
    v_cmd = command[0]
    vel = np.array([[r["vx"], r["vy"], r["vz"]] for r in rows])
    steady = (vel[last] - v_cmd).mean(axis=0) if last.any() else np.full(3, np.nan)
    out = {
        "test": test, "seed": rows[0]["seed"], "decisions": len(rows) - 1, "stopped_reason": stopped,
        "command_velocity": [float(x) for x in v_cmd], "command_yaw": command[1], "at": at,
        "last10_trend_change_mps": change, "last10_trend": label,
        "last10_mean_speed": float(speed[last].mean()) if last.any() else None,
        "last10_mean_vel_error": float(err[last].mean()) if last.any() else None,
        "last10_roll_p2p_deg": float(np.ptp(roll[last])) if last.any() else None,
        "last10_pitch_p2p_deg": float(np.ptp(pitch[last])) if last.any() else None,
        "max_abs_roll_pitch_deg_10_20s": tilt_mid, "max_abs_roll_pitch_deg_20_30s": tilt_last,
        "attitude_diverges": diverges,
        "steady_state_velocity_error_mps": [float(x) for x in steady],
        "max_abs_integrators": {k: float(max(abs(r[k]) for r in rows)) for k in
                                ("int_x", "int_y", "int_height", "int_att_roll", "int_att_pitch", "int_att_yaw")},
        "saturated_steps": int(sum(r["saturated_steps"] for r in rows)),
        "max_speed_jump_per_decision": float(np.max(np.linalg.norm(np.diff(vel, axis=0), axis=1))) if len(vel) > 1
        else 0.0,
        "min_height": float(min(r["pz"] for r in rows)),
        "travel_xy_m": float(np.hypot(rows[-1]["px"] - rows[0]["px"], rows[-1]["py"] - rows[0]["py"])),
    }
    if label == "growing" or diverges:
        rule = "3 (grows or attitude diverges: real defect)"
    elif test != "zero":
        rule = "n/a (rules 1-2 are for the zero test; rule 3 not triggered)"
    elif at["30s"] is not None and at["30s"]["speed"] < RULE1_SPEED and label == "falling":
        rule = "1 (decays below 0.02 m/s and still falling)"
    elif label == "flat" and RULE2_RANGE[0] <= out["last10_mean_speed"] <= RULE2_RANGE[1]:
        rule = "2 (roughly constant 0.02-0.15 m/s: steady bias)"
    else:
        rule = "UNCLASSIFIED (for Claire)"
    out["rule"] = rule
    return out


def fmt(x, nd=4):
    return "  n/a " if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------- suite m8b analysis

M8B_PERIOD = 2.8
BAND_DECAY, BAND_GROW = 0.8, 1.25
M8B_TESTS = (   # name, arm, gusts, seconds, seeds (None = all validation seeds)
    ("T1", "direct", False, 120.0, None),
    ("T2", "direct", True, 120.0, (400_000, 400_001, 400_006)),
    ("T3", "residual", False, 30.0, None),
    ("T4", "residual", True, 30.0, None),
)


def _arrays(rows):
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0] if k != "seed"}


def window_amplitude(a, lo, hi):
    """RMS of the horizontal velocity error in [lo, hi), each axis linearly detrended (None if < 3 samples)."""
    t = a["t"]
    m = (t >= lo - 1e-9) & (t < hi - 1e-9) if hi < t[-1] + 1e-9 else (t >= lo - 1e-9)
    if m.sum() < 3:
        return None
    out = 0.0
    for e in (a["vx"] - a["cmd_vx"], a["vy"] - a["cmd_vy"]):
        x, y = t[m], e[m]
        out += float(np.mean((y - np.polyval(np.polyfit(x, y, 1), x)) ** 2))
    return float(np.sqrt(out))


def tilt_response(a, f0=1.0 / M8B_PERIOD, start=10.0):
    t = a["t"]
    m = t >= start - 1e-9
    A = np.column_stack([np.cos(2 * np.pi * f0 * t[m]), np.sin(2 * np.pi * f0 * t[m]), np.ones(m.sum()), t[m]])

    def comp(y):
        c = np.linalg.lstsq(A, y[m], rcond=None)[0]
        return c[0] - 1j * c[1]

    out = {}
    for axis in ("pitch", "roll"):
        act, cmd = comp(a[f"{axis}_deg"]), comp(a[f"cmd_{axis}_deg"])
        out[axis] = {"gain": float(abs(act) / abs(cmd)) if abs(cmd) > 0 else None,
                     "phase_deg": float(np.degrees(np.angle(act / cmd))) if abs(cmd) > 0 else None,
                     "actual_amp_deg": float(abs(act)), "commanded_amp_deg": float(abs(cmd))}
    y = a["pitch_deg"][m] - np.polyval(np.polyfit(t[m], a["pitch_deg"][m], 1), t[m])
    n = 8192
    f = np.fft.rfftfreq(n, t[1] - t[0])
    p = np.abs(np.fft.rfft(y * np.hanning(len(y)), n))
    i = 1 + int(np.argmax(p[1:]))
    out["dominant_pitch_period_s"] = float(1.0 / f[i])
    return out


def summarize_m8b(name, rows, stopped, seconds):
    a = _arrays(rows)
    t = a["t"]
    windows = [(lo, lo + 10.0) for lo in np.arange(0.0, seconds - 1e-9, 10.0)]
    amps = [(lo, hi, window_amplitude(a, lo, hi)) for lo, hi in windows]
    out = {"test": name, "seed": rows[0]["seed"], "decisions": len(rows) - 1, "stopped_reason": stopped,
           "window_amplitude_mps": {f"{lo:g}-{hi:g}": amp for lo, hi, amp in amps},
           "max_wind_mps": float(a["wind_max"].max()), "saturated_steps": int(a["saturated_steps"].sum()),
           "max_abs_integrators": {k: float(np.max(np.abs(a[k]))) for k in ("int_x", "int_y", "int_height",
                                                                           "int_att_roll", "int_att_pitch",
                                                                           "int_att_yaw")},
           "max_tilt_deg": float(a["tilt_deg"].max())}
    out["speed_at"] = {}
    for s in (30.0, 60.0, 90.0, 120.0):
        idx = np.where(np.isclose(t, s))[0]
        if idx.size:
            out["speed_at"][f"{s:g}s"] = {"speed": float(a["speed"][idx[0]]),
                                          "window_amplitude": window_amplitude(a, s - 10.0, s)}
    last = t >= t[-1] - 10.0 + 1e-9
    out["last10_mean_velocity_error"] = [float((a[f"v{c}"] - a[f"cmd_v{c}"])[last].mean()) for c in "xyz"]
    fit = [(0.5 * (lo + hi), amp) for lo, hi, amp in amps if lo >= 10.0 and amp]
    if len(fit) >= 2:
        slope = float(np.polyfit([c for c, _ in fit], np.log([v for _, v in fit]), 1)[0])
        out["envelope_slope_per_s"] = slope
        out["envelope_tau_s"] = float(-1.0 / slope) if slope != 0 else float("inf")
    else:
        out["envelope_slope_per_s"] = out["envelope_tau_s"] = None
    if seconds >= 120.0:
        early, late = window_amplitude(a, 20.0, 30.0), window_amplitude(a, 110.0, 120.0 + 1e-6)
    else:
        early, late = window_amplitude(a, 10.0, 20.0), window_amplitude(a, 20.0, seconds + 1e-6)
    ratio = late / early if early and late is not None else None
    out["amplitude_ratio"] = ratio
    out["verdict"] = (None if ratio is None else "decays" if ratio < BAND_DECAY else
                      "grows" if ratio > BAND_GROW else "holds")
    out["tilt_response"] = tilt_response(a)
    return out


def geometry(sim):
    """What the sim uses for the steady-bias question (read-only)."""
    from drone_diagnostics import read_total_com_in_body_frame

    g = {"rotor_positions_m": np.asarray(sim.rotor_positions, dtype=float).tolist(),
         "vehicle_com_from_body_origin_m": np.asarray(read_total_com_in_body_frame(sim.body, sim.vehicle_bodies),
                                                      dtype=float).tolist(),
         "body_com_local_m": np.asarray(sim.body_com, dtype=float).tolist(),
         "masses_kg": np.asarray(sim.vehicle_bodies.get_masses(), dtype=float).reshape(-1).tolist(),
         "total_mass_kg": float(sim.total_mass)}
    for owner, view in (("body", sim.body), ("vehicle_bodies", sim.vehicle_bodies)):
        try:
            g[f"{owner}_inertias"] = np.asarray(view.get_inertias(), dtype=float).tolist()
        except Exception as exc:   # informational only
            g[f"{owner}_inertias"] = f"unavailable: {type(exc).__name__}: {exc}"
    return g


def run_m8b(direct, shared, sim, protocol):
    record = {"suite": "m8b", "tests": {}, "geometry": geometry(sim),
              "bands": {"decays_below": BAND_DECAY, "grows_above": BAND_GROW}, "period_s": M8B_PERIOD}
    say(f"geometry: {json.dumps(record['geometry'], default=str)}")
    zero = np.zeros(4, dtype=np.float32)
    for name, arm, gusts, seconds, seeds in M8B_TESTS:
        env = direct if arm == "direct" else shared
        seeds = seeds or tuple(protocol.VALIDATION_SEEDS)
        decisions = int(round(seconds * 10))
        say(f"--- {name}: {arm} arm, zero {'command' if arm == 'direct' else 'residual'}, gusts "
            f"{'ON' if gusts else 'OFF'}, {seconds:g} s, seeds {seeds[0]}..{seeds[-1]} ({len(seeds)})")
        all_rows, summaries = [], []
        for seed in seeds:
            rows, stopped, _ = fly(env, sim, zero, seed, protocol, decisions=decisions, gusts=gusts)
            all_rows += rows
            s = summarize_m8b(name, rows, stopped, seconds)
            summaries.append(s)
            tr = s["tilt_response"]
            sp = "/".join(f"{v['speed']:.4f}" for v in s["speed_at"].values())
            say(f"{name} seed {seed}: verdict {s['verdict']} (ratio {fmt(s['amplitude_ratio'], 3)}); envelope tau "
                f"{fmt(s['envelope_tau_s'], 1)} s; |v| at {'/'.join(s['speed_at'])} = {sp} m/s; window amplitude "
                + "/".join(fmt(v['window_amplitude']) for v in s["speed_at"].values())
                + f"; pitch gain {fmt(tr['pitch']['gain'], 2)} phase {fmt(tr['pitch']['phase_deg'], 0)} deg, roll gain "
                f"{fmt(tr['roll']['gain'], 2)} phase {fmt(tr['roll']['phase_deg'], 0)} deg; dominant pitch period "
                f"{tr['dominant_pitch_period_s']:.2f} s; mean v err last 10 s "
                f"[{', '.join(f'{x:+.4f}' for x in s['last10_mean_velocity_error'])}]; max wind "
                f"{s['max_wind_mps']:.3f} m/s; saturated {s['saturated_steps']}; stopped {stopped}")
        with open(OUT_DIR / f"hold_m8b_{name}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        counts = {}
        for s in summaries:
            counts[s["verdict"]] = counts.get(s["verdict"], 0) + 1
        say(f"{name}: verdict counts {counts}")
        record["tests"][name] = {"arm": arm, "gusts": gusts, "seconds": seconds, "seeds": summaries,
                                 "verdict_counts": counts}
    (OUT_DIR / "hold_m8b.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    say("written: residual_ppo/results/m8/hold_m8b.json, hold_m8b_T1.csv .. hold_m8b_T4.csv")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("v1", "m8b"), default="v1")
    args = parser.parse_args()
    check_line = rc.preflight(say)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import eval_protocol_v1 as protocol
        from direct_control.direct_env import make_direct_env

        if protocol.protocol_record()["protocol_file_sha256"] != rc.EXPECTED_PROTOCOL_SHA:
            raise SystemExit("protocol hash differs from the frozen one")
        protocol.register_profiles()
        env = make_direct_env(app, log=say)
        outside = rc.loaded_outside_residual()
        if outside:
            raise SystemExit(f"shared modules loaded outside residual_ppo: {outside}")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if args.suite == "m8b":
            from path_sac_env import PathSacEnv

            shared = PathSacEnv(env.sim, log=say)   # residual arm on the same sim (as in check_direct_env.py)
            run_m8b(env, shared, env.sim, protocol)
            say("RESULT: DONE (diagnostic only; the decision is Claire's)")
            return
        record = {"checks": check_line, "total_mass_kg": float(env.sim.total_mass), "decisions": DECISIONS,
                  "seeds": list(protocol.VALIDATION_SEEDS), "rule_thresholds": {
                      "trend_eps_mps": TREND_EPS, "rule1_speed": RULE1_SPEED, "rule2_range": RULE2_RANGE,
                      "diverge_factor": DIVERGE_FACTOR, "diverge_min_deg": DIVERGE_MIN_DEG}, "tests": {}}
        for test, action in TESTS.items():
            all_rows, summaries = [], []
            say(f"--- test {test}: action {action.tolist()}")
            for seed in protocol.VALIDATION_SEEDS:
                rows, stopped, command = fly(env, env.sim, action, seed, protocol)
                all_rows += rows
                s = summarize(test, rows, stopped, command)
                summaries.append(s)
                a = s["at"]
                ss = s["steady_state_velocity_error_mps"]
                say(f"{test} seed {seed}: |v| 5/10/20/30 s = "
                    + "/".join(fmt(None if a[k] is None else a[k]["speed"]) for k in ("5s", "10s", "20s", "30s"))
                    + " m/s; |v-cmd| 30 s " + fmt(None if a["30s"] is None else a["30s"]["vel_error"])
                    + f"; last 10 s trend {s['last10_trend']} ({s['last10_trend_change_mps']:+.4f} m/s); roll/pitch "
                    f"p2p {fmt(s['last10_roll_p2p_deg'], 3)}/{fmt(s['last10_pitch_p2p_deg'], 3)} deg; steady error "
                    f"[{ss[0]:+.4f}, {ss[1]:+.4f}, {ss[2]:+.4f}] m/s; stopped {stopped}; rule {s['rule']}")
            with open(OUT_DIR / f"hold_diagnostic_{test}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
                w.writeheader()
                w.writerows(all_rows)
            counts = {}
            for s in summaries:
                counts[s["rule"]] = counts.get(s["rule"], 0) + 1
            say(f"{test}: rule counts {counts}")
            record["tests"][test] = {"action": action.tolist(), "seeds": summaries, "rule_counts": counts}
        (OUT_DIR / "hold_diagnostic.json").write_text(json.dumps(record, indent=2, default=str) + "\n",
                                                       encoding="utf-8")
        say("written: residual_ppo/results/m8/hold_diagnostic.json, hold_diagnostic_zero.csv, hold_diagnostic_vx1.csv")
        say("RESULT: DONE (diagnostic only; the decision is Claire's, by the rule in the docstring)")
    except BaseException:
        import traceback
        traceback.print_exc()
        say("RESULT: FAIL (exception)")
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
