"""M8 Isaac checks for the direct-control env, BEFORE any direct training (plan sections 2e, 4; prespec A5). No model.

    # Directory: <repo root>
    & "<Isaac Sim>\\python.bat" residual_ppo\\tools\\check_direct_env.py *> residual_ppo\\results\\m8\\check_direct_env.log

Part R, realizations (D-env-5). One sim, two envs on it: the shared PathSacEnv and DirectEnv. For every cell of the
validation seeds 400000-400009 (6 scenarios x 6 profiles = 360 resets; reset only, nothing is flown):
  R1  PathSacEnv reproduces all 5 key fields of the PID validation reference (results/eval/eval-v1/pid_validation/
      realizations.json), through the frozen run_protocol realization guard: the check itself is sound.
  R2  DirectEnv reproduces the 4 non-observation key fields (wind_seed, wind_speed, route_seconds, max_decisions)
      through the same guard, given reference rows cut down to those 4 fields.
      WHY initial_obs_sha256 is excluded: features 18-20 and 24-27 of the observation are controller-derived (inner
      loop under the policy's command vs the PID's lead-point command), and no shadow PID is allowed (plan 3b), so
      the hash cannot match between arms. Physical identity is proven instead by R3 (simulator state and mean wind
      bit-identical) and R4 (every controller-independent feature bit-identical).
      Fix 2026-11-16 (after the first run, 01:04): the first version passed the FULL stored reference rows (5
      fields) with key = 4 fields. The frozen row_differences (protocol_stats.py:345-355) counts a field missing on
      one side as a difference, so every cell failed for lacking initial_obs_sha256 (0/360). The rows are now cut to
      the 4 compared fields before they reach the guard. Nothing else changed.
  R3  DirectEnv's drone state after reset is bit-identical to PathSacEnv's (position, orientation, velocities, rotor
      speeds: every value read_drone_state() returns), and so is the wind field's mean wind.
  R4  DirectEnv's initial observation equals PathSacEnv's bit-for-bit in every controller-independent feature
      (per frame 0-17, 21-23, 28-31); only features 18-20 / 24-27 (inner loop vs PID) may differ.
  R5  negative test: DirectEnv reset with seed + 1 on one cell must be REFUSED by the guard. R5 only means something
      when R2 passes: in the first run the broken R2 guard refused EVERY reset, so its R5 PASS was vacuous.
Part H, zero velocity command (gate 4, Isaac half). DirectEnv, scenario hover, profile still_air (no wind, no gusts:
the still_air reset sets gust scale 0, wind.py:196-197), validation seeds 400000-400009, action 0.
  H1  episode flown through PathSacEnv.step (failure rules on, route truncation, 54-80 decisions): no failure.
  H2  limit-cycle amplitude within the measured band (regression gate, not a settling test).
  H3  limit-cycle amplitude ratio stable (regression gate, not a settling test).
  H4  in the H1 episodes: xy and height integrators exactly 0, attitude integrator finite.
  H2/H3 fly the hold_diagnostic.py m8b T1 protocol unchanged (its fly / summarize_m8b / window_amplitude): 120 s
  (1,200 decisions), route truncation and failure rules bypassed, state checked every decision.
  - amplitude = RMS of the horizontal velocity error, each axis linearly detrended, in the window 110-120 s.
  - ratio = amplitude(110-120 s) / amplitude(20-30 s).
  - H2 passes if every seed's amplitude is in [0.0539, 0.0874] m/s AND the logged point wind is exactly 0 in every
    flight. H3 passes if every seed's ratio is in [0.8265, 1.0977] AND no flight stopped early.
  DERIVATION (auditable; from results/m8/hold_m8b.json / hold_m8b_T1.csv, Claire's m8b run of 2026-11-16 03:05,
  recomputed from the CSV with the same functions, matching the JSON to 1e-12):
  - measured amplitude 110-120 s across the 10 seeds: min 0.06739, max 0.07287 m/s. Bound = [0.8 x min, 1.2 x max]
    = [0.0539, 0.0874] m/s: a stated margin of 20 % (relative) on each measured extreme.
  - measured ratio across the 10 seeds: min 0.9265, max 0.9977. Bound = [min - 0.10, max + 0.10] = [0.8265, 1.0977]:
    a stated margin of 0.10 (absolute) on each measured extreme.
  - Why margins at all: the physics has been bit-identical between runs on this machine (gate runs 1 and 2), so on
    the same build the gate reproduces T1 exactly. The margins allow platform-level floating-point drift (another
    Isaac build) while still catching a regression: a damped mode (ratio < 0.83), a growing one (> 1.10), or an
    amplitude change above 20 % (for scale: the residual arm's PID shows 0.087-0.092 m/s, above the upper bound).
  - Seeds 400000-400009 are VALIDATION seeds, never the 300000-300029 EVAL seeds, and this is a zero-command
    code-correctness gate with no model, so setting its bounds from them does not touch the experiment.
  History: H2/H3 were first "drift after 3 s <= 0.10 m" and "final speed <= 0.05 m/s" (Claude's guesses from an
  ideal 0.83 s velocity loop). They failed in runs 1 and 2 (max 0.2118 m, 0.1387 m/s) because the plant has a
  bounded limit cycle (period 2.61 s, present in the residual arm too), not because the env or the test was wrong;
  prespec A6/A7. Replaced 2026-11-16 by Claire's decision (rule 2 of the hold-diagnostic decision rule).
Prints PASS/FAIL per check and `RESULT: PASS` or `RESULT: FAIL`; writes results/m8/check_direct_env.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))
sys.path.insert(2, str(RESIDUAL_DIR / "training"))
sys.path.insert(3, str(RESIDUAL_DIR / "tools"))

import run_common as rc  # noqa: E402

REFERENCE = "residual_ppo/results/eval/eval-v1/pid_validation/realizations.json"
OUT = RESIDUAL_DIR / "results" / "m8" / "check_direct_env.json"
FRAME = 32
INDEPENDENT = [*range(0, 18), *range(21, 24), *range(28, 32)]
INNER_LOOP = [*range(18, 21), *range(24, 28)]
HOLD_SETTLE_DECISIONS = 30          # 3 s; drift after it is still LOGGED (information only, no longer a check)
# H2/H3 limit-cycle regression bounds, derived in the docstring from the m8b T1 run on validation seeds 400000-400009
# (never the 300000-series eval seeds): measured extremes +- the stated margin.
LC_SECONDS = 120.0
LC_MEASURED_AMPLITUDE = (0.06739, 0.07287)          # m/s, amplitude 110-120 s, min / max over the 10 seeds
LC_AMPLITUDE_BAND = (0.8 * LC_MEASURED_AMPLITUDE[0], 1.2 * LC_MEASURED_AMPLITUDE[1])   # 20 % relative margin
LC_MEASURED_RATIO = (0.9265, 0.9977)                # amplitude(110-120) / amplitude(20-30), min / max
LC_RATIO_BAND = (LC_MEASURED_RATIO[0] - 0.10, LC_MEASURED_RATIO[1] + 0.10)               # 0.10 absolute margin


def say(msg=""):
    print(f"[M8.check] {msg}", flush=True)


def realized(env, info, observation):
    import hashlib
    return {"wind_seed": info["wind_seed"], "wind_speed": float(np.linalg.norm(info["wind_mean"])),
            "route_seconds": info["route_seconds"],
            "initial_obs_sha256": hashlib.sha256(np.ascontiguousarray(observation).tobytes()).hexdigest(),
            "max_decisions": int(env.max_decisions)}


def state_bytes(sim):
    return {k: np.asarray(v, dtype=float).tobytes() for k, v in sim.read_drone_state().items()}


def main():
    results, record = [], {"reference": REFERENCE}

    def check(name, ok, detail=""):
        results.append(bool(ok))
        say(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    check_line = rc.preflight(say)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import eval_protocol_v1 as protocol
        import path_sac_config
        import run_protocol
        from direct_control.direct_env import DirectEnv, integrator_magnitudes
        from hover_env import make_hover_sim
        from path_sac_env import PathSacEnv

        if protocol.protocol_record()["protocol_file_sha256"] != rc.EXPECTED_PROTOCOL_SHA:
            raise SystemExit("protocol hash differs from the frozen one")
        protocol.register_profiles()
        sim = make_hover_sim(app, path_sac_config.BASE_WIND, log=say)
        sim.world.play()
        shared = PathSacEnv(sim, log=say)
        direct = DirectEnv(sim, log=say)
        outside = rc.loaded_outside_residual()
        check("shared modules loaded from residual_ppo only", not outside, str(outside or ""))
        record.update(checks=check_line, total_mass_kg=float(sim.total_mass))

        # ---------------- Part R ----------------
        reference = json.loads((REPO_DIR / REFERENCE).read_text(encoding="utf-8"))
        guard5 = run_protocol.make_realization_guard(reference)
        # initial_obs_sha256 excluded (controller-derived features 18-20 / 24-27, no shadow PID); R3 + R4 prove the
        # physical identity instead. The rows must be cut too: row_differences counts a missing field as a difference.
        four = [k for k in reference["key"] if k != "initial_obs_sha256"]
        guard4 = run_protocol.make_realization_guard({
            **reference, "key": four,
            "flights": {cell: {k: row[k] for k in four} for cell, row in reference["flights"].items()}})
        n = r1 = r2 = r3 = r4 = 0
        bad = {"R1": [], "R2": [], "R3": [], "R4": []}
        inner_diff_cells = 0
        for scenario in protocol.SCENARIOS:
            for profile in protocol.PROFILES:
                for seed in protocol.VALIDATION_SEEDS:
                    options = protocol.reset_options(scenario, profile)
                    key = f"{scenario}|{profile}|{seed}"
                    obs_s, info_s = shared.reset(seed=seed, options=options)
                    st_s, wind_s = state_bytes(sim), np.array(sim.wind.mean_wind, dtype=float).tobytes()
                    obs_d, info_d = direct.reset(seed=seed, options=options)
                    st_d, wind_d = state_bytes(sim), np.array(sim.wind.mean_wind, dtype=float).tobytes()
                    n += 1
                    for tag, guard, env, info, obs in (("R1", guard5, shared, info_s, obs_s),
                                                       ("R2", guard4, direct, info_d, obs_d)):
                        try:
                            guard(scenario, profile, seed, realized(env, info, obs))
                            r1 += tag == "R1"
                            r2 += tag == "R2"
                        except run_protocol.RealizationMismatch as exc:
                            bad[tag].append(str(exc)[:160])
                    if st_s == st_d and wind_s == wind_d:
                        r3 += 1
                    else:
                        bad["R3"].append(key)
                    fs = np.asarray(obs_s).reshape(-1, FRAME)
                    fd = np.asarray(obs_d).reshape(-1, FRAME)
                    if fs[:, INDEPENDENT].tobytes() == fd[:, INDEPENDENT].tobytes():
                        r4 += 1
                    else:
                        bad["R4"].append(key)
                    inner_diff_cells += int(fs[:, INNER_LOOP].tobytes() != fd[:, INNER_LOOP].tobytes())
        check(f"R1 PathSacEnv reproduces all 5 key fields of the PID validation reference", r1 == n,
              f"{r1}/{n} {bad['R1'][:2]}")
        check(f"R2 DirectEnv reproduces {four}", r2 == n, f"{r2}/{n} {bad['R2'][:2]}")
        check("R3 DirectEnv drone state and mean wind after reset bit-identical to PathSacEnv", r3 == n,
              f"{r3}/{n} {bad['R3'][:3]}")
        check("R4 controller-independent observation features (0-17, 21-23, 28-31) bit-identical", r4 == n,
              f"{r4}/{n} {bad['R4'][:3]}; cells where features 18-20/24-27 differ (expected, inner loop vs PID): "
              f"{inner_diff_cells}/{n}")
        scenario, profile, seed = "turns", "training_range", protocol.VALIDATION_SEEDS[0]
        obs_d, info_d = direct.reset(seed=seed + 1, options=protocol.reset_options(scenario, profile))
        try:
            guard4(scenario, profile, seed, realized(direct, info_d, obs_d))
            refused = False
        except run_protocol.RealizationMismatch:
            refused = True
        check("R5 negative: seed + 1 is refused by the guard (meaningful only if R2 passed)", refused and r2 == n)
        record["realizations"] = {"resets": n, "R1": r1, "R2": r2, "R3": r3, "R4": r4, "inner_loop_features_differ":
                                  inner_diff_cells, "bad": bad, "negative_refused": refused}

        # ---------------- Part H ----------------
        hold = []
        for seed in protocol.VALIDATION_SEEDS:
            direct.reset(seed=seed, options=protocol.reset_options("hover", protocol.STILL_AIR_NAME))
            positions, reason, steps = [], None, 0
            while True:
                _, _, terminated, truncated, info = direct.step(np.zeros(4, dtype=np.float32))
                steps += 1
                state = sim.read_drone_state()
                positions.append(np.asarray(state["position"], dtype=float))
                reason = info["failure_reason"]
                if terminated or truncated:
                    break
            p = np.array(positions)
            drift = float(np.max(np.linalg.norm(p[HOLD_SETTLE_DECISIONS:] - p[HOLD_SETTLE_DECISIONS - 1], axis=1)))
            speed = float(np.linalg.norm(state["linear_velocity"]))
            integ = integrator_magnitudes(direct.core.pid.pid)
            hold.append({"seed": seed, "decisions": steps, "failure_reason": reason, "drift_after_3s_m": drift,
                         "final_speed_mps": speed, "integrators": integ})
            say(f"hold seed {seed} (information only): {steps} decisions, failure {reason}, drift after 3 s "
                f"{drift:.4f} m, final speed {speed:.4f} m/s, integrators {integ}")
        # Limit-cycle runs: the m8b T1 protocol, unchanged (hold_diagnostic.fly / summarize_m8b / window_amplitude).
        import hold_diagnostic as hd

        cycle = []
        for seed in protocol.VALIDATION_SEEDS:   # validation seeds, never the 300000-series eval seeds
            rows, stopped, _ = hd.fly(direct, sim, np.zeros(4, dtype=np.float32), seed, protocol,
                                      decisions=int(round(LC_SECONDS * 10)), gusts=False)
            s = hd.summarize_m8b("H", rows, stopped, LC_SECONDS)
            amp = hd.window_amplitude(hd._arrays(rows), LC_SECONDS - 10.0, LC_SECONDS + 1e-6)
            cycle.append({"seed": seed, "amplitude_110_120_mps": amp, "ratio": s["amplitude_ratio"],
                          "stopped_reason": stopped, "max_wind_mps": s["max_wind_mps"],
                          "dominant_pitch_period_s": s["tilt_response"]["dominant_pitch_period_s"]})
            say(f"limit cycle seed {seed}: amplitude 110-120 s {amp:.5f} m/s, ratio {s['amplitude_ratio']:.4f}, "
                f"period {cycle[-1]['dominant_pitch_period_s']:.2f} s, max wind {s['max_wind_mps']:.3f} m/s, "
                f"stopped {stopped}")
        lo, hi = LC_AMPLITUDE_BAND
        rlo, rhi = LC_RATIO_BAND
        check("H1 zero command, still air, hover: no failure", all(h["failure_reason"] is None for h in hold))
        check(f"H2 limit-cycle amplitude within the measured band [{lo:.4f}, {hi:.4f}] m/s (still air)",
              all(c["amplitude_110_120_mps"] is not None and lo <= c["amplitude_110_120_mps"] <= hi
                  and c["max_wind_mps"] == 0.0 for c in cycle),
              f"amplitude {min(c['amplitude_110_120_mps'] for c in cycle):.5f}-"
              f"{max(c['amplitude_110_120_mps'] for c in cycle):.5f} m/s; max wind "
              f"{max(c['max_wind_mps'] for c in cycle):.3f} m/s")
        check(f"H3 limit-cycle amplitude ratio stable [{rlo:.4f}, {rhi:.4f}] (110-120 s / 20-30 s)",
              all(c["ratio"] is not None and rlo <= c["ratio"] <= rhi and c["stopped_reason"] is None
                  for c in cycle),
              f"ratio {min(c['ratio'] for c in cycle):.4f}-{max(c['ratio'] for c in cycle):.4f}; stopped "
              f"{[c['stopped_reason'] for c in cycle if c['stopped_reason']] or 'none'}")
        check("H4 xy / height integrators exactly 0, attitude finite",
              all(h["integrators"]["xy_abs_max"] == 0.0 and h["integrators"]["height_abs"] == 0.0
                  and np.isfinite(h["integrators"]["attitude_abs_max"]) for h in hold))
        record["hold"] = hold
        record["limit_cycle"] = {"bands": {"amplitude_mps": LC_AMPLITUDE_BAND, "ratio": LC_RATIO_BAND},
                                 "measured_in_m8b_T1": {"amplitude_mps": LC_MEASURED_AMPLITUDE,
                                                        "ratio": LC_MEASURED_RATIO},
                                 "seeds": cycle}
        ok = all(results)
        record.update(passed=sum(results), checks_total=len(results), result="PASS" if ok else "FAIL")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
        say(f"written: {OUT.relative_to(REPO_DIR).as_posix()}")
        say(f"{sum(results)}/{len(results)} checks passed")
        say("RESULT: PASS" if ok else "RESULT: FAIL")
    except BaseException:
        import traceback
        traceback.print_exc()
        say("RESULT: FAIL (exception)")
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
