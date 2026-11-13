"""M2.1 headless speed measurement. NO training: zero residual through PathSacEnv, plus an untrained
policy's inference time. Run with Isaac python.bat from the repo root:

    # Directory: <repo root> (drone_final_project1)
    & "<Isaac Sim>\\python.bat" residual_ppo\\tools\\measure_speed.py *> residual_ppo\\results\\m2\\speed.log

Writes residual_ppo/results/m2/speed.json and speed_summary.txt (repo-relative paths only).
Timing seeds start at eval_protocol_v1.TIMING_SEEDS_START (disjoint from eval and validation seeds).
"""
import contextlib
import importlib.util
import io
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = RESIDUAL_DIR / "results" / "m2"
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))

WARMUP_DECISIONS = 200
REPEATS = 3
DECISIONS_PER_REPEAT = 2_000
TIMED_PROFILES = ("training_range", "still_air")
TRAINING_DECISIONS = 2_000_000
PPO_N_STEPS = 2048
INFERENCE_CALLS = 5_000

lines = []


def say(msg=""):
    line = f"[M2.1] {msg}"
    print(line, flush=True)
    lines.append(line)


def load_sync_shared():
    spec = importlib.util.spec_from_file_location("sync_shared", RESIDUAL_DIR / "tools" / "sync_shared.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sync_shared"] = mod
    spec.loader.exec_module(mod)
    return mod


def hardware():
    info = {"platform": platform.platform(), "cpu": platform.processor()}
    try:
        out = subprocess.run(["wmic", "cpu", "get", "Name,NumberOfCores,NumberOfLogicalProcessors", "/format:list"],
                             capture_output=True, text=True, timeout=20).stdout
        info["cpu_wmic"] = {k: v for k, v in (l.split("=", 1) for l in out.splitlines() if "=" in l)}
    except Exception as exc:  # informational only
        info["cpu_wmic"] = f"unavailable: {exc}"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        info["gpu"] = out
    except Exception as exc:
        info["gpu"] = f"unavailable: {exc}"
    return info


# ---------- before Isaac: shared files must be intact ----------
sync_shared = load_sync_shared()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    rc = sync_shared.check() if sync_shared.team_repo_configured() else sync_shared.check_local()
check_line = buf.getvalue().strip().splitlines()[-1]
say(f"checks: {check_line} (rc={rc})")
if rc != 0:
    print(buf.getvalue())
    raise SystemExit("shared files do not match the manifest; not measuring")

t_app0 = time.perf_counter()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
t_app = time.perf_counter() - t_app0
env = None
result = {}
try:
    import numpy as np
    import torch
    import gymnasium
    import stable_baselines3
    from stable_baselines3.common.utils import get_device

    import eval_protocol_v1 as protocol
    import path_sac_env

    registered = protocol.register_profiles()
    say(f"registered in memory (D1): {registered}")

    t_env0 = time.perf_counter()
    env = path_sac_env.make_path_sac_env(app)
    t_env = time.perf_counter() - t_env0
    say(f"startup: SimulationApp {t_app:.1f} s, env build {t_env:.1f} s")
    assert env.core.decimation == 12

    seed_counter = [protocol.TIMING_SEEDS_START]

    def next_seed():
        seed_counter[0] += 1
        return seed_counter[0]

    def timed_run(profile, decisions):
        """Zero action for `decisions`, resetting when an episode ends (resets inside the window)."""
        zero = np.zeros(4, np.float32)
        options = {"route_stage": protocol.ROUTE_STAGE, "wind_profile": profile}
        reset_times, physics_steps, episodes, failures = [], 0, 0, 0
        t_r = time.perf_counter()
        env.reset(seed=next_seed(), options=options)
        reset_times.append(time.perf_counter() - t_r)
        t0 = time.perf_counter()
        for _ in range(decisions):
            _, _, term, trunc, info = env.step(zero)
            physics_steps += info["physics_steps"]
            if term or trunc:
                episodes += 1
                failures += int(term)
                t_r = time.perf_counter()
                env.reset(seed=next_seed(), options=options)
                reset_times.append(time.perf_counter() - t_r)
        elapsed = time.perf_counter() - t0
        return {"decisions": decisions, "seconds": elapsed, "decisions_per_s": decisions / elapsed,
                "physics_steps_per_s": physics_steps / elapsed, "episodes_ended": episodes,
                "failures": failures, "resets": len(reset_times),
                "reset_mean_s": statistics.mean(reset_times), "reset_max_s": max(reset_times)}

    # ---------- 1-3. Env throughput ----------
    say(f"warm-up {WARMUP_DECISIONS} decisions (discarded)")
    timed_run("training_range", WARMUP_DECISIONS)
    throughput = {}
    for profile in TIMED_PROFILES:
        runs = [timed_run(profile, DECISIONS_PER_REPEAT) for _ in range(REPEATS)]
        rates = [r["decisions_per_s"] for r in runs]
        mean_rate = statistics.mean(rates)
        spread = (max(rates) - min(rates)) / mean_rate
        throughput[profile] = {"repeats": runs, "decisions_per_s_mean": mean_rate,
                               "decisions_per_s_min": min(rates), "decisions_per_s_max": max(rates),
                               "relative_spread": spread,
                               "within_10pct": all(abs(r - mean_rate) / mean_rate <= 0.10 for r in rates)}
        for i, r in enumerate(runs):
            say(f"{profile:<15} repeat {i}: {r['decisions_per_s']:.2f} decisions/s, {r['physics_steps_per_s']:.1f} "
                f"physics steps/s, {r['episodes_ended']} episodes ended ({r['failures']} failures), "
                f"reset mean {r['reset_mean_s']*1000:.1f} ms")
        say(f"{profile:<15} mean {mean_rate:.2f} decisions/s (min {min(rates):.2f}, max {max(rates):.2f}); "
            f"all repeats within +-10% of mean: {throughput[profile]['within_10pct']}")

    # ---------- 4. Policy inference overhead (untrained, CPU, no gradient step) ----------
    # A plain torch MLP with PPO MlpPolicy's actor shape (128-256-256-4, Tanh), forward pass only.
    # Not SB3's ActorCriticPolicy: building it creates an Adam optimizer, which imports torch._dynamo, and after
    # SimulationApp that resolves to a second torch in omni.isaac.ml_archive (run 1 ImportError; see M2.1 notes).
    # SB3's predict() adds numpy/tensor conversion overhead, so this is a lower bound.
    device = get_device("auto")
    infer_s = None
    try:
        mlp = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.Tanh(), torch.nn.Linear(256, 256),
                                  torch.nn.Tanh(), torch.nn.Linear(256, 4)).to("cpu").eval()
        obs, _ = env.reset(seed=next_seed(),
                           options={"route_stage": protocol.ROUTE_STAGE, "wind_profile": "still_air"})
        with torch.no_grad():
            for _ in range(200):
                mlp(torch.as_tensor(obs).unsqueeze(0)).numpy()
            t0 = time.perf_counter()
            for _ in range(INFERENCE_CALLS):
                mlp(torch.as_tensor(obs).unsqueeze(0)).numpy()
        infer_s = (time.perf_counter() - t0) / INFERENCE_CALLS
        say(f"policy forward (untrained 128-256-256-4 Tanh MLP, cpu, no_grad; lower bound for SB3 predict): "
            f"{infer_s*1e3:.3f} ms/decision; SB3 get_device('auto') = {device}")
    except Exception as exc:
        say(f"policy inference timing FAILED: {type(exc).__name__}: {exc}")

    # ---------- 6. Eval episode caps for the 30 eval seeds (profile-independent) ----------
    caps, rng_mismatch = {}, []
    for scenario in protocol.SCENARIOS:
        caps[scenario] = []
        for seed in protocol.EVAL_SEEDS:
            obs_a, info_a = env.reset(seed=seed, options=protocol.reset_options(scenario, "training_range"))
            cap = env.max_decisions
            obs_b, info_b = env.reset(seed=seed, options=protocol.reset_options(scenario, "still_air"))
            if (info_a["route_seconds"] != info_b["route_seconds"] or env.max_decisions != cap
                    or not np.array_equal(obs_a, obs_b) or info_b["wind_mean"].any()):
                rng_mismatch.append((scenario, seed))
            caps[scenario].append(cap)
    total_caps = sum(sum(v) for v in caps.values())
    say(f"eval caps (30 seeds x 6 scenarios): total {total_caps} decisions per profile; per scenario mean "
        + ", ".join(f"{s} {statistics.mean(v):.0f}" for s, v in caps.items()))
    say(f"still_air vs training_range same route/start/caps and zero still-air wind on all 180: {not rng_mismatch}"
        + (f" MISMATCHES: {rng_mismatch}" if rng_mismatch else ""))

    # ---------- 5-6. Estimates ----------
    dps = throughput["training_range"]["decisions_per_s_mean"]
    reset_s = statistics.mean(r["reset_mean_s"] for p in throughput.values() for r in p["repeats"])
    rollouts = -(-TRAINING_DECISIONS // PPO_N_STEPS)
    env_train_h = TRAINING_DECISIONS / dps / 3600
    infer_train_h = TRAINING_DECISIONS * infer_s / 3600 if infer_s is not None else float("nan")
    n_profiles = len(protocol.PROFILES)
    flights = n_profiles * len(protocol.SCENARIOS) * len(protocol.EVAL_SEEDS)
    eval_s_upper = n_profiles * total_caps / dps + flights * reset_s
    per_seed_s = eval_s_upper / len(protocol.EVAL_SEEDS)
    say(f"ESTIMATE training, {TRAINING_DECISIONS:,} decisions: env {env_train_h:.1f} h + policy inference "
        f"{infer_train_h:.2f} h + {rollouts} PPO updates x t_update (unmeasured; M3 smoke run)")
    say(f"ESTIMATE eval, {flights} flights ({n_profiles} profiles x 6 scenarios x {len(protocol.EVAL_SEEDS)} seeds): "
        f"<= {eval_s_upper/3600:.2f} h per controller (upper bound: no early failures); "
        f"per 10 seeds {per_seed_s*10/3600:.2f} h; per seed {per_seed_s/60:.1f} min")

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "M2.1 headless speed measurement (no training)",
        "checks": check_line,
        "protocol": protocol.protocol_record(),
        "registered_profiles": registered,
        "python": platform.python_version(),
        "packages": {m.__name__: m.__version__ for m in (np, gymnasium, stable_baselines3, torch)},
        "torch_cuda_available": torch.cuda.is_available(),
        "sb3_auto_device": str(device),
        "hardware": hardware(),
        "startup_s": {"simulation_app": t_app, "env_build": t_env},
        "settings": {"warmup": WARMUP_DECISIONS, "repeats": REPEATS, "decisions_per_repeat": DECISIONS_PER_REPEAT,
                     "route_stage": protocol.ROUTE_STAGE, "action": "zero", "headless": True},
        "throughput": throughput,
        "policy_inference_s_per_decision": infer_s,
        "eval_caps_per_scenario": caps,
        "eval_caps_total_per_profile": total_caps,
        "rng_consistency_still_vs_training": {"ok": not rng_mismatch, "mismatches": rng_mismatch},
        "estimates": {
            "training_decisions": TRAINING_DECISIONS, "ppo_rollouts": rollouts,
            "env_hours": env_train_h, "policy_inference_hours": infer_train_h,
            "ppo_update_hours": "unmeasured (needs a gradient step; M3 smoke run)",
            "eval_flights": flights, "eval_hours_upper_bound": eval_s_upper / 3600,
            "eval_hours_per_10_seeds": per_seed_s * 10 / 3600, "eval_minutes_per_seed": per_seed_s / 60,
            "basis": "training_range mean decisions/s; zero action; reset time added per flight",
        },
    }
except Exception:
    import traceback
    traceback.print_exc()
    say("RESULT: FAIL (exception)")
    raise
finally:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if result:
        (OUT_DIR / "speed.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
        say("written: residual_ppo/results/m2/speed.json, speed_summary.txt")
        say("RESULT: DONE")
    (OUT_DIR / "speed_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if env is not None:
        env.close()
    app.close()
