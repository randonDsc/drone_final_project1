"""M5b S8: torch-origin evidence for EVALUATION flights, without touching the frozen protocol (not frozen itself).

Mirrors the load order of the frozen `run_protocol.py fly` (mode_fly) as closely as possible:
  preflight -> import torch, torch._dynamo -> controllers.make_controller(name, model)   [all BEFORE SimulationApp]
  -> SimulationApp (NO torch import pin, exactly like fly) -> gymnasium, stable_baselines3, eval_protocol_v1,
  path_sac_env -> register_profiles -> make_path_sac_env -> controller.bind_env(env) -> reset + act/step loop.
Around that it adds evidence only:
  1. actions of the controller on a fixed seeded batch of observations BEFORE SimulationApp and AFTER the flight
     must be bit-identical;
  2. after the flight, every loaded torch module must come from the kit torch (ppo_setup.torch_origin_report),
     and every torch module first imported after SimulationApp is listed with its origin.
One short flight on validation seed 400000 (hover / training_range); no tracking metrics are computed or stored.

    # Directory: <repo root>
    & "<Isaac Sim>\\python.bat" residual_ppo\\tools\\check_eval_torch_origin.py --controller pid_ppo `
        --model residual_ppo/runs/ppo_path/smoke_ppo_s0_4/final_model.zip *> residual_ppo\\results\\m5\\s8_pid_ppo.log

Writes residual_ppo/results/m5/s8_eval_torch_origin_<controller>.{json,txt}. Success = RESULT: PASS in the log and
the output files (never the exit code).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))
sys.path.insert(2, str(RESIDUAL_DIR / "training"))

import run_common as rc  # noqa: E402

SEED, SCENARIO, PROFILE, MAX_DECISIONS = 400_000, "hover", "training_range", 60
BATCH_SEED, BATCH = 20_261_114, 64

parser = argparse.ArgumentParser()
parser.add_argument("--controller", choices=("pid_ppo", "pid_sac"), required=True)
parser.add_argument("--model", required=True)
args = parser.parse_args()
lines = []


def say(msg=""):
    line = f"[S8] {msg}"
    print(line, flush=True)
    lines.append(line)


def torch_names():
    return {n for n in sys.modules if n == "torch" or n.startswith("torch.")}


def batch_actions(controller):
    obs = np.random.default_rng(BATCH_SEED).uniform(-1.0, 1.0, (BATCH, 128)).astype(np.float32)
    return np.stack([np.asarray(controller.act(o), dtype=np.float32) for o in obs])


# ---------- same order as mode_fly, before SimulationApp ----------
check_line = rc.preflight(say)
import torch  # noqa: E402  before SimulationApp (as in mode_fly)
import torch._dynamo  # noqa: E402,F401

import controllers  # noqa: E402

controller = controllers.make_controller(args.controller, model=args.model)
say(f"controller {controller.name} loaded before SimulationApp (model: {args.model})")
before = batch_actions(controller)
names_before = torch_names()
say(f"batch of {BATCH} seeded observations -> actions before SimulationApp: finite {bool(np.isfinite(before).all())}; "
    f"{len(names_before)} torch modules loaded")

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})   # NO pin_torch_imports(): the frozen fly mode has none
env = None
result = {}
try:
    import gymnasium  # noqa: F401
    import stable_baselines3

    import eval_protocol_v1 as protocol
    import path_sac_env

    protocol.register_profiles()
    env = path_sac_env.make_path_sac_env(app)
    controller.bind_env(env)
    obs, info = env.reset(seed=SEED, options=protocol.reset_options(SCENARIO, PROFILE))
    decisions, bad, ended = 0, 0, None
    while decisions < MAX_DECISIONS:
        action = np.asarray(controller.act(obs), dtype=np.float32)
        bad += int(action.shape != (4,) or not np.all(np.isfinite(action)))
        obs, _, terminated, truncated, info = env.step(action)
        decisions += 1
        if terminated or truncated:
            ended = "terminated" if terminated else "truncated"
            break
    say(f"flew {decisions} decisions on validation seed {SEED} ({SCENARIO} / {PROFILE}); non-finite or bad-shape "
        f"actions {bad}; episode end: {ended or 'cut at ' + str(MAX_DECISIONS)}")
    after = batch_actions(controller)
    identical = before.tobytes() == after.tobytes() and before.shape == after.shape
    say(f"same batch after the flight: bit-identical to before = {identical} "
        f"(max |diff| {float(np.abs(before - after).max()):.3e})")

    import ppo_setup

    origins = ppo_setup.torch_origin_report()
    root = Path(torch.__file__).resolve().parent
    late = []
    for name in sorted(torch_names() - names_before):
        f = getattr(sys.modules.get(name), "__file__", None)
        on_disk = isinstance(f, str) and Path(f).is_absolute()   # placeholders (torch.ops: "_ops.py") are not files
        late.append({"module": name, "file": rc.redact_isaac(f) if on_disk else f, "on_disk": on_disk,
                     "inside_kit_torch": on_disk and root in Path(f).resolve().parents})
    late_outside = [m for m in late if m["on_disk"] and not m["inside_kit_torch"]]
    clean = origins["outside_package_dir"] == 0 and origins["modules_from_ml_archive_pip_prebundle"] == 0
    say(f"torch origin at the end: {origins['torch_modules_with_file']} modules with a file, "
        f"{origins['outside_package_dir']} outside {origins['torch_package_dir']} {origins['outside_examples']}; "
        f"from the ml_archive pip_prebundle: {origins['modules_from_ml_archive_pip_prebundle']} "
        f"{origins['prebundle_examples']}")
    say(f"torch modules first imported AFTER SimulationApp: {len(late)} ({len(late_outside)} outside the kit torch)"
        + (f": {[m['module'] for m in late][:12]}" if late else ""))
    ok = identical and clean and bad == 0 and not late_outside
    record = controller.record()
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "M5b S8: torch origin evidence for evaluation flights (frozen fly load order, no pin)",
        "pass": ok, "controller": record, "checks": check_line,
        "flight": {"seed": SEED, "scenario": SCENARIO, "profile": PROFILE, "decisions": decisions,
                   "nonfinite_or_bad_shape_actions": bad, "episode_end": ended,
                   "note": "no tracking metrics computed or stored"},
        "batch": {"observations": BATCH, "rng_seed": BATCH_SEED, "bit_identical_before_after": identical,
                  "max_abs_diff": float(np.abs(before - after).max())},
        "torch_origins_end": origins, "torch_modules_imported_after_app": late,
        "torch_modules_before_app": len(names_before), "pin_used": False,
        "versions": {"torch": torch.__version__, "torch_file": rc.redact_isaac(torch.__file__),
                     "stable_baselines3": stable_baselines3.__version__, "numpy": np.__version__},
        "isaac_sim": rc.isaac_version(), "total_mass_kg": float(env.sim.total_mass), "git": rc.git_info(),
    }
    say(f"RESULT: {'PASS' if ok else 'FAIL'}")
except Exception:
    import traceback

    traceback.print_exc()
    say("RESULT: FAIL (exception)")
finally:
    out = RESIDUAL_DIR / "results" / "m5" / f"s8_eval_torch_origin_{args.controller}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if result:
        out.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if env is not None:
        env.close()
    app.close()
