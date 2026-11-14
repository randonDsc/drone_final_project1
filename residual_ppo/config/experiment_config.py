"""Run layout, checkpoint rule and run ids for residual PPO (M3 design, confirmed by Claire 2026-11-14).

Paths are relative to residual_ppo/ (committed records use repo-relative paths). No Isaac, no torch.

Primary PPO result (M4/E6): the FINAL model at 1,501,184 decisions (like the team SAC final model; no checkpoint
selection). Checkpoints 150k/400k/800k are evaluated on validation seeds 400000-400009 as a secondary learning
curve only.
"""
import re
from pathlib import Path

from config import ppo_config

RESIDUAL_DIR = Path(__file__).resolve().parents[1]

# ---------- run layout ----------
RUNS_DIR = Path("runs") / "ppo_path"            # residual_ppo/runs/ppo_path/<run_id>/ (model zips gitignored)
MAIN_RUN_ID = "ppo_s0_1p5m"                     # the M4 training run (seed 0, 1.5M decisions)
SMOKE_RUN_PREFIX = "smoke_ppo_s0_"              # M5 end-to-end tests of the M4 training script (never evaluated)

FINAL_MODEL = "final_model"                     # final_model.zip (+ ppo_contract.json next to it)
CHECKPOINT_DIR = "checkpoints"                  # checkpoints/ppo_<steps>_steps.zip (+ ppo_contract.json)
CHECKPOINT_NAME = "ppo_{steps}_steps"
CONTRACT_FILE = "ppo_contract.json"             # must equal evaluation/controllers.PPO_CONTRACT_FILE
STATUS_FILE = "status.json"                     # running / done / failed + last step (exit codes cannot be trusted)
RUN_META_FILE = "run_meta.json"                 # versions, device, total_mass_kg, HESAI, hardware, wall clock, counts
LOG_DIR = "logs"                                # SB3 progress.csv (CSV logger) + monitor.csv

# ---------- checkpoints ----------
# Saved when num_timesteps % CHECKPOINT_INTERVAL == 0, i.e. mid-rollout: the weights are those of the last update.
CHECKPOINT_INTERVAL = 50_000
LEARNING_CURVE_CHECKPOINTS = (150_000, 400_000, 800_000)   # secondary learning curve (validation seeds only)
SAC_CHECKPOINT_STEPS = (150_000, 400_000, 800_000, 1_500_000)
assert all(s % CHECKPOINT_INTERVAL == 0 for s in SAC_CHECKPOINT_STEPS)

# ---------- M3 ----------
VERIFY_RESULTS = Path("results") / "m3" / "ppo_setup_check"      # .json / .txt (.log gitignored)
TIMING_PROBE_ID = "m3_timing_probe_s0_8k"       # separate id; model discarded (never saved), never evaluated
TIMING_PROBE_DECISIONS = 4 * ppo_config.PPO["n_steps"]           # 8,192 decisions = 4 rollouts with updates
TIMING_PROBE_RESULTS = Path("results") / "m3" / "timing_probe"   # .json / .txt / _progress csv folder


# ---------- M4: run settings (D1, smoke override (5), D5, D8) ----------
STOP_FILE = "STOP"                              # D5: create runs/ppo_path/<id>/STOP to stop cleanly
SUMMARY_FILE = "training_summary.txt"
SMOKE_OVERRIDE = {"total_decisions": 10_240, "checkpoint_interval": 4_096}   # smoke_ppo_s0_<n> only
_SMOKE_RE = re.compile(re.escape(SMOKE_RUN_PREFIX) + r"\d+")


def is_smoke(run_id):
    return bool(_SMOKE_RE.fullmatch(run_id))


def run_settings(run_id, curriculum_offset=0):
    """D1: only the main id or smoke_ppo_s0_<n>. The smoke override and the curriculum offset (D8) are smoke-only."""
    if run_id == MAIN_RUN_ID:
        if curriculum_offset:
            raise ValueError("--curriculum-offset is smoke-only")
        return {"run_id": run_id, "smoke": False, "total_decisions": ppo_config.TOTAL_DECISIONS,
                "checkpoint_interval": CHECKPOINT_INTERVAL, "curriculum_offset": 0}
    if is_smoke(run_id):
        return {"run_id": run_id, "smoke": True, "total_decisions": SMOKE_OVERRIDE["total_decisions"],
                "checkpoint_interval": SMOKE_OVERRIDE["checkpoint_interval"],
                "curriculum_offset": int(curriculum_offset)}
    raise ValueError(f"run id {run_id!r} not allowed (D1): use {MAIN_RUN_ID!r} or {SMOKE_RUN_PREFIX}<n>")


def run_contract(settings):
    """ppo_contract.json for a run: ppo_config.contract(), plus smoke_override in the training group for smoke runs
    (so a smoke run and the main run can never resume into each other)."""
    contract = ppo_config.contract()
    if settings["smoke"]:
        contract["training"]["smoke_override"] = {"total_decisions": settings["total_decisions"],
                                                  "checkpoint_interval": settings["checkpoint_interval"],
                                                  "curriculum_offset": settings["curriculum_offset"]}
    return contract


def run_dir(run_id):
    return RESIDUAL_DIR / RUNS_DIR / run_id


def checkpoint_name(steps):
    return CHECKPOINT_NAME.format(steps=steps)


def checkpoint_steps(total=ppo_config.TOTAL_DECISIONS):
    """Decision counts at which checkpoints are saved during a run of `total` decisions (30 for 1.5M)."""
    last = ppo_config.planned_decisions(total)
    return tuple(range(CHECKPOINT_INTERVAL, last + 1, CHECKPOINT_INTERVAL))


def decisions_trained_on(num_timesteps):
    """Decisions the weights saved at `num_timesteps` were trained on, for a mid-rollout checkpoint.

    The callback fires during rollout k = ceil(n / n_steps); the last update used rollouts 1..k-1.
    (At n = 150,000: 73 complete rollouts = 149,504.) The final model is saved after the last update, so it is
    trained on all of its num_timesteps.
    """
    n_steps = ppo_config.PPO["n_steps"]
    return ((num_timesteps - 1) // n_steps) * n_steps


def forbidden_reset_seeds():
    """Seeds that must never reach a training reset (E6): the 30 eval seeds and the 10 validation seeds.

    Needs residual_ppo/evaluation on sys.path (read-only use of the frozen protocol constants).
    """
    import eval_protocol_v1

    return frozenset(eval_protocol_v1.EVAL_SEEDS) | frozenset(eval_protocol_v1.VALIDATION_SEEDS)
