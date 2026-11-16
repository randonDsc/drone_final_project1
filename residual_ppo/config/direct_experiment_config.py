"""Run layout and run ids for PPO DIRECT control (M8). Same rules as config/experiment_config.py (checkpoint interval,
smoke override, checkpoint naming, trained-on formula, forbidden seeds); only the folder and the run ids differ.

Run ids (Claire, 2026-11-16): ppo_direct_s0_1p5m (main) and smoke_direct_s0_<n>. They may be renamed later to match
the team convention. Folder: residual_ppo/runs/ppo_direct/<run_id>/.
"""
import re
from pathlib import Path

from config import direct_ppo_config as ppo_config
from config import experiment_config as _residual
from config.experiment_config import (  # noqa: F401  (same rules, re-exported for the training script / audit)
    CHECKPOINT_DIR, CHECKPOINT_INTERVAL, CHECKPOINT_NAME, CONTRACT_FILE, FINAL_MODEL, LEARNING_CURVE_CHECKPOINTS,
    LOG_DIR, RESIDUAL_DIR, RUN_META_FILE, SMOKE_OVERRIDE, STATUS_FILE, STOP_FILE, SUMMARY_FILE, checkpoint_name,
    checkpoint_steps, decisions_trained_on, forbidden_reset_seeds)

ARM = "direct"
RUNS_DIR = Path("runs") / "ppo_direct"
MAIN_RUN_ID = "ppo_direct_s0_1p5m"
SMOKE_RUN_PREFIX = "smoke_direct_s0_"
_SMOKE_RE = re.compile(re.escape(SMOKE_RUN_PREFIX) + r"\d+")

assert ppo_config.PPO["n_steps"] == _residual.ppo_config.PPO["n_steps"]   # decisions_trained_on uses it


def is_smoke(run_id):
    return bool(_SMOKE_RE.fullmatch(run_id))


def run_settings(run_id, curriculum_offset=0):
    """Only the main id or smoke_direct_s0_<n>. Smoke override and curriculum offset are smoke-only."""
    if run_id == MAIN_RUN_ID:
        if curriculum_offset:
            raise ValueError("--curriculum-offset is smoke-only")
        return {"run_id": run_id, "smoke": False, "total_decisions": ppo_config.TOTAL_DECISIONS,
                "checkpoint_interval": CHECKPOINT_INTERVAL, "curriculum_offset": 0}
    if is_smoke(run_id):
        return {"run_id": run_id, "smoke": True, "total_decisions": SMOKE_OVERRIDE["total_decisions"],
                "checkpoint_interval": SMOKE_OVERRIDE["checkpoint_interval"],
                "curriculum_offset": int(curriculum_offset)}
    raise ValueError(f"run id {run_id!r} not allowed for the direct arm: use {MAIN_RUN_ID!r} or {SMOKE_RUN_PREFIX}<n>")


def run_contract(settings):
    contract = ppo_config.contract()
    if settings["smoke"]:
        contract["training"]["smoke_override"] = {"total_decisions": settings["total_decisions"],
                                                  "checkpoint_interval": settings["checkpoint_interval"],
                                                  "curriculum_offset": settings["curriculum_offset"]}
    return contract


def run_dir(run_id):
    return RESIDUAL_DIR / RUNS_DIR / run_id
