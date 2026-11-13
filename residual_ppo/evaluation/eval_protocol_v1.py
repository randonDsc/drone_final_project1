"""Evaluation protocol v1 constants (docs/milestone2_plan.md, section E). Pure Python, no Isaac.

Every evaluation of PID, PID + SAC and PID + PPO imports these. Changing anything here makes a new
protocol version. The SHA-256 of this file is recorded in every results file (protocol_record()).
"""
import copy
import hashlib
from pathlib import Path

import path_sac_config

VERSION = "eval-v1"

# E1. Eval seeds: 30 (D3, decided 2026-11-13 after M2.1). The first 10 are the team's EVAL_SEEDS.
EVAL_SEEDS = tuple(range(300_000, 300_030))
TEAM_EVAL_SEEDS = tuple(path_sac_config.EVAL_SEEDS)
assert EVAL_SEEDS[:10] == TEAM_EVAL_SEEDS, "first 10 eval seeds must be the team's EVAL_SEEDS"

# E6. Validation seeds for checkpoint selection; disjoint from eval seeds.
VALIDATION_SEEDS = tuple(range(400_000, 400_010))
assert not set(VALIDATION_SEEDS) & set(EVAL_SEEDS)

# Seeds used only for timing (M2.1); disjoint from eval and validation seeds.
TIMING_SEEDS_START = 500_000

SCENARIOS = dict(path_sac_config.EVAL_SCENARIOS)       # scenario -> route_type (6, team)
ROUTE_STAGE = path_sac_config.EVAL_ROUTE_STAGE         # "full"
TEAM_PROFILES = tuple(path_sac_config.EVAL_WIND_PROFILES)  # training_range, unseen_gusts, outside_range

# E2. Still air (D1): training_range ranges with no_wind_probability = 1.0. Every draw is still made,
# so routes and starts match the windy profiles; mean wind, gust scale and turn rate become 0.
STILL_AIR_NAME = "still_air"
STILL_AIR = {
    "no_wind_probability": 1.0,
    "mean_speed_range": (2.0, 7.0),
    "gust_scale_range": (0.5, 1.0),
    "direction_rate_range": (0.0, 0.2),
}
OUR_PROFILES = {STILL_AIR_NAME: STILL_AIR}
PROFILES = (STILL_AIR_NAME,) + TEAM_PROFILES  # evaluation order


def register_profiles():
    """Add our profiles to path_sac_config.EVAL_WIND_PROFILES in memory only (D1). Never overwrites."""
    registry = path_sac_config.EVAL_WIND_PROFILES
    for name, profile in OUR_PROFILES.items():
        if name in registry and registry[name] != profile:
            raise RuntimeError(f"{name!r} already exists in EVAL_WIND_PROFILES with a different definition")
        registry.setdefault(name, copy.deepcopy(profile))
    return {name: copy.deepcopy(registry[name]) for name in OUR_PROFILES}


def reset_options(scenario, profile):
    return {"route_stage": ROUTE_STAGE, "route_type": SCENARIOS[scenario], "wind_profile": profile}


def protocol_record():
    """What every snapshot / results file stores about the protocol (E7)."""
    source = Path(__file__).read_bytes()
    return {
        "version": VERSION,
        "protocol_file_sha256": hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest(),
        "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1], len(EVAL_SEEDS)],
        "validation_seeds": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1], len(VALIDATION_SEEDS)],
        "route_stage": ROUTE_STAGE,
        "scenarios": SCENARIOS,
        "team_profiles": {n: path_sac_config.EVAL_WIND_PROFILES[n] for n in TEAM_PROFILES},
        "our_profiles": copy.deepcopy(OUR_PROFILES),
    }
