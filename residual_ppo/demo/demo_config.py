"""M9 visual demo settings. NumPy only. DESCRIPTIVE DEMO, NOT eval-v1.

Decided by Claire (2026-11-17):
- indoor = sim_scene/dij3d_boxes.usd (the training scene and drone asset) with /World/BoxWarehouse ACTIVE; only
  /World/Cube (launch pad) is deactivated. outdoor = sim_scene/forest.usd (synced asset), same rule.
- winds: 0 (truly calm: gust scale 0, as eval-v1 still_air), 3.5 and 7 m/s mean, gust scale 0.75, no turning,
  a fixed direction per track (crosswind to the main leg) and a fixed wind seed per level.
- indoor wind is labelled "artificial uniform disturbance (no sheltering)": wind.py applies the same field everywhere.
- forest: two edge tracks at >= 2 m clearance plus one weaving track at >= 1 m; indoor tracks >= 2 m.
- seeds from 600000 up; never the eval (300000-300029) or validation (400000-400009) seeds.
- no controller perceives obstacles (the observation has none); collisions are reported, not avoided.
"""
import numpy as np

import path_sac_config

LABEL = "descriptive demo, not eval-v1"
NO_PERCEPTION_NOTE = "no controller perceives obstacles"

# ---------- Scenes (repo-relative) ----------

SCENES = {
    "indoor": {
        "usd": "sim_scene/dij3d_boxes.usd",
        "deactivate": ("/World/Cube",),
        "obstacle_root": "/World/BoxWarehouse",
        "wind_label": "artificial uniform disturbance (no sheltering)",
    },
    "forest": {
        "usd": "sim_scene/forest.usd",
        "deactivate": ("/World/Cube",),
        "obstacle_root": "/World/Forest",
        "wind_label": "uniform wind field (no sheltering by trees)",
    },
}

# ---------- Tracks: the `full` curriculum stage's limits (in distribution) ----------

_FULL = path_sac_config.stage_by_name("full")["route"]
TRACK_LIMITS = {
    "speed": 1.5,                                            # full: speed_range (1.0, 2.0)
    "acceleration": 0.55,                                    # full: acceleration_range (0.5, 0.6)
    "lateral_acceleration": float(_FULL["lateral_acceleration"]),   # 0.6
    "vertical_speed": float(_FULL["vertical_speed"]),               # 1.0
    "corner_radius": 0.8,                                    # full: corner_radius_range (0.0, 1.0)
    "start_hold": 0.5,                                       # as sample_reference for non-hover routes
    "end_hold": float(path_sac_config.EPISODE["settle_seconds"]),   # 3.0
    "spacing": float(path_sac_config.EPISODE["spacing"]),           # 0.05
}
FULL_RANGES = {   # for the in-distribution report
    "speed_range": tuple(_FULL["speed_range"]),
    "acceleration_range": tuple(_FULL["acceleration_range"]),
    "length_range": tuple(_FULL["length_range"]),
    "height_change_range": tuple(_FULL["height_change_range"]),
    "waypoint_count_range": tuple(_FULL["waypoint_count_range"]),
    "turn_degrees_range": tuple(_FULL["turn_degrees_range"]),
    "corner_radius_range": tuple(_FULL["corner_radius_range"]),
    "height_range": tuple(path_sac_config.EPISODE["height_range"]),
}

# ---------- Collision convention (team NAVIGATION["collision_half_size"]) ----------

COLLISION_HALF_SIZE = np.array(path_sac_config.NAVIGATION["collision_half_size"], dtype=float)  # (0.25, 0.25, 0.06)

# ---------- Wind ----------

WIND_LEVELS = {
    #  name     mean [m/s]  gust scale  wind seed
    "calm": {"speed": 0.0, "gust_scale": 0.0, "seed": 600100},
    "w3p5": {"speed": 3.5, "gust_scale": 0.75, "seed": 600101},
    "w7": {"speed": 7.0, "gust_scale": 0.75, "seed": 600102},
}


def demo_wind(level, direction_xy):
    """The wind dict PathCore.apply_wind takes (hover_task.sample_stage_wind's keys), fixed instead of drawn."""
    spec = WIND_LEVELS[level]
    d = np.asarray(direction_xy, dtype=float).reshape(2)
    d = d / np.linalg.norm(d)
    return {
        "mean_wind": np.array([spec["speed"] * d[0], spec["speed"] * d[1], 0.0]),
        "gust_scale": float(spec["gust_scale"]),
        "seed": int(spec["seed"]),
        "direction_rate": 0.0,
    }


# Registered in memory only, like eval-v1's still_air (register_demo_profiles); degenerate ranges record the level.
# The fixed direction is not representable in a profile; the demo reset passes demo_wind(...) directly.
DEMO_PROFILES = {
    f"m9_{name}": {
        "no_wind_probability": 1.0 if spec["speed"] == 0.0 else 0.0,
        "mean_speed_range": (spec["speed"], spec["speed"]),
        "gust_scale_range": (spec["gust_scale"], spec["gust_scale"]),
        "direction_rate_range": (0.0, 0.0),
    }
    for name, spec in WIND_LEVELS.items()
}


def register_demo_profiles():
    """Add the demo profiles to path_sac_config.EVAL_WIND_PROFILES in memory only. Never overwrites."""
    registry = path_sac_config.EVAL_WIND_PROFILES
    for name, profile in DEMO_PROFILES.items():
        if name in registry and registry[name] != profile:
            raise RuntimeError(f"profile {name!r} already registered with different settings")
        registry[name] = profile
    return sorted(DEMO_PROFILES)


# ---------- Seeds ----------

SEED_FLOOR = 600000
FORBIDDEN_SEEDS = {"eval-v1 eval": (300000, 300029), "eval-v1 validation": (400000, 400009)}


def assert_demo_seed(seed):
    """Refuse any seed below 600000 or inside the eval / validation ranges."""
    seed = int(seed)
    for name, (lo, hi) in FORBIDDEN_SEEDS.items():
        if lo <= seed <= hi:
            raise ValueError(f"seed {seed} is an {name} seed; the demo must not use it")
    if seed < SEED_FLOOR:
        raise ValueError(f"seed {seed} is below the demo floor {SEED_FLOOR}")
    return seed


for _spec in WIND_LEVELS.values():
    assert_demo_seed(_spec["seed"])
