"""Settings for the hover SAC residual: training, evaluation, and its use in
motor_physics.py. The path-following SAC has its own, in path_sac_config.py.

Units: meters, seconds, Newtons, radians unless named degrees. PID gains are in
sim_config.PID. See HOVER_SAC.md.
"""
from pathlib import Path

import sim_config


RUN_DIR = "runs/sac_hover"

# ---------- Timing ----------

# Must match drone_sim.PHYSICS_DT. PID runs at this rate.
PHYSICS_RATE_HZ = 120

# The policy decides at this rate and its correction is held in between.
# PHYSICS_RATE_HZ / RL_RATE_HZ must be a whole number of physics steps.
RL_RATE_HZ = 10

# ---------- Task ----------

TASK = {
    "target_position": (0.0, 0.0, 2.0),    # fixed hover target, world frame

    "episode_seconds": 20.0,

    # Uniform in +-range around hover at the target.
    "initial_position_range": (0.2, 0.2, 0.2),
    "initial_velocity_range": (0.1, 0.1, 0.1),
    "initial_angle_range_degrees": (3.0, 3.0, 5.0),   # roll, pitch, yaw
    "initial_angular_velocity_range": (0.1, 0.1, 0.1),
}

# ---------- Action ----------

# Correction at action = +-1. correction = action * limit, added to the PID
# command. All four at full scale take about 0.6 N of a rotor's 3.1 N headroom.
RESIDUAL_LIMITS = {
    "thrust": 1.5,             # [N]
    "roll_torque": 0.04,       # [N m]
    "pitch_torque": 0.04,      # [N m]
    "yaw_torque": 0.02,        # [N m]
}

# ---------- Observation ----------

# Frames stacked, oldest first: 5 frames at 10 Hz is 0.4 s of past.
HISTORY_LENGTH = 5

# Each raw value is divided by its scale, then clipped to +-OBSERVATION_CLIP.
# SAC replays old data, so these stay fixed instead of learned statistics.
OBSERVATION_SCALES = {
    "position_error": 1.0,         # [m]
    "linear_velocity": 1.0,        # [m/s]
    "attitude_error": 0.5,         # [rad]
    "angular_velocity": 2.0,       # [rad/s]
    "pid_thrust": 5.0,             # [N], offset from hover thrust
    "pid_torque": 0.1,             # [N m]
}

OBSERVATION_CLIP = 10.0

# In flight, a waypoint can be meters away. Training saw at most the failure
# limit, so the position error given to the policy stops here [m].
NAVIGATION_ERROR_LIMIT = 1.0

# ---------- Reward ----------

# reward = -scale * (sum of weighted costs + failure penalty if it failed)
REWARD = {
    "position": 1.0,                   # |error|^2 / (1 m)^2
    "position_linear": 0.5,            # |error| / 1 m; still pulls when small
    "speed": 0.1,                      # |velocity|^2 / (1 m/s)^2
    "attitude": 0.2,                   # (roll^2 + pitch^2) / reference^2
    "angular_rate": 0.02,              # |rate|^2 / (1 rad/s)^2
    "action": 0.01,                    # |normalized action|^2
    "action_rate": 0.05,               # |change in action|^2
    "saturation": 0.5,                 # share of this decision's steps saturated
    "persistent_saturation": 2.0,      # per decision, once persistent

    "attitude_reference_degrees": 20.0,

    "scale": 0.1,
}

# Fully saturated decisions in a row before persistent_saturation applies.
PERSISTENT_SATURATION_DECISIONS = 5

# Crossing any limit ends the episode as a failure. There is no contact sensor:
# in the open hover scene, the only thing to hit is the ground, so min_height
# stands in for a crash.
FAILURE = {
    "max_position_error": 3.0,     # [m]
    "min_height": 0.5,             # [m] world Z
    "max_tilt_degrees": 45.0,
    "max_speed": 4.0,              # [m/s]
    "max_angular_speed": 5.0,      # [rad/s]
}

# ---------- Wind ----------

# gust_scale multiplies these. Episode mean speed and direction come from a stage.
BASE_WIND = {
    "enabled": True,
    "mean_wind": (0.0, 0.0, 0.0),
    "gust_std": (1.0, 1.0, 0.35),          # [m/s]
    "local_gust_std": (0.3, 0.3, 0.15),    # [m/s]
}

# Used in order. A stage applies until the total training step reaches
# until_step; None never ends. direction_rate is how fast the mean wind turns,
# in a random direction [rad/s]. Wind stays on in a no-wind episode, at zero
# speed, so drag still acts on the drone's own motion.
WIND_STAGES = (
    {
        "name": "light",
        "until_step": 100_000,
        "no_wind_probability": 0.5,
        "mean_speed_range": (0.0, 2.0),
        "gust_scale_range": (0.0, 0.25),
        "direction_rate_range": (0.0, 0.0),
    },
    {
        "name": "steady",
        "until_step": 300_000,
        "no_wind_probability": 0.2,
        "mean_speed_range": (0.0, 4.0),
        "gust_scale_range": (0.0, 0.5),
        "direction_rate_range": (0.0, 0.05),
    },
    {
        "name": "gusty",
        "until_step": 600_000,
        "no_wind_probability": 0.1,
        "mean_speed_range": (1.0, 6.0),
        "gust_scale_range": (0.25, 0.75),
        "direction_rate_range": (0.0, 0.1),
    },
    {
        "name": "strong",
        "until_step": None,
        "no_wind_probability": 0.1,
        "mean_speed_range": (2.0, 8.0),
        "gust_scale_range": (0.5, 1.0),
        "direction_rate_range": (0.0, 0.2),
    },
)

# ---------- SAC ----------

SAC = {
    "learning_rate": 3e-4,
    "buffer_size": 200_000,
    "learning_starts": 2_000,
    "batch_size": 256,
    "tau": 0.005,

    # At 10 Hz, 0.99 looks about 10 s ahead.
    "gamma": 0.99,

    # One gradient update per environment step.
    "train_freq": 1,
    "gradient_steps": 1,

    "ent_coef": "auto",
    "policy_kwargs": {"net_arch": [256, 256]},
    "device": "auto",
}

TRAINING = {
    "checkpoint_interval": 100_000,    # [environment steps]
    "seed": 0,
}

# Used for both PID-only and PID + SAC. A seed fixes wind and start state.
EVAL_SEEDS = tuple(range(200_000, 200_020))


def physics_steps_per_decision():
    steps = PHYSICS_RATE_HZ / RL_RATE_HZ

    if abs(steps - round(steps)) > 1e-9:
        raise ValueError("RL_RATE_HZ must divide PHYSICS_RATE_HZ exactly.")

    return int(round(steps))


def stage_by_name(name):
    for index, stage in enumerate(WIND_STAGES):
        if stage["name"] == name:
            return index

    names = ", ".join(stage["name"] for stage in WIND_STAGES)
    raise ValueError(f"Unknown wind stage {name}; choose from {names}.")


def replay_buffer_path_for(model_path):
    """Replay buffer saved with a model, by SB3's and train_hover_sac's names."""
    model_path = Path(model_path)
    name = model_path.stem

    if name.endswith("_model"):
        prefix = name.removesuffix("_model")
        return model_path.with_name(f"{prefix}_replay_buffer.pkl")

    head, steps, tail = name.rsplit("_", 2)
    return model_path.with_name(f"{head}_replay_buffer_{steps}_{tail}.pkl")


def contract():
    """Everything a saved policy depends on. Any change needs a new model."""
    return {
        "algorithm": "sac",
        "version": 1,
        "rl_rate_hz": RL_RATE_HZ,
        "physics_rate_hz": PHYSICS_RATE_HZ,
        "history_length": HISTORY_LENGTH,
        "observation_scales": OBSERVATION_SCALES,
        "observation_clip": OBSERVATION_CLIP,
        "residual_limits": RESIDUAL_LIMITS,
        "pid": sim_config.PID,
    }
