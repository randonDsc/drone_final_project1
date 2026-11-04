"""Settings for PPO residual hover training and evaluation.

Units follow sim_config.py: meters, seconds, Newtons, and radians unless
explicitly named degrees. Physics and controller gains are not set here.
"""
from pathlib import Path


# ---------- Scene ----------

# Relative to the project root.
SCENES = {
    # Same drone and physics scene as the demo: IMU and LiDAR at the old mount.
    "original": "sim_scene/dij3d_boxes.usd",

    # The same scene with sim_asset/iris_indoor.usda: raised LiDAR, rangefinder.
    # Physics is identical; see drone_code/check_sensor_assets.py.
    "indoor": "sim_scene/dij3d_boxes_indoor.usda",
}

ACTIVE_SCENE = "original"
SCENE = SCENES[ACTIVE_SCENE]

# Disabled in the loaded stage only, leaving the open ground plane.
#
# With the warehouse gone, LiDAR can only see the floor, which gives height but
# no x, y, or yaw. LiDAR odometry needs it active, and then the hover target must
# move: (0, 0, 2) is 0.24 m from structure, inside the 0.34 m rotor reach.
DEACTIVATE_PRIMS = (
    "/World/BoxWarehouse",
    "/World/Cube",
)

# ---------- Control state ----------

# "imu": the controller and policy use the IMU and estimator (sensor_config.py).
# "ground_truth": they use perfect simulator state, the old baseline.
STATE_SOURCES = ("imu", "ground_truth")
DEFAULT_STATE_SOURCE = "imu"

# ---------- Task ----------

HOVER = {
    # Fixed hover target, world frame [m]; yaw target is 0, as in the demo.
    "target_position": (0.0, 0.0, 2.0),

    # Physics stays at 120 Hz. One PPO decision per 4 steps = 30 Hz.
    "control_decimation": 4,

    # 300 decisions at 30 Hz = 10 s.
    "max_decisions": 300,

    # Episode start, uniform in +-range around hover at the target. Small
    # enough to stay in the controller's near-hover regime.
    "initial_position_range": (0.2, 0.2, 0.2),
    "initial_velocity_range": (0.1, 0.1, 0.1),
    "initial_angle_range_degrees": (3.0, 3.0, 5.0),   # roll, pitch, yaw
    "initial_angular_velocity_range": (0.1, 0.1, 0.1),
}

# Residual limits at action = +-1, added to the PID command.
#
# Hover needs 15.88 N from 28.26 N available, leaving each rotor about
# 3.1 N of upward headroom. PID alone commands about 0.05 N m near hover, so
# these limits are set from what the PID loops need, not from motor authority:
#
#   thrust +-1.5 N: about 10% of hover weight.
#   roll/pitch +-0.04 N m: against the attitude gain kp = 0.2 N m/rad this
#       shifts the equilibrium lean by 11 degrees, a little more than the
#       10 degree tilt limit, which is 1.7 m/s^2 of sustained wind push.
#   yaw +-0.02 N m: the same 11 degrees against kp = 0.1 N m/rad.
#
# All four at their limits on one rotor total about 0.6 N, roughly 20% of its
# headroom, so the residual alone cannot saturate a rotor at hover.
RESIDUAL_LIMITS = {
    "thrust": 1.5,
    "roll_torque": 0.04,
    "pitch_torque": 0.04,
    "yaw_torque": 0.02,
}

# Observation scales: each raw quantity is divided by its scale so VecNormalize
# starts from sensible magnitudes.
OBSERVATION_SCALES = {
    "position_error": 1.0,         # [m]
    "linear_velocity": 1.0,        # [m/s]
    "angular_velocity": 2.0,       # [rad/s]
    "pd_thrust": 5.0,              # [N], offset from hover thrust
    "pd_torque": 0.1,              # [N m]
}

# ---------- Reward ----------

REWARD = {
    "position": 1.0,           # |position error|^2 / (1 m)^2
    "speed": 0.2,              # |linear velocity|^2 / (1 m/s)^2
    "attitude": 0.2,           # (roll^2 + pitch^2) / radians(20)^2
    "angular_rate": 0.05,      # |body angular velocity|^2 / (1 rad/s)^2
    "action": 0.01,            # |normalized residual|^2
    "action_rate": 0.05,       # |change in normalized residual|^2

    "attitude_reference_degrees": 20.0,

    # Constant multiplier on the reward PPO sees, including the failure
    # penalty, so value targets stay near +-10. It does not change the
    # optimal policy; logged components stay unscaled.
    "scale": 0.1,
}

# ---------- Failure ----------

# Crossing any limit terminates the episode. Because every cost term is bounded
# inside these limits, they also fix the failure penalty; see
# hover_task.failure_penalty.
FAILURE = {
    "max_position_error": 2.0,     # [m] leaving the box around the target
    "min_height": 0.5,             # [m] world Z, well above ground contact
    "max_tilt_degrees": 45.0,      # beyond the controller's near-hover regime
    "max_speed": 3.0,              # [m/s]
    "max_angular_speed": 5.0,      # [rad/s]
}

# ---------- Wind ----------

# Each episode draws a mean wind and a gust strength from these ranges.
# sim_config.WIND supplies everything else (timescales, drag, sample points).
#
# Drag is 0.018 * v^2 N, so "weak" pushes with at most about 0.3 N. The demo's
# 8 m/s mean with 2 m/s gusts is close to the 1.5 m/s^2 horizontal acceleration
# cap in controller.py, too hard a place to start learning.
WIND_PROFILES = {
    "none": {
        "enabled": False,
        "mean_speed_range": (0.0, 0.0),
        "gust_std": (0.0, 0.0, 0.0),
        "local_gust_std": (0.0, 0.0, 0.0),
        "gust_scale_range": (0.0, 0.0),
    },
    "weak": {
        "enabled": True,
        "mean_speed_range": (0.0, 3.0),
        "gust_std": (0.5, 0.5, 0.2),
        "local_gust_std": (0.15, 0.15, 0.08),
        "gust_scale_range": (0.5, 1.0),
    },
    "moderate": {
        "enabled": True,
        "mean_speed_range": (2.0, 5.0),
        "gust_std": (1.0, 1.0, 0.35),
        "local_gust_std": (0.3, 0.3, 0.15),
        "gust_scale_range": (0.5, 1.0),
    },
}

DEFAULT_WIND_PROFILE = "weak"

# ---------- PPO ----------

PPO = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,

    # Decisions are 1/30 s, so 0.995 looks about 7 s ahead, longer than the
    # PID position loop's 2 s time constant.
    "gamma": 0.995,
    "gae_lambda": 0.95,

    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    "policy_kwargs": {
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},

        # SB3 already starts the mean near zero, so the policy begins close to
        # PID-only. std = 0.22 keeps early exploration inside the limits.
        "log_std_init": -1.5,
    },

    # An MLP this small trains faster on the CPU than on the GPU.
    "device": "cpu",
}

# ---------- Training run ----------

TRAINING = {
    "checkpoint_interval": 50_000,     # [environment decisions]
    "seed": 0,
    "observation_clip": 10.0,          # applied after VecNormalize
}

RUN_DIR = "runs/ppo_hover"


def vecnormalize_path_for(checkpoint):
    """<prefix>_<N>_steps.zip -> <prefix>_vecnormalize_<N>_steps.pkl."""
    checkpoint = Path(checkpoint)
    name = checkpoint.stem

    if name == "final_model" or name == "interrupted_model":
        prefix = name.removesuffix("_model")
        return checkpoint.with_name(f"{prefix}_vecnormalize.pkl")

    head, steps, tail = name.rsplit("_", 2)
    return checkpoint.with_name(f"{head}_vecnormalize_{steps}_{tail}.pkl")


# ---------- Evaluation ----------

# Fixed seeds, used for both PID-only and PID+PPO. Each seed gives one mean wind,
# gust sequence, and start state, none of which depend on the policy.
EVAL_SEEDS = tuple(range(100_000, 100_020))
