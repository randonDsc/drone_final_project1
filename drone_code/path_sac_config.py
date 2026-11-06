"""Settings for the path-following SAC residual workflow. See PATH_SAC.md.

The path scripts take no arguments: edit this file, then run them. Nothing here
is read by the hover SAC, hover PPO, or navigation workflows.

Units: meters, seconds, Newtons, radians unless named degrees. Values are project
defaults chosen for this drone, not values from any paper. Standard library plus
sim_config, so scripts can check it before Isaac Sim starts.
"""
import json
import os
from pathlib import Path

if __package__:
    from . import model_contract, sim_config
else:
    import model_contract
    import sim_config


# Saved files keep this workflow's earlier name, "flight", so runs trained
# before the rename still load and resume: this folder, the contract file, the
# "workflow" value in policy_contract(), and flight_sac_* checkpoint names.
RUN_DIR = "runs/flight_sac"
CONTRACT_FILE = "flight_contract.json"

# Bump when the meaning of an observation, the PID adapter, or the reference
# convention changes.
VERSION = 1

# ---------- What to run ----------

# train_path_sac.py
PATH_TRAIN = {
    # "fresh": a new randomly initialized model. Refuses a run folder that already
    #     holds a model. Never loads hover weights or a hover replay buffer.
    # "resume": continues resume_model. Refuses a missing or incompatible one.
    "mode": "resume",
    "resume_model": "runs/flight_sac/flight1/final_model.zip",

    # True: resume with an empty replay buffer if its .pkl is missing.
    "allow_empty_replay_buffer": False,

    # Saved under runs/flight_sac/<run_name>/.
    "run_name": "flight1",

    # SAC decisions to reach, 10 per simulated second. About 24 hours at the 17
    # decisions per second the hover run reached.
    "total_timesteps": 2_000_000,

    "seed": 0,
    "checkpoint_interval": 50_000,      # [decisions], about 50 minutes

    # Each buffer can reach about 520 MB; see SAC below. Older ones are deleted
    # from this run's checkpoints folder. Models are all kept.
    "keep_replay_buffers": 2,

    "gui": False,
}

# eval_path_sac.py
PATH_EVAL = {
    "run_name": "flight1",

    # Model .zip, or None for the run's final_model.zip.
    "model": None,

    # True: fly PID only, with no model.
    "pid_only": False,

    # Use only the first N of EVAL_SEEDS, or None for all.
    "episodes": None,

    # False skips the Dijkstra route, which loads a different scene.
    "include_dijkstra": True,

    "gui": False,
}

# navigate_path_sac.py
PATH_NAV = {
    # "pid" or "pid_sac". pid_sac stops with an error if the model is missing or
    # incompatible; it never quietly flies PID only.
    "mode": "pid_sac",

    "run_name": "flight1",

    # Model .zip, or None for the run's final_model.zip.
    "model": None,

    # A name from EVAL_WIND_PROFILES.
    "wind_profile": "training_range",
    "seed": 18,

    "gui": True,
}

# sweep_path_sac.py. Each variant is a separately trained model, in its own
# folder, multiplying the nominal outer weights. Rescoring one model would not
# test what training on those weights does.
PATH_SWEEP = {
    "base_run_name": "sweep",
    "total_timesteps": 500_000,
    "variants": {
        "nominal": {"lambda_v": 1.0, "lambda_a": 1.0},
        "v0.5": {"lambda_v": 0.5, "lambda_a": 1.0},
        "v2": {"lambda_v": 2.0, "lambda_a": 1.0},
        "a0.5": {"lambda_v": 1.0, "lambda_a": 0.5},
        "a2": {"lambda_v": 1.0, "lambda_a": 2.0},
    },
}

# Set by sweep_path_sac.py for the process it launches.
SWEEP_VARIANT_VARIABLE = "PATH_SAC_SWEEP_VARIANT"

# ---------- Timing ----------

# Must match drone_sim.PHYSICS_DT. PID runs at this rate.
PHYSICS_RATE_HZ = 120

# The policy decides at this rate and its correction is held in between.
# PHYSICS_RATE_HZ / RL_RATE_HZ must be a whole number of physics steps.
RL_RATE_HZ = 10

# ---------- Residual ----------

# Reference scale from a related method, per rotor: |delta_T_i| <= 0.02 * T_lim,
# with T_lim = 1.5 * m * g. For this 1.62 kg drone that is 0.48 N per rotor. It is
# used here only as the budget for the check below.
ROTOR_CORRECTION_FRACTION = 0.02
ROTOR_LIMIT_MASS_FACTOR = 1.5

# Correction at action = +-1, before the mixer. One rotor receives approximately
#
#     thrust / 4  +  roll / (4 * 0.21)  +  pitch / (4 * 0.13)  +  yaw / (4 * 0.06)
#
# with all four at their limits: 0.20 + 0.06 + 0.06 + 0.08 = 0.40 N. That leaves
# margin below 0.48 N for the uneven arms and COM shift. path_sac_policy checks the
# exact sum against the real allocation matrix at startup, and refuses to run if
# any rotor can exceed the budget.
RESIDUAL_LIMITS = {
    "thrust": 0.8,             # [N]
    "roll_torque": 0.05,       # [N m]
    "pitch_torque": 0.03,      # [N m]
    "yaw_torque": 0.02,        # [N m]
}

# ---------- Observation ----------

# Frames stacked, oldest first: 4 frames at 10 Hz is 0.3 s of past.
HISTORY_LENGTH = 4

# Each raw value is divided by its scale, then clipped to +-OBSERVATION_CLIP. With
# the 1 m position scale that passes +-10 m, far past the failure limit, so tracking
# error is never capped the way the hover policy's input was in flight.
OBSERVATION_SCALES = {
    "position_error": 1.0,         # [m]
    "velocity_error": 1.0,         # [m/s]
    "reference_velocity": 2.0,     # [m/s]
    "reference_acceleration": 1.0, # [m/s^2]
    "attitude_error": 0.35,        # [rad]
    "angular_velocity": 2.0,       # [rad/s]
    "pid_thrust": 5.0,             # [N], offset from hover thrust
    "pid_torque": 0.1,             # [N m]
}

OBSERVATION_CLIP = 10.0

# ---------- Reward ----------

# r = -scale * min(sum of costs, cost_cap)  (and the failure penalty on failure)
#
# Each tracking error is divided by its scale first, which makes the terms
# dimensionless and comparable, then goes through the Huber loss. So with delta = 1
# a term turns from quadratic to linear when the error reaches its scale.
REWARD = {
    # Nominal outer weights, from the reference method.
    "lambda_v": 1.0,
    "lambda_a": 1.0,

    # Project choices from here on.
    "position": {"scale": 0.25, "delta": 1.0},     # [m]
    "speed": {"scale": 0.5, "delta": 1.0},         # [m/s]
    "direction": {"scale": 1.0, "delta": 1.0},     # already dimensionless
    "attitude": {"scale": 0.2, "delta": 1.0},      # [rad], about 11 degrees
    "rate": {"scale": 1.0, "delta": 1.0},          # [rad/s]

    # w_dir = direction_weight * g(|v_ref|) * g(|v|), g = smoothstep(x / threshold).
    # The direction of a nearly stopped vector means nothing, so this fades out.
    "direction_weight": 1.0,
    "direction_speed_threshold": 0.3,              # [m/s]
    "direction_epsilon": 1e-6,                     # [m/s], guards the unit vectors

    # Largest desired body rate kept in omega_ref, per axis [rad/s].
    "reference_rate_limit": 1.0,

    # Added costs, applied outside the outer weights.
    "action": 0.005,                   # |normalized action|^2
    "action_rate": 0.02,               # |change in normalized action|^2
    "saturation": 0.5,                 # share of this decision's steps saturated
    "persistent_saturation": 2.0,      # per decision, once persistent
    "unsafe_tilt": 1.0,                # ((tilt - soft) / soft)^2 above soft
    "unsafe_rate": 0.5,                # ((rate - soft) / soft)^2 above soft
    "route_deviation": 1.0,            # ((error - soft) / soft)^2 above soft

    "soft_tilt_degrees": 20.0,
    "soft_rate": 3.0,                  # [rad/s]
    "soft_deviation": 1.0,             # [m]

    # Per-decision cost ceiling. It bounds the cost of any surviving flight, which
    # is what lets the failure penalty below be exact and still a usable size.
    "cost_cap": 10.0,

    # Constant multiplier on the reward SAC sees. Logged costs stay unscaled.
    "scale": 0.1,
}

# Fully saturated decisions in a row before persistent_saturation applies.
PERSISTENT_SATURATION_DECISIONS = 5

# Crossing any limit ends the episode as a failure. Failure costs
# cost_cap / (1 - gamma), the most any continued flight could ever cost, so
# crashing can never beat continuing. There is no contact sensor: in the open scene
# the only thing to hit is the ground, so min_height stands in for a crash.
FAILURE = {
    "max_position_error": 3.0,     # [m] from the reference point
    "min_height": 0.5,             # [m] world Z
    "max_tilt_degrees": 45.0,
    "max_speed": 7.0,              # [m/s]
    "max_angular_speed": 6.0,      # [rad/s]
}

# Arrival: within these of the final point for dwell seconds.
COMPLETION = {
    "radius": 0.15,                # [m]
    "speed": 0.2,                  # [m/s]
    "dwell": 0.5,                  # [s]
}

# ---------- Episodes ----------

# Fixed for all episodes. Spacing is the reference's arc-length step.
EPISODE = {
    "settle_seconds": 3.0,         # hold at the end, part of every episode
    "max_seconds": 60.0,           # time limit, a truncation
    "spacing": 0.05,               # [m]
    "start_xy": 3.0,               # route start, +- this in x and y [m]
    "height_range": (1.2, 4.8),    # routes stay inside this world Z [m]

    # Uniform in +-range around the first reference point.
    "initial_position_range": (0.1, 0.1, 0.1),
    "initial_velocity_range": (0.1, 0.1, 0.1),
    "initial_angle_range_degrees": (3.0, 3.0, 5.0),   # roll, pitch, yaw
    "initial_angular_velocity_range": (0.1, 0.1, 0.1),
}

# The hover scene with the warehouse removed; wind.py's drag at every physics step.
BASE_WIND = {
    "enabled": True,
    "mean_wind": (0.0, 0.0, 0.0),
    "gust_std": (1.0, 1.0, 0.35),          # [m/s], multiplied by gust scale
    "local_gust_std": (0.3, 0.3, 0.15),    # [m/s], multiplied by gust scale
}

# controller.py caps horizontal demand at 1.5 m/s^2, and drag is 0.018 * v^2 N on
# 1.62 kg. In the last stage: 2 m/s into 7 m/s wind is 0.9 m/s^2 of drag, and
# accelerating or turning at 0.6 m/s^2 still fits under the cap.
#
# A stage applies until the total training step reaches until_step. direction_rate
# is how fast the mean wind turns [rad/s].
CURRICULUM = (
    {
        "name": "short_slow",
        "until_step": 150_000,
        "route": {
            "weights": {
                "hover": 0.2, "straight": 0.45, "altitude": 0.25,
                "waypoints": 0.1, "circle": 0.0, "figure_eight": 0.0,
            },
            "speed_range": (0.3, 0.7),
            "acceleration_range": (0.3, 0.4),
            "lateral_acceleration": 0.3,
            "vertical_speed": 0.4,
            "length_range": (1.0, 3.0),
            "height_change_range": (0.0, 0.5),
            "waypoint_count_range": (3, 3),
            "turn_degrees_range": (20.0, 60.0),
            "corner_radius_range": (0.2, 0.5),
            "loop_radius_range": (1.0, 1.5),
            "hover_seconds_range": (4.0, 8.0),
        },
        "wind": {
            "no_wind_probability": 0.5,
            "mean_speed_range": (0.0, 2.0),
            "gust_scale_range": (0.0, 0.25),
            "direction_rate_range": (0.0, 0.0),
        },
    },
    {
        "name": "turns",
        "until_step": 400_000,
        "route": {
            "weights": {
                "hover": 0.1, "straight": 0.25, "altitude": 0.15,
                "waypoints": 0.35, "circle": 0.15, "figure_eight": 0.0,
            },
            "speed_range": (0.5, 1.2),
            "acceleration_range": (0.4, 0.5),
            "lateral_acceleration": 0.4,
            "vertical_speed": 0.6,
            "length_range": (2.0, 6.0),
            "height_change_range": (0.0, 1.0),
            "waypoint_count_range": (3, 4),
            "turn_degrees_range": (30.0, 90.0),
            "corner_radius_range": (0.0, 0.6),
            "loop_radius_range": (1.0, 2.5),
            "hover_seconds_range": (3.0, 6.0),
        },
        "wind": {
            "no_wind_probability": 0.2,
            "mean_speed_range": (0.0, 4.0),
            "gust_scale_range": (0.0, 0.5),
            "direction_rate_range": (0.0, 0.05),
        },
    },
    {
        "name": "climbs_loops",
        "until_step": 800_000,
        "route": {
            "weights": {
                "hover": 0.05, "straight": 0.15, "altitude": 0.15,
                "waypoints": 0.3, "circle": 0.2, "figure_eight": 0.15,
            },
            "speed_range": (0.8, 1.6),
            "acceleration_range": (0.5, 0.6),
            "lateral_acceleration": 0.5,
            "vertical_speed": 0.8,
            "length_range": (3.0, 10.0),
            "height_change_range": (0.0, 1.5),
            "waypoint_count_range": (3, 5),
            "turn_degrees_range": (30.0, 120.0),
            "corner_radius_range": (0.0, 0.8),
            "loop_radius_range": (1.5, 3.5),
            "hover_seconds_range": (2.0, 5.0),
        },
        "wind": {
            "no_wind_probability": 0.1,
            "mean_speed_range": (1.0, 6.0),
            "gust_scale_range": (0.25, 0.75),
            "direction_rate_range": (0.0, 0.1),
        },
    },
    {
        "name": "full",
        "until_step": None,
        "route": {
            "weights": {
                "hover": 0.05, "straight": 0.15, "altitude": 0.15,
                "waypoints": 0.3, "circle": 0.175, "figure_eight": 0.175,
            },
            "speed_range": (1.0, 2.0),
            "acceleration_range": (0.5, 0.6),
            "lateral_acceleration": 0.6,
            "vertical_speed": 1.0,
            "length_range": (4.0, 16.0),
            "height_change_range": (0.0, 2.5),
            "waypoint_count_range": (3, 6),
            "turn_degrees_range": (30.0, 150.0),
            "corner_radius_range": (0.0, 1.0),
            "loop_radius_range": (2.0, 4.0),
            "hover_seconds_range": (2.0, 5.0),
        },
        "wind": {
            "no_wind_probability": 0.1,
            "mean_speed_range": (2.0, 7.0),
            "gust_scale_range": (0.5, 1.0),
            "direction_rate_range": (0.0, 0.2),
        },
    },
)

# ---------- SAC ----------

# stable-baselines3 SAC: twin critics, target networks, automatic temperature.
SAC = {
    "learning_rate": 3e-4,

    # At 10 Hz, 0.99 looks about 10 s ahead, long enough to see braking.
    "gamma": 0.99,

    "tau": 0.005,
    "batch_size": 256,

    # Holds the last 14 simulated hours. About 520 MB in memory and on disk per
    # buffer: 500,000 * 2 * 128 values * 4 bytes, plus actions and rewards.
    "buffer_size": 500_000,

    "learning_starts": 5_000,

    # One gradient update per environment step.
    "train_freq": 1,
    "gradient_steps": 1,

    "ent_coef": "auto",
    "target_entropy": -4.0,            # minus the action size, SAC's usual choice

    "policy_kwargs": {"net_arch": {"pi": [256, 256], "qf": [256, 256]}},

    # The RTX 4070 SUPER. The simulator's own load is what limits the rate.
    "device": "auto",
}

# ---------- Evaluation ----------

# Used by both modes, separate from training seeds. A seed fixes route, start, and
# wind, so PID-only and PID + SAC fly the same realization.
EVAL_SEEDS = tuple(range(300_000, 300_010))

# Scenario -> route type, drawn with the limits of EVAL_ROUTE_STAGE.
EVAL_SCENARIOS = {
    "hover": "hover",
    "straight": "straight",
    "turns": "waypoints",
    "altitude": "altitude",
    "circle": "circle",
    "figure_eight": "figure_eight",
}

EVAL_ROUTE_STAGE = "full"

# training_range: new seeds from the final training distribution.
# unseen_gusts: faster, stronger gusts than training ever used.
# outside_range: mean wind above anything in training.
EVAL_WIND_PROFILES = {
    "training_range": {
        "no_wind_probability": 0.0,
        "mean_speed_range": (2.0, 7.0),
        "gust_scale_range": (0.5, 1.0),
        "direction_rate_range": (0.0, 0.2),
    },
    "unseen_gusts": {
        "no_wind_probability": 0.0,
        "mean_speed_range": (2.0, 6.0),
        "gust_scale_range": (1.2, 1.5),
        "direction_rate_range": (0.1, 0.3),
        "gust_timescale": 0.7,
        "local_gust_timescale": 0.15,
    },
    "outside_range": {
        "no_wind_probability": 0.0,
        "mean_speed_range": (8.0, 10.0),
        "gust_scale_range": (0.5, 1.0),
        "direction_rate_range": (0.0, 0.2),
    },
}

# ---------- Dijkstra route ----------

# Same scene, destination, and height as motor_physics.py, which keeps them inside
# its main(). Update these together if that destination changes.
NAVIGATION = {
    "scene": "sim_scene/dij3d_boxes.usd",
    # Far end of the warehouse: about 41 m, climbing over the racks to 6.6 m.
    # The earlier, short destination was (5.93, 1.12) at 1.12 m.
    "goal_offset_xy": (-2.0, 20.0),    # from the USD start [m]
    "goal_height": 1.2,                # world Z [m]

    "speed": 0.8,                      # [m/s]
    "acceleration": 0.5,               # [m/s^2]
    "lateral_acceleration": 0.5,       # [m/s^2]
    "vertical_speed": 0.5,             # [m/s]

    # Corners are rounded by up to this, and only where every chord passes the
    # planner's own line_of_sight check. Otherwise the corner stays sharp, and the
    # speed profile nearly stops there [m].
    "corner_radius": 0.4,

    "start_hold": 1.0,                 # [s]
    "settle_seconds": 5.0,             # [s]

    # Stops the flight with a failure. Tighter than training: walls are close.
    "max_position_error": 1.0,         # [m]

    # The policy never trained with ground contact, so its correction stays zero
    # until the launch departure has been flown.
    "residual_after_launch": True,

    # Half-size around the body origin that counts as touching an obstacle [m].
    # Tighter than the planner's conservative clearance, which is reported apart.
    "collision_half_size": (0.25, 0.25, 0.06),
}

# ---------- Derived ----------

# In controller.py; horizontal demand above this is cut [m/s^2].
PID_ACCELERATION_CAP = 1.5


def physics_steps_per_decision():
    steps = PHYSICS_RATE_HZ / RL_RATE_HZ

    if abs(steps - round(steps)) > 1e-9:
        raise ValueError("RL_RATE_HZ must divide PHYSICS_RATE_HZ exactly.")

    return int(round(steps))


def stage_by_name(name):
    for stage in CURRICULUM:
        if stage["name"] == name:
            return stage

    names = ", ".join(stage["name"] for stage in CURRICULUM)
    raise ValueError(f"Unknown curriculum stage {name}; choose from {names}.")


def stage_for_step(step):
    for stage in CURRICULUM:
        limit = stage["until_step"]

        if limit is None or step < limit:
            return stage

    return CURRICULUM[-1]


def sweep_variant():
    """The active sweep variant's name, or None for an ordinary run."""
    name = os.environ.get(SWEEP_VARIANT_VARIABLE)

    if name is not None and name not in PATH_SWEEP["variants"]:
        raise ValueError(f"Unknown sweep variant: {name}")

    return name


def train_settings():
    """PATH_TRAIN, or its sweep version: fresh, with its own run folder."""
    settings = dict(PATH_TRAIN)
    variant = sweep_variant()

    if variant is not None:
        settings.update(
            mode="fresh",
            resume_model=None,
            run_name=f"{PATH_SWEEP['base_run_name']}_{variant}",
            total_timesteps=PATH_SWEEP["total_timesteps"],
        )

    return settings


def eval_settings():
    settings = dict(PATH_EVAL)
    variant = sweep_variant()

    if variant is not None:
        settings.update(
            model=None,
            pid_only=False,
            include_dijkstra=False,
            run_name=f"{PATH_SWEEP['base_run_name']}_{variant}",
        )

    return settings


def reward_settings():
    """REWARD with the active sweep variant's multipliers."""
    reward = dict(REWARD)
    variant = sweep_variant()

    if variant is not None:
        factors = PATH_SWEEP["variants"][variant]
        reward["lambda_v"] = REWARD["lambda_v"] * factors["lambda_v"]
        reward["lambda_a"] = REWARD["lambda_a"] * factors["lambda_a"]

    return reward


def policy_contract():
    """What a saved policy needs in order to fly. Any change needs a new model."""
    return {
        "workflow": "flight_sac",
        "version": VERSION,
        "rl_rate_hz": RL_RATE_HZ,
        "physics_rate_hz": PHYSICS_RATE_HZ,
        "history_length": HISTORY_LENGTH,
        "observation_scales": OBSERVATION_SCALES,
        "observation_clip": OBSERVATION_CLIP,
        "residual_limits": RESIDUAL_LIMITS,
        "pid": sim_config.PID,
    }


def training_contract():
    """What must also stay the same for training to continue."""
    return {
        "reward": reward_settings(),
        "failure": FAILURE,
        "episode": EPISODE,
        "curriculum": CURRICULUM,
        "sac": SAC,
    }


def write_contract(run_dir):
    (Path(run_dir) / CONTRACT_FILE).write_text(
        json.dumps(
            {"policy": policy_contract(), "training": training_contract()},
            indent=2,
        ),
        encoding="utf-8",
    )


def incompatibility(model_path, check_training=False):
    """Why this model cannot be used here, or None if it can."""
    path = model_contract.run_folder(model_path) / CONTRACT_FILE

    if not path.is_file():
        return (
            f"{model_path}\n"
            f"has no {CONTRACT_FILE} in its run folder, so it is not a flight "
            "model from train_path_sac.py. Hover SAC and PPO models are refused."
        )

    saved = json.loads(path.read_text(encoding="utf-8"))
    groups = [("policy", policy_contract())]

    if check_training:
        groups.append(("training", training_contract()))

    different = []

    for group, expected in groups:
        expected = json.loads(json.dumps(expected))
        stored = saved.get(group, {})

        different += sorted(
            f"{group}.{name}"
            for name, value in expected.items()
            if stored.get(name) != value
        )

    if different:
        return (
            f"{model_path}\n"
            f"does not match the current settings: {', '.join(different)}.\n"
            "Train a new model, or restore the settings it was trained with."
        )

    return None


def replay_buffer_path_for(model_path):
    """The replay buffer saved with a model, by this workflow's names."""
    model_path = Path(model_path)
    name = model_path.stem

    if name.endswith("_model"):
        prefix = name.removesuffix("_model")
        return model_path.with_name(f"{prefix}_replay_buffer.pkl")

    steps = name.rsplit("_", 2)[1]
    return model_path.with_name(f"flight_replay_buffer_{steps}_steps.pkl")
