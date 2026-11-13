"""Hover task definition: observations, reward, failure, episode sampling.

NumPy only, so all of it can be tested without Isaac Sim. hover_env.py connects
these functions to the simulator.

Observation (float32, 23 values, before VecNormalize), scales in hover_ppo_config:

    0:3    target position - position, world frame
    3:6    linear velocity, world frame
    6:12   orientation: first two columns of R, the body-to-world rotation.
           Unlike a quaternion it has no sign ambiguity (q and -q are the same
           attitude) and no jump at any attitude.
    12:15  angular velocity, body frame
    15:19  PID command: thrust minus hover thrust, then roll/pitch/yaw torque
    19:23  previous normalized action

The wind is deliberately not observed; the policy must infer it from motion.
"""
import numpy as np

from wind import quaternion_to_matrix


OBSERVATION_SIZE = 23
GRAVITY = 9.81


def residual_limits_array(limits):
    return np.array([
        limits["thrust"],
        limits["roll_torque"],
        limits["pitch_torque"],
        limits["yaw_torque"],
    ], dtype=float)


def roll_pitch(R):
    """Roll and pitch with the same convention as controller.py."""
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    return roll, pitch


def euler_to_quaternion(roll, pitch, yaw):
    """Quaternion [w, x, y, z] for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def state_is_valid(state):
    return all(
        np.all(np.isfinite(state[key]))
        for key in (
            "position",
            "orientation",
            "linear_velocity",
            "angular_velocity",
        )
    ) and np.linalg.norm(state["orientation"]) > 1e-8


def build_observation(state, target_position, pd_command, hover_thrust,
                      previous_action, scales):
    R = quaternion_to_matrix(state["orientation"])
    body_rate = R.T @ np.asarray(state["angular_velocity"], dtype=float)

    observation = np.concatenate([
        (target_position - state["position"]) / scales["position_error"],
        np.asarray(state["linear_velocity"]) / scales["linear_velocity"],
        R[:, 0],
        R[:, 1],
        body_rate / scales["angular_velocity"],
        [(pd_command[0] - hover_thrust) / scales["pd_thrust"]],
        np.asarray(pd_command[1:]) / scales["pd_torque"],
        previous_action,
    ])

    return observation.astype(np.float32)


def step_costs(state, target_position, action, previous_action, weights):
    """Weighted per-decision cost terms, each >= 0. Keys match REWARD."""
    R = quaternion_to_matrix(state["orientation"])
    roll, pitch = roll_pitch(R)

    reference = np.deg2rad(weights["attitude_reference_degrees"])
    error = target_position - state["position"]
    change = action - previous_action

    terms = {
        "position": np.sum(error**2),
        "speed": np.sum(np.square(state["linear_velocity"])),
        "attitude": (roll**2 + pitch**2) / reference**2,
        # A rotation does not change a vector's length, so this equals the
        # squared body angular speed.
        "angular_rate": np.sum(np.square(state["angular_velocity"])),
        "action": np.sum(action**2),
        "action_rate": np.sum(change**2),
    }

    return {
        name: float(weights[name] * value)
        for name, value in terms.items()
    }


def failure_reason(state, target_position, limits):
    """Return why the episode must end, or None for a healthy state."""
    if not state_is_valid(state):
        return "invalid_state"

    R = quaternion_to_matrix(state["orientation"])
    position = np.asarray(state["position"], dtype=float)

    if position[2] < limits["min_height"]:
        return "too_low"

    error = np.linalg.norm(target_position - position)

    if error > limits["max_position_error"]:
        return "out_of_bounds"

    if np.arccos(np.clip(R[2, 2], -1.0, 1.0)) > np.deg2rad(
        limits["max_tilt_degrees"]
    ):
        return "excessive_tilt"

    if np.linalg.norm(state["linear_velocity"]) > limits["max_speed"]:
        return "too_fast"

    if np.linalg.norm(state["angular_velocity"]) > limits["max_angular_speed"]:
        return "spinning"

    return None


def max_step_cost(weights, limits):
    """Upper bound on the total cost of any non-failed decision."""
    tilt = np.deg2rad(limits["max_tilt_degrees"])
    reference = np.deg2rad(weights["attitude_reference_degrees"])

    # cos(tilt) = cos(roll) cos(pitch) bounds each angle by the tilt.
    worst_attitude = 2.0 * tilt**2 / reference**2

    return float(
        weights["position"] * limits["max_position_error"]**2
        + weights["speed"] * limits["max_speed"]**2
        + weights["attitude"] * worst_attitude
        + weights["angular_rate"] * limits["max_angular_speed"]**2
        + weights["action"] * 4.0            # four actions of magnitude 1
        + weights["action_rate"] * 16.0      # four changes of magnitude 2
    )


def failure_penalty(weights, limits, gamma):
    """Terminal cost that makes crashing worse than any way of staying alive.

    With only costs, termination is attractive: it stops future costs from
    accumulating. A flight inside the limits costs at most c = max_step_cost
    per decision, so its discounted future cost from any moment is below
    c / (1 - gamma). Charging exactly that on failure means crashing can never
    beat continuing, even from the worst state still inside the limits.

    It is the infinite-horizon sum, not one for the steps left, because PPO
    bootstraps the value at a time limit: a surviving flight still carries
    future cost at step 299, so a crash there must not look cheap.
    """
    return max_step_cost(weights, limits) / (1.0 - gamma)


def sample_wind(rng, profile):
    """Draw one episode's wind: mean vector, gust scale, and gust seed.

    A profile with its own "mean_wind" is fixed: only the gust seed changes.
    """
    if "mean_wind" in profile:
        return {
            "mean_wind": np.asarray(profile["mean_wind"], dtype=float).copy(),
            "gust_scale": 1.0,
            "seed": int(rng.integers(0, 2**63 - 1)),
        }

    low, high = profile["mean_speed_range"]
    speed = rng.uniform(low, high)
    direction = rng.uniform(-np.pi, np.pi)

    low, high = profile["gust_scale_range"]

    return {
        "mean_wind": np.array([
            speed * np.cos(direction),
            speed * np.sin(direction),
            0.0,
        ]),
        "gust_scale": float(rng.uniform(low, high)),
        # A fresh seed per episode, so gusts never repeat in training.
        "seed": int(rng.integers(0, 2**63 - 1)),
    }


def sample_stage_wind(rng, stage):
    """Draw one episode's wind from a hover_sac_config.WIND_STAGES entry.

    Always makes the same draws, so the start state after it does not depend on
    which kind of wind came up. direction_rate turns the mean wind [rad/s].
    """
    calm = rng.random() < stage["no_wind_probability"]
    speed = rng.uniform(*stage["mean_speed_range"])
    direction = rng.uniform(-np.pi, np.pi)
    gust_scale = float(rng.uniform(*stage["gust_scale_range"]))
    rate = float(rng.uniform(*stage["direction_rate_range"]))
    rate *= 1.0 if rng.random() < 0.5 else -1.0
    seed = int(rng.integers(0, 2**63 - 1))

    if calm:
        speed, gust_scale, rate = 0.0, 0.0, 0.0

    return {
        "mean_wind": np.array([
            speed * np.cos(direction),
            speed * np.sin(direction),
            0.0,
        ]),
        "gust_scale": gust_scale,
        "seed": seed,
        "direction_rate": rate,
    }


def sample_initial_state(rng, task):
    """Airborne start near the target, independent of wind and policy."""
    position_range = np.asarray(task["initial_position_range"], dtype=float)
    velocity_range = np.asarray(task["initial_velocity_range"], dtype=float)
    rate_range = np.asarray(
        task["initial_angular_velocity_range"], dtype=float
    )
    angle_range = np.deg2rad(
        np.asarray(task["initial_angle_range_degrees"], dtype=float)
    )

    return {
        "position": (
            np.asarray(task["target_position"], dtype=float)
            + rng.uniform(-position_range, position_range)
        ),
        "orientation": euler_to_quaternion(
            *rng.uniform(-angle_range, angle_range)
        ),
        "linear_velocity": rng.uniform(-velocity_range, velocity_range),
        "angular_velocity": rng.uniform(-rate_range, rate_range),
    }
