"""SAC residual observations, and running a trained policy in flight.

One frame, normalized by sac_config.OBSERVATION_SCALES and clipped:

    0:3    target position - position, world frame
    3:6    linear velocity, world frame. There is no target velocity at hover.
    6:9    attitude error from level at the target yaw: rotation vector in the
           body frame, taking the short way around, so it has no jump at +-pi
           yaw and no quaternion sign ambiguity
    9:12   angular velocity, body frame
    12:16  PID command: thrust minus hover thrust, then roll/pitch/yaw torque
    16:20  previous normalized action, the one applied up to this frame

The observation is the last HISTORY_LENGTH frames, oldest first, one per RL
decision. All are measured at or before the decision, never after, and the first
frame of an episode fills the whole history. Wind is never observed.

NumPy only, apart from loading a model.
"""
from collections import deque
from pathlib import Path

import numpy as np

import hover_task
import model_contract
import sac_config
import sim_config

from state_estimation import quaternion_multiply
from wind import quaternion_to_matrix


FRAME_SIZE = 20
OBSERVATION_SIZE = FRAME_SIZE * sac_config.HISTORY_LENGTH


def attitude_error(quaternion, target_yaw):
    """Rotation vector from level at target_yaw to the attitude, body frame."""
    q = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)

    target = hover_task.euler_to_quaternion(0.0, 0.0, target_yaw)
    target[1:] *= -1.0

    error = quaternion_multiply(target, q)

    if error[0] < 0.0:
        error = -error

    size = np.linalg.norm(error[1:])

    if size < 1e-9:
        return 2.0 * error[1:]

    return error[1:] * (2.0 * np.arctan2(size, error[0]) / size)


def build_frame(state, target_position, target_yaw, pid_command, hover_thrust,
                previous_action, position_limit=None):
    scales = sac_config.OBSERVATION_SCALES
    R = quaternion_to_matrix(state["orientation"])

    position_error = (
        np.asarray(target_position, dtype=float)
        - np.asarray(state["position"], dtype=float)
    )

    if position_limit is not None:
        position_error = np.clip(
            position_error, -position_limit, position_limit
        )

    frame = np.concatenate([
        position_error / scales["position_error"],
        np.asarray(state["linear_velocity"]) / scales["linear_velocity"],
        attitude_error(state["orientation"], target_yaw)
        / scales["attitude_error"],
        R.T @ np.asarray(state["angular_velocity"])
        / scales["angular_velocity"],
        [(pid_command[0] - hover_thrust) / scales["pid_thrust"]],
        np.asarray(pid_command[1:]) / scales["pid_torque"],
        previous_action,
    ])

    return np.clip(
        frame,
        -sac_config.OBSERVATION_CLIP,
        sac_config.OBSERVATION_CLIP,
    ).astype(np.float32)


class ObservationHistory:
    def __init__(self):
        self.frames = deque(maxlen=sac_config.HISTORY_LENGTH)

    def clear(self):
        self.frames.clear()

    def push(self, frame):
        """Add a frame; return the stacked observation, oldest first."""
        if not self.frames:
            self.frames.extend([frame] * sac_config.HISTORY_LENGTH)
        else:
            self.frames.append(frame)

        return np.concatenate(self.frames)


def load_sac_policy(model_path):
    """Load a trained model; return a deterministic observation -> action."""
    from stable_baselines3 import SAC

    model = SAC.load(str(model_path), device="cpu")

    def policy(observation):
        action, _ = model.predict(observation, deterministic=True)
        return action

    return policy


class FlightResidual:
    """The trained correction, for the navigation loop.

    Called every physics step; decides at RL_RATE_HZ and holds in between.
    Trained on hover only. Tracking moving waypoints is a separate evaluation.
    """

    def __init__(self, model_path, hover_thrust):
        self.policy = load_sac_policy(model_path)
        self.hover_thrust = hover_thrust
        self.decimation = sac_config.physics_steps_per_decision()
        self.limits = hover_task.residual_limits_array(
            sac_config.RESIDUAL_LIMITS
        )
        self.history = ObservationHistory()
        self.reset()

    def reset(self):
        self.history.clear()
        self.step = 0
        self.action = np.zeros(4, dtype=np.float32)
        self.correction = np.zeros(4)

    def update(self, state, waypoint, pid_command, target_yaw):
        """The correction [N, N m, N m, N m] to add to pid_command."""
        if self.step % self.decimation == 0:
            observation = self.history.push(build_frame(
                state,
                waypoint,
                target_yaw,
                pid_command,
                self.hover_thrust,
                self.action,
                position_limit=sac_config.NAVIGATION_ERROR_LIMIT,
            ))

            self.action = np.asarray(
                self.policy(observation), dtype=np.float32
            )
            self.correction = self.action.astype(float) * self.limits

        self.step += 1

        return self.correction


def load_flight_residual(project_dir, sim, log):
    """The configured policy, or None and a clear notice to fly PID only."""
    model_path = Path(project_dir) / sim_config.SAC_MODEL_PATH

    if not model_path.is_file():
        log(
            f"FALLBACK to PID only: no SAC model at {model_path}. "
            "Train one, or set sim_config.SAC_MODEL_PATH."
        )
        return None

    problem = model_contract.sac_incompatibility(
        model_path, sac_config.contract()
    )

    if problem:
        log(f"FALLBACK to PID only: {problem}")
        return None

    log(f"SAC correction loaded: {model_path}")

    return FlightResidual(
        model_path,
        sim.total_mass * hover_task.GRAVITY,
    )
