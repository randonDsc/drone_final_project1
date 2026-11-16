"""PPO direct control: the policy's action is a world-frame velocity and a yaw angle (M8, design
docs/milestone8_direct_control_plan.md; prespec docs/residual_vs_direct_prespec.md, A4/A5).

    action a in Box(-1, 1, (4,))  ->  v_cmd = (3 a0, 3 a1, 1.5 a2) m/s (world frame), yaw_cmd = pi a3 rad

The inner loop is the SHARED, unmodified PIDController (controller.py), capped at 1.5 m/s^2 horizontally (D-env-1 (a),
prespec A5). It takes only a position target, so the velocity command enters by TARGET SUBSTITUTION, the technique the
shared ReferencePID itself uses (path_sac_policy.py:112-122):

    PIDController request:  v_des = kp/kd * (target - p)        (controller.py:152-153, 195-200)
    we pass:                target = p + (kd/kp) * v_cmd        ->  v_des = v_cmd (to ~1e-15 m/s; floating point)

Nothing else differs from the residual arm (PathSacEnv): reset (wind, route, initial state, in the same RNG order),
physics substeps, observation layout, reward code and constants, failure and completion rules are all inherited. The
held residual correction is always 0, so the inner loop's command is what reaches the mixer.

Features 18-20 and 24-27 and the attitude / rate reward references come from the inner loop under the policy's own
command, exactly as they come from the PID in the residual arm (D-env-2; no shadow PID anywhere).

Integrators (D-env-3): reading the command uses dt = 0 at the substituted target, which changes no state. The
integrate call targets the CURRENT position, so the xy and height position errors it sees are exactly 0 and those two
integrators stay at 0 (controller.py:25: step = ki * 0 * dt). The yaw attitude integrator (ki = 0.02) still integrates
the true yaw error; roll/pitch have ki = 0. Integrating at the substituted target instead would wind the xy and height
integrators up on the command (plan section 2e / D-env-3).

NumPy + gymnasium only at import (like path_sac_env); make_direct_env needs a running SimulationApp.
"""
import numpy as np

import path_sac_config
from hover_env import make_hover_sim
from path_sac_env import PathSacEnv
from path_sac_policy import PathCore, ReferencePID

VELOCITY_LIMITS = np.array([3.0, 3.0, 1.5])   # [m/s] world frame, |a| <= 1 -> |v| <= limit
YAW_LIMIT = float(np.pi)                      # [rad] yaw ANGLE command (controller.py wraps the error, :267-270)
INTEGRATOR_HANDLING = "integrate at the current position: xy/height integrators stay 0, yaw integrates"


def action_to_command(action):
    """Normalized action -> (world velocity command [m/s], yaw angle command [rad]). Clips to [-1, 1] first."""
    a = np.clip(np.asarray(action, dtype=np.float32).reshape(4), -1.0, 1.0)
    velocity = a[:3].astype(float) * VELOCITY_LIMITS
    return velocity, float(a[3]) * YAW_LIMIT


class VelocityPID(ReferencePID):
    """The shared PIDController, commanded a velocity and a yaw angle through its position-target interface."""

    def __init__(self, base_target, mass):
        super().__init__(base_target, mass)
        self.set_command(np.zeros(3), 0.0)

    def reset(self):
        super().reset()
        self.set_command(np.zeros(3), 0.0)

    def set_command(self, velocity, yaw):
        self.velocity_command = np.asarray(velocity, dtype=float).reshape(3).copy()
        self.yaw_command = float(yaw)

    def substituted_target(self, position):
        """Position target whose PID velocity request is the velocity command: p + (kd/kp) v_cmd per axis."""
        return np.asarray(position, dtype=float).reshape(3) + self.velocity_lead * self.velocity_command

    def _target(self, position):
        target = super()._target(position)
        target["yaw"] = self.yaw_command   # the yaw ANGLE itself; the controller takes the short way (:267-270)
        return target

    def read(self, state, sample, saturated):
        """The inner loop's command for the held velocity/yaw command; changes no controller state (dt = 0).

        `sample` (the reference) is not used: the reference never reaches the inner loop."""
        return self._update(state, self.substituted_target(state["position"]), 0.0, saturated)

    def integrate(self, state, sample, dt, saturated):
        """Advance the integrators with zero position error (target = current position); output discarded."""
        self._update(state, np.asarray(state["position"], dtype=float).reshape(3), dt, saturated)


class DirectCore(PathCore):
    """PathCore with the velocity-commanded inner loop. Only the action path differs; step/observe are inherited."""

    def __init__(self, sim, log=print):
        super().__init__(sim, log=log)
        self.pid = VelocityPID(sim.base_target(), sim.total_mass)

    def apply_action(self, action):
        """Hold this normalized action until the next decision: it sets the inner loop's command."""
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(4), -1.0, 1.0)

        self.previous_action = self.action
        self.action = action
        self.correction = np.zeros(4)   # no residual: the inner loop's command reaches the mixer unchanged
        self.pid.set_command(*action_to_command(action))


class DirectEnv(PathSacEnv):
    """PathSacEnv with DirectCore. Spaces, reset, step, reward and termination are inherited unchanged."""

    def __init__(self, sim, render_sim=False, log=print):
        super().__init__(sim, render_sim=render_sim, log=log)
        self.core = DirectCore(sim, log=log)

    def step(self, action):
        """PathSacEnv.step, plus the inner-loop integrator magnitudes in info (read-only; for divergence checks)."""
        observation, reward, terminated, truncated, info = super().step(action)
        info["integrators"] = integrator_magnitudes(self.core.pid.pid)
        return observation, reward, terminated, truncated, info


def integrator_magnitudes(pid):
    """|integrator| of a PIDController: xy (max of 2 axes) [m/s^2], height [N], attitude (max of 3 axes) [N m]."""
    return {"xy_abs_max": float(np.max(np.abs(pid.xy_integral))),
            "height_abs": abs(float(pid.height_integral)),
            "attitude_abs_max": float(np.max(np.abs(pid.attitude_integral)))}


def make_direct_env(simulation_app, render_sim=False, log=print):
    """As path_sac_env.make_path_sac_env, with DirectEnv. Needs SimulationApp running."""
    sim = make_hover_sim(simulation_app, path_sac_config.BASE_WIND, log=log)
    sim.world.play()

    return DirectEnv(sim, render_sim=render_sim, log=log)
