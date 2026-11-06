"""The original main.py cascade, with explicit documented physical adaptation."""
from copy import deepcopy

import numpy as np
from simple_pid import PID

from drone_code.controller import integrate
from drone_code.path_sac_policy import ReferencePID, euler_angles, wrap_angle
from drone_code.wind import quaternion_to_matrix
from drone_code import sim_config


class OriginalAdaptedPID:
    """main.py's position -> velocity -> attitude cascade, in SI coordinates.

    Keep original gains and convert the old per-rotor corrections to a wrench.
    Euler feedback / 2 retains the original quaternion-vector small-angle gain.
    Yaw uses a signed angle; q.w in the original is not a heading measurement.
    Integrals use simulation dt; feedforward supports a moving reference.
    """
    def __init__(self, sim):
        self.sim = sim
        self.reset()

    def reset(self):
        self.position_integral = np.zeros(2)
        self.velocity_integral = np.zeros(2)
        self.height_integral = 0.0
        self.attitude_integral = np.zeros(3)
        self.errors = None
        self.desired = np.zeros(3)

    def read(self, state, sample, saturated):
        cfg = self.sim.config
        R = quaternion_to_matrix(state["orientation"])
        angles = euler_angles(R)
        velocity = state["linear_velocity"]
        position_error = sample.position - state["position"]
        velocity_error = sample.velocity - velocity
        # Original planner 2/.15/1.5; derivative feedback on tracking error
        # plus v_ref avoids braking against the reference's own motion.
        desired_velocity = sample.velocity[:2] + 2*position_error[:2] + 1.5*velocity_error[:2] + self.position_integral
        desired_velocity = np.clip(desired_velocity, -cfg.max_xy_speed, cfg.max_xy_speed)
        vel_error = desired_velocity - velocity[:2]
        # Original velocity gains .1/.003/.02, original half-angle convention.
        acceleration = state.get("origin_acceleration", np.zeros(3))
        half_tilt = np.clip(0.1*vel_error + self.velocity_integral - 0.02*acceleration[:2], -0.1, 0.1)
        desired_acceleration = 2*9.81*half_tilt + sample.acceleration[:2]
        magnitude = np.linalg.norm(desired_acceleration)
        cap = min(1.5, 9.81*np.tan(np.deg2rad(cfg.max_tilt_degrees)))
        if magnitude > cap:
            desired_acceleration *= cap / magnitude
        self.desired = np.array([-np.arctan2(desired_acceleration[1], 9.81),
                                  np.arctan2(desired_acceleration[0], 9.81), 0.0])
        attitude_error = wrap_angle(self.desired - angles)
        body_rate = R.T @ state["angular_velocity"]
        half_error = attitude_error * [0.5, 0.5, 1.0]
        half_rate = body_rate * [0.5, 0.5, 1.0]
        corrections = (np.array([2.6785, 2.6785, 0.54])*half_error
                       - np.array([1.2508, 1.2508, 5.358333])*half_rate
                       + self.attitude_integral)
        corrections = np.clip(corrections, [-1, -1, -3], [1, 1, 3])
        # Original motor mixing maps positive stabilizing corrections as below.
        moments = corrections * [4*0.18, 4*0.14, 4*0.0201]
        thrust = (self.sim.total_mass*9.81 + 4*(5.50844*position_error[2]
                  + 1.2*velocity_error[2] + self.height_integral)
                  + self.sim.total_mass*sample.acceleration[2]) / max(R[2, 2], 0.5)
        self.errors = (position_error, vel_error, half_error, magnitude >= cap)
        return np.r_[np.clip(thrust, 0, self.sim.max_total_thrust), moments]

    def integrate(self, state, sample, dt, saturated):
        ep, ev, ea, capped = self.errors
        blocked = saturated or capped
        self.position_integral = integrate(self.position_integral, ep[:2], 0.15, 1.0, 2.0, dt, blocked)
        self.velocity_integral = integrate(self.velocity_integral, ev, 0.003, 0.08, 2.0, dt, blocked)
        self.height_integral = float(integrate(self.height_integral, ep[2], 0.57871, 0.75, 0.5, dt, saturated))
        self.attitude_integral = integrate(self.attitude_integral, ea, [0.56871, 0.56871, 0.0], [0.1, 0.1, 0.1], 0.35, dt, saturated)

    def desired_attitude(self, state, command):
        return self.desired.copy()

    def acceleration_capped(self, state, command, desired):
        return self.errors[3]


class OriginalLiteralPID:
    """Diagnostic replay of main.py's feedback/mixer, using deterministic dt.

    Preserves quaternion-component feedback, yaw setpoint 1, integer planner
    velocity assignment and the original motor correction signs. Never used
    as evidence of a validated baseline. GUI/wall-clock scheduling is replaced
    by the original simulated 100 Hz inner / 5 Hz outer timing.
    """
    def __init__(self, sim):
        self.sim = sim
        self.reset()

    def reset(self):
        self.axes = dict(
            x=PID(2, 0.15, 1.5, output_limits=(-2, 2), sample_time=None),
            y=PID(2, 0.15, 1.5, output_limits=(-2, 2), sample_time=None),
            vx=PID(0.1, 0.003, 0.02, output_limits=(-0.1, 0.1), sample_time=None),
            vy=PID(0.1, 0.003, 0.02, output_limits=(-0.1, 0.1), sample_time=None),
            height=PID(5.50844, 0.57871, 1.2, sample_time=None),
            roll=PID(2.6785, 0.56871, 1.2508, output_limits=(-1, 1), sample_time=None),
            pitch=PID(2.6785, 0.56871, 1.2508, output_limits=(-1, 1), sample_time=None),
            yaw=PID(0.54, 0, 5.358333, setpoint=1, output_limits=(-3, 3), sample_time=None))
        self.time = 0.0
        self.next_outer = 0.2
        self.desired = np.zeros(3)

    def _compute(self, axes, state, sample):
        p, v, q = state["position"], state["linear_velocity"], state["orientation"]
        if self.time + 1e-9 >= self.next_outer:
            axes["x"].setpoint, axes["y"].setpoint = sample.position[:2]
            desired_velocity = np.array([0, 0, 0])  # intentional original dtype
            desired_velocity[0] = axes["x"](p[0], dt=0.2)
            desired_velocity[1] = axes["y"](p[1], dt=0.2)
            axes["vx"].setpoint, axes["vy"].setpoint = desired_velocity[:2]
            axes["pitch"].setpoint = axes["vx"](v[0], dt=0.2)
            axes["roll"].setpoint = -axes["vy"](v[1], dt=0.2)
            distance = sample.position[2] - p[2]
            if distance > 0.5:
                steps = max(1, int((distance/2)/0.25))
                axes["height"].setpoint = p[2] + 2*distance/steps
            else:
                axes["height"].setpoint = sample.position[2]
        thrust = axes["height"](p[2], dt=self.sim.dt) + 3.2495
        roll = -axes["roll"](q[1], dt=self.sim.dt)
        pitch = axes["pitch"](q[2], dt=self.sim.dt)
        yaw = -axes["yaw"](q[0], dt=self.sim.dt)
        controls = np.array([thrust+roll+pitch-yaw, thrust-roll+pitch+yaw,
                             thrust-roll-pitch-yaw, thrust+roll-pitch+yaw])
        return self.sim.allocation_matrix @ controls

    def read(self, state, sample, saturated):
        return self._compute(deepcopy(self.axes), state, sample)

    def integrate(self, state, sample, dt, saturated):
        self._compute(self.axes, state, sample)
        if self.time + 1e-9 >= self.next_outer:
            self.next_outer += 0.2
        self.time += dt

    def desired_attitude(self, state, command):
        return np.array([self.axes["roll"].setpoint, self.axes["pitch"].setpoint, 0.0])

    def acceleration_capped(self, state, command, desired):
        return False


def make_controller(sim):
    if sim.config.controller == "original_adapted":
        return OriginalAdaptedPID(sim)
    if sim.config.controller == "original_literal":
        return OriginalLiteralPID(sim)
    gains = deepcopy(sim_config.PID)
    # Retain Isaac outer loops, use the compiled X2 inertia for attitude tuning.
    inertia = np.diag(np.array(sim.audit()["inertia_body_kg_m2"]))
    frequency = np.array([8.0, 8.0, 5.0])
    gains["attitude"]["kp"] = (inertia*frequency**2).tolist()
    gains["attitude"]["kd"] = (2*0.9*inertia*frequency).tolist()
    return ReferencePID(sim.base_target(), sim.total_mass, gains=gains)
