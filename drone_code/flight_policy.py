"""Shared flight loop: PID adapter, observations, residual timing, and reward.

Training, evaluation, and navigation all fly through FlightCore, so they cannot
disagree about references, observations, action scaling, or timing:

    TimedReference -> ReferencePID -> PID command
                                          + held SAC correction, added once here
                                          -> DroneSim.step_command: mixer, motors,
                                             wind drag, one physics step

Observation frame, normalized by flight_config.OBSERVATION_SCALES and clipped:

    0:3    position error p_ref - p, world frame
    3:6    velocity error v_ref - v, world frame
    6:9    reference velocity, world frame
    9:12   reference acceleration, world frame
    12:18  orientation: first two columns of R, the body-to-world rotation, which
           has no wrap and no quaternion sign ambiguity
    18:21  attitude error from the PID's desired attitude, rotation vector, body
           frame, taking the short way around
    21:24  angular velocity, body frame
    24:28  PID command: thrust minus hover thrust, then roll/pitch/yaw torque
    28:32  previous normalized action, the one applied up to this frame

The observation is the last HISTORY_LENGTH frames, oldest first, one per decision.
The first frame of an episode fills the whole history. Wind is never observed.

NumPy only apart from loading a model. Needs no running simulator: FlightCore takes
any object with DroneSim's methods.
"""
from collections import deque

import numpy as np

import flight_config
import hover_task
import sim_config

from controller import PIDController
from state_estimation import quaternion_multiply
from wind import quaternion_to_matrix


FRAME_SIZE = 32
OBSERVATION_SIZE = FRAME_SIZE * flight_config.HISTORY_LENGTH


def wrap_angle(angle):
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def huber(error, delta):
    """H(e) = 0.5 e^2 for |e| <= delta, else delta (|e| - 0.5 delta)."""
    size = np.abs(error)

    return np.where(
        size <= delta,
        0.5 * size**2,
        delta * (size - 0.5 * delta),
    )


def smoothstep(value):
    share = np.clip(value, 0.0, 1.0)

    return share**2 * (3.0 - 2.0 * share)


def euler_angles(R):
    """Roll, pitch, yaw with the same convention as controller.py."""
    roll, pitch = hover_task.roll_pitch(R)

    return np.array([roll, pitch, np.arctan2(R[1, 0], R[0, 0])])


def attitude_error(quaternion, reference_quaternion):
    """Rotation vector from the reference to the attitude, body frame [rad]."""
    q = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)

    inverse = np.asarray(reference_quaternion, dtype=float).copy()
    inverse[1:] *= -1.0

    error = quaternion_multiply(inverse, q)

    if error[0] < 0.0:
        error = -error

    size = np.linalg.norm(error[1:])

    if size < 1e-9:
        return 2.0 * error[1:]

    return error[1:] * (2.0 * np.arctan2(size, error[0]) / size)


def rotor_correction_bound(allocation_matrix, limits, com_offset):
    """Largest correction any rotor can get, all four actions at worst [N].

    The mixer also shifts torque by the COM offset times the thrust, so a thrust
    correction carries a small torque with it, included here. The sum of absolute
    values is the exact worst case over every combination of signs.
    """
    shift = np.eye(4)
    shift[1, 0] = com_offset[1]
    shift[2, 0] = -com_offset[0]

    to_rotors = np.linalg.inv(allocation_matrix) @ shift

    return np.abs(to_rotors) @ limits


class ReferencePID:
    """Flies a reference through the unchanged position-only PIDController.

    The PID's velocity request is kp/kd * (target - p). Commanding the lead point

        target = p_ref + (kd/kp) * v_ref + a_ref * (1/kp or mass/kp)

    turns its horizontal output into 0.6 e_p + 1.2 e_v + a_ref, and its height
    output into m g + 10 e_z + 5 e_vz + m a_z,ref: position and velocity feedback
    on the tracking errors, with acceleration feedforward. So the PID is commanded
    the same v_ref the reward measures, through its existing interface.

    Its integrators must not see that lead, or they would wind up in the direction
    of travel. Each step therefore makes two calls: update(dt=0) with the lead
    target reads the command and changes nothing, then update(dt) with p_ref itself
    advances the integrators on the true tracking error, and that output is
    discarded. Saturation still reaches them.
    """

    def __init__(self, base_target, mass):
        self.pid = PIDController()
        self.base_target = dict(base_target)
        self.mass = float(mass)

        gains = self.pid.gains
        xy = gains["xy"]
        height = gains["height"]

        self.velocity_lead = np.array([
            xy["kd"] / xy["kp"],
            xy["kd"] / xy["kp"],
            height["kd"] / height["kp"],
        ])
        self.acceleration_lead = np.array([
            1.0 / xy["kp"],
            1.0 / xy["kp"],
            self.mass / height["kp"],
        ])

        attitude = gains["attitude"]
        self.attitude_kp = np.asarray(attitude["kp"], dtype=float)
        self.attitude_kd = np.asarray(attitude["kd"], dtype=float)

    def reset(self):
        self.pid.reset()

    def _target(self, position):
        target = dict(self.base_target)
        target["xy"] = np.asarray(position[:2], dtype=float).copy()
        target["height"] = float(position[2])

        return target

    def lead_point(self, sample):
        return (
            sample.position
            + self.velocity_lead * sample.velocity
            + self.acceleration_lead * sample.acceleration
        )

    def _update(self, state, position, dt, saturated):
        return np.array(
            self.pid.update(
                positions=state["position"],
                orientations=state["orientation"],
                velocities=state["linear_velocity"],
                angular_velocities=state["angular_velocity"],
                target=self._target(position),
                dt=dt,
                actuators_saturated=saturated,
            ),
            dtype=float,
        )

    def read(self, state, sample, saturated):
        """The PID command for this reference; changes no controller state."""
        return self._update(state, self.lead_point(sample), 0.0, saturated)

    def integrate(self, state, sample, dt, saturated):
        """Advance the integrators on p_ref - p."""
        self._update(state, sample.position, dt, saturated)

    def desired_attitude(self, state, command):
        """The roll, pitch, yaw the PID was steering toward [rad].

        The PID does not expose them, but its torque output is exactly
        kp * error - kd * body rate + integral, so this inverts that. It reads the
        real command, so it cannot drift from what the PID did.
        """
        R = quaternion_to_matrix(state["orientation"])
        body_rate = R.T @ np.asarray(state["angular_velocity"], dtype=float)

        error = (
            command[1:]
            + self.attitude_kd * body_rate
            - self.pid.attitude_integral
        ) / self.attitude_kp

        return wrap_angle(euler_angles(R) + error)

    def acceleration_capped(self, state, command, desired):
        """Whether the lead target's horizontal demand reached the PID's cap.

        The discarded integrator call has no lead in it, so it cannot see this
        itself. Rebuilds the demand from the desired tilt, as the PID formed it.
        """
        R = quaternion_to_matrix(state["orientation"])

        vertical = max(
            command[0] * max(float(R[2, 2]), 0.5) / self.mass,
            0.5 * hover_task.GRAVITY,
        )

        forward = vertical * np.tan(desired[1])
        sideways = -np.tan(desired[0]) * np.hypot(forward, vertical)

        cap = min(
            flight_config.PID_ACCELERATION_CAP,
            vertical * np.tan(np.deg2rad(sim_config.MAX_TILT_DEGREES)),
        )

        return bool(np.hypot(forward, sideways) >= 0.98 * cap)


def build_frame(state, sample, desired_attitude_angles, pid_command, hover_thrust,
                previous_action):
    """One observation frame. See the module docstring for its layout."""
    scales = flight_config.OBSERVATION_SCALES
    R = quaternion_to_matrix(state["orientation"])

    desired = hover_task.euler_to_quaternion(*desired_attitude_angles)

    frame = np.concatenate([
        (sample.position - state["position"]) / scales["position_error"],
        (sample.velocity - state["linear_velocity"])
        / scales["velocity_error"],
        sample.velocity / scales["reference_velocity"],
        sample.acceleration / scales["reference_acceleration"],
        R[:, 0],
        R[:, 1],
        attitude_error(state["orientation"], desired)
        / scales["attitude_error"],
        R.T @ np.asarray(state["angular_velocity"])
        / scales["angular_velocity"],
        [(pid_command[0] - hover_thrust) / scales["pid_thrust"]],
        np.asarray(pid_command[1:]) / scales["pid_torque"],
        previous_action,
    ])

    return np.clip(
        frame,
        -flight_config.OBSERVATION_CLIP,
        flight_config.OBSERVATION_CLIP,
    ).astype(np.float32)


def physics_terms(state, sample, desired_attitude_angles, omega_ref, reward):
    """Unweighted costs at one physics step, before any outer weight.

    L_vec is position + speed + direction; L_att is attitude + rate. Each error
    is divided by its scale, so terms are dimensionless, then Huber-penalized.
    """
    def term(error, name):
        settings = reward[name]

        return float(np.sum(huber(
            np.asarray(error, dtype=float) / settings["scale"],
            settings["delta"],
        )))

    R = quaternion_to_matrix(state["orientation"])
    velocity = np.asarray(state["linear_velocity"], dtype=float)
    position_error = sample.position - state["position"]

    speed = np.linalg.norm(velocity)
    reference_speed = np.linalg.norm(sample.velocity)

    threshold = reward["direction_speed_threshold"]
    epsilon = reward["direction_epsilon"]

    direction_weight = (
        reward["direction_weight"]
        * smoothstep(reference_speed / threshold)
        * smoothstep(speed / threshold)
    )
    direction_error = direction_weight * (
        1.0
        - float(
            (sample.velocity / max(reference_speed, epsilon))
            @ (velocity / max(speed, epsilon))
        )
    )

    desired = hover_task.euler_to_quaternion(*desired_attitude_angles)
    body_rate = R.T @ np.asarray(state["angular_velocity"], dtype=float)

    tilt = float(np.arccos(np.clip(R[2, 2], -1.0, 1.0)))
    soft_tilt = np.deg2rad(reward["soft_tilt_degrees"])
    rate = float(np.linalg.norm(body_rate))
    deviation = float(np.linalg.norm(position_error))

    def excess(value, soft):
        return (max(value - soft, 0.0) / soft)**2

    return {
        "position": term(position_error, "position"),
        "speed": term(abs(reference_speed - speed), "speed"),
        "direction": term(direction_error, "direction"),
        "attitude": term(
            attitude_error(state["orientation"], desired), "attitude"
        ),
        "rate": term(omega_ref - body_rate, "rate"),
        "unsafe_tilt": excess(tilt, soft_tilt),
        "unsafe_rate": excess(rate, reward["soft_rate"]),
        "route_deviation": excess(deviation, reward["soft_deviation"]),
    }


def decision_costs(mean_terms, action, previous_action, saturated_share,
                   persistent, reward):
    """Weighted cost of one decision, from physics terms averaged over it.

    Averaging keeps one decision's size independent of the physics rate. Keys are
    the individual logged components. Each is >= 0.
    """
    change = action - previous_action

    costs = {
        name: float(reward["lambda_v"] * mean_terms[name])
        for name in ("position", "speed", "direction")
    }
    costs.update({
        name: float(reward["lambda_a"] * mean_terms[name])
        for name in ("attitude", "rate")
    })
    costs.update({
        name: float(reward[name] * mean_terms[name])
        for name in ("unsafe_tilt", "unsafe_rate", "route_deviation")
    })
    costs.update({
        "action": float(reward["action"] * np.sum(action**2)),
        "action_rate": float(reward["action_rate"] * np.sum(change**2)),
        "saturation": float(reward["saturation"] * saturated_share),
        "persistent_saturation": float(
            reward["persistent_saturation"] * bool(persistent)
        ),
    })

    return costs


def failure_penalty(reward, gamma):
    """Terminal cost that makes crashing worse than any way of staying alive.

    A surviving decision costs at most cost_cap, so continuing from anywhere costs
    less than cost_cap / (1 - gamma) in discounted total. Charging that on failure
    means ending the episode early can never avoid cost.
    """
    return reward["cost_cap"] / (1.0 - gamma)


class CompletionTracker:
    """Arrival: after the motion ends, near the last point, slow, for dwell."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.settled = 0.0
        self.completed = False
        self.completion_time = None

    def update(self, state, sample, final_position, dt, time):
        limits = flight_config.COMPLETION

        close = (
            sample.finished
            and np.linalg.norm(state["position"] - final_position)
            <= limits["radius"]
            and np.linalg.norm(state["linear_velocity"]) <= limits["speed"]
        )

        self.settled = self.settled + dt if close else 0.0

        if not self.completed and self.settled >= limits["dwell"]:
            self.completed = True
            self.completion_time = float(time)


class FlightCore:
    """One drone following one reference: the loop shared by every flight script.

    At a decision: frame = observe(state), then apply_action(policy(frame)).
    Every physics step: step(state). PID and the reference update at the physics
    rate; the correction is held between decisions.
    """

    def __init__(self, sim, log=print):
        self.sim = sim
        self.decimation = flight_config.physics_steps_per_decision()

        if abs(self.decimation * sim.dt - 1.0 / flight_config.RL_RATE_HZ) > 1e-9:
            raise ValueError(
                "flight_config.PHYSICS_RATE_HZ does not match the simulation."
            )

        self.limits = hover_task.residual_limits_array(
            flight_config.RESIDUAL_LIMITS
        )
        self.check_residual_budget(log)

        self.hover_thrust = sim.total_mass * hover_task.GRAVITY
        self.hover_motor_speed = np.sqrt(
            self.hover_thrust / (4 * sim.motors.kf)
        )

        self.pid = ReferencePID(sim.base_target(), sim.total_mass)
        self.frames = deque(maxlen=flight_config.HISTORY_LENGTH)
        self.completion = CompletionTracker()

        self.base_gust_timescale = sim.wind.gust_timescale
        self.base_local_gust_timescale = sim.wind.local_gust_timescale

        self.reference = None
        self.clear()

    def check_residual_budget(self, log):
        """Refuse limits the real geometry turns into too much rotor authority."""
        from drone_diagnostics import read_total_com_in_body_frame

        sim = self.sim

        com = read_total_com_in_body_frame(sim.body, sim.vehicle_bodies)

        bound = rotor_correction_bound(sim.allocation_matrix, self.limits, com)

        budget = (
            flight_config.ROTOR_CORRECTION_FRACTION
            * flight_config.ROTOR_LIMIT_MASS_FACTOR
            * sim.total_mass
            * hover_task.GRAVITY
        )

        log(
            f"Residual: worst rotor correction per rotor {np.round(bound, 3)} N; "
            f"budget {budget:.3f} N; rotor limit {sim.max_rotor_thrust:.2f} N"
        )

        if np.any(bound > budget):
            raise ValueError(
                "flight_config.RESIDUAL_LIMITS exceed the rotor correction "
                f"budget: {np.round(bound, 3)} N against {budget:.3f} N."
            )

    def clear(self):
        self.time = 0.0
        self.saturated = False
        self.direction_rate = 0.0
        self.action = np.zeros(4, dtype=np.float32)
        self.previous_action = self.action.copy()
        self.correction = np.zeros(4)
        self.previous_desired = None
        self.desired = np.zeros(3)
        self.omega_ref = np.zeros(3)

    def begin(self, reference):
        """Start a flight: clear PID, history, held action, and completion."""
        self.reference = reference
        self.pid.reset()
        self.frames.clear()
        self.completion.reset()
        self.clear()

    def apply_wind(self, wind, profile):
        """Start one episode's wind, from hover_task.sample_stage_wind."""
        wind = dict(wind)
        self.direction_rate = wind.pop("direction_rate")

        field = self.sim.wind
        field.gust_timescale = float(
            profile.get("gust_timescale", self.base_gust_timescale)
        )
        field.local_gust_timescale = float(
            profile.get("local_gust_timescale", self.base_local_gust_timescale)
        )
        field.reset_episode(**wind)

    def sample(self):
        return self.reference.sample(self.time)

    def observe(self, state):
        """Build the policy's observation. Call once per decision.

        Also refreshes omega_ref: how fast the PID's own desired attitude is moving,
        which is what the reward's rate term compares against.
        """
        sample = self.sample()
        command = self.pid.read(state, sample, self.saturated)
        desired = self.pid.desired_attitude(state, command)

        if self.previous_desired is not None:
            limit = flight_config.REWARD["reference_rate_limit"]

            self.omega_ref = np.clip(
                wrap_angle(desired - self.previous_desired)
                * flight_config.RL_RATE_HZ,
                -limit,
                limit,
            )

        self.previous_desired = desired

        frame = build_frame(
            state,
            sample,
            desired,
            command,
            self.hover_thrust,
            self.action,
        )

        if not self.frames:
            self.frames.extend([frame] * flight_config.HISTORY_LENGTH)
        else:
            self.frames.append(frame)

        return np.concatenate(self.frames)

    def apply_action(self, action):
        """Hold this normalized action until the next decision."""
        action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(4), -1.0, 1.0
        )

        self.previous_action = self.action
        self.action = action
        self.correction = action.astype(float) * self.limits

    def step(self, state, render=False):
        """One physics step; returns whether the mixer clipped a rotor."""
        sim = self.sim

        if self.direction_rate != 0.0:
            # Turn the mean wind; gusts keep fluctuating around it.
            angle = self.direction_rate * sim.dt
            c, s = np.cos(angle), np.sin(angle)
            x, y, z = sim.wind.mean_wind

            sim.wind.mean_wind = np.array([c*x - s*y, s*x + c*y, z])

        sample = self.sample()
        command = self.pid.read(state, sample, self.saturated)

        self.desired = self.pid.desired_attitude(state, command)

        self.pid.integrate(
            state,
            sample,
            sim.dt,
            self.saturated
            or self.pid.acceleration_capped(state, command, self.desired),
        )

        # u_final = u_PID + delta_u_SAC, added once, before the mixer.
        final = command + self.correction

        _, saturated, _, _ = sim.step_command(state, *final, render=render)

        self.saturated = bool(saturated)
        self.time += sim.dt

        return self.saturated


class FlightTrace:
    """One flight's record, sampled at each decision, for its metrics."""

    def __init__(self):
        self.positions = []
        self.errors = []
        self.action_norms = []
        self.action_changes = []
        self.saturated_steps = 0
        self.physics_steps = 0

    def record(self, position, error, action, previous_action):
        self.positions.append(np.array(position, dtype=float))
        self.errors.append(float(error))
        self.action_norms.append(float(np.linalg.norm(action)))
        self.action_changes.append(
            float(np.linalg.norm(np.asarray(action) - previous_action))
        )

    def metrics(self, path, failure_reason, collided, completion):
        """Metrics over what was flown. Failed flights are reported, not dropped.

        position_rmse: time-indexed, sqrt(mean |p_ref(t) - p(t)|^2).
        cross_track: geometric, distance to the nearest point of the path whatever
            the time. Falling behind on the line adds nothing to it.
        max_deviation: largest time-indexed error.
        """
        from flight_reference import distances_to_path

        errors = np.array(self.errors, dtype=float)
        finite = errors[np.isfinite(errors)]

        if len(finite) == 0:
            finite = np.array([float("nan")])

        positions = np.array(self.positions, dtype=float).reshape(-1, 3)
        positions = positions[np.all(np.isfinite(positions), axis=1)]

        cross = distances_to_path(positions, path)

        if len(cross) == 0:
            cross = np.array([float("nan")])

        return {
            "decisions": len(errors),
            "failed": failure_reason is not None,
            "failure_reason": failure_reason,
            "collided": bool(collided),
            "completed": completion.completed,
            "completion_time": completion.completion_time,
            "position_rmse": float(np.sqrt(np.mean(finite**2))),
            "cross_track_rmse": float(np.sqrt(np.mean(cross**2))),
            "cross_track_max": float(np.max(cross)),
            "max_deviation": float(np.max(finite)),
            "saturation_rate": self.saturated_steps
            / max(self.physics_steps, 1),
            "residual_norm": float(np.mean(self.action_norms))
            if self.action_norms else 0.0,
            "residual_change": float(np.mean(self.action_changes))
            if self.action_changes else 0.0,
        }


def summarize(rows):
    """Means over every flight, failed ones included."""
    def mean(name):
        return float(np.nanmean([row[name] for row in rows]))

    times = [
        row["completion_time"]
        for row in rows
        if row["completion_time"] is not None
    ]

    return {
        "flights": len(rows),
        "position_rmse": mean("position_rmse"),
        "cross_track_rmse": mean("cross_track_rmse"),
        "cross_track_max": float(np.nanmax([r["cross_track_max"] for r in rows])),
        "max_deviation": float(np.nanmax([r["max_deviation"] for r in rows])),
        "completion_rate": mean("completed"),
        "completion_time": float(np.mean(times)) if times else None,
        "failure_rate": mean("failed"),
        "collision_rate": mean("collided"),
        "saturation_rate": mean("saturation_rate"),
        "residual_norm": mean("residual_norm"),
        "residual_change": mean("residual_change"),
    }


def load_flight_policy(model_path):
    """A frozen deterministic observation -> action. No training updates."""
    from sac_policy import load_sac_policy

    return load_sac_policy(model_path)
