"""One thrust budget shared by every axis: when one axis takes more, the others
get less. NumPy and sim_config only, so the controller, the planner, and tests all
use the same numbers. See sim_config.THRUST_ENVELOPE.

The rotors can make one thrust vector of limited length. Holding height takes
m g of it, so

    (horizontal force)^2 + (vertical force)^2 <= T_usable^2

with T_usable = usable_fraction * max total thrust. The rest is kept for attitude
torque. Priority is attitude > vertical > horizontal: vertical acceleration is
limited first, but never so far that horizontal keeps less than min_xy_acceleration,
and horizontal gets what is left. Running out of thrust then means drifting
sideways, not sinking.

Speeds: with drag k v^2 against the motion, steady flight at climb angle gamma
needs thrust (k v^2 cos gamma, m g + k v^2 sin gamma). Its length reaches the
budget at

    k v^2 = -W sin gamma + sqrt(T^2 - W^2 cos^2 gamma),   W = m g

so full horizontal speed means no climb, and the other way round. That formula is
for still air. In wind it only shapes the speed the controller asks for; the
acceleration limits act on the real thrust, so wind and the integrators use up the
budget by themselves.
"""
import numpy as np

import sim_config


GRAVITY = 9.81


def drag_coefficient(wind=None):
    """k in drag = k v^2 [N/(m/s)^2], from the wind model's settings."""
    wind = sim_config.WIND if wind is None else wind

    return (
        0.5 * wind["air_density"] * wind["drag_coefficient"]
        * wind["effective_area"]
    )


class ThrustEnvelope:
    """Limits for one vehicle. All accelerations are world frame [m/s^2]."""

    def __init__(self, mass, max_total_thrust, settings=None, drag=None,
                 max_xy_speed=None, max_z_speed=None, max_tilt_degrees=None):
        settings = sim_config.THRUST_ENVELOPE if settings is None else settings

        self.mass = float(mass)
        self.max_total_thrust = float(max_total_thrust)
        self.usable_thrust = settings["usable_fraction"] * self.max_total_thrust
        self.drag = drag_coefficient() if drag is None else float(drag)

        self.max_up = float(settings["max_up_acceleration"])
        self.max_down = float(settings["max_down_acceleration"])
        self.min_xy = float(settings["min_xy_acceleration"])
        self.reserve = float(settings["planning_reserve_acceleration"])

        self.max_xy_speed = float(
            sim_config.MAX_XY_SPEED if max_xy_speed is None else max_xy_speed
        )
        self.max_z_speed = float(
            sim_config.MAX_Z_SPEED if max_z_speed is None else max_z_speed
        )
        self.tan_tilt = np.tan(np.deg2rad(float(
            sim_config.MAX_TILT_DEGREES
            if max_tilt_degrees is None else max_tilt_degrees
        )))

        # Specific thrust: the most acceleration the budget gives, gravity included.
        self.budget = self.usable_thrust / self.mass

        if self.budget**2 < GRAVITY**2 + self.min_xy**2:
            raise ValueError(
                "THRUST_ENVELOPE leaves no horizontal acceleration while "
                "holding height. Raise usable_fraction or lower "
                "min_xy_acceleration."
            )

    # ---------- Controller ----------

    def vertical_limits(self):
        """(lowest, highest) vertical acceleration.

        Climbing never takes so much that horizontal keeps less than min_xy.
        Down is limited by thrust >= 0 as well, which is at most -g.
        """
        up = min(
            self.max_up,
            np.sqrt(self.budget**2 - self.min_xy**2) - GRAVITY,
        )
        down = min(self.max_down, GRAVITY)

        return -down, up

    def horizontal_limit(self, vertical_acceleration):
        """Most horizontal acceleration left at this vertical acceleration.

        Also kept under the tilt cap: tan(tilt) = a_xy / (g + a_z).
        """
        lift = GRAVITY + float(vertical_acceleration)
        left = np.sqrt(max(self.budget**2 - lift**2, 0.0))

        return max(min(left, max(lift, 0.0) * self.tan_tilt), 0.0)

    def steady_speed(self, direction, thrust=None):
        """Fastest steady still-air speed along direction [m/s], within the
        request caps. thrust defaults to the usable budget [N]."""
        thrust = self.usable_thrust if thrust is None else float(thrust)
        direction = np.asarray(direction, dtype=float).reshape(3)
        size = np.linalg.norm(direction)

        if size < 1e-9:
            return 0.0

        sine = direction[2] / size
        cosine = np.hypot(direction[0], direction[1]) / size
        weight = self.mass * GRAVITY

        root = thrust**2 - (weight * cosine)**2
        drag = -weight * sine + np.sqrt(max(root, 0.0))

        speed = np.sqrt(max(drag, 0.0) / self.drag) if self.drag > 0 else np.inf

        # Descending: drag pulls up, and with zero thrust the drone falls at
        # sqrt(W / k). Steeper than that has no steady state, so the caps decide.
        if cosine > 1e-9:
            speed = min(speed, self.max_xy_speed / cosine)

        if abs(sine) > 1e-9:
            speed = min(speed, self.max_z_speed / abs(sine))

        return float(speed)

    def limit_velocity(self, velocity):
        """Scale a requested world velocity, keeping its direction, into the
        envelope. Returns (velocity, whether it was cut)."""
        velocity = np.asarray(velocity, dtype=float).reshape(3)
        speed = np.linalg.norm(velocity)

        if speed < 1e-9:
            return velocity, False

        most = self.steady_speed(velocity)

        if speed <= most:
            return velocity, False

        return velocity * (most / speed), True

    # ---------- Planning ----------

    def planning_thrust(self):
        """Budget a reference may use: what is left after the reserve
        acceleration the controller keeps for feedback and wind [N]."""
        return self.usable_thrust - self.mass * self.reserve

    def planning_acceleration(self):
        """Largest reference acceleration that fits in any direction [m/s^2].

        Worst case is straight up plus sideways: (g + a)^2 + a^2 = A^2, with A
        the planning budget. Also inside the vertical and tilt limits.
        """
        budget = self.planning_thrust() / self.mass
        root = 2.0 * budget**2 - GRAVITY**2

        if root <= GRAVITY**2:
            raise ValueError(
                "planning_reserve_acceleration leaves the planner no "
                "acceleration."
            )

        a = 0.5 * (-GRAVITY + np.sqrt(root))
        _, up = self.vertical_limits()

        return float(min(a, up, GRAVITY * self.tan_tilt))

    def planning_speed(self, direction):
        """Fastest reference speed along direction, inside the planning budget."""
        return self.steady_speed(direction, self.planning_thrust())


def desaturate(allocation_matrix, max_rotor_thrust, wrench, com_offset):
    """The mixer; returns (rotor thrusts [N], saturated). See mix."""
    rotors, saturated, _ = mix(
        allocation_matrix, max_rotor_thrust, wrench, com_offset
    )

    return rotors, saturated


def mix(allocation_matrix, max_rotor_thrust, wrench, com_offset):
    """The mixer: rotor thrusts for wrench [T, roll, pitch, yaw] about the COM.

    Returns (rotor thrusts [N], saturated, torque_fits). When the exact wrench
    does not fit
    between 0 and the rotor limit, torque is kept and total thrust gives way: all
    rotors move together until every one fits, as PX4 and Betaflight mixers do.
    The drone briefly loses some lift and keeps steering. Only if the torque alone
    cannot fit does it fall back to clipping each rotor. saturated is True
    whenever the delivered wrench differs from the request. torque_fits is False
    only in that fallback: no total thrust delivers the requested torque, so the
    drone is out of control authority, not just short of lift.

    allocation_matrix maps rotor thrusts to the wrench about the body origin.
    Thrust applied there also turns the vehicle about its COM, by
    com x [0, 0, T], which is linear in T, so it is part of the thrust direction.
    """
    allocation_matrix = np.asarray(allocation_matrix, dtype=float)
    com_offset = np.asarray(com_offset, dtype=float).reshape(3)
    wrench = np.asarray(wrench, dtype=float).reshape(4)

    thrust_part = np.linalg.solve(
        allocation_matrix,
        np.array([1.0, com_offset[1], -com_offset[0], 0.0]),
    )
    torque_part = np.linalg.solve(
        allocation_matrix,
        np.array([0.0, *wrench[1:]]),
    )

    requested = float(wrench[0])
    limit = float(max_rotor_thrust)
    rotors = torque_part + requested * thrust_part

    if np.all(rotors >= 0.0) and np.all(rotors <= limit):
        return rotors, False, True

    # Every rotor bounds T: 0 <= torque_i + T * thrust_i <= limit. thrust_part is
    # positive for every rotor of a normal quad frame.
    if np.any(thrust_part <= 0.0):
        return np.clip(rotors, 0.0, limit), True, False

    lowest = np.max(-torque_part / thrust_part)
    highest = np.min((limit - torque_part) / thrust_part)
    torque_fits = bool(lowest <= highest)

    if torque_fits:
        thrust = float(np.clip(requested, lowest, highest))
    else:
        thrust = 0.5 * (lowest + highest)

    return (
        np.clip(torque_part + thrust * thrust_part, 0.0, limit),
        True,
        torque_fits,
    )


def for_target(target):
    """The envelope for a controller target dictionary."""
    return ThrustEnvelope(
        target.get("mass", 1.5),
        target.get("max_total_thrust", 4 * 5.84e-6 * 1100.0**2),
        max_xy_speed=target.get("max_xy_speed"),
        max_z_speed=target.get("max_z_speed"),
        max_tilt_degrees=target.get("max_tilt_degrees"),
    )
