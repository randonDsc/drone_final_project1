"""Low-level near-hover PID flight controller; no maps, no search.

Input state is world position (m), unit quaternion [w,x,y,z], world linear
velocity (m/s), and world angular velocity (rad/s). The target dictionary holds
ONE active waypoint. navigation.py chooses that waypoint from the global route.
Output is body +Z thrust (N) and body roll/pitch/yaw torque (N m).

Gains are in sim_config.PID. See CONTROL.md for the layers. The integrators are
stored here, in world axes for position and body axes for attitude, and each one
is kept in the units of the output it adds to.

Speed, vertical acceleration, horizontal acceleration, and tilt share one thrust
budget (thrust_envelope.py, sim_config.THRUST_ENVELOPE): each limit is recomputed
every call from what the others are using. Vertical comes before horizontal.
"""
import numpy as np

import sim_config
import thrust_envelope


def integrate(integral, error, ki, limit, zone, dt, blocked):
    """One integrator step, per axis, with anti-windup.

    An axis holds its value while its error is outside zone, so a far-away
    waypoint cannot wind it up, and while blocked, if the step would grow the
    integral. blocked means the output is already saturated. The integral can
    always shrink, and it is clamped to +-limit.
    """
    step = np.asarray(ki, dtype=float) * error * dt

    step = np.where(np.abs(error) <= zone, step, 0.0)

    growing = np.abs(integral + step) > np.abs(integral)
    step = np.where(blocked & growing, 0.0, step)

    return np.clip(integral + step, -np.asarray(limit), np.asarray(limit))


class PIDController:
    """The flight controller. One instance per drone; it holds integrators."""

    def __init__(self, gains=None):
        self.gains = sim_config.PID if gains is None else gains
        self.reset()

    def reset(self):
        """Clear all integrators. Call on every reset or controller restart.

        Not called between waypoints: the wind does not change because the
        target did, so the learned wind correction should carry over.
        """
        self.height_integral = 0.0
        self.xy_integral = np.zeros(2)
        self.attitude_integral = np.zeros(3)
        self.limits = None

    def _envelope(self, target):
        """The thrust envelope for this target's vehicle, built once per vehicle."""
        key = tuple(
            target.get(name)
            for name in ("mass", "max_total_thrust", "max_xy_speed",
                         "max_z_speed", "max_tilt_degrees")
        )

        if getattr(self, "_envelope_key", None) != key:
            self._envelope_key = key
            self._envelope_cache = thrust_envelope.for_target(target)

        return self._envelope_cache

    def update(
        self,
        positions,
        orientations,
        velocities,
        angular_velocities,
        target,
        dt=0.0,
        actuators_saturated=False,
    ):
        """Return (thrust, roll torque, pitch torque, yaw torque).

        dt is the time since the last control step [s]. The output uses the
        integrals from before this step. dt = 0 changes nothing, so the same
        call can read the current command without integrating twice.

        actuators_saturated: the mixer clipped a rotor on the last step.
        """
        position = np.asarray(positions, dtype=float).reshape(3)
        quaternion = np.asarray(orientations, dtype=float).reshape(4)
        velocity = np.asarray(velocities, dtype=float).reshape(3)
        angular_velocity_world = np.asarray(
            angular_velocities, dtype=float
        ).reshape(3)

        # Reject bad state before NaNs propagate into forces or rotor commands.
        if not all(np.all(np.isfinite(v)) for v in
                   (position, quaternion, velocity, angular_velocity_world)):
            raise ValueError("Drone state contains NaN or infinity.")

        if not np.isfinite(dt) or dt < 0:
            raise ValueError(f"Control timestep must not be negative: {dt}")

        # ---------- Quaternion → rotation matrix ----------

        norm = np.linalg.norm(quaternion)

        if not np.isfinite(norm) or norm < 1e-8:
            raise ValueError("Invalid orientation quaternion.")

        w, x, y, z = quaternion / norm

        # R converts body-frame vectors into world-frame vectors.
        R = np.array([
            [
                1 - 2 * (y*y + z*z),
                2 * (x*y - w*z),
                2 * (x*z + w*y),
            ],
            [
                2 * (x*y + w*z),
                1 - 2 * (x*x + z*z),
                2 * (y*z - w*x),
            ],
            [
                2 * (x*z - w*y),
                2 * (y*z + w*x),
                1 - 2 * (x*x + y*y),
            ],
        ])

        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        yaw = np.arctan2(R[1, 0], R[0, 0])

        angular_velocity_body = R.T @ angular_velocity_world

        # ---------- Targets ----------

        target_height = float(target["height"])
        target_roll = float(target.get("roll", 0.0))
        target_pitch = float(target.get("pitch", 0.0))
        target_yaw = float(target.get("yaw", 0.0))

        mass = float(target.get("mass", 1.5))
        gravity = 9.81
        if (not np.all(np.isfinite([target_height, target_roll, target_pitch,
                                   target_yaw, mass])) or mass <= 0):
            raise ValueError("Targets must be finite and mass must be positive.")

        # ---------- Controller gains ----------

        height_gains = self.gains["height"]
        xy_gains = self.gains["xy"]
        attitude_gains = self.gains["attitude"]

        kp_height = height_gains["kp"]
        kd_height = height_gains["kd"]

        kp_attitude = np.asarray(attitude_gains["kp"], dtype=float)
        kd_attitude = np.asarray(attitude_gains["kd"], dtype=float)

        envelope = self._envelope(target)
        has_xy = "xy" in target

        # ---------- Velocity request, inside the thrust envelope ----------

        # Each position loop is rewritten as a velocity request. With no limit
        # this is algebraically the usual kp*error - kd*velocity. The request is
        # then scaled as one vector, keeping its direction, to the fastest speed
        # the thrust budget sustains that way: full horizontal speed leaves no
        # climb, and the other way round. It shapes the request only; it is not a
        # hard guarantee on actual speed.
        height_error = target_height - position[2]
        desired_velocity = np.array([
            0.0, 0.0, kp_height / kd_height * height_error,
        ])

        if has_xy:
            target_xy = np.asarray(
                target["xy"], dtype=float
            ).reshape(2)

            if not np.all(np.isfinite(target_xy)):
                raise ValueError(
                    "Target xy must contain two finite world coordinates."
                )

            xy_error = target_xy - position[:2]
            desired_velocity[:2] = (
                xy_gains["kp"] / xy_gains["kd"] * xy_error
            )

        desired_velocity, speed_capped = envelope.limit_velocity(
            desired_velocity
        )

        # ---------- Height → total thrust ----------

        # Vertical comes first in the budget, inside its own limits.
        requested_vertical = (
            kd_height * (desired_velocity[2] - velocity[2])
            + self.height_integral
        ) / mass

        lowest, highest = envelope.vertical_limits()
        vertical_acceleration_command = float(
            np.clip(requested_vertical, lowest, highest)
        )
        vertical_capped = vertical_acceleration_command != requested_vertical

        vertical_force = mass * (gravity + vertical_acceleration_command)

        # Compensate for the reduced vertical thrust when tilted.
        requested_thrust = vertical_force / max(float(R[2, 2]), 0.5)

        # The simulation supplies its actual configured motor limit. The default
        # preserves the original controller when this function is used standalone.
        max_total_thrust = float(target.get(
            "max_total_thrust", 4 * 5.84e-6 * 1100.0**2
        ))

        total_thrust = float(
            np.clip(requested_thrust, 0.0, max_total_thrust)
        )

        height_blocked = (
            actuators_saturated
            or vertical_capped
            or total_thrust != requested_thrust
        )

        # ---------- Step 10: horizontal position control ----------

        max_acceleration = 0.0
        xy_capped = False

        # Without "xy", the original manual roll/pitch targets are used.
        if has_xy:
            # Position error pulls toward the target.
            # Velocity feedback brakes horizontal motion.
            acceleration_xy = (
                xy_gains["kd"] * (desired_velocity[:2] - velocity[:2])
                + self.xy_integral
            )

            # Horizontal gets what the vertical command left of the budget,
            # and stays under the tilt cap.
            vertical_acceleration = max(
                vertical_force / mass,
                0.5 * gravity,
            )

            max_acceleration = envelope.horizontal_limit(
                vertical_acceleration - gravity
            )

            magnitude = np.linalg.norm(acceleration_xy)
            xy_capped = magnitude > max_acceleration

            xy_blocked = actuators_saturated or xy_capped

            if magnitude > max_acceleration:
                acceleration_xy *= max_acceleration / magnitude

            # Convert world acceleration into axes at the desired yaw.
            c = np.cos(target_yaw)
            s = np.sin(target_yaw)

            ax, ay = acceleration_xy

            ax_heading = c * ax + s * ay
            ay_heading = -s * ax + c * ay

            # Desired pitch and roll for local +Z thrust.
            target_pitch = np.arctan2(
                ax_heading,
                vertical_acceleration,
            )

            target_roll = np.arctan2(
                -ay_heading,
                np.hypot(ax_heading, vertical_acceleration),
            )

            self.xy_integral = integrate(
                self.xy_integral,
                xy_error,
                xy_gains["ki"],
                xy_gains["integral_limit"],
                xy_gains["integral_zone"],
                dt,
                xy_blocked,
            )

        # ---------- Attitude → body-frame torques ----------
        # This is the original small-angle Euler controller, appropriate near
        # hover. It is not a large-angle recovery controller. R.T above converts
        # world angular velocity to the body axes in which these commanded
        # torques are applied.

        attitude_error = np.array([
            target_roll - roll,
            target_pitch - pitch,
            target_yaw - yaw,
        ])

        # Use the shortest angular difference.
        attitude_error = (
            attitude_error + np.pi
        ) % (2 * np.pi) - np.pi

        torques = (
            kp_attitude * attitude_error
            - kd_attitude * angular_velocity_body
            + self.attitude_integral
        )

        self.height_integral = float(integrate(
            self.height_integral,
            height_error,
            height_gains["ki"],
            height_gains["integral_limit"],
            height_gains["integral_zone"],
            dt,
            height_blocked,
        ))

        self.attitude_integral = integrate(
            self.attitude_integral,
            attitude_error,
            attitude_gains["ki"],
            attitude_gains["integral_limit"],
            attitude_gains["integral_zone"],
            dt,
            actuators_saturated,
        )

        # What the envelope allowed on this call, for logging and for callers
        # that need to know a cap was reached. Not controller state: the next
        # call recomputes it.
        self.limits = {
            "max_xy_acceleration": float(max_acceleration),
            "vertical_acceleration_range": (float(lowest), float(highest)),
            "vertical_acceleration": vertical_acceleration_command,
            "xy_capped": bool(xy_capped),
            "vertical_capped": bool(vertical_capped),
            "speed_capped": bool(speed_capped),
        }

        return (
            total_thrust,
            float(torques[0]),
            float(torques[1]),
            float(torques[2]),
        )
