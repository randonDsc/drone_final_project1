"""Low-level near-hover flight controller; it does not build maps or run search.

Input state is world position (m), unit quaternion [w,x,y,z], world linear
velocity (m/s), and world angular velocity (rad/s). The target dictionary holds
ONE active waypoint. navigation.py chooses that waypoint from the global route.
Output is body +Z thrust (N) and body roll/pitch/yaw torque (N m).
The existing PD gains and quaternion convention are retained from your code.
"""
import numpy as np


def controller(
    positions,
    orientations,
    velocities,
    angular_velocities,
    target,
):
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

    kp_height = 10.0
    kd_height = 5.0

    kp_attitude = np.array([0.2, 0.2, 0.1])
    kd_attitude = np.array([0.05, 0.05, 0.05])

    # ---------- Height → total thrust ----------

    height_error = target_height - position[2]

    # Rewrite PD as a velocity request, then cap it. With no saturation this
    # is algebraically the original kp*error - kd*velocity. Capping the request
    # reduces speed on long segments; it is not a hard guarantee on actual speed.
    max_z_speed = float(target.get("max_z_speed", 0.4))
    desired_vz = np.clip(kp_height / kd_height * height_error,
                         -max_z_speed, max_z_speed)
    vertical_force = mass * gravity + kd_height * (desired_vz - velocity[2])

    # Compensate for the reduced vertical thrust when tilted.
    total_thrust = vertical_force / max(float(R[2, 2]), 0.5)

    # The simulation supplies its actual configured motor limit. The default
    # preserves the original controller when this function is used standalone.
    max_total_thrust = float(target.get(
        "max_total_thrust", 4 * 5.84e-6 * 1100.0**2
    ))

    total_thrust = float(
        np.clip(total_thrust, 0.0, max_total_thrust)
    )

    # ---------- Step 10: horizontal position control ----------

    # Without "xy", the original manual roll/pitch targets are used.
    if "xy" in target:
        target_xy = np.asarray(
            target["xy"], dtype=float
        ).reshape(2)

        if not np.all(np.isfinite(target_xy)):
            raise ValueError(
                "Target xy must contain two finite world coordinates."
            )

        kp_xy = 0.6
        kd_xy = 1.2

        # Position error pulls toward the target.
        # Velocity feedback brakes horizontal motion.
        desired_velocity_xy = kp_xy / kd_xy * (target_xy - position[:2])
        requested_speed = np.linalg.norm(desired_velocity_xy)
        max_xy_speed = float(target.get("max_xy_speed", 0.6))
        if requested_speed > max_xy_speed:
            desired_velocity_xy *= max_xy_speed / requested_speed
        acceleration_xy = kd_xy * (desired_velocity_xy - velocity[:2])

        # Near-hover controller: limit desired tilt to 10 degrees
        # and horizontal acceleration to 1.5 m/s².
        vertical_acceleration = max(
            vertical_force / mass,
            0.5 * gravity,
        )

        max_acceleration = min(
            1.5,
            vertical_acceleration * np.tan(np.deg2rad(float(target.get("max_tilt_degrees", 10.0)))),
        )

        magnitude = np.linalg.norm(acceleration_xy)

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

    # ---------- Attitude → body-frame torques ----------
    # This is the original small-angle Euler PD, appropriate near hover. It is
    # not a large-angle recovery controller. R.T above converts world angular
    # velocity to the body axes in which these commanded torques are applied.

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
    )

    return (
        total_thrust,
        float(torques[0]),
        float(torques[1]),
        float(torques[2]),
    )
