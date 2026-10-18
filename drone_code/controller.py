import numpy as np


def controller(
    positions,
    orientations,
    velocities,
    angular_velocities,
    target,
):
    # Accept either one drone's vector or a one-drone batch.
    position = np.asarray(positions, dtype=float).reshape(3)
    quaternion = np.asarray(orientations, dtype=float).reshape(4)
    velocity = np.asarray(velocities, dtype=float).reshape(3)
    angular_velocity_world = np.asarray(
        angular_velocities, dtype=float
    ).reshape(3)

    # ---------- Quaternion → rotation matrix ----------

    norm = np.linalg.norm(quaternion)

    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid orientation quaternion.")

    w, x, y, z = quaternion / norm

    # R transforms body-frame vectors into world-frame vectors.
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

    # Roll, pitch, yaw in radians: R = Rz(yaw) Ry(pitch) Rx(roll).
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # Motor torques act along BODY axes.
    angular_velocity_body = R.T @ angular_velocity_world

    # ---------- Targets ----------

    target_height = float(target["height"])
    target_roll = float(target.get("roll", 0.0))
    target_pitch = float(target.get("pitch", 0.0))
    target_yaw = float(target.get("yaw", 0.0))

    # Initial estimate; should eventually include the whole vehicle.
    mass = float(target.get("mass", 1.5))
    gravity = 9.81

    # ---------- PD gains: starting values, need tuning ----------

    kp_height = 10.0
    kd_height = 5.0

    # Order: roll, pitch, yaw.
    kp_attitude = np.array([0.2, 0.2, 0.1])
    kd_attitude = np.array([0.05, 0.05, 0.05])

    # ---------- Height → total thrust ----------

    height_error = target_height - position[2]

    vertical_force = (
        mass * gravity
        + kp_height * height_error
        - kd_height * velocity[2]
    )

    # When tilted, only R[2, 2] of local thrust points upward.
    # Bound the compensation; this controller is for near-level flight.
    total_thrust = vertical_force / max(float(R[2, 2]), 0.5)

    # Combined thrust limit from your four modeled motors.
    max_total_thrust = 4 * 5.84e-6 * 1100.0**2
    total_thrust = float(
        np.clip(total_thrust, 0.0, max_total_thrust)
    )

    # ---------- Attitude → body-frame torques ----------

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