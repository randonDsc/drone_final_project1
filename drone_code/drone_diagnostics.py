import numpy as np


def rotate_vector(quaternion, vector):
    """Rotate a vector using a quaternion ordered [w, x, y, z]."""
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(q)

    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid quaternion.")

    q = q / norm
    vector = np.asarray(vector, dtype=float).reshape(3)

    return (
        vector
        + 2 * q[0] * np.cross(q[1:], vector)
        + 2 * np.cross(q[1:], np.cross(q[1:], vector))
    )


def read_total_com_in_body_frame(body, mass_views):
    """Return the vehicle COM relative to the body origin, in body axes."""
    weighted_position = np.zeros(3)
    total_mass = 0.0

    for view in mass_views:
        mass = float(
            np.asarray(view.get_masses()).reshape(-1)[0]
        )

        positions, orientations = view.get_world_poses()
        local_coms, _ = view.get_coms()

        world_com = (
            np.asarray(positions).reshape(3)
            + rotate_vector(
                np.asarray(orientations).reshape(4),
                np.asarray(local_coms).reshape(3),
            )
        )

        weighted_position += mass * world_com
        total_mass += mass

    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Invalid total vehicle mass.")

    total_com_world = weighted_position / total_mass

    body_positions, body_orientations = body.get_world_poses()
    body_position = np.asarray(body_positions).reshape(3)
    body_quaternion = np.asarray(body_orientations).reshape(4)

    inverse_quaternion = body_quaternion.copy()
    inverse_quaternion[1:] *= -1

    return rotate_vector(
        inverse_quaternion,
        total_com_world - body_position,
    )


def diagnose_thrust_offset(body, mass_views, motors, speeds):
    """Return COM offset and the corresponding thrust-offset torque."""
    com_offset = read_total_com_in_body_frame(body, mass_views)

    thrusts, _ = motors.calculate(speeds)
    applied_force = np.array([0.0, 0.0, thrusts.sum()])

    # Torque reference shift: body origin → entire vehicle COM.
    offset_torque = np.cross(-com_offset, applied_force)

    return com_offset, offset_torque