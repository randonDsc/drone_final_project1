"""Vehicle COM and torque-reference checks retained from the uploaded project.

Every rigid body contributes mass * world COM. Averaging those contributions
finds the full vehicle COM; inverse body rotation expresses its offset in body
axes, which the motor mixer uses to compensate for off-center applied thrust.
"""
import numpy as np


def cross(a, b):
    """Same result as np.cross for (..., 3) arrays.

    np.cross dominated the cost of a physics step: this runs many times per
    step, and the training rate is limited by Python here, not by PhysX.
    """
    return np.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
    ], axis=-1)


def rotate_vector(quaternion, vector):
    """Rotate vectors using quaternions ordered [w, x, y, z].

    Accepts one pair of shapes (4,) and (3,), or N pairs (N, 4) and (N, 3).
    """
    q = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)

    if not np.all(np.isfinite(norm)) or np.any(norm < 1e-8):
        raise ValueError("Invalid quaternion.")

    q = q / norm
    vector = np.asarray(vector, dtype=float)

    w = q[..., :1]
    xyz = q[..., 1:]

    # Quaternion rotation: v_rot = v + 2*w*(q_xyz x v)
    #                                  + 2*q_xyz x (q_xyz x v).
    return (
        vector
        + 2 * w * cross(xyz, vector)
        + 2 * cross(xyz, cross(xyz, vector))
    )


def read_total_com_in_body_frame(body, vehicle_bodies):
    """Return the vehicle COM relative to the body origin, in body axes.

    vehicle_bodies is one view holding every rigid body of the vehicle.
    """
    masses = np.asarray(vehicle_bodies.get_masses(), dtype=float).reshape(-1)

    positions, orientations = vehicle_bodies.get_world_poses()
    local_coms, _ = vehicle_bodies.get_coms()

    # get_coms() returns a LOCAL center of mass; rotate and translate
    # it before averaging COM locations from different rigid bodies.
    world_coms = (
        np.asarray(positions, dtype=float).reshape(-1, 3)
        + rotate_vector(
            np.asarray(orientations, dtype=float).reshape(-1, 4),
            np.asarray(local_coms, dtype=float).reshape(-1, 3),
        )
    )

    total_mass = masses.sum()

    if not np.isfinite(total_mass) or total_mass <= 0:
        raise ValueError("Invalid total vehicle mass.")

    total_com_world = (masses[:, None] * world_coms).sum(axis=0) / total_mass

    body_positions, body_orientations = body.get_world_poses()
    body_position = np.asarray(body_positions).reshape(3)
    body_quaternion = np.asarray(body_orientations).reshape(4)

    # Unit quaternion inverse = conjugate: [w,-x,-y,-z].
    inverse_quaternion = body_quaternion.copy()
    inverse_quaternion[1:] *= -1

    return rotate_vector(
        inverse_quaternion,
        total_com_world - body_position,
    )


def diagnose_thrust_offset(body, vehicle_bodies, motors, speeds):
    """Return COM offset and the corresponding thrust-offset torque."""
    com_offset = read_total_com_in_body_frame(body, vehicle_bodies)

    thrusts, _ = motors.calculate(speeds)
    applied_force = np.array([0.0, 0.0, thrusts.sum()])

    # Torque reference shift: body origin → entire vehicle COM.
    offset_torque = np.cross(-com_offset, applied_force)

    return com_offset, offset_torque
