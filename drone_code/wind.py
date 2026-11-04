"""Gusting wind and distributed aerodynamic drag (approximate model, not CFD).

Wind at sample point i, world frame [m/s]:

    wind_i = mean_wind + shared_gust + local_gust_i

shared_gust
    One Ornstein-Uhlenbeck (OU) vector felt by the whole drone. Exact
    timestep-aware update, so its statistics do not depend on dt:

        a = exp(-dt / gust_timescale)
        gust = a * gust + gust_std * sqrt(1 - a*a) * normal(3)

local_gust_i
    Smaller, faster OU fluctuations that differ across the airframe.
    Each point has unit-variance OU states z_i (one per world axis) with
    local_gust_timescale. They are mixed by a fixed matrix M with
    M @ M.T = C, where

        C_ij = exp(-|p_i - p_j| / correlation_length)

    is an exponential spatial correlation over the body-frame distance
    between sample points. local_gust = local_gust_std * (M @ z) keeps each
    point's variance at local_gust_std**2, gives nearby points correlated
    fluctuations, and stays temporally smooth because M is constant and
    the OU update is linear.

Approximation: the local field is "frozen" to the airframe (sample-point
distances never change) instead of being advected past the drone, and the
three axes are independent. Shared and local parts are independent, so per
axis

    variance(wind_i)        = gust_std**2 + local_gust_std**2
    covariance(wind_i, w_j) = gust_std**2 + local_gust_std**2 * C_ij

Drag at each point uses the point's own airspeed, including rotation:

    v_point = v_com + omega x r            (r measured from the COM)
    v_rel   = v_point - wind_i
    F_i     = -0.5 * rho * Cd * A_i * |v_rel| * v_rel

A_i = area_weight_i * effective_area with weights normalized to sum to 1,
so adding points does not multiply total drag. Unequal F_i produce torque
through the moment arms; orientation is never modified directly.
"""
import numpy as np


def quaternion_to_matrix(quaternion):
    """Body-to-world rotation from a quaternion ordered [w, x, y, z]."""
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(q)

    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid quaternion.")

    w, x, y, z = q / norm

    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def spatial_mixing_matrix(offsets, correlation_length):
    """Return M with M @ M.T = exp(-distance / correlation_length).

    correlation_length <= 0 makes points independent (M = I); infinity
    makes them identical. A symmetric eigen square root is used instead of
    Cholesky so the fully correlated (singular) limit still works.
    """
    offsets = np.asarray(offsets, dtype=float).reshape(-1, 3)
    count = len(offsets)

    if correlation_length <= 0:
        return np.eye(count)

    distances = np.linalg.norm(
        offsets[:, None, :] - offsets[None, :, :],
        axis=2,
    )
    correlation = np.exp(-distances / correlation_length)

    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    return eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))


def drag_forces(relative_velocities, areas, air_density, drag_coefficient):
    """Quadratic drag F = -0.5 rho Cd A |v_rel| v_rel for each point [N]."""
    v = np.asarray(relative_velocities, dtype=float).reshape(-1, 3)
    speed = np.linalg.norm(v, axis=1, keepdims=True)

    return (
        -0.5
        * air_density
        * drag_coefficient
        * np.asarray(areas, dtype=float).reshape(-1, 1)
        * speed
        * v
    )


class GustingWind:
    """Time-varying wind field and its drag wrench on one rigid body.

    All vectors are world frame except offsets, which are body frame and
    measured from the body origin (the same reference as rotor_positions).
    """

    def __init__(
        self,
        mean_wind,
        gust_std,
        gust_timescale,
        local_gust_std,
        local_gust_timescale,
        correlation_length,
        air_density,
        drag_coefficient,
        effective_area,
        sample_points,
        seed=None,
        enabled=True,
    ):
        self.enabled = bool(enabled)
        self.mean_wind = np.asarray(mean_wind, dtype=float).reshape(3)
        self.gust_std = np.asarray(gust_std, dtype=float).reshape(3)
        self.local_gust_std = np.asarray(
            local_gust_std, dtype=float
        ).reshape(3)
        self.gust_timescale = float(gust_timescale)
        self.local_gust_timescale = float(local_gust_timescale)
        self.correlation_length = float(correlation_length)
        self.air_density = float(air_density)
        self.drag_coefficient = float(drag_coefficient)
        self.effective_area = float(effective_area)
        self.seed = seed

        # reset_episode() scales from these, so scaling never compounds.
        self.base_gust_std = self.gust_std.copy()
        self.base_local_gust_std = self.local_gust_std.copy()

        # sample_points: sequence of (name, (x, y, z) body offset, weight).
        self.point_names = [str(name) for name, _, _ in sample_points]
        self.offsets = np.array(
            [offset for _, offset, _ in sample_points], dtype=float
        ).reshape(-1, 3)
        weights = np.array(
            [weight for _, _, weight in sample_points], dtype=float
        )

        if (
            len(self.offsets) == 0
            or np.any(weights < 0)
            or weights.sum() <= 0
            or self.gust_timescale <= 0
            or self.local_gust_timescale <= 0
            or np.any(self.gust_std < 0)
            or np.any(self.local_gust_std < 0)
            or self.air_density < 0
            or self.drag_coefficient < 0
            or self.effective_area < 0
            or not np.all(np.isfinite(self.offsets))
            or not np.all(np.isfinite(self.mean_wind))
        ):
            raise ValueError("Invalid wind configuration.")

        # Partition the total area; weights only set the split.
        self.area_weights = weights / weights.sum()
        self.areas = self.area_weights * self.effective_area

        self.mixing = spatial_mixing_matrix(
            self.offsets,
            self.correlation_length,
        )

        self.reset()

    def reset(self):
        """Restart the gust sequence from the configured seed.

        Gust states start from their stationary distribution, so the
        statistics at t = 0 already match the configured strengths.
        """
        self.rng = np.random.default_rng(self.seed)
        self.time = 0.0
        self.shared_gust = self.gust_std * self.rng.normal(size=3)
        self.local_states = self.rng.normal(size=(len(self.offsets), 3))
        self.point_winds = self._compose()

    def reset_episode(self, seed, mean_wind, gust_scale=1.0):
        """Start a new wind realization with its own seed and strength."""
        self.seed = seed
        self.mean_wind = np.asarray(mean_wind, dtype=float).reshape(3)
        self.gust_std = float(gust_scale) * self.base_gust_std
        self.local_gust_std = float(gust_scale) * self.base_local_gust_std
        self.reset()

    def _compose(self):
        local_gusts = self.local_gust_std * (self.mixing @ self.local_states)
        return self.mean_wind + self.shared_gust + local_gusts

    def step(self, dt):
        """Advance both OU processes by dt seconds; return point winds."""
        dt = float(dt)

        if not np.isfinite(dt) or dt <= 0:
            raise ValueError(f"Wind timestep must be positive: {dt}")

        a = np.exp(-dt / self.gust_timescale)
        self.shared_gust = (
            a * self.shared_gust
            + self.gust_std * np.sqrt(1 - a*a) * self.rng.normal(size=3)
        )

        a_local = np.exp(-dt / self.local_gust_timescale)
        self.local_states = (
            a_local * self.local_states
            + np.sqrt(1 - a_local*a_local)
            * self.rng.normal(size=self.local_states.shape)
        )

        self.time += dt
        self.point_winds = self._compose()
        return self.point_winds

    def point_forces(
        self,
        body_position,
        body_orientation,
        com_velocity,
        angular_velocity,
        com_offset_body=np.zeros(3),
    ):
        """Return world sample positions, world forces, and body offsets r.

        body_position/orientation: body origin pose (orientation [w,x,y,z]).
        com_velocity: world linear velocity of the body's center of mass.
        angular_velocity: world angular velocity [rad/s].
        com_offset_body: body COM relative to the body origin, body frame.
        """
        R = quaternion_to_matrix(body_orientation)
        origin = np.asarray(body_position, dtype=float).reshape(3)
        v_com = np.asarray(com_velocity, dtype=float).reshape(3)
        omega = np.asarray(angular_velocity, dtype=float).reshape(3)
        com = np.asarray(com_offset_body, dtype=float).reshape(3)

        # Moment arms in world axes, from the origin (for the applied
        # wrench) and from the COM (for rigid-body point velocity).
        r_origin = self.offsets @ R.T
        r_com = (self.offsets - com) @ R.T

        positions = origin + r_origin
        velocities = v_com + np.cross(omega, r_com)

        forces = drag_forces(
            velocities - self.point_winds,
            self.areas,
            self.air_density,
            self.drag_coefficient,
        )

        return positions, forces, r_origin

    def body_wrench(
        self,
        body_position,
        body_orientation,
        com_velocity,
        angular_velocity,
        com_offset_body=np.zeros(3),
    ):
        """Net drag force and torque about the body origin, in body axes.

        Applying this wrench at the body origin is exactly equivalent to
        applying every point force at its own position:
            F = sum F_i,  tau = sum r_i x F_i.
        """
        if not self.enabled:
            return np.zeros(3), np.zeros(3)

        _, forces, r_origin = self.point_forces(
            body_position,
            body_orientation,
            com_velocity,
            angular_velocity,
            com_offset_body,
        )

        R = quaternion_to_matrix(body_orientation)
        force_world = forces.sum(axis=0)
        torque_world = np.cross(r_origin, forces).sum(axis=0)

        return R.T @ force_world, R.T @ torque_world
