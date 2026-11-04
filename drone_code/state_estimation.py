"""IMU noise model and IMU-only state estimator. NumPy only.

The estimator takes gyro and accelerometer samples and nothing else. Isaac's IMU
also reports an orientation, but that is simulator ground truth, so it is never
read. Frames and signs:

    gyro            body angular velocity, sensor axes [rad/s]
    accelerometer   specific force f = R^T (a - g_vector), sensor axes [m/s^2]:
                    +g on the up axis at rest, 0 in free fall
    R               body-to-world rotation, from q = [w, x, y, z]
    g_vector        (0, 0, -g), world frame

Steps, once per physics step:

    1. Mount: rotate sensor axes into body axes and remove the centripetal term
       w x (w x r) from the lever arm r. Angular acceleration is not removed,
       because differentiating the gyro would amplify its noise.
    2. Attitude: complementary (Mahony) filter. Integrate the gyro, and steer
       predicted "up" in the body frame toward the measured specific force.
    3. Translation: strapdown integration. a = R f + g_vector, then integrate
       once to velocity and again to position.

What an IMU alone can and cannot observe:

    Roll, pitch     Only weakly observable on a multirotor. Unaccelerated flight
                    reads gravity correctly, wind drag included, but thrust acts
                    along body z, so without drag the accelerometer reads "up"
                    at ANY tilt: a tilted drone accelerating sideways looks
                    level. Only drag brings the true direction back, and at low
                    airspeed drag is tiny. Gyro bias therefore turns into a
                    slowly growing true tilt the filter does not see.
    Yaw             Not observable. Gyro bias integrates forever; a magnetometer
                    or similar heading reference is needed.
    Velocity        Not observable. Bias b gives error b t. Tilt error e gives
                    g e t: 1 degree is 0.17 m/s^2.
    Position        Not observable. The start offset is never corrected, and
                    errors grow as t^2 or faster.

The two errors hide each other. An accelerometer bias looks like a tilt of b / g,
and the filter's "level" pull cancels the horizontal acceleration it would have
integrated. The estimate stays calm at the target while the real drone drifts
away, so a steady estimate says nothing about the real position. Only an outside
position or velocity reference can reveal this. See PPO_RESIDUAL.md.
"""
import numpy as np

from wind import quaternion_to_matrix


def quaternion_multiply(a, b):
    """Hamilton product of [w, x, y, z] quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b

    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def tilt_quaternion_from_gravity(specific_force, yaw=0.0):
    """Attitude whose body frame reads this specific force at rest."""
    fx, fy, fz = specific_force / np.linalg.norm(specific_force)

    roll = np.arctan2(fy, fz)
    pitch = np.arctan2(-fx, np.hypot(fy, fz))

    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def yaw_of(quaternion):
    R = quaternion_to_matrix(quaternion)
    return float(np.arctan2(R[1, 0], R[0, 0]))


class ImuNoise:
    """Bias and white noise added to clean IMU samples."""

    def __init__(self, settings, dt):
        self.settings = settings
        self.dt = float(dt)
        rate = 1.0 / self.dt

        self.gyro_std = settings["gyro_noise_density"] * np.sqrt(rate)
        self.accel_std = settings["accel_noise_density"] * np.sqrt(rate)
        self.gyro_walk = settings["gyro_bias_walk"] * np.sqrt(self.dt)
        self.accel_walk = settings["accel_bias_walk"] * np.sqrt(self.dt)

        self.reset(np.random.default_rng(0))

    def reset(self, rng):
        """New switch-on bias and noise sequence, independent of the policy."""
        self.rng = rng
        self.gyro_bias = (
            self.settings["gyro_bias_std"] * rng.normal(size=3)
        )
        self.accel_bias = (
            self.settings["accel_bias_std"] * rng.normal(size=3)
        )

    def apply(self, gyro, accel):
        """Return (gyro, accel) as the sensor would report them."""
        rng = self.rng

        self.gyro_bias = self.gyro_bias + self.gyro_walk * rng.normal(size=3)
        self.accel_bias = self.accel_bias + self.accel_walk * rng.normal(size=3)

        return (
            gyro + self.gyro_bias + self.gyro_std * rng.normal(size=3),
            accel + self.accel_bias + self.accel_std * rng.normal(size=3),
        )


class ImuEstimator:
    """Attitude filter plus strapdown integration; see the module docstring."""

    def __init__(
        self,
        settings,
        dt,
        gravity,
        mount_rotation=np.eye(3),
        mount_offset=np.zeros(3),
    ):
        """mount_rotation: sensor-to-body rotation.

        mount_offset: sensor position in the body frame, measured from the
        body's center of mass [m].
        """
        self.kp = float(settings["attitude_kp"])
        self.ki = float(settings["attitude_ki"])
        self.accel_gate = float(settings["accel_gate"])
        self.ignore_samples = int(settings["reset_ignore_samples"])
        self.alignment_samples = int(settings["alignment_samples"])
        self.leak_time = float(settings["velocity_leak_time"])

        if (
            self.kp < 0
            or self.ki < 0
            or self.accel_gate <= 0
            or self.ignore_samples < 0
            or self.alignment_samples < 1
            or self.leak_time < 0
        ):
            raise ValueError("Invalid estimator settings.")

        self.dt = float(dt)
        self.gravity = float(gravity)
        self.mount_rotation = np.asarray(mount_rotation, dtype=float)
        self.mount_offset = np.asarray(mount_offset, dtype=float)

        self.reset(np.zeros(3))

    def reset(self, position, yaw=0.0):
        """Restart from a prior, never from the simulator's state.

        position and yaw are what the operator assumes at the start. Velocity
        is assumed zero; tilt comes from the first accepted accelerometer
        samples. Until alignment finishes the estimate stays at this prior.
        """
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.zeros(3)
        self.yaw_prior = float(yaw)
        self.orientation = tilt_quaternion_from_gravity(
            np.array([0.0, 0.0, 1.0]), self.yaw_prior
        )
        self.gyro_bias = np.zeros(3)
        self.angular_velocity_body = np.zeros(3)
        self.specific_force_body = np.array([0.0, 0.0, self.gravity])

        self.samples_to_ignore = self.ignore_samples
        self.alignment_sum = np.zeros(3)
        self.alignment_count = 0

    @property
    def aligned(self):
        return self.alignment_count >= self.alignment_samples

    def update(self, gyro, accel, dt=None):
        """Advance by one IMU sample, in sensor axes."""
        dt = self.dt if dt is None else float(dt)

        rate = self.mount_rotation @ np.asarray(gyro, dtype=float)
        force = self.mount_rotation @ np.asarray(accel, dtype=float)

        # Stored during alignment too, so the controller still damps rotation.
        self.angular_velocity_body = rate - self.gyro_bias

        if self.samples_to_ignore > 0:
            self.samples_to_ignore -= 1
            return

        force = force - np.cross(
            rate, np.cross(rate, self.mount_offset)
        )
        self.specific_force_body = force

        if not self.aligned:
            self.alignment_sum += force
            self.alignment_count += 1

            if self.aligned:
                self.orientation = tilt_quaternion_from_gravity(
                    self.alignment_sum, self.yaw_prior
                )

            return

        self._update_attitude(rate, force, dt)
        self._update_translation(force, dt)

    def _update_attitude(self, rate, force, dt):
        R = quaternion_to_matrix(self.orientation)
        magnitude = np.linalg.norm(force)
        correction = np.zeros(3)

        if (
            magnitude > 1e-6
            and abs(magnitude - self.gravity) < self.accel_gate
        ):
            predicted_up = R[2, :]
            error = np.cross(force / magnitude, predicted_up)

            correction = self.kp * error
            self.gyro_bias = self.gyro_bias - self.ki * error * dt

        omega = rate - self.gyro_bias + correction
        self.angular_velocity_body = rate - self.gyro_bias

        half_angle = 0.5 * np.linalg.norm(omega) * dt
        scale = 0.5 * dt * np.sinc(half_angle / np.pi)

        step = np.concatenate([[np.cos(half_angle)], scale * omega])
        orientation = quaternion_multiply(self.orientation, step)

        self.orientation = orientation / np.linalg.norm(orientation)

    def _update_translation(self, force, dt):
        R = quaternion_to_matrix(self.orientation)

        acceleration = R @ force
        acceleration[2] -= self.gravity

        velocity = self.velocity + acceleration * dt

        if self.leak_time > 0:
            velocity *= np.exp(-dt / self.leak_time)

        self.position = (
            self.position + 0.5 * (self.velocity + velocity) * dt
        )
        self.velocity = velocity

    def state(self):
        """Estimate in the same format as DroneSim.read_drone_state."""
        R = quaternion_to_matrix(self.orientation)

        return {
            "position": self.position.copy(),
            "orientation": self.orientation.copy(),
            "linear_velocity": self.velocity.copy(),
            "angular_velocity": R @ self.angular_velocity_body,
        }
