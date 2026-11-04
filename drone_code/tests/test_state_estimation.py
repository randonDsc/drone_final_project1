"""IMU estimator checks: run with Isaac Sim's python, from drone_code:

    C:\\isaacsim\\python.bat -m unittest discover -s tests -p test_state_estimation.py -v

No SimulationApp needed. Flights use a small rigid-body model with the real
controller.py, so these check frames, signs, and limits, not Isaac's IMU.
"""
import unittest

import numpy as np

import sensor_config

from controller import PIDController
from state_estimation import (
    ImuEstimator,
    ImuNoise,
    quaternion_multiply,
    tilt_quaternion_from_gravity,
    yaw_of,
)
from wind import quaternion_to_matrix


GRAVITY = 9.81
DT = 1.0 / 120.0
MASS = 1.6
INERTIA = np.array([0.03, 0.03, 0.05])
DRAG = 0.5 * 1.225 * 0.03       # sim_config.WIND density * Cd * area
QUIET = dict(sensor_config.ESTIMATOR, reset_ignore_samples=0, alignment_samples=1)
ZERO_NOISE = {name: 0.0 for name in sensor_config.IMU_NOISE}


class RigidBody:
    """Rigid body with an ideal IMU at the origin, in body axes."""

    def __init__(self, position=(0.0, 0.0, 2.0), orientation=(1.0, 0.0, 0.0, 0.0)):
        self.position = np.array(position, dtype=float)
        self.velocity = np.zeros(3)
        self.orientation = np.array(orientation, dtype=float)
        self.rate = np.zeros(3)
        self.force = np.array([0.0, 0.0, MASS * GRAVITY])

    def imu(self):
        """Specific force and body rate, as the sensor would read them."""
        return self.rate.copy(), self.force / MASS

    def step(self, thrust, torque, wind=np.zeros(3)):
        R = quaternion_to_matrix(self.orientation)

        # The same quadratic drag as wind.py, on airspeed, in calm air too.
        airspeed = wind - self.velocity
        drag = DRAG * np.linalg.norm(airspeed) * airspeed

        total = R @ np.array([0.0, 0.0, thrust]) + drag
        self.force = R.T @ total

        acceleration = total / MASS - np.array([0.0, 0.0, GRAVITY])
        self.velocity = self.velocity + acceleration * DT
        self.position = self.position + self.velocity * DT

        spin = np.cross(self.rate, INERTIA * self.rate)
        self.rate = self.rate + (np.asarray(torque) - spin) / INERTIA * DT

        half = 0.5 * self.rate * DT
        step = np.array([1.0, *half])
        orientation = quaternion_multiply(self.orientation, step)
        self.orientation = orientation / np.linalg.norm(orientation)

    def truth(self):
        R = quaternion_to_matrix(self.orientation)

        return {
            "position": self.position,
            "orientation": self.orientation,
            "linear_velocity": self.velocity,
            "angular_velocity": R @ self.rate,
        }


def tilt_between(a, b):
    """Angle between two attitudes' up axes [rad]."""
    return float(np.arccos(np.clip(
        quaternion_to_matrix(a)[:, 2] @ quaternion_to_matrix(b)[:, 2],
        -1.0,
        1.0,
    )))


def pid_target(target):
    return {
        "xy": target[:2],
        "height": target[2],
        "yaw": 0.0,
        "mass": MASS,
        "max_xy_speed": 5.0,
        "max_z_speed": 5.0,
        "max_tilt_degrees": 10.0,
        "max_total_thrust": 28.0,
    }


def fly(seconds, noise=None, seed=0, wind=(0.0, 0.0, 0.0),
        vibration=0.0, settings=sensor_config.ESTIMATOR,
        start_offset=(0.0, 0.0, 0.0), use_truth=False):
    """Closed loop on the estimate. Returns final truth, estimate, and errors."""
    target = np.array([0.0, 0.0, 2.0])
    rng = np.random.default_rng(seed)

    body = RigidBody(target + np.asarray(start_offset, dtype=float))
    estimator = ImuEstimator(settings, DT, GRAVITY)
    estimator.reset(target)

    sensor = ImuNoise(noise or ZERO_NOISE, DT)
    sensor.reset(np.random.default_rng(seed + 1))

    gains = pid_target(target)
    pid = PIDController()
    true_errors = []
    tilts = []

    for _ in range(int(seconds / DT)):
        state = body.truth() if use_truth else estimator.state()
        thrust, *torque = pid.update(
            positions=state["position"],
            orientations=state["orientation"],
            velocities=state["linear_velocity"],
            angular_velocities=state["angular_velocity"],
            target=gains,
            dt=DT,
        )

        body.step(thrust, torque, np.asarray(wind, dtype=float))

        gyro, accel = sensor.apply(*body.imu())
        accel = accel + vibration * rng.normal(size=3)
        estimator.update(gyro, accel)

        true_errors.append(np.linalg.norm(body.position - target))
        tilts.append(tilt_between(estimator.orientation, body.orientation))

    return body, estimator, np.array(true_errors), np.array(tilts)


class Frames(unittest.TestCase):
    def test_level_rest_reports_no_motion(self):
        estimator = ImuEstimator(QUIET, DT, GRAVITY)
        estimator.reset(np.array([1.0, 2.0, 3.0]))

        for _ in range(1200):
            estimator.update(np.zeros(3), np.array([0.0, 0.0, GRAVITY]))

        np.testing.assert_allclose(estimator.position, [1.0, 2.0, 3.0], atol=1e-9)
        np.testing.assert_allclose(estimator.velocity, 0.0, atol=1e-9)

    def test_free_fall_integrates_down_at_g(self):
        settings = dict(QUIET, attitude_kp=0.0, attitude_ki=0.0)
        estimator = ImuEstimator(settings, DT, GRAVITY)
        estimator.reset(np.array([0.0, 0.0, 100.0]))
        estimator.update(np.zeros(3), np.array([0.0, 0.0, GRAVITY]))

        for _ in range(120):
            estimator.update(np.zeros(3), np.zeros(3))

        self.assertAlmostEqual(estimator.velocity[2], -GRAVITY, places=6)
        self.assertAlmostEqual(
            estimator.position[2], 100.0 - 0.5 * GRAVITY, places=6
        )

    def test_climb_acceleration_has_gravity_removed(self):
        estimator = ImuEstimator(QUIET, DT, GRAVITY)
        estimator.reset(np.zeros(3))
        estimator.update(np.zeros(3), np.array([0.0, 0.0, GRAVITY]))

        for _ in range(120):
            estimator.update(np.zeros(3), np.array([0.0, 0.0, GRAVITY + 1.0]))

        self.assertAlmostEqual(estimator.velocity[2], 1.0, places=6)

    def test_alignment_recovers_roll_and_pitch(self):
        truth = tilt_quaternion_from_gravity(
            np.array([0.05, -0.04, 1.0])
        )
        force = quaternion_to_matrix(truth).T @ np.array([0.0, 0.0, GRAVITY])

        estimator = ImuEstimator(QUIET, DT, GRAVITY)
        estimator.reset(np.zeros(3))
        estimator.update(np.zeros(3), force)

        self.assertLess(tilt_between(truth, estimator.orientation), 1e-6)

    def test_filter_corrects_a_wrong_tilt(self):
        settings = dict(QUIET, attitude_ki=0.0)
        estimator = ImuEstimator(settings, DT, GRAVITY)
        estimator.reset(np.zeros(3))
        estimator.update(np.zeros(3), np.array([0.0, 0.0, GRAVITY]))

        truth = tilt_quaternion_from_gravity(np.array([0.1, 0.0, 1.0]))
        force = quaternion_to_matrix(truth).T @ np.array([0.0, 0.0, GRAVITY])
        start = tilt_between(truth, estimator.orientation)

        for _ in range(int(6.0 / DT)):
            estimator.update(np.zeros(3), force)

        self.assertGreater(start, 0.09)
        self.assertLess(tilt_between(truth, estimator.orientation), 0.01)

    def test_rotated_mount_gives_body_axes(self):
        # Sensor z along body z, sensor x along body y.
        mount = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        estimator = ImuEstimator(QUIET, DT, GRAVITY, mount_rotation=mount)
        estimator.reset(np.zeros(3))
        estimator.update(np.zeros(3), np.array([0.0, 0.0, GRAVITY]))
        estimator.update(mount.T @ np.array([0.1, 0.2, 0.3]),
                         mount.T @ np.array([0.0, 0.0, GRAVITY]))

        np.testing.assert_allclose(
            estimator.angular_velocity_body, [0.1, 0.2, 0.3], atol=1e-3
        )

    def test_lever_arm_centripetal_term_is_removed(self):
        offset = np.array([0.1, 0.0, 0.0])
        rate = np.array([0.0, 0.0, 3.0])

        estimator = ImuEstimator(
            dict(QUIET, attitude_kp=0.0, attitude_ki=0.0),
            DT,
            GRAVITY,
            mount_offset=offset,
        )
        estimator.reset(np.zeros(3))
        estimator.update(
            rate,
            np.array([0.0, 0.0, GRAVITY]) + np.cross(
                rate, np.cross(rate, offset)
            ),
        )

        np.testing.assert_allclose(
            estimator.specific_force_body, [0.0, 0.0, GRAVITY], atol=1e-9
        )

    def test_reset_discards_a_teleport_spike(self):
        estimator = ImuEstimator(
            dict(sensor_config.ESTIMATOR, reset_ignore_samples=2,
                 alignment_samples=2),
            DT,
            GRAVITY,
        )
        estimator.reset(np.zeros(3))

        estimator.update(np.zeros(3), np.array([500.0, 0.0, 0.0]))
        estimator.update(np.zeros(3), np.array([500.0, 0.0, 0.0]))

        for _ in range(4):
            estimator.update(np.zeros(3), np.array([0.0, 0.0, GRAVITY]))

        np.testing.assert_allclose(estimator.velocity, 0.0, atol=1e-9)
        np.testing.assert_allclose(estimator.position, 0.0, atol=1e-9)

    def test_gyro_rate_is_used_directly_for_damping(self):
        estimator = ImuEstimator(QUIET, DT, GRAVITY)
        estimator.reset(np.zeros(3))
        estimator.update(
            np.array([0.2, -0.1, 0.05]), np.array([0.0, 0.0, GRAVITY])
        )

        np.testing.assert_allclose(
            estimator.angular_velocity_body, [0.2, -0.1, 0.05], atol=1e-3
        )


class Limits(unittest.TestCase):
    """These pin down what an IMU cannot do. They should keep passing."""

    def test_yaw_bias_is_not_observable(self):
        bias = 0.01
        estimator = ImuEstimator(QUIET, DT, GRAVITY)
        estimator.reset(np.zeros(3))

        for _ in range(int(20.0 / DT)):
            estimator.update(
                np.array([0.0, 0.0, bias]), np.array([0.0, 0.0, GRAVITY])
            )

        self.assertAlmostEqual(yaw_of(estimator.orientation), 0.2, delta=0.01)

    def test_start_offset_is_never_corrected(self):
        offset = np.array([0.2, -0.15, 0.1])
        body, estimator, errors, _ = fly(10.0, start_offset=offset)

        self.assertAlmostEqual(
            float(np.linalg.norm(body.position - estimator.position)),
            float(np.linalg.norm(offset)),
            delta=0.01,
        )

    def test_accel_bias_looks_like_tilt_and_the_drone_drifts_unseen(self):
        bias = np.array([0.02, 0.0, 0.0])
        target = np.array([0.0, 0.0, 2.0])

        body = RigidBody(target)
        estimator = ImuEstimator(QUIET, DT, GRAVITY)
        estimator.reset(target)
        gains = pid_target(target)
        pid = PIDController()

        for _ in range(int(30.0 / DT)):
            state = estimator.state()
            thrust, *torque = pid.update(
                positions=state["position"],
                orientations=state["orientation"],
                velocities=state["linear_velocity"],
                angular_velocities=state["angular_velocity"],
                target=gains,
                dt=DT,
            )
            body.step(thrust, torque)
            gyro, accel = body.imu()
            estimator.update(gyro, accel + bias)

        estimate_error = np.linalg.norm(estimator.position - target)
        true_error = np.linalg.norm(body.position - target)

        self.assertLess(estimate_error, 0.25 * true_error)
        self.assertGreater(true_error, 0.5)

    def test_tilt_during_acceleration_reads_level(self):
        # No airspeed yet, so no drag: thrust alone, along body z.
        tilted = tilt_quaternion_from_gravity(np.array([0.1, 0.0, 1.0]))
        body = RigidBody(orientation=tilted)
        body.step(MASS * GRAVITY, np.zeros(3))

        _, accel = body.imu()

        np.testing.assert_allclose(accel[:2], 0.0, atol=1e-9)


class Flight(unittest.TestCase):
    def test_noise_free_hover_holds_the_target(self):
        _, _, errors, _ = fly(10.0)

        self.assertLess(errors.max(), 0.01)

    def test_truth_control_is_the_reference(self):
        _, _, errors, _ = fly(
            10.0, wind=(2.0, 0.0, 0.0), use_truth=True
        )

        self.assertLess(errors.max(), 0.5)

    def test_noisy_hover_keeps_flying_for_one_episode(self):
        # A limit on tilt only: position may legitimately drift a long way.
        for seed in range(3):
            _, _, _, tilts = fly(
                10.0, noise=sensor_config.IMU_NOISE, seed=seed
            )

            self.assertLess(np.rad2deg(tilts.max()), 5.0)


if __name__ == "__main__":
    unittest.main()
