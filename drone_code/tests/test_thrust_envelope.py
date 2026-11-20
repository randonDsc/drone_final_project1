"""Thrust envelope, desaturating mixer, and their use in the controller and the
reference planner. NumPy only, no Isaac Sim."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim_config

from controller import PIDController
from path_reference import TimedReference
from thrust_envelope import GRAVITY, ThrustEnvelope, desaturate, mix


MASS = 1.62
MOTOR = sim_config.MOTOR
ROTOR_MAX = MOTOR["thrust_coefficient"] * MOTOR["max_speed"]**2
TOTAL_MAX = 4 * ROTOR_MAX


def envelope():
    return ThrustEnvelope(MASS, TOTAL_MAX)


def allocation():
    positions = np.array(sim_config.ROTOR_POSITIONS)
    signs = np.array(MOTOR["reaction_signs"], dtype=float)

    return np.vstack([
        np.ones(4),
        positions[:, 1],
        -positions[:, 0],
        signs * MOTOR["torque_coefficient"] / MOTOR["thrust_coefficient"],
    ])


def target(**extra):
    return {
        "height": 3.0, "xy": np.zeros(2), "yaw": 0.0, "mass": MASS,
        "max_xy_speed": sim_config.MAX_XY_SPEED,
        "max_z_speed": sim_config.MAX_Z_SPEED,
        "max_tilt_degrees": sim_config.MAX_TILT_DEGREES,
        "max_total_thrust": TOTAL_MAX,
        **extra,
    }


def update(pid, target, position=(0.0, 0.0, 3.0), velocity=(0.0, 0.0, 0.0)):
    return pid.update(
        np.array(position, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]),
        np.array(velocity, dtype=float), np.zeros(3), target, dt=0.0,
    )


class Budget(unittest.TestCase):
    def test_the_whole_command_stays_inside_the_usable_thrust(self):
        e = envelope()
        low, high = e.vertical_limits()

        for a_z in np.linspace(low, high, 25):
            a_xy = e.horizontal_limit(a_z)
            used = MASS * np.hypot(a_xy, GRAVITY + a_z)
            self.assertLessEqual(used, e.usable_thrust + 1e-9)

    def test_more_climb_leaves_less_horizontal(self):
        e = envelope()
        _, high = e.vertical_limits()
        values = [e.horizontal_limit(a) for a in np.linspace(0.0, high, 10)]

        self.assertTrue(np.all(np.diff(values) < 0))
        self.assertAlmostEqual(e.horizontal_limit(0.0), 9.9, delta=0.1)

    def test_climbing_never_takes_the_horizontal_minimum(self):
        e = envelope()
        _, high = e.vertical_limits()

        self.assertGreaterEqual(
            e.horizontal_limit(high),
            sim_config.THRUST_ENVELOPE["min_xy_acceleration"] - 1e-9,
        )

    def test_tilt_cap_applies_when_descending(self):
        e = envelope()
        low, _ = e.vertical_limits()
        lift = GRAVITY + low
        tilt = np.degrees(np.arctan(e.horizontal_limit(low) / lift))

        self.assertLessEqual(tilt, sim_config.MAX_TILT_DEGREES + 1e-6)


class Speeds(unittest.TestCase):
    def test_level_and_vertical_corners(self):
        e = ThrustEnvelope(MASS, TOTAL_MAX, max_xy_speed=1e3, max_z_speed=1e3)

        self.assertAlmostEqual(e.steady_speed((1, 0, 0)), 29.6, delta=0.2)
        self.assertAlmostEqual(e.steady_speed((0, 0, 1)), 19.1, delta=0.2)

    def test_steady_flight_at_the_edge_uses_the_whole_budget(self):
        e = ThrustEnvelope(MASS, TOTAL_MAX, max_xy_speed=1e3, max_z_speed=1e3)

        for angle in np.radians([0, 20, 45, 70, 90]):
            u = np.array([np.cos(angle), 0.0, np.sin(angle)])
            v = e.steady_speed(u)
            drag = e.drag * v**2 * u
            thrust = drag + np.array([0.0, 0.0, MASS * GRAVITY])
            self.assertAlmostEqual(
                np.linalg.norm(thrust), e.usable_thrust, delta=1e-6
            )

    def test_limiting_keeps_direction_and_respects_caps(self):
        e = envelope()
        request = np.array([30.0, 40.0, 25.0])
        limited, capped = e.limit_velocity(request)

        self.assertTrue(capped)
        np.testing.assert_allclose(
            limited / np.linalg.norm(limited),
            request / np.linalg.norm(request),
        )
        self.assertLessEqual(np.hypot(*limited[:2]), sim_config.MAX_XY_SPEED)
        self.assertLessEqual(abs(limited[2]), sim_config.MAX_Z_SPEED)

        small = np.array([0.5, 0.0, 0.2])
        same, capped = e.limit_velocity(small)
        self.assertFalse(capped)
        np.testing.assert_allclose(same, small)


class Mixer(unittest.TestCase):
    def test_reachable_wrench_is_exact(self):
        wrench = np.array([MASS * GRAVITY, 0.1, -0.05, 0.02])
        rotors, saturated = desaturate(allocation(), ROTOR_MAX, wrench, np.zeros(3))

        self.assertFalse(saturated)
        np.testing.assert_allclose(allocation() @ rotors, wrench, atol=1e-9)

    def test_full_thrust_keeps_torque_and_gives_up_lift(self):
        torques = np.array([0.8, -0.4, 0.1])
        wrench = np.array([TOTAL_MAX, *torques])
        rotors, saturated = desaturate(allocation(), ROTOR_MAX, wrench, np.zeros(3))

        self.assertTrue(saturated)
        self.assertTrue(np.all(rotors >= 0.0))
        self.assertTrue(np.all(rotors <= ROTOR_MAX + 1e-9))

        delivered = allocation() @ rotors
        np.testing.assert_allclose(delivered[1:], torques, atol=1e-9)
        self.assertLess(delivered[0], TOTAL_MAX)

    def test_com_offset_still_cancelled_after_desaturating(self):
        com = np.array([0.01, -0.005, 0.0])
        torques = np.array([0.5, 0.3, 0.0])
        rotors, _ = desaturate(
            allocation(), ROTOR_MAX, np.array([TOTAL_MAX, *torques]), com
        )

        delivered = allocation() @ rotors
        thrust = delivered[0]
        about_com = delivered[1:] - np.array([com[1], -com[0], 0.0]) * thrust
        np.testing.assert_allclose(about_com, torques, atol=1e-9)

    def test_torque_fits_until_no_thrust_can_deliver_it(self):
        # Lift given up, torque kept: saturated but still in control.
        _, saturated, fits = mix(
            allocation(), ROTOR_MAX, np.array([TOTAL_MAX, 0.8, -0.4, 0.1]),
            np.zeros(3),
        )
        self.assertTrue(saturated)
        self.assertTrue(fits)

        # A roll torque no rotor split can make: out of control authority.
        _, saturated, fits = mix(
            allocation(), ROTOR_MAX, np.array([15.0, 10.0, 0.0, 0.0]),
            np.zeros(3),
        )
        self.assertTrue(saturated)
        self.assertFalse(fits)

        # Normal hover.
        _, saturated, fits = mix(
            allocation(), ROTOR_MAX, np.array([MASS * GRAVITY, 0, 0, 0]),
            np.zeros(3),
        )
        self.assertFalse(saturated)
        self.assertTrue(fits)


class Controller(unittest.TestCase):
    def test_far_target_request_follows_the_envelope_corners(self):
        pid = PIDController()
        update(pid, target(xy=np.array([1000.0, 0.0])))
        self.assertTrue(pid.limits["speed_capped"])

        pid = PIDController()
        update(pid, target(height=3.0 + 0.1))
        self.assertFalse(pid.limits["speed_capped"])

    def test_vertical_acceleration_is_capped(self):
        pid = PIDController()
        thrust, *_ = update(pid, target(height=50.0), velocity=(0, 0, 0))

        _, high = envelope().vertical_limits()
        self.assertTrue(pid.limits["vertical_capped"])
        self.assertAlmostEqual(thrust, MASS * (GRAVITY + high), places=6)

    def test_horizontal_limit_shrinks_while_climbing(self):
        pid = PIDController()
        update(pid, target(xy=np.array([50.0, 0.0])))
        level = pid.limits["max_xy_acceleration"]

        pid = PIDController()
        update(pid, target(xy=np.array([50.0, 0.0]), height=50.0))
        climbing = pid.limits["max_xy_acceleration"]

        self.assertTrue(pid.limits["xy_capped"])
        self.assertLess(climbing, level)

    def test_never_asks_more_than_the_usable_thrust_level(self):
        e = envelope()
        pid = PIDController()

        for goal in ([50, 0, 50], [50, 50, 3], [0, 0, 50], [-30, 10, -20]):
            thrust, *_ = update(
                pid, target(xy=np.array(goal[:2], float), height=goal[2])
            )
            # Level attitude: thrust is the vertical force alone.
            self.assertLessEqual(thrust, e.usable_thrust + 1e-9)


class Planning(unittest.TestCase):
    def test_without_envelope_nothing_changes(self):
        points = np.array([[0, 0, 2], [5, 0, 2], [5, 5, 4]], dtype=float)
        a = TimedReference(points, 1.5, 0.5, 0.6, 1.0)
        b = TimedReference(points, 1.5, 0.5, 0.6, 1.0, envelope=None)

        np.testing.assert_allclose(a.times, b.times)

    def test_slow_routes_are_not_changed_by_the_envelope(self):
        points = np.array([[0, 0, 2], [5, 0, 2], [5, 5, 4]], dtype=float)
        a = TimedReference(points, 0.8, 0.5, 0.5, 0.5)
        b = TimedReference(points, 0.8, 0.5, 0.5, 0.5, envelope=envelope())

        np.testing.assert_allclose(a.times, b.times)

    def test_fast_request_is_held_inside_the_planning_budget(self):
        e = envelope()
        # Long enough to reach top speed at the planner's acceleration.
        points = np.array([[0, 0, 2], [800, 0, 2], [800, 0, 400]], dtype=float)
        ref = TimedReference(
            points, 100.0, 50.0, 50.0, 100.0, spacing=0.5, envelope=e
        )

        level = e.planning_speed((1, 0, 0))
        climb = e.planning_speed((0, 0, 1))
        speeds = np.linalg.norm(ref.velocities, axis=1)
        vertical = np.abs(ref.velocities[:, 2])

        self.assertLessEqual(speeds.max(), level + 1e-6)
        self.assertLessEqual(vertical.max(), climb + 1e-6)
        self.assertGreater(speeds.max(), 0.9 * level)

        accelerations = np.linalg.norm(ref.accelerations[1:-1], axis=1)
        self.assertLessEqual(
            np.percentile(accelerations, 99),
            e.planning_acceleration() + 1e-6,
        )


if __name__ == "__main__":
    unittest.main()
