"""Unit tests for the flight workflow's NumPy parts. No Isaac Sim."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import path_sac_config
import hover_task
import sim_config

from path_sac_policy import (
    ReferencePID,
    attitude_error,
    failure_penalty,
    huber,
    rotor_correction_bound,
)
from path_reference import (
    ReferenceSample,
    TimedReference,
    distances_to_path,
    round_corners,
    sample_reference,
)


def level_state(position=(0.0, 0.0, 2.0), velocity=(0.0, 0.0, 0.0)):
    return {
        "position": np.array(position, dtype=float),
        "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
        "linear_velocity": np.array(velocity, dtype=float),
        "angular_velocity": np.zeros(3),
    }


class ReferenceTests(unittest.TestCase):
    def test_limits_hold_and_it_starts_and_ends_at_rest(self):
        reference = TimedReference(
            [(0, 0, 2), (5, 0, 2), (5, 5, 3)],
            speed=1.5,
            acceleration=0.5,
            lateral_acceleration=0.5,
            vertical_speed=0.4,
        )

        speeds = np.linalg.norm(reference.velocities, axis=1)

        self.assertLessEqual(speeds.max(), 1.5 + 1e-9)
        self.assertLessEqual(np.abs(reference.velocities[:, 2]).max(), 0.4 + 1e-3)
        self.assertEqual(speeds[0], 0.0)
        self.assertEqual(speeds[-1], 0.0)

    def test_velocity_is_the_derivative_of_position(self):
        reference = TimedReference(
            [(0, 0, 2), (6, 0, 2)],
            speed=1.0,
            acceleration=0.5,
            lateral_acceleration=0.5,
            vertical_speed=1.0,
        )

        dt = 1e-3
        before = reference.sample(3.0)
        after = reference.sample(3.0 + dt)

        np.testing.assert_allclose(
            (after.position - before.position) / dt,
            before.velocity,
            atol=1e-2,
        )

    def test_hover_route_holds(self):
        reference = TimedReference(
            [(1, 2, 3)], 1.0, 1.0, 1.0, 1.0, start_hold=2.0, end_hold=1.0
        )

        sample = reference.sample(1.0)

        np.testing.assert_array_equal(sample.position, (1, 2, 3))
        np.testing.assert_array_equal(sample.velocity, np.zeros(3))
        self.assertEqual(reference.duration, 3.0)

    def test_every_route_type_builds_inside_height_range(self):
        stage = path_sac_config.CURRICULUM[-1]
        episode = path_sac_config.EPISODE
        rng = np.random.default_rng(0)

        for route_type in path_sac_config.EVAL_SCENARIOS.values():
            for _ in range(5):
                _, reference = sample_reference(
                    rng, stage["route"], episode, route_type
                )

                low, high = episode["height_range"]

                self.assertGreaterEqual(reference.path[:, 2].min(), low - 1e-9)
                self.assertLessEqual(reference.path[:, 2].max(), high + 1e-9)
                self.assertTrue(np.all(np.isfinite(reference.accelerations)))

    def test_same_seed_gives_the_same_route(self):
        stage = path_sac_config.CURRICULUM[-1]

        first = sample_reference(
            np.random.default_rng(3), stage["route"], path_sac_config.EPISODE
        )
        second = sample_reference(
            np.random.default_rng(3), stage["route"], path_sac_config.EPISODE
        )

        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1].path, second[1].path)


class CornerTests(unittest.TestCase):
    corner = [(0, 0, 1), (2, 0, 1), (2, 2, 1)]

    def test_unsafe_arc_keeps_the_sharp_corner(self):
        points = round_corners(
            self.corner, 0.5, is_safe=lambda a, b: False
        )

        np.testing.assert_array_equal(points, np.array(self.corner, dtype=float))

    def test_locked_corner_stays_sharp(self):
        points = round_corners(self.corner, 0.5, locked=(1,))

        np.testing.assert_array_equal(points, np.array(self.corner, dtype=float))

    def test_safe_arc_stays_near_the_corner(self):
        points = round_corners(self.corner, 0.5)

        self.assertGreater(len(points), 3)
        self.assertLess(
            distances_to_path(points, self.corner).max(), 0.5
        )


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.mass = 1.62
        self.adapter = ReferencePID(
            {
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "mass": self.mass,
                "max_xy_speed": 100.0, "max_z_speed": 100.0,
                "max_tilt_degrees": 10.0, "max_total_thrust": 28.0,
            },
            self.mass,
        )

    def test_lead_makes_height_output_the_tracking_law(self):
        sample = ReferenceSample(
            np.array([0.0, 0.0, 2.1]),
            np.array([0.0, 0.0, 0.3]),
            np.array([0.0, 0.0, 0.2]),
            False,
        )

        command = self.adapter.read(
            level_state(velocity=(0.0, 0.0, 0.1)), sample, False
        )

        gains = sim_config.PID["height"]
        expected = (
            self.mass * 9.81
            + gains["kp"] * 0.1
            + gains["kd"] * (0.3 - 0.1)
            + self.mass * 0.2
        )

        self.assertAlmostEqual(command[0], expected, places=6)

    def test_integrators_see_tracking_error_not_the_lead(self):
        moving = ReferenceSample(
            np.array([0.0, 0.0, 2.0]),
            np.array([1.0, 0.0, 0.0]),
            np.zeros(3),
            False,
        )

        self.adapter.read(level_state(), moving, False)
        self.adapter.integrate(level_state(), moving, 1.0 / 120.0, False)

        np.testing.assert_array_equal(self.adapter.pid.xy_integral, np.zeros(2))

    def test_desired_attitude_inverts_the_torque_output(self):
        sample = ReferenceSample(
            np.array([1.0, 0.0, 2.0]), np.zeros(3), np.zeros(3), False
        )

        state = level_state()
        command = self.adapter.read(state, sample, False)
        desired = self.adapter.desired_attitude(state, command)

        # Forward target at zero yaw: nose down is positive pitch here.
        self.assertGreater(desired[1], 0.0)
        self.assertAlmostEqual(desired[0], 0.0, places=9)


class RewardTests(unittest.TestCase):
    def test_huber_meets_at_delta(self):
        np.testing.assert_allclose(huber(1.0, 1.0), 0.5)
        np.testing.assert_allclose(huber(3.0, 1.0), 2.5)
        np.testing.assert_allclose(huber(-3.0, 1.0), 2.5)

    def test_attitude_error_takes_the_short_way(self):
        a = hover_task.euler_to_quaternion(0.0, 0.0, 3.1)
        b = hover_task.euler_to_quaternion(0.0, 0.0, -3.1)

        self.assertLess(np.linalg.norm(attitude_error(a, b)), 0.1)
        self.assertLess(np.linalg.norm(attitude_error(a, -a)), 1e-6)

    def test_failure_costs_more_than_any_continuation(self):
        reward = path_sac_config.REWARD
        gamma = path_sac_config.SAC["gamma"]

        self.assertGreaterEqual(
            failure_penalty(reward, gamma),
            reward["cost_cap"] * sum(gamma**k for k in range(5000)),
        )


class ResidualTests(unittest.TestCase):
    def test_limits_fit_the_budget_on_the_real_geometry(self):
        positions = np.array(sim_config.ROTOR_POSITIONS)
        motor = sim_config.MOTOR

        allocation = np.vstack([
            np.ones(4),
            positions[:, 1],
            -positions[:, 0],
            np.array(motor["reaction_signs"])
            * motor["torque_coefficient"] / motor["thrust_coefficient"],
        ])

        bound = rotor_correction_bound(
            allocation,
            hover_task.residual_limits_array(path_sac_config.RESIDUAL_LIMITS),
            np.zeros(3),
        )

        budget = 0.02 * 1.5 * 1.62 * 9.81

        self.assertTrue(np.all(bound <= budget), bound)


if __name__ == "__main__":
    unittest.main()
