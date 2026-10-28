"""Wind model checks: run `python -m unittest discover -s tests -v` here.

NumPy only; no Isaac Sim needed.
"""
import unittest
import numpy as np
import sim_config
from wind import GustingWind, drag_forces, quaternion_to_matrix

RHO, CD, AREA = 1.225, 1.0, 0.03
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def make_wind(points=(("c", (0, 0, 0), 1.0),), **overrides):
    config = dict(
        mean_wind=(0.0, 0.0, 0.0), gust_std=(0.0, 0.0, 0.0),
        gust_timescale=1.0, local_gust_std=(0.0, 0.0, 0.0),
        local_gust_timescale=0.3, correlation_length=0.4,
        air_density=RHO, drag_coefficient=CD, effective_area=AREA,
        sample_points=points, seed=0,
    )
    config.update(overrides)
    return GustingWind(**config)


def wrench(wind, orientation=IDENTITY, velocity=(0, 0, 0), omega=(0, 0, 0),
           com=(0, 0, 0)):
    return wind.body_wrench(np.zeros(3), orientation, velocity, omega, com)


class DragDirection(unittest.TestCase):
    def test_wind_pushes_drone_downwind(self):
        force, torque = wrench(make_wind(mean_wind=(10, 0, 0)))
        expected = 0.5 * RHO * CD * AREA * 100.0
        np.testing.assert_allclose(force, [expected, 0, 0])
        np.testing.assert_allclose(torque, 0, atol=1e-12)

    def test_motion_in_still_air_is_opposed(self):
        force, _ = wrench(make_wind(), velocity=(0, -4, 0))
        self.assertGreater(force[1], 0)
        np.testing.assert_allclose(force[[0, 2]], 0)

    def test_flying_with_the_wind_has_no_drag(self):
        force, _ = wrench(make_wind(mean_wind=(3, 1, 0)), velocity=(3, 1, 0))
        np.testing.assert_allclose(force, 0, atol=1e-12)

    def test_body_frame_output_follows_rotation(self):
        # 90 deg yaw: world +X wind is body -Y.
        yaw90 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        force, _ = wrench(make_wind(mean_wind=(10, 0, 0)), yaw90)
        R = quaternion_to_matrix(yaw90)
        np.testing.assert_allclose(R @ force, [0.5 * RHO * CD * AREA * 100, 0, 0],
                                   atol=1e-12)
        self.assertLess(force[1], 0)


class OffCenterTorque(unittest.TestCase):
    def test_single_offset_point_gives_r_cross_f(self):
        wind = make_wind((("p", (0, 0.2, 0), 1.0),), mean_wind=(10, 0, 0))
        force, torque = wrench(wind)
        np.testing.assert_allclose(torque, np.cross([0, 0.2, 0], force))
        self.assertLess(torque[2], 0)  # +X push at +Y arm yaws negative.

    def test_wrench_equals_sum_of_point_forces_when_rotated(self):
        wind = make_wind(sim_config.WIND["sample_points"],
                         mean_wind=(10, 0, 0), local_gust_std=(1, 1, 1))
        q = np.array([0.9, 0.1, -0.2, 0.3])
        q /= np.linalg.norm(q)
        args = (np.array([1.0, 2.0, 3.0]), q, (0.5, -1, 0.2), (0.3, -0.4, 1.0))
        positions, forces, _ = wind.point_forces(*args)
        force_b, torque_b = wind.body_wrench(*args)
        R = quaternion_to_matrix(q)
        # Each force applied at its own world position, torque about origin.
        np.testing.assert_allclose(R @ force_b, forces.sum(axis=0))
        np.testing.assert_allclose(
            R @ torque_b,
            np.cross(positions - args[0], forces).sum(axis=0), atol=1e-12)

    def test_uneven_wind_creates_torque(self):
        points = (("l", (0, 0.2, 0), 1.0), ("r", (0, -0.2, 0), 1.0))
        wind = make_wind(points, mean_wind=(10, 0, 0))
        _, even = wrench(wind)
        np.testing.assert_allclose(even, 0, atol=1e-12)
        wind.point_winds = np.array([[11.0, 0, 0], [9.0, 0, 0]])
        _, uneven = wrench(wind)
        self.assertLess(uneven[2], 0)

    def test_rotation_in_still_air_is_damped(self):
        points = (("l", (0, 0.2, 0), 1.0), ("r", (0, -0.2, 0), 1.0))
        force, torque = wrench(make_wind(points), omega=(0, 0, 5.0))
        np.testing.assert_allclose(force, 0, atol=1e-12)
        self.assertLess(torque[2], 0)

    def test_point_velocity_uses_com_moment_arm(self):
        # Spin about a COM at the sample point: that point has no airspeed.
        wind = make_wind((("p", (0, 0.2, 0), 1.0),))
        force, _ = wrench(wind, omega=(0, 0, 5.0), com=(0, 0.2, 0))
        np.testing.assert_allclose(force, 0, atol=1e-12)


class AreaPartition(unittest.TestCase):
    def test_more_points_do_not_multiply_drag(self):
        one, _ = wrench(make_wind(mean_wind=(10, 0, 0)))
        many = make_wind(tuple((f"p{i}", (0.1 * i, 0, 0), 1.0) for i in range(9)),
                         mean_wind=(10, 0, 0))
        nine, _ = wrench(many)
        np.testing.assert_allclose(nine, one)
        self.assertAlmostEqual(many.areas.sum(), AREA)

    def test_configured_points_sum_to_effective_area(self):
        wind = GustingWind(**sim_config.WIND)
        self.assertAlmostEqual(wind.areas.sum(), sim_config.WIND["effective_area"])
        np.testing.assert_allclose(sim_config.WIND["mean_wind"], [10, 0, 0])

    def test_drag_is_quadratic(self):
        f1 = drag_forces([[2.0, 0, 0]], [1.0], 1.0, 1.0)
        f2 = drag_forces([[4.0, 0, 0]], [1.0], 1.0, 1.0)
        np.testing.assert_allclose(f2, 4 * f1)


class GustStatistics(unittest.TestCase):
    def series(self, dt, steps, **overrides):
        config = dict(mean_wind=(10, 0, 0), gust_std=(2.0, 1.0, 0.5),
                      gust_timescale=2.0, seed=3)
        config.update(overrides)
        wind = make_wind(**config)
        return np.array([wind.step(dt)[0] for _ in range(steps)])

    def test_mean_and_std_match_configuration(self):
        samples = self.series(0.05, 200_000)
        np.testing.assert_allclose(samples.mean(axis=0), [10, 0, 0], atol=0.1)
        np.testing.assert_allclose(samples.std(axis=0), [2.0, 1.0, 0.5], rtol=0.05)

    def test_statistics_do_not_depend_on_timestep(self):
        tau = 2.0
        for dt in (1 / 120, 1 / 30, 0.5):
            lag = int(round(tau / dt))
            x = self.series(dt, int(4000 * tau / dt) if dt < 0.1 else 80_000,
                            seed=11)[:, 0] - 10
            self.assertAlmostEqual(x.std(), 2.0, delta=0.2, msg=f"dt={dt}")
            corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
            self.assertAlmostEqual(corr, np.exp(-1), delta=0.06, msg=f"dt={dt}")

    def test_gusts_are_smooth_not_white_noise(self):
        x = self.series(1 / 120, 20_000)[:, 0]
        step_change = np.abs(np.diff(x)).mean()
        self.assertLess(step_change, 0.1 * x.std())

    def test_seed_and_reset_reproduce_sequence(self):
        wind = make_wind(gust_std=(1, 1, 1), local_gust_std=(0.5, 0.5, 0.5),
                         sample_points=sim_config.WIND["sample_points"], seed=5)
        first = [wind.step(1 / 120).copy() for _ in range(50)]
        wind.reset()
        again = [wind.step(1 / 120).copy() for _ in range(50)]
        np.testing.assert_array_equal(first, again)

    def test_invalid_timestep_is_rejected(self):
        with self.assertRaises(ValueError):
            make_wind().step(0.0)


class SpatialCorrelation(unittest.TestCase):
    def test_nearby_points_are_more_correlated(self):
        points = (("a", (0, 0, 0), 1), ("near", (0.1, 0, 0), 1),
                  ("far", (1.0, 0, 0), 1))
        wind = make_wind(points, local_gust_std=(1, 1, 1),
                         correlation_length=0.4, seed=2)
        x = np.array([wind.step(0.05)[:, 0] for _ in range(100_000)])
        near = np.corrcoef(x[:, 0], x[:, 1])[0, 1]
        far = np.corrcoef(x[:, 0], x[:, 2])[0, 1]
        self.assertAlmostEqual(near, np.exp(-0.1 / 0.4), delta=0.05)
        self.assertAlmostEqual(far, np.exp(-1.0 / 0.4), delta=0.05)
        np.testing.assert_allclose(x.std(axis=0), 1.0, rtol=0.05)

    def test_shared_and_local_variances_add(self):
        points = (("a", (0, 0, 0), 1), ("b", (0.3, 0, 0), 1))
        wind = make_wind(points, gust_std=(1.5, 0, 0), local_gust_std=(0.5, 0, 0),
                         correlation_length=0.3, seed=4)
        x = np.array([wind.step(0.1)[:, 0] for _ in range(200_000)])
        self.assertAlmostEqual(x[:, 0].var(), 1.5**2 + 0.5**2, delta=0.15)
        expected_cov = 1.5**2 + 0.5**2 * np.exp(-1.0)
        self.assertAlmostEqual(np.cov(x[:, 0], x[:, 1])[0, 1], expected_cov,
                               delta=0.15)


class Switch(unittest.TestCase):
    def test_disabled_wind_applies_nothing(self):
        force, torque = wrench(make_wind(mean_wind=(10, 0, 0), enabled=False))
        np.testing.assert_array_equal(force, 0)
        np.testing.assert_array_equal(torque, 0)


if __name__ == "__main__":
    unittest.main()
