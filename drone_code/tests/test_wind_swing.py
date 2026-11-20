"""Swinging wind direction and speed-scaled gusts."""
import unittest

import numpy as np

import hover_task
from wind import DirectionSwing


BASE = {
    "no_wind_probability": 0.0,
    "mean_speed_range": (8.0, 12.0),
    "gust_scale_range": (0.5, 1.0),
    "direction_rate_range": (0.0, 0.2),
}
EXTRAS = {
    "turbulence_intensity_range": (0.1, 0.3),
    "swing_degrees_range": (10.0, 30.0),
    "swing_timescale_range": (1.0, 3.0),
}


class DirectionSwingTest(unittest.TestCase):
    def test_swings_around_its_center_at_the_set_size(self):
        swing = DirectionSwing((0.0, 10.0, 0.0), np.deg2rad(20.0), 1.5, seed=3)
        dt = 1 / 120
        angles, speeds = [], []

        for _ in range(int(600 / dt)):
            wind = swing.step(dt)
            angles.append(np.arctan2(wind[1], wind[0]))
            speeds.append(np.linalg.norm(wind))

        offsets = np.degrees(np.array(angles) - np.pi / 2)
        self.assertAlmostEqual(np.mean(offsets), 0.0, delta=3.0)
        self.assertAlmostEqual(np.std(offsets), 20.0, delta=3.0)
        np.testing.assert_allclose(speeds, 10.0)

        # It goes both ways, many times: not a full turn, a back-and-forth.
        crossings = np.sum(np.diff(np.sign(offsets)) != 0)
        self.assertGreater(crossings, 100)

    def test_same_seed_same_wind(self):
        a = DirectionSwing((10.0, 0.0, 0.0), 0.3, 1.0, seed=[5, 2])
        b = DirectionSwing((10.0, 0.0, 0.0), 0.3, 1.0, seed=[5, 2])
        for _ in range(100):
            np.testing.assert_array_equal(a.step(0.01), b.step(0.01))

    def test_turn_rate_still_turns_the_center(self):
        swing = DirectionSwing((10.0, 0.0, 0.0), 0.0, 1.0, seed=1, turn_rate=0.5)
        swing.step(1.0)
        self.assertAlmostEqual(swing.center, 0.5)


class SampleWindExtrasTest(unittest.TestCase):
    def test_old_profiles_draw_exactly_as_before(self):
        plain = hover_task.sample_stage_wind(np.random.default_rng(7), BASE)
        self.assertEqual(set(plain), {"mean_wind", "gust_scale", "seed", "direction_rate"})

    def test_extras_do_not_shift_the_main_draws(self):
        rng_a, rng_b = np.random.default_rng(7), np.random.default_rng(7)
        plain = hover_task.sample_stage_wind(rng_a, BASE)
        extra = hover_task.sample_stage_wind(rng_b, {**BASE, **EXTRAS})

        np.testing.assert_array_equal(plain["mean_wind"], extra["mean_wind"])
        self.assertEqual(plain["seed"], extra["seed"])
        # The route and start drawn next see the same generator state.
        self.assertEqual(rng_a.random(), rng_b.random())

        self.assertTrue(0.1 <= extra["turbulence_intensity"] <= 0.3)
        self.assertTrue(np.deg2rad(10) <= extra["swing_std"] <= np.deg2rad(30))
        self.assertTrue(1.0 <= extra["swing_timescale"] <= 3.0)

    def test_calm_has_no_gusts_or_swing(self):
        calm = hover_task.sample_stage_wind(
            np.random.default_rng(1), {**BASE, **EXTRAS, "no_wind_probability": 1.0}
        )
        self.assertEqual(calm["turbulence_intensity"], 0.0)
        self.assertEqual(calm["swing_std"], 0.0)


if __name__ == "__main__":
    unittest.main()
