"""Indoor sensor processing checks: run with Isaac Sim's python, from drone_code:

    C:\\isaacsim\\python.bat -m unittest discover -s tests -p test_indoor_sensors.py -v

No SimulationApp needed. The readers themselves need Isaac: inspect_sensors.py.
"""
import os
import unittest

import numpy as np

import hover_ppo_config
import sensor_config

from indoor_sensors import RangefinderModel, channel_counts, clean_scan


QUIET = dict(
    sensor_config.RANGEFINDER,
    noise_std=0.0,
    noise_fraction=0.0,
    resolution=0.0,
    dropout_probability=0.0,
)


def scan_arrays():
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, -3.0],
    ])
    stamps = np.array([5_000_000, 5_000_000, 5_020_000, 5_100_000])
    distances = np.linalg.norm(points, axis=1)
    return points, stamps, distances, np.ones(4), np.array([0.2, 0.0, 1.1, -15.9])


class Scans(unittest.TestCase):
    def test_zero_distance_points_are_dropped(self):
        scan = clean_scan(*scan_arrays(), min_range=0.1)

        self.assertEqual(len(scan), 3)
        self.assertGreater(np.linalg.norm(scan.points, axis=1).min(), 0.9)

    def test_point_times_run_from_the_start_of_the_sweep(self):
        scan = clean_scan(*scan_arrays(), min_range=0.1)

        np.testing.assert_allclose(scan.point_times, [0.0, 2e-5, 1e-4])
        self.assertEqual(scan.start_time_ns, 5_000_000)

    def test_timestamps_keep_nanosecond_precision(self):
        points, _, distances, intensity, elevation = scan_arrays()
        stamps = np.array([8_600_030_608] * 2 + [8_600_030_708] * 2)

        scan = clean_scan(
            points, stamps, distances, intensity, elevation, min_range=0.1
        )

        self.assertAlmostEqual(scan.point_times.max(), 1e-7, places=12)

    def test_channels_are_counted_by_elevation(self):
        scan = clean_scan(*scan_arrays(), min_range=0.1)

        self.assertEqual(channel_counts(scan), {1: 1, 0: 1, -16: 1})

    def test_empty_or_all_invalid_scan_gives_nothing(self):
        self.assertIsNone(
            clean_scan(np.zeros((0, 3)), [], [], [], [], min_range=0.1)
        )
        self.assertIsNone(
            clean_scan(np.zeros((1, 3)), [1], [0.0], [1.0], [0.0], min_range=0.1)
        )

    def test_recorded_scan_format_is_read(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "sim_scene", "recordings",
            "20261016_001324_964195", "lidar_00200.npz",
        )

        if not os.path.exists(path):
            self.skipTest("recording not present")

        data = np.load(path)
        scan = clean_scan(
            data["points"],
            data["timestamp"].astype(np.int64),
            data["distance"],
            data["intensity"],
            data["elevation"],
            sensor_config.LIDAR["min_valid_range"],
        )

        self.assertGreater(len(scan), 10_000)
        self.assertGreater(np.linalg.norm(scan.points, axis=1).min(), 0.09)
        self.assertAlmostEqual(scan.point_times.max(), 0.1, delta=0.01)


class Rangefinder(unittest.TestCase):
    def model(self, **changes):
        return RangefinderModel(
            dict(QUIET, **changes), np.random.default_rng(0)
        )

    def test_clean_depth_passes_through(self):
        self.assertAlmostEqual(self.model().measure(2.0), 2.0)

    def test_a_miss_is_no_return_rather_than_a_range(self):
        model = self.model()

        self.assertIsNone(model.measure(12.0))
        self.assertIsNone(model.measure(100.0))

    def test_closer_than_minimum_is_no_return(self):
        self.assertIsNone(self.model().measure(0.01))

    def test_invalid_depth_is_no_return(self):
        self.assertIsNone(self.model().measure(float("nan")))

    def test_noise_has_the_configured_size(self):
        model = self.model(noise_std=0.02, noise_fraction=0.005)
        samples = np.array([model.measure(4.0) for _ in range(20_000)])

        self.assertAlmostEqual(samples.std(), 0.04, delta=0.002)
        self.assertAlmostEqual(samples.mean(), 4.0, delta=0.002)

    def test_resolution_quantizes(self):
        self.assertAlmostEqual(self.model(resolution=0.1).measure(2.04), 2.0)

    def test_dropouts_happen_at_the_configured_rate(self):
        model = self.model(dropout_probability=0.25)
        missing = sum(model.measure(2.0) is None for _ in range(20_000))

        self.assertAlmostEqual(missing / 20_000, 0.25, delta=0.02)

    def test_rate_holds_at_a_faster_physics_step(self):
        model = self.model(update_rate_hz=50.0)
        samples = sum(model.due(step / 120.0) for step in range(1200))

        self.assertIn(samples, (500, 501))


class Selection(unittest.TestCase):
    def test_original_scene_stays_the_default(self):
        self.assertEqual(hover_ppo_config.ACTIVE_SCENE, "original")
        self.assertEqual(hover_ppo_config.SCENE, "sim_scene/dij3d_boxes.usd")

    def test_both_scenes_exist(self):
        root = os.path.join(os.path.dirname(__file__), "..", "..")

        for scene in hover_ppo_config.SCENES.values():
            self.assertTrue(os.path.isfile(os.path.join(root, scene)), scene)


if __name__ == "__main__":
    unittest.main()
