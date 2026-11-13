"""Task-2 regression checks, runnable without Isaac Sim."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wind_events import WindEvent, event_metrics, trajectory_event_time
from path_sac_policy import PathCore


class WindEventTests(unittest.TestCase):
    def test_direction_steps_preserve_speed(self):
        for name, expected in [("turn90", [0, 4, 0]), ("turn180", [-4, 0, 0])]:
            event = WindEvent(name, 0, 5)
            np.testing.assert_allclose(event.at(4.99), [4, 0, 0])
            np.testing.assert_allclose(event.at(5), expected, atol=1e-12)

    def test_speed_steps_preserve_heading_and_control_is_constant(self):
        for name, speeds in [("speed_up", [2, 6]), ("speed_down", [6, 2]), ("control", [4, 4])]:
            event = WindEvent(name, np.pi / 2)
            for t, speed in zip([0, 6], speeds):
                np.testing.assert_allclose(event.at(t), [0, speed, 0], atol=1e-12)

    def test_core_applies_schedule_to_physics_and_preserves_legacy_rotation(self):
        seen = []
        sim = SimpleNamespace(dt=0.1, wind=SimpleNamespace(mean_wind=np.array([4., 0., 0.])))
        sim.step_command = lambda *a, **k: (seen.append(sim.wind.mean_wind.copy()), False, None, None)
        core = PathCore.__new__(PathCore)
        core.clear()
        core.sim = sim
        core.pid = SimpleNamespace(read=lambda *a: np.zeros(4),
            desired_attitude=lambda *a: np.zeros(3), integrate=lambda *a: None,
            acceleration_capped=lambda *a: False)
        core.sample = lambda: None
        core.wind_event = WindEvent("turn90", 0, 5)
        core.time = 5
        core.step({})
        np.testing.assert_allclose(seen[-1], [0, 4, 0], atol=1e-12)
        core.clear()
        core.direction_rate = 1.0
        sim.wind.mean_wind = np.array([4., 0., 0.])
        core.step({})
        np.testing.assert_allclose(seen[-1], [4*np.cos(.1), 4*np.sin(.1), 0])

    def test_reset_clears_event(self):
        core = PathCore.__new__(PathCore)
        core.wind_event = WindEvent("turn90", 0)
        core.clear()
        self.assertIsNone(core.wind_event)

    def test_recovery_requires_dwell_after_peak(self):
        result = event_metrics([1, 2, 3, 4, 5, 6, 7], [.01, .01, .3, .05, .2, .04, .03], 2)
        self.assertEqual(result["recovery_time_s"], 4)

    def test_failure_and_short_runs_never_count_as_recovered(self):
        for errors, failed in [([.2, .01, .01], True), ([.2, np.nan, .01], False)]:
            result = event_metrics([1, 2, 3], errors, .5, failed=failed)
            self.assertTrue(result["failed"])
            self.assertIsNone(result["recovery_time_s"])
        result = event_metrics([.1, .2], [.01, .01], 5)
        self.assertFalse(result["event_reached"])
        self.assertFalse(result["recovered"])

    def test_peak_prevents_premature_recovery(self):
        result = event_metrics([1, 2, 3, 4, 5], [.01, .01, .01, .4, .02], .5)
        self.assertIsNone(result["recovery_time_s"])

    def test_moving_event_falls_within_motion_and_window(self):
        reference = SimpleNamespace(motion_end=15, sample=lambda t: SimpleNamespace(velocity=np.array([1., 0., 0.])))
        event = trajectory_event_time(reference, 20)
        self.assertGreater(event, .5)
        self.assertLess(event, 15)
        self.assertLessEqual(event, 18)
        self.assertEqual(trajectory_event_time(reference, 20, 6), 6)

    def test_stationary_or_late_moving_event_rejected(self):
        reference = SimpleNamespace(motion_end=15, sample=lambda t: SimpleNamespace(velocity=np.zeros(3)))
        with self.assertRaises(ValueError):
            trajectory_event_time(reference, 20)
        with self.assertRaises(ValueError):
            trajectory_event_time(reference, 20, 19)

    def test_invalid_schedule_rejected(self):
        with self.assertRaises(ValueError):
            WindEvent("turn90", float("nan"))


if __name__ == "__main__":
    unittest.main()
