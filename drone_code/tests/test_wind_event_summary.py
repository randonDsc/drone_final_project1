"""Reject incomplete or altered results instead of silently summarizing them."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import contextlib
import io
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summarize_wind_events import analyze
from wind_events import event_metrics


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.manifest = {"completed": True, "model": None, "scenarios": ["turn90"],
                         "seeds": [1], "duration_s": 3, "sampling_hz": 1,
                         "event_s": 1, "contract": {"physics_rate_hz": 120}}
        self.flight = {"scenario": "turn90", "seed": 1, "controller": "pid_only",
                       "heading_rad": 0, "failure_reason": None,
                       **event_metrics([1, 2, 3], [.2, .05, .02], 1)}
        self.samples = [
            {"time_s": 1, "error_m": .2, "mean_wind_x_m_s": 4, "mean_wind_y_m_s": 0, "mean_wind_z_m_s": 0},
            {"time_s": 2, "error_m": .05, "mean_wind_x_m_s": 0, "mean_wind_y_m_s": 4, "mean_wind_z_m_s": 0},
            {"time_s": 3, "error_m": .02, "mean_wind_x_m_s": 0, "mean_wind_y_m_s": 4, "mean_wind_z_m_s": 0},
        ]

    def save(self):
        (self.folder / "manifest.json").write_text(json.dumps(self.manifest))
        (self.folder / "flights.json").write_text(json.dumps([self.flight]))
        with (self.folder / "turn90_1_pid_only.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.samples[0]))
            w.writeheader()
            w.writerows(self.samples)

    def test_valid_results(self):
        self.save()
        with contextlib.redirect_stdout(io.StringIO()):
            rows = analyze(self.folder)
        self.assertEqual(rows[0]["stayed_below_threshold"], 1)
        self.assertEqual(rows[0]["failed"], 0)

    def test_incomplete_run_rejected(self):
        self.manifest["completed"] = False
        self.save()
        with self.assertRaises(ValueError):
            analyze(self.folder)

    def test_wrong_wind_rejected(self):
        self.samples[1]["mean_wind_y_m_s"] = 0
        self.save()
        with self.assertRaises(AssertionError):
            analyze(self.folder)

    def test_wrong_score_rejected(self):
        self.flight["position_rmse_m"] = 0
        self.save()
        with self.assertRaises(AssertionError):
            analyze(self.folder)

    def test_short_successful_flight_rejected(self):
        self.samples.pop()
        self.save()
        with self.assertRaises(AssertionError):
            analyze(self.folder)

    def moving_fixture(self):
        self.manifest.update(route="straight", event_s=None)
        self.flight.update(event_s=1, reference_speed_at_event_m_s=1)
        for sample in self.samples:
            sample["reference_speed_m_s"] = 1
            for axis in "xyz":
                sample[f"target_{axis}_m"] = 0
                sample[f"position_{axis}_m"] = sample["error_m"] if axis == "x" else 0

    def test_moving_run_uses_per_flight_event_and_checks_positions(self):
        self.moving_fixture()
        self.save()
        with contextlib.redirect_stdout(io.StringIO()):
            rows = analyze(self.folder)
        self.assertAlmostEqual(rows[0]["mean_moving_rmse_m"], self.flight["position_rmse_m"])
        self.samples[0]["position_x_m"] += 1
        self.save()
        with self.assertRaises(AssertionError):
            analyze(self.folder)

    def test_stationary_moving_event_rejected(self):
        self.moving_fixture()
        self.flight["reference_speed_at_event_m_s"] = 0
        self.save()
        with self.assertRaises(ValueError):
            analyze(self.folder)
