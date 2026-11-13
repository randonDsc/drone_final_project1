"""M2.3: protocol_stats (aggregation, README reproduction check) and controllers. NumPy only, no Isaac.

Run from residual_ppo/ with either interpreter:
    python -m unittest discover -s tests -p "test_*.py" -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))

import controllers  # noqa: E402
import protocol_stats as ps  # noqa: E402

SCENARIOS = ("hover", "straight", "turns", "altitude", "circle", "figure_eight")


def flight(scenario, profile, seed, rmse, maxdev=1.0, failed=False, completed=False, reason=None):
    return {"scenario": scenario, "wind_profile": profile, "seed": seed, "controller": "pid",
            "position_rmse": rmse, "cross_track_rmse": rmse / 2, "cross_track_max": maxdev, "max_deviation": maxdev,
            "completed": completed, "completion_time": 10.0 if completed else None, "failed": failed,
            "failure_reason": reason if failed else None, "collided": False, "saturation_rate": 0.0,
            "residual_norm": 0.0, "residual_change": 0.0, "decisions": 100}


def profile_flights(profile, rmse_by_scenario, n_failed=0, n_completed=0, maxdev_by_scenario=None):
    out, k_f, k_c = [], 0, 0
    for scenario in SCENARIOS:
        for seed in range(300_000, 300_010):
            failed = k_f < n_failed
            completed = (not failed) and k_c < n_completed
            k_f += int(failed)
            k_c += int(completed)
            md = (maxdev_by_scenario or {}).get(scenario, 1.0) + (seed - 300_000) * 0.01
            out.append(flight(scenario, profile, seed, rmse_by_scenario[scenario], md, failed, completed,
                              "out_of_bounds"))
    return out


class AggregationTest(unittest.TestCase):
    def test_mean_over_scenario_means_and_both_maxdev_aggregations(self):
        rmse = dict(zip(SCENARIOS, (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)))
        maxdev = dict(zip(SCENARIOS, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)))
        agg = ps.profile_aggregate(profile_flights("training_range", rmse, maxdev_by_scenario=maxdev),
                                   "training_range", SCENARIOS)
        self.assertEqual(agg["flights"], 60)
        self.assertAlmostEqual(agg["position_rmse"], 0.35)
        # per-scenario max = base + 0.09; mean of those = 3.5 + 0.09; overall max = 6.09
        self.assertAlmostEqual(agg["max_deviation_mean_of_scenario_max"], 3.59)
        self.assertAlmostEqual(agg["max_deviation_overall_max"], 6.09)

    def test_counts_and_failure_reasons(self):
        rows = profile_flights("outside_range", dict.fromkeys(SCENARIOS, 1.0), n_failed=37, n_completed=2)
        agg = ps.profile_aggregate(rows, "outside_range", SCENARIOS)
        self.assertEqual((agg["failed_flights"], agg["completed_flights"]), (37, 2))
        self.assertEqual(agg["failure_reasons"], {"out_of_bounds": 37})
        self.assertAlmostEqual(agg["failure_rate"] * 60, 37)

    def test_cell_summary_matches_team_summarize_and_adds_iqr(self):
        rows = [flight("turns", "training_range", s, r) for s, r in zip(range(4), (0.1, 0.2, 0.3, 0.4))]
        out = ps.cell_summary(rows)
        self.assertAlmostEqual(out["position_rmse"], 0.25)
        self.assertAlmostEqual(out["position_rmse_median"], 0.25)
        self.assertAlmostEqual(out["position_rmse_q25"], 0.175)
        self.assertEqual(out["flights"], 4)


class ReproductionCheckTest(unittest.TestCase):
    def build(self, tr_rmse=0.502, ug_rmse=0.497, or_rmse=1.653, tr_completed=3, or_failed=37, drop=0):
        rows = (profile_flights("training_range", dict.fromkeys(SCENARIOS, tr_rmse), n_completed=tr_completed)
                + profile_flights("unseen_gusts", dict.fromkeys(SCENARIOS, ug_rmse), n_completed=3)
                + profile_flights("outside_range", dict.fromkeys(SCENARIOS, or_rmse), n_failed=or_failed))
        return rows[drop:]

    def test_exact_readme_values_pass(self):
        rep = ps.reproduction_check(self.build(), SCENARIOS)
        self.assertTrue(rep["pass"], rep)

    def test_rmse_boundary(self):
        self.assertTrue(ps.reproduction_check(self.build(tr_rmse=0.502 * 1.0499), SCENARIOS)["pass"])
        rep = ps.reproduction_check(self.build(tr_rmse=0.502 * 1.051), SCENARIOS)
        self.assertFalse(rep["pass"])
        self.assertFalse(rep["profiles"]["training_range"]["checks"]["position_rmse"])
        self.assertTrue(rep["profiles"]["unseen_gusts"]["pass"])

    def test_count_boundaries(self):
        # README 62% of 60 = 37.2 expected failures: 35..40 pass, 34 and 41 do not.
        for failed, ok in ((34, False), (35, True), (40, True), (41, False)):
            rep = ps.reproduction_check(self.build(or_failed=failed), SCENARIOS)
            self.assertEqual(rep["profiles"]["outside_range"]["checks"]["failed_flights"], ok, failed)
        # README 5% of 60 = 3 completed: 0..6 pass, 7 does not.
        self.assertTrue(ps.reproduction_check(self.build(tr_completed=6), SCENARIOS)["pass"])
        self.assertFalse(ps.reproduction_check(self.build(tr_completed=7), SCENARIOS)["pass"])

    def test_incomplete_profile_fails(self):
        rep = ps.reproduction_check(self.build(drop=1), SCENARIOS)
        self.assertFalse(rep["profiles"]["training_range"]["checks"]["complete"])
        self.assertFalse(rep["pass"])

    def test_text_report_renders(self):
        text = ps.reproduction_text(ps.reproduction_check(self.build(), SCENARIOS))
        self.assertIn("overall: PASS", text)


class ControllersTest(unittest.TestCase):
    def test_pid_returns_float32_zeros(self):
        a = controllers.make_controller("pid").act(np.ones(128, np.float32))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(a.shape, (4,))
        self.assertFalse(a.any())

    def test_future_controllers_not_implemented(self):
        for name in ("pid_ppo", "pid_sac"):
            with self.assertRaises(NotImplementedError):
                controllers.make_controller(name)
        with self.assertRaises(ValueError):
            controllers.make_controller("nope")


if __name__ == "__main__":
    unittest.main()
