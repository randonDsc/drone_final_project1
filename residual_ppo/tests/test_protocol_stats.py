"""M2.3-M2.5: protocol_stats and controllers (incl. PPO/SAC adapter refusals; SB3 + torch, no Isaac).

Run from residual_ppo/ with either interpreter:
    python -m unittest discover -s tests -p "test_*.py" -v
"""
import json
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


def grid(mse_still, mse_wind, seeds=range(5), fail=()):
    """mse_*: {scenario: [per-seed MSE]}. fail: set of (scenario, profile, seed) that failed."""
    rows = []
    for sc in SCENARIOS:
        for p, table in (("still_air", mse_still), ("training_range", mse_wind)):
            for i, s in enumerate(seeds):
                f = flight(sc, p, s, float(np.sqrt(table[sc][i])), failed=(sc, p, s) in fail)
                f.update(route_seconds=10.0 + s, max_decisions=100 + s, initial_obs_sha256=f"h{sc}{s}",
                         wind_speed=0.0 if p == "still_air" else 3.0,
                         wind_mean=[0.0, 0.0, 0.0] if p == "still_air" else [3.0, 0.0, 0.0])
                rows.append(f)
    return rows


class StillVsWindH1Test(unittest.TestCase):
    PROFILES = ("still_air", "training_range")

    def test_wind_share_point_and_decision_rule(self):
        # still MSE 1, wind MSE 4 everywhere -> share = 3/4 for every resample -> CI [0.75, 0.75] -> rejected
        rows = grid({sc: [1.0] * 5 for sc in SCENARIOS}, {sc: [4.0] * 5 for sc in SCENARIOS})
        h1 = ps.h1_analysis(rows, SCENARIOS, range(5))
        p = h1["primary_wind_share_mse"]
        self.assertAlmostEqual(p["value"], 0.75)
        self.assertAlmostEqual(p["ci95"][0], 0.75)
        self.assertEqual(p["verdict"], "rejected")
        self.assertTrue(h1["primary_and_sensitivity_agree"])
        self.assertEqual(ps.h1_verdict([0.1, 0.49]), "supported")
        self.assertEqual(ps.h1_verdict([0.4, 0.6]), "inconclusive")

    def test_ratio_of_sums_not_mean_of_ratios_and_negative_kept(self):
        still = {sc: [1.0, 1.0, 0.0, 2.0, 1.0] for sc in SCENARIOS}
        wind = {sc: [2.0, 1.0, 0.5, 1.0, 1.5] for sc in SCENARIOS}
        h1 = ps.h1_analysis(grid(still, wind), SCENARIOS, range(5))
        # sum(wind - still) = 1 - 0 + 0.5 - 1 + 0.5 = 1.0; sum(wind) = 6.0
        self.assertAlmostEqual(h1["primary_wind_share_mse"]["value"], 1.0 / 6.0)
        still2 = {sc: [2.0] * 5 for sc in SCENARIOS}
        wind2 = {sc: [1.0] * 5 for sc in SCENARIOS}
        self.assertAlmostEqual(ps.h1_analysis(grid(still2, wind2), SCENARIOS, range(5))
                               ["primary_wind_share_mse"]["value"], -1.0)

    def test_bootstrap_reproducible_and_paired(self):
        a = ps.bootstrap_mean([0.1, 0.5, 0.2, 0.9, 0.3])
        b = ps.bootstrap_mean([0.1, 0.5, 0.2, 0.9, 0.3])
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci95"][0], a["value"])
        self.assertGreaterEqual(a["ci95"][1], a["value"])

    def test_failure_sensitivity_excludes_seed_from_both_profiles(self):
        still = {sc: [1.0] * 5 for sc in SCENARIOS}
        wind = {sc: [2.0, 2.0, 2.0, 2.0, 100.0] for sc in SCENARIOS}
        rows = grid(still, wind, fail={("turns", "training_range", 4)})
        h1 = ps.h1_analysis(rows, SCENARIOS, range(5))
        sens = h1["sensitivity_excluding_failed_seeds"]
        self.assertEqual(sens["excluded_seeds"], [4])
        self.assertEqual(sens["n_seeds"], 4)
        self.assertAlmostEqual(sens["value"], 0.5)
        self.assertEqual(len(h1["failed_flights"]), 1)

    def test_hover_sanity_flag(self):
        still = {sc: [1.0] * 5 for sc in SCENARIOS}
        wind = {sc: [1.1] * 5 for sc in SCENARIOS}  # hover share 0.09 < 0.8 -> suspect
        h1 = ps.h1_analysis(grid(still, wind), SCENARIOS, range(5))
        self.assertTrue(h1["hover_sanity"]["suspect"])
        self.assertFalse(h1["h1_interpretable"])
        self.assertTrue(h1["verdict"].startswith("not interpreted"))

    def test_consistency_check(self):
        rows = grid({sc: [1.0] * 5 for sc in SCENARIOS}, {sc: [2.0] * 5 for sc in SCENARIOS})
        self.assertTrue(ps.consistency_check(rows, SCENARIOS, self.PROFILES, range(5))["pass"])
        rows[7]["initial_obs_sha256"] = "different"
        bad = ps.consistency_check(rows, SCENARIOS, self.PROFILES, range(5))
        self.assertFalse(bad["pass"])
        self.assertEqual(bad["problems"][0]["field"], "initial_obs_sha256")
        rows = grid({sc: [1.0] * 5 for sc in SCENARIOS}, {sc: [2.0] * 5 for sc in SCENARIOS})
        self.assertFalse(ps.consistency_check(rows[1:], SCENARIOS, self.PROFILES, range(5))["pass"])

    def test_still_vs_wind_table_paired_differences(self):
        rows = grid({sc: [1.0] * 5 for sc in SCENARIOS}, {sc: [4.0] * 5 for sc in SCENARIOS})
        table = ps.still_vs_wind_table(rows, SCENARIOS, self.PROFILES, range(5))
        self.assertEqual(len(table), 12)
        tr = [r for r in table if r["wind_profile"] == "training_range"][0]
        self.assertAlmostEqual(tr["position_rmse_diff_vs_still_mean"], 1.0)  # sqrt(4) - sqrt(1)
        self.assertNotIn("position_rmse_diff_vs_still_mean", table[0])


class EqualityChecksTest(unittest.TestCase):
    """M2.4: pilot flights must equal the 30-seed run bit-for-bit; 1-cell re-run must equal the stored row."""

    def test_exact_equality_nan_and_types(self):
        a = {"x": 0.1, "y": float("nan"), "z": [1.0, 2.0], "b": True, "s": None}
        self.assertEqual(ps.row_differences(a, dict(a)), {})
        self.assertIn("x", ps.row_differences(a, {**a, "x": float(np.nextafter(0.1, 1.0))}))  # 1 ulp
        self.assertIn("z", ps.row_differences(a, {**a, "z": [1.0, 2.0000000000000004]}))
        self.assertIn("b", ps.row_differences(a, {**a, "b": 1}))      # bool vs int not conflated
        self.assertIn("y", ps.row_differences(a, {**a, "y": 0.0}))
        self.assertIn("new", ps.row_differences(a, {**a, "new": 1}))  # missing on one side
        self.assertEqual(ps.row_differences(a, {**a, "x": 9.0}, ignore=("x",)), {})

    def test_subset_equality(self):
        ref = [flight("hover", "still_air", s, 0.1) for s in (1, 2)]
        cand = ref + [flight("hover", "still_air", 3, 0.1)]
        r = ps.subset_equality(ref, cand)
        self.assertTrue(r["pass"])
        self.assertEqual((r["compared"], r["identical"]), (2, 2))
        bad = [dict(cand[0], position_rmse=0.1000000001)] + cand[1:]
        r = ps.subset_equality(ref, bad)
        self.assertFalse(r["pass"])
        self.assertEqual(list(r["differences"]), ["hover|still_air|1"])
        r = ps.subset_equality(ref, cand[1:])
        self.assertEqual(r["missing_in_candidate"], [["hover", "still_air", 1]])
        self.assertFalse(r["pass"])


class D4ProfilesTest(unittest.TestCase):
    """D4: strong_range / beyond_strong = outside_range with only mean_speed_range changed; secondary endpoints."""

    def test_definitions_and_status(self):
        import copy

        import eval_protocol_v1 as protocol
        import path_sac_config

        outside = path_sac_config.EVAL_WIND_PROFILES["outside_range"]
        for name, rng in (("strong_range", (10.0, 14.0)), ("beyond_strong", (14.0, 16.0))):
            prof = protocol.OUR_PROFILES[name]
            self.assertEqual(prof["mean_speed_range"], rng)
            self.assertEqual({k: v for k, v in prof.items() if k != "mean_speed_range"},
                             {k: v for k, v in outside.items() if k != "mean_speed_range"})
            self.assertEqual(set(prof), set(outside))
        self.assertEqual(protocol.PROFILES, ("still_air", "training_range", "unseen_gusts", "outside_range",
                                             "strong_range", "beyond_strong"))
        self.assertEqual(protocol.PRIMARY_PROFILES, ("training_range", "unseen_gusts", "outside_range"))
        self.assertEqual(protocol.SECONDARY_PROFILES, ("strong_range", "beyond_strong"))
        before = copy.deepcopy(path_sac_config.EVAL_WIND_PROFILES)
        registered = protocol.register_profiles()
        self.assertEqual(set(registered), {"still_air", "strong_range", "beyond_strong"})
        for team in protocol.TEAM_PROFILES:  # team profiles untouched
            self.assertEqual(path_sac_config.EVAL_WIND_PROFILES[team], before[team])

    def test_same_draws_for_any_mean_speed_range(self):
        """Matched realizations: only the mean speed differs between outside_range and the D4 profiles."""
        import eval_protocol_v1 as protocol
        import hover_task
        import path_sac_config

        outside = path_sac_config.EVAL_WIND_PROFILES["outside_range"]
        for seed in range(5):
            ref = hover_task.sample_stage_wind(np.random.default_rng(seed), outside)
            for name in ("strong_range", "beyond_strong"):
                w = hover_task.sample_stage_wind(np.random.default_rng(seed), protocol.OUR_PROFILES[name])
                self.assertEqual((w["gust_scale"], w["seed"], w["direction_rate"]),
                                 (ref["gust_scale"], ref["seed"], ref["direction_rate"]))
                lo, hi = protocol.OUR_PROFILES[name]["mean_speed_range"]
                self.assertTrue(lo <= np.linalg.norm(w["mean_wind"]) <= hi)
                np.testing.assert_allclose(w["mean_wind"][:2] / np.linalg.norm(w["mean_wind"]),
                                           ref["mean_wind"][:2] / np.linalg.norm(ref["mean_wind"]))


class ReportingRulesTest(unittest.TestCase):
    """M2.5 E5 rules: exact McNemar, Clopper-Pearson, median decisions, controller report."""

    def test_binom_cdf_exact(self):
        self.assertAlmostEqual(ps.binom_cdf(3, 10, 0.5), 176 / 1024, places=15)
        self.assertEqual(ps.binom_cdf(-1, 10, 0.3), 0.0)
        self.assertEqual(ps.binom_cdf(10, 10, 0.3), 1.0)

    def test_mcnemar_exact(self):
        self.assertEqual(ps.mcnemar_exact(0, 0)["p_value"], 1.0)
        self.assertAlmostEqual(ps.mcnemar_exact(3, 7)["p_value"], 2 * 176 / 1024, places=15)
        self.assertAlmostEqual(ps.mcnemar_exact(7, 3)["p_value"], 2 * 176 / 1024, places=15)  # symmetric
        self.assertAlmostEqual(ps.mcnemar_exact(10, 0)["p_value"], 2 / 1024, places=15)
        self.assertEqual(ps.mcnemar_exact(5, 5)["p_value"], 1.0)  # capped at 1

    def test_clopper_pearson_known_values(self):
        lo, hi = ps.clopper_pearson(5, 10)
        self.assertAlmostEqual(lo, 0.187086028, places=8)
        self.assertAlmostEqual(hi, 0.812913972, places=8)
        lo, hi = ps.clopper_pearson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 1 - 0.025 ** 0.1, places=10)  # closed form for k = 0
        lo, hi = ps.clopper_pearson(10, 10)
        self.assertAlmostEqual(lo, 0.025 ** 0.1, places=10)
        self.assertEqual(hi, 1.0)
        # coverage check: the CI bounds are where the tail probabilities equal alpha/2
        lo, hi = ps.clopper_pearson(110, 180)
        self.assertAlmostEqual(1 - ps.binom_cdf(109, 180, lo), 0.025, places=9)
        self.assertAlmostEqual(ps.binom_cdf(110, 180, hi), 0.025, places=9)

    def _pair_runs(self):
        seeds = (1, 2, 3, 4)
        pid, ctrl = [], []
        for sc in ("hover", "straight"):
            for s in seeds:
                pid.append({**flight(sc, "outside_range", s, 1.0, failed=(s <= 3), reason="out_of_bounds"),
                            "decisions": 40 if s <= 3 else 160})
                ctrl.append({**flight(sc, "outside_range", s, 0.8, failed=(s == 1), reason="out_of_bounds"),
                             "decisions": 50 if s == 1 else 160, "controller": "x"})
        return pid, ctrl, seeds

    def test_paired_failure_comparison_and_decisions(self):
        pid, ctrl, seeds = self._pair_runs()
        fc = ps.paired_failure_comparison(ctrl, pid, "outside_range", ("hover", "straight"), seeds)
        self.assertEqual((fc["pairs"], fc["both_failed"], fc["controller_only_failed"], fc["pid_only_failed"],
                          fc["neither_failed"]), (8, 2, 0, 4, 2))
        self.assertAlmostEqual(fc["mcnemar"]["p_value"], 2 * 0.5 ** 4, places=15)
        d = ps.decisions_median_iqr(pid)
        self.assertEqual((d["median"], d["q25"], d["q75"], d["n"]), (40.0, 40.0, 70.0, 8))
        with self.assertRaises(ValueError):
            ps.paired_failure_comparison(ctrl[1:], pid, "outside_range", ("hover", "straight"), seeds)

    def test_controller_report_rules(self):
        pid, ctrl, seeds = self._pair_runs()
        still_pid = [{**flight(sc, "still_air", s, 0.2), "decisions": 160} for sc in ("hover", "straight") for s in seeds]
        rep = ps.controller_report(ctrl + still_pid, pid + still_pid, ("still_air", "outside_range"),
                                   ("hover", "straight"), seeds, primary=("outside_range",), secondary=(),
                                   reported_not_primary=("still_air",))
        still, out = rep
        self.assertEqual(still["status"], "reported, not primary")
        self.assertFalse(still["rmse_requires_failure_comparison"])  # rule 2: no failures anywhere
        self.assertNotIn("primary_endpoint", still)
        self.assertTrue(out["rmse_requires_failure_comparison"])    # rule 2: failures present
        pe = out["primary_endpoint"]
        self.assertAlmostEqual(pe["value"], -0.2, places=12)
        self.assertTrue(pe["improves_on_pid"])
        self.assertEqual(pe["interpret_only_with"], "paired_failure_comparison")
        self.assertEqual(out["controller"]["failures"]["failed"], 2)
        self.assertEqual(out["pid"]["failures"]["failed"], 6)
        # rule 3: secondary profile -> fixed endpoint order and RMSE label
        rep2 = ps.controller_report(ctrl, pid, ("outside_range",), ("hover", "straight"), seeds, primary=(),
                                    secondary=("outside_range",), reported_not_primary=())
        self.assertEqual(rep2[0]["rmse_label"], ps.RMSE_FLOWN_LABEL)
        self.assertEqual(rep2[0]["endpoint_order"][0], "failure count (exact CI)")
        self.assertNotIn("primary_endpoint", rep2[0])

    def test_pid_vs_itself_is_null(self):
        pid, _, seeds = self._pair_runs()
        r = ps.controller_report(pid, pid, ("outside_range",), ("hover", "straight"), seeds,
                                 primary=("outside_range",), secondary=(), reported_not_primary=())[0]
        self.assertEqual(r["paired_failure_comparison"]["mcnemar"]["p_value"], 1.0)
        self.assertEqual(r["primary_endpoint"]["ci95"], [0.0, 0.0])
        self.assertFalse(r["primary_endpoint"]["improves_on_pid"])


class ControllersTest(unittest.TestCase):
    def test_pid_returns_float32_zeros(self):
        a = controllers.make_controller("pid").act(np.ones(128, np.float32))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual(a.shape, (4,))
        self.assertFalse(a.any())

    def test_unknown_and_missing_model(self):
        with self.assertRaises(ValueError):
            controllers.make_controller("nope")
        for name in ("pid_ppo", "pid_sac"):
            with self.assertRaises(controllers.ControllerRefused):
                controllers.make_controller(name)              # no --model
            with self.assertRaises(controllers.ControllerRefused):
                controllers.make_controller(name, model="does/not/exist.zip")
        with self.assertRaises(controllers.ControllerRefused):
            controllers.make_controller("pid", model="x.zip")  # pid takes no model


class _Spaces:
    def __init__(self, obs_shape=(128,), act_high=1.0):
        import gymnasium as gym

        self.observation_space = gym.spaces.Box(-10.0, 10.0, obs_shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(-act_high, act_high, (4,), dtype=np.float32)


def _dummy_env():
    import gymnasium as gym

    class DummyPathEnv(gym.Env):
        """Only the PathSacEnv spaces (Box(-10, 10, (128,)), Box(-1, 1, (4,))); never stepped by these tests."""

        def __init__(self):
            sp = _Spaces()
            self.observation_space, self.action_space = sp.observation_space, sp.action_space

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            return np.zeros(128, np.float32), {}

        def step(self, action):
            return np.zeros(128, np.float32), 0.0, True, False, {}

    return DummyPathEnv()


class AdapterRefusalTest(unittest.TestCase):
    """M2.5 (a): PPO / SAC adapters refuse before loading anything; load and act deterministically when allowed."""

    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _garbage_model(self, sub):
        d = self.dir / sub
        d.mkdir()
        m = d / "model.zip"
        m.write_bytes(b"not a zip: a refusal must happen before SB3 ever reads this")
        return m

    def _write_ppo_contract(self, folder, policy):
        (folder / controllers.PPO_CONTRACT_FILE).write_text(json.dumps({"policy": policy}), encoding="utf-8")

    def test_ppo_refusals_before_load(self):
        import path_sac_config

        m = self._garbage_model("no_contract")
        with self.assertRaisesRegex(controllers.ControllerRefused, "no ppo_contract.json"):
            controllers.make_controller("pid_ppo", model=str(m))
        m = self._garbage_model("bad_policy")
        policy = json.loads(json.dumps(path_sac_config.policy_contract()))
        policy["residual_limits"] = [0.9, 0.05, 0.03, 0.02]
        self._write_ppo_contract(m.parent, policy)
        with self.assertRaisesRegex(controllers.ControllerRefused, "residual_limits"):
            controllers.make_controller("pid_ppo", model=str(m))
        m = self._garbage_model("extra_key")
        self._write_ppo_contract(m.parent, {**json.loads(json.dumps(path_sac_config.policy_contract())), "x": 1})
        with self.assertRaisesRegex(controllers.ControllerRefused, r"\['x'\]"):
            controllers.make_controller("pid_ppo", model=str(m))
        m = self._garbage_model("no_policy_group")
        (m.parent / controllers.PPO_CONTRACT_FILE).write_text(json.dumps({"training": {}}), encoding="utf-8")
        with self.assertRaisesRegex(controllers.ControllerRefused, "missing policy group"):
            controllers.make_controller("pid_ppo", model=str(m))

    def test_sac_refusals_before_load(self):
        import path_sac_config

        m = self._garbage_model("sac_no_contract")
        with self.assertRaisesRegex(controllers.ControllerRefused, "incompatibility"):
            controllers.make_controller("pid_sac", model=str(m))
        m = self._garbage_model("sac_training_differs")
        path_sac_config.write_contract(m.parent)
        c = json.loads((m.parent / path_sac_config.CONTRACT_FILE).read_text(encoding="utf-8"))
        c["training"]["sac"]["gamma"] = 0.98  # policy group fine, training group differs -> check_training refuses
        (m.parent / path_sac_config.CONTRACT_FILE).write_text(json.dumps(c), encoding="utf-8")
        with self.assertRaisesRegex(controllers.ControllerRefused, "training.sac"):
            controllers.make_controller("pid_sac", model=str(m))

    def test_check_spaces(self):
        ok, bad_obs, bad_act = _Spaces(), _Spaces(obs_shape=(127,)), _Spaces(act_high=2.0)
        controllers.check_spaces(ok.observation_space, ok.action_space, ok.observation_space, ok.action_space)
        with self.assertRaisesRegex(controllers.ControllerRefused, "observation space"):
            controllers.check_spaces(bad_obs.observation_space, ok.action_space, ok.observation_space, ok.action_space)
        with self.assertRaisesRegex(controllers.ControllerRefused, "action space"):
            controllers.check_spaces(ok.observation_space, bad_act.action_space, ok.observation_space, ok.action_space)

    def test_ppo_loads_acts_and_binds(self):
        import path_sac_config
        from stable_baselines3 import PPO

        folder = self.dir / "ppo_ok"
        folder.mkdir()
        PPO("MlpPolicy", _dummy_env(), policy_kwargs={"net_arch": [8]}, device="cpu", seed=0).save(folder / "model")
        self._write_ppo_contract(folder, path_sac_config.policy_contract())
        ctrl = controllers.make_controller("pid_ppo", model=str(folder / "model.zip"))
        obs = np.linspace(-1, 1, 128, dtype=np.float32)
        a1, a2 = ctrl.act(obs), ctrl.act(obs)
        self.assertEqual((a1.dtype, a1.shape), (np.float32, (4,)))
        self.assertTrue(np.array_equal(a1, a2) and np.all(np.isfinite(a1)))
        ctrl.bind_env(_Spaces())
        with self.assertRaises(controllers.ControllerRefused):
            ctrl.bind_env(_Spaces(obs_shape=(127,)))
        rec = ctrl.record()
        for k in ("model", "model_sha256", "saved_sb3_version", "runtime_sb3_version", "num_timesteps",
                  "contract_sha256", "device"):
            self.assertIn(k, rec)
        self.assertEqual(rec["device"], "cpu")

    def test_realization_guard(self):
        import run_protocol

        ref = {"key": ["wind_seed", "wind_speed", "route_seconds", "initial_obs_sha256", "max_decisions"],
               "flights": {"turns|training_range|400000": {
                   "wind_seed": 123, "wind_speed": 4.25, "route_seconds": 15.0, "initial_obs_sha256": "ab",
                   "max_decisions": 180}}}
        guard = run_protocol.make_realization_guard(ref)
        same = dict(ref["flights"]["turns|training_range|400000"])
        guard("turns", "training_range", 400000, {**same, "extra": 1})  # extra fields ignored
        self.assertEqual(guard.checked["resets"], 1)
        with self.assertRaisesRegex(run_protocol.RealizationMismatch, "wind_speed"):
            guard("turns", "training_range", 400000, {**same, "wind_speed": float(np.nextafter(4.25, 5.0))})
        with self.assertRaisesRegex(run_protocol.RealizationMismatch, "initial_obs_sha256"):
            guard("turns", "training_range", 400000, {**same, "initial_obs_sha256": "ac"})
        with self.assertRaisesRegex(run_protocol.RealizationMismatch, "no reference realization"):
            guard("turns", "training_range", 400001, same)
        self.assertEqual(guard.checked["resets"], 1)

    def test_sac_loads_and_acts_through_team_loader(self):
        import path_sac_config
        from stable_baselines3 import SAC

        folder = self.dir / "sac_ok"
        folder.mkdir()
        SAC("MlpPolicy", _dummy_env(), policy_kwargs={"net_arch": [8]}, buffer_size=10, device="cpu",
            seed=0).save(folder / "final_model")
        path_sac_config.write_contract(folder)
        ctrl = controllers.make_controller("pid_sac", model=str(folder / "final_model.zip"))
        obs = np.linspace(-1, 1, 128, dtype=np.float32)
        a = ctrl.act(obs)
        self.assertEqual((a.dtype, a.shape), (np.float32, (4,)))
        self.assertTrue(np.array_equal(a, ctrl.act(obs)))
        ctrl.bind_env(_Spaces())
        self.assertIn("load_path_policy", ctrl.record()["action"])


if __name__ == "__main__":
    unittest.main()
