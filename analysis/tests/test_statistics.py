import copy
import io
import json
import zipfile

import numpy as np
import pytest

from analysis.archive_io import pair_rows, parse_rows, parse_boolean, model_metadata
from analysis.run_analysis import DEFAULTS
from analysis.stats_tools import (binary_statistics, cluster_overview, deterministic_seed,
                                  effect_statistics, holm_adjust)


def config():
    return dict(DEFAULTS, bootstrap_resamples=1000)


def test_holm_known_values_preserves_order_and_missing_slots():
    assert holm_adjust([.04, .001, .03]) == pytest.approx([.06, .003, .06])
    assert holm_adjust([.04, None, .001]) == [.08, None, .003]
    with pytest.raises(ValueError):
        holm_adjust([float("nan")])


def test_exact_signed_rank_and_sign_tests_for_ten_consistent_improvements():
    pid = np.arange(1, 11, dtype=float)
    sac = np.zeros(10)
    result = effect_statistics(pid, sac, config(), "ten_improvements")
    assert result["mean_difference"] == -5.5
    assert result["sac_lower"] == 10
    assert result["wilcoxon_p"] == 2/(2**10)
    assert result["sign_test_p"] == 2/(2**10)
    assert result["mean_difference_ci_high"] < 0


def test_bootstrap_repeats_and_reversing_controllers_reverses_effect():
    pid = np.array([.2, .5, .1, .4, .7, .9, .3, .8])
    sac = np.array([.15, .3, .2, .1, .9, .4, .1, .6])
    a = effect_statistics(pid, sac, config(), "repeat")
    b = effect_statistics(pid, sac, config(), "repeat")
    reverse = effect_statistics(sac, pid, config(), "repeat")
    assert a == b
    assert reverse["mean_difference"] == pytest.approx(-a["mean_difference"])
    assert reverse["wilcoxon_p"] == a["wilcoxon_p"]
    assert reverse["sign_test_p"] == a["sign_test_p"]
    assert reverse["mean_difference_ci_low"] == pytest.approx(-a["mean_difference_ci_high"])
    assert reverse["mean_difference_ci_high"] == pytest.approx(-a["mean_difference_ci_low"])


def test_zero_differences_and_single_observation_are_labeled():
    equal = effect_statistics([0]*10, [0]*10, config(), "zeros")
    assert equal["wilcoxon_p"] == 1
    assert equal["ties"] == 10
    assert equal["ci_status"] == "degenerate_observed_sample_not_equivalence"
    assert equal["relative_mean_reduction_percent"] is None
    single = effect_statistics([1], [0], config(), "one")
    assert single["ci_status"] == "insufficient_pairs"
    assert single["mean_difference_ci_low"] is None
    empty = effect_statistics([], [], config(), "empty")
    assert empty["n_pairs"] == 0 and empty["wilcoxon_p"] is None


def test_ties_and_zero_ranks_use_permutation_instead_of_invalid_exact_formula():
    result = effect_statistics([1,1,1,1], [0,0,1,1], config(), "ties")
    assert result["wilcoxon_method"] == "exhaustive_permutation_ties_or_zeros"
    assert result["sac_lower"] == 2 and result["ties"] == 2
    assert 0 <= result["wilcoxon_p"] <= 1


def test_exact_mcnemar_uses_only_discordant_pairs_and_swap_symmetry():
    pid = [False]*10
    sac = [True]*10
    result = binary_statistics(pid, sac)
    assert result["mcnemar_exact_p"] == 2/(2**10)
    assert result["sac_only_true"] == 10
    assert result["rate_difference"] == 1
    assert result["pid_rate_exact_ci_high"] > 0
    assert result["sac_rate_exact_ci_low"] < 1
    reverse = binary_statistics(sac, pid)
    assert reverse["mcnemar_exact_p"] == result["mcnemar_exact_p"]
    assert reverse["rate_difference"] == -1
    equal = binary_statistics([False,True], [False,True])
    assert equal["discordant_pairs"] == 0 and equal["mcnemar_exact_p"] == 1


def flight(seed, controller, *, wind_seed=9223372036854775801):
    return dict(scenario="hover", wind_profile="training_range", seed=seed,
                controller=controller, wind_seed=wind_seed, route_seconds=10., wind_speed=3.,
                residual_norm=0., residual_change=0., decisions=100, failed=False,
                position_rmse=float(seed+1) if controller=="pid_only" else .1)


def test_pairing_rejects_duplicates_missing_members_and_wind_mismatch():
    rows = [flight(s,c) for s in range(3) for c in ("pid_only","pid_sac")]
    pairs = pair_rows(rows)
    assert len(pairs) == 3
    assert pairs[0]["wind_seed"] == 9223372036854775801
    with pytest.raises(ValueError, match="Duplicate"):
        pair_rows(rows+[rows[0]])
    with pytest.raises(ValueError, match="Missing"):
        pair_rows(rows[:-1])
    changed = copy.deepcopy(rows)
    changed[1]["wind_seed"] += 1
    with pytest.raises(ValueError, match="wind seed mismatch"):
        pair_rows(changed)


def test_cluster_overview_counts_seeds_instead_of_repeated_cases():
    pairs = []
    for scenario in ("hover", "straight", "dijkstra"):
        for profile in ("training_range", "unseen_gusts", "outside_range"):
            for seed in range(3):
                pairs.append(dict(scenario=scenario, wind_profile=profile, seed=seed,
                                  pid={"position_rmse":float(seed+1)}, sac={"position_rmse":.1}))
    rows, values = cluster_overview(pairs, config())
    overall = next(r for r in rows if r["benchmark"]=="open_air_all_profiles")
    assert overall["paired_cases"] == 18
    assert overall["n_pairs"] == overall["independent_resampling_units"] == 3
    assert overall["mean_difference"] == pytest.approx(-1.9)
    assert len([r for r in values if r["benchmark"]=="open_air_all_profiles"]) == 3


def test_model_inspection_reads_metadata_without_unpickling_weight_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("data", json.dumps(dict(num_timesteps=1500000, _n_updates=1494999, seed=0,
                         observation_space={"_shape":[128], ":serialized:":"intentionally not a pickle"},
                         action_space={"_shape":[4]})))
        z.writestr("policy.pth", b"not executable model data")
        z.writestr("_stable_baselines3_version", "2.9.0")
        z.writestr("system_info.txt", "example")
    result = model_metadata(buffer.getvalue())
    assert result["timesteps"] == 1500000
    assert result["observation_shape"] == [128]
    assert result["deserialized_model"] is False


def test_outcome_parser_does_not_treat_false_string_as_true():
    assert parse_boolean("False") is False
    assert parse_boolean("True") is True
    with pytest.raises(ValueError):
        parse_boolean("unknown")


def test_metric_validation_rejects_bad_schema_and_nonfinite_values():
    with pytest.raises(ValueError, match="Missing evaluation columns"):
        parse_rows(b"seed,position_rmse\n1,nan\n")
    with pytest.raises(ValueError):
        effect_statistics([1, float("nan")], [0,1], config(), "bad")


def test_group_random_seeds_are_stable_and_distinct():
    assert deterministic_seed(42,"hover") == deterministic_seed(42,"hover")
    assert deterministic_seed(42,"hover") != deterministic_seed(42,"circle")
