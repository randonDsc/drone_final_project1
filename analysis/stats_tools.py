"""Seed-paired inference with explicitly scoped multiple-testing families."""
import hashlib
import math
import warnings

import numpy as np
from scipy import stats


PRIMARY = "position_rmse"
SECONDARY = ("cross_track_rmse", "max_deviation", "cross_track_max", "saturation_rate")
UNITS = {"position_rmse": "m", "cross_track_rmse": "m", "max_deviation": "m",
         "cross_track_max": "m", "saturation_rate": "fraction"}


def deterministic_seed(master, label):
    return int.from_bytes(hashlib.sha256(f"{master}:{label}".encode()).digest()[:8], "big")


def holm_adjust(pvalues):
    """Step-down Bonferroni, preserving order; absent tests retain family slots."""
    values = np.asarray([1.0 if p is None else float(p) for p in pvalues])
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must be finite and in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(values[order]*(len(values)-np.arange(len(values)))))
    result = np.empty(len(values))
    result[order] = adjusted
    return [None if p is None else float(result[i]) for i, p in enumerate(pvalues)]


def effect_statistics(pid, sac, config, label):
    pid, sac = np.asarray(pid, float), np.asarray(sac, float)
    if pid.ndim != 1 or pid.shape != sac.shape or not np.all(np.isfinite([pid, sac])):
        raise ValueError("Finite, paired one-dimensional metric arrays are required")
    seed = deterministic_seed(config["resampling_seed"], label)
    result = dict(n_pairs=len(pid), pid_mean=None, sac_mean=None, mean_difference=None,
                  median_difference=None, difference_sd=None, mean_difference_ci_low=None,
                  mean_difference_ci_high=None, ci_status="no_observations", sac_lower=0, pid_lower=0, ties=0,
                  relative_mean_reduction_percent=None, wilcoxon_statistic=None, wilcoxon_p=None,
                  sign_test_p=None, wilcoxon_method="not_tested", bootstrap_seed=seed)
    if not len(pid):
        return result
    difference = sac-pid
    result.update(pid_mean=float(pid.mean()), sac_mean=float(sac.mean()), mean_difference=float(difference.mean()),
                  median_difference=float(np.median(difference)),
                  difference_sd=float(difference.std(ddof=1)) if len(pid)>1 else None,
                  relative_mean_reduction_percent=float(100*(pid.mean()-sac.mean())/pid.mean()) if pid.mean()>0 else None)
    tolerance = config["difference_tolerance"]
    result.update(sac_lower=int(np.sum(difference < -tolerance)), pid_lower=int(np.sum(difference > tolerance)),
                  ties=int(np.sum(abs(difference) <= tolerance)))
    if len(pid) < 2:
        result["ci_status"] = "insufficient_pairs"
    elif float(np.ptp(difference)) <= tolerance:
        result.update(mean_difference_ci_low=float(difference.mean()), mean_difference_ci_high=float(difference.mean()),
                      ci_status="degenerate_observed_sample_not_equivalence")
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ci = stats.bootstrap((difference,), np.mean, confidence_level=config["confidence_level"],
                                 n_resamples=config["bootstrap_resamples"], batch=2000,
                                 method="BCa", rng=np.random.default_rng(seed)).confidence_interval
        if np.all(np.isfinite(ci)):
            result.update(mean_difference_ci_low=float(ci.low), mean_difference_ci_high=float(ci.high), ci_status="BCa")
        else:
            # A silent switch to another interval would change the stated method.
            result["ci_status"] = "BCa_unavailable_inspect_sample"
        result["bootstrap_warning"] = "; ".join(sorted({str(w.message) for w in caught}))
    # Rounding applies only to rank tests, not estimates or bootstrap intervals.
    ranked = np.round(difference, config["rank_round_decimals"])
    ranked[abs(difference) <= tolerance] = 0
    if not np.any(ranked):
        result.update(wilcoxon_statistic=0.0, wilcoxon_p=1.0, sign_test_p=1.0,
                      wilcoxon_method="all_zero_no_rank_information")
    else:
        nonzero = ranked[ranked != 0]
        distinct = len(np.unique(abs(nonzero))) == len(nonzero)
        if len(nonzero) == len(ranked) and distinct:
            method, method_name = "exact", "exact_nonzero_distinct"
        else:
            # Exhaustive sign permutations for the small tied/zero samples here.
            # For future larger evaluations, use fixed-seed Monte Carlo sampling.
            samples = np.inf if len(ranked) <= 15 else config["bootstrap_resamples"]
            method = stats.PermutationMethod(n_resamples=samples, rng=np.random.default_rng(seed))
            method_name = "exhaustive_permutation_ties_or_zeros" if len(ranked)<=15 else "monte_carlo_permutation_ties_or_zeros"
        test = stats.wilcoxon(ranked, zero_method="wilcox", alternative="two-sided", method=method)
        result.update(wilcoxon_statistic=float(test.statistic), wilcoxon_p=float(test.pvalue), wilcoxon_method=method_name,
                      sign_test_p=float(stats.binomtest(result["sac_lower"], result["sac_lower"]+result["pid_lower"], p=.5).pvalue))
    return result


def adjust_family(rows, family, config, tests=("wilcoxon", "sign_test")):
    for test in tests:
        raw = [r[test+"_p"] for r in rows]
        for row, p in zip(rows, holm_adjust(raw)):
            row[test+"_p_holm"] = p
            row[test+"_reject_holm"] = p is not None and p <= config["alpha"]
    for row in rows:
        row.update(test_family=family, family_size=len(rows), intervals="pointwise_not_multiplicity_adjusted")


def binary_statistics(pid, sac, confidence=.95):
    pid, sac = np.asarray(pid, bool), np.asarray(sac, bool)
    if pid.ndim != 1 or pid.shape != sac.shape or not len(pid):
        raise ValueError("Nonempty paired binary arrays are required")
    both = int(np.sum(pid & sac))
    pid_only = int(np.sum(pid & ~sac))
    sac_only = int(np.sum(~pid & sac))
    neither = int(np.sum(~pid & ~sac))
    discordant = pid_only+sac_only
    # Exact McNemar conditional test = binomial test on discordant pairs.
    p = float(stats.binomtest(sac_only, discordant, p=.5).pvalue) if discordant else 1.0
    result = dict(n_pairs=len(pid), both_true=both, pid_only_true=pid_only, sac_only_true=sac_only,
                  both_false=neither, discordant_pairs=discordant,
                  pid_true=int(pid.sum()), sac_true=int(sac.sum()),
                  pid_rate=float(pid.mean()), sac_rate=float(sac.mean()), rate_difference=float(sac.mean()-pid.mean()),
                  mcnemar_exact_p=p)
    for name, values in (("pid", pid), ("sac", sac)):
        ci = stats.binomtest(int(values.sum()), len(values)).proportion_ci(confidence_level=confidence, method="exact")
        result[name+"_rate_exact_ci_low"], result[name+"_rate_exact_ci_high"] = float(ci.low), float(ci.high)
    return result


def groups_in_order(pairs, scenarios, profiles):
    for scenario in scenarios:
        for profile in profiles:
            group = sorted([p for p in pairs if p["scenario"]==scenario and p["wind_profile"]==profile], key=lambda p:p["seed"])
            yield scenario, profile, group


def continuous_table(pairs, metrics, config, family, scenarios, profiles, subset=False):
    rows = []
    for metric in metrics:
        for scenario, profile, group in groups_in_order(pairs, scenarios, profiles):
            total = len(group)
            if subset:
                group = [p for p in group if not p["pid"]["failed"] and not p["sac"]["failed"]
                         and p["pid"]["decisions"]==p["sac"]["decisions"]]
            row = dict(scenario=scenario, wind_profile=profile, metric=metric, unit=UNITS[metric],
                       eligible_pairs=total, excluded_pairs=total-len(group),
                       population="both_nonfailed_equal_recorded_lengths" if subset else "all_recorded_pairs")
            row.update(effect_statistics([p["pid"][metric] for p in group], [p["sac"][metric] for p in group],
                                         config, f"{family}:{metric}:{scenario}:{profile}"))
            rows.append(row)
    adjust_family(rows, family, config)
    return rows


def secondary_table(pairs, config, scenarios, profiles):
    """Metric-wise Holm families with a joint-family sensitivity calculation."""
    # Keep stable resampling labels so paired estimates and BCa draws are fixed.
    rows = continuous_table(pairs, SECONDARY, config, "secondary_continuous_84", scenarios, profiles)
    for row in rows:
        row.update(joint_test_family="secondary_joint_sensitivity", joint_family_size=len(rows))
        for test in ("wilcoxon", "sign_test"):
            row[test+"_p_holm_joint"] = row[test+"_p_holm"]
            row[test+"_reject_holm_joint"] = row[test+"_reject_holm"]
    for metric in SECONDARY:
        group = [row for row in rows if row["metric"]==metric]
        adjust_family(group, "secondary_"+metric, config)
    return rows


def outcome_table(pairs, config, scenarios, profiles):
    rows = []
    for outcome in ("completed", "failed", "collided"):
        family = []
        for scenario, profile, group in groups_in_order(pairs, scenarios, profiles):
            row = dict(scenario=scenario, wind_profile=profile, outcome=outcome,
                       favorable_direction="positive" if outcome=="completed" else "negative",
                       test_family="secondary_binary_"+outcome,
                       interval_scope="marginal_controller_rates_not_paired_rate_difference")
            row.update(binary_statistics([p["pid"][outcome] for p in group], [p["sac"][outcome] for p in group], config["confidence_level"]))
            family.append(row)
        for row, p in zip(family, holm_adjust([r["mcnemar_exact_p"] for r in family])):
            row.update(mcnemar_p_holm=p, mcnemar_reject_holm=p <= config["alpha"], family_size=len(family))
        rows += family
    return rows


def cluster_overview(pairs, config):
    buckets = [("open_air_all_profiles", lambda p:p["scenario"]!="dijkstra"),
               ("dijkstra_all_profiles", lambda p:p["scenario"]=="dijkstra")]
    buckets += [("open_air_"+profile, lambda p,w=profile:p["scenario"]!="dijkstra" and p["wind_profile"]==w)
                for profile in ("training_range", "unseen_gusts", "outside_range")]
    rows, seed_rows = [], []
    for name, select in buckets:
        selected = [p for p in pairs if select(p)]
        seeds = sorted({p["seed"] for p in selected})
        pid, sac = [], []
        for seed in seeds:
            group = [p for p in selected if p["seed"]==seed]
            a, b = (float(np.mean([p[c][PRIMARY] for p in group])) for c in ("pid", "sac"))
            pid.append(a); sac.append(b)
            seed_rows.append(dict(benchmark=name, seed=seed, cases=len(group), pid_mean_flight_rmse=a,
                                  sac_mean_flight_rmse=b, difference=b-a))
        row = dict(benchmark=name, metric="mean_of_flight_position_rmse", unit="m", paired_cases=len(selected),
                   independent_resampling_units=len(seeds), weighting="equal_weight_fixed_conditions_within_each_seed")
        row.update(effect_statistics(pid, sac, config, "cluster:"+name))
        rows.append(row)
    adjust_family(rows, "exploratory_seed_cluster_overviews", config)
    return rows, seed_rows


def diagnostic_tables(pairs, scenarios, profiles):
    lengths, effort = [], []
    for scenario, profile, group in groups_in_order(pairs, scenarios, profiles):
        lengths.append(dict(scenario=scenario, wind_profile=profile, pairs=len(group),
                            different_decision_counts=sum(p["pid"]["decisions"]!=p["sac"]["decisions"] for p in group),
                            either_failed=sum(p["pid"]["failed"] or p["sac"]["failed"] for p in group),
                            both_failed=sum(p["pid"]["failed"] and p["sac"]["failed"] for p in group),
                            both_nonfailed_equal_counts=sum(not p["pid"]["failed"] and not p["sac"]["failed"]
                                                         and p["pid"]["decisions"]==p["sac"]["decisions"] for p in group),
                            sac_failed_zero_mean_residual=sum(p["sac"]["failed"] and p["sac"]["residual_norm"]==0 for p in group)))
        for controller, key in (("pid_only", "pid"), ("pid_sac", "sac")):
            rows = [p[key] for p in group]
            effort.append(dict(scenario=scenario, wind_profile=profile, controller=controller, flights=len(rows),
                               mean_residual_norm=float(np.mean([r["residual_norm"] for r in rows])),
                               mean_residual_change=float(np.mean([r["residual_change"] for r in rows])),
                               mean_saturation_rate=float(np.mean([r["saturation_rate"] for r in rows])),
                               mean_clearance_violation_rate=float(np.mean([r["clearance_violation_rate"] for r in rows])) if rows[0]["clearance_violation_rate"] is not None else None,
                               interpretation="descriptive_effort_not_energy_or_vector_alignment"))
    return lengths, effort
