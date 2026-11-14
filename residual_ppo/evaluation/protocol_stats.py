"""Protocol statistics (docs/milestone2_plan.md, section E and M2.3). NumPy only, no Isaac.

Per-cell summaries (team summarize() + median/IQR + failure reasons), README-style per-profile aggregation, the M2.3
team README section 2 reproduction check, the M2.3 consistency check, the still-vs-wind table (paired per-seed
differences with seed bootstrap) and the pre-registered H1 analysis (wind_share_MSE, decision rule, failure sensitivity,
secondary shares, hover sanity check). Bootstrap: 10,000 seed resamples, percentile 95% CI, np.random.default_rng(0)
started afresh for each statistic.
"""
from collections import Counter

import numpy as np

from path_sac_policy import summarize

# Team README section 2 (61d2722), open scene, PID only, 10 seeds x 6 route types.
TEAM_README_PID = {
    "training_range": {"position_rmse": 0.502, "max_deviation": 1.77, "completed_pct": 5.0, "failed_pct": 0.0},
    "unseen_gusts": {"position_rmse": 0.497, "max_deviation": 1.51, "completed_pct": 5.0, "failed_pct": 0.0},
    "outside_range": {"position_rmse": 1.653, "max_deviation": 3.11, "completed_pct": 0.0, "failed_pct": 62.0},
}
# Pre-registered pass range (plan, M2.3 first check, (a)).
RMSE_REL_TOL = 0.05
COUNT_TOL = 3
FLIGHTS_PER_PROFILE = 60
_EPS = 1e-12


def group(flights, keys):
    groups = {}
    for f in flights:
        groups.setdefault(tuple(f[k] for k in keys), []).append(f)
    return groups


def _median_iqr(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": None, "q25": None, "q75": None}
    q25, med, q75 = np.percentile(v, [25, 50, 75])
    return {"median": float(med), "q25": float(q25), "q75": float(q75)}


def cell_summary(rows):
    """Team summarize() (means over all flights, failed included) plus n, counts, median/IQR, failure reasons."""
    out = dict(summarize(rows))
    out["failed_flights"] = int(sum(bool(r["failed"]) for r in rows))
    out["completed_flights"] = int(sum(bool(r["completed"]) for r in rows))
    out["failure_reasons"] = dict(Counter(r["failure_reason"] for r in rows if r["failure_reason"] is not None))
    for name in ("position_rmse", "cross_track_rmse"):
        stats = _median_iqr([r[name] for r in rows])
        out.update({f"{name}_{k}": v for k, v in stats.items()})
    return out


def summaries(flights):
    """One row per (scenario, wind_profile, controller), in first-seen order."""
    return [{"scenario": s, "wind_profile": p, "controller": c, **cell_summary(rows)}
            for (s, p, c), rows in group(flights, ("scenario", "wind_profile", "controller")).items()]


def profile_aggregate(flights, profile, scenarios):
    """README section 2 style: per profile, the mean over scenarios of per-scenario summarize() values."""
    per = {}
    for scenario in scenarios:
        rows = [f for f in flights if f["wind_profile"] == profile and f["scenario"] == scenario]
        if rows:
            per[scenario] = cell_summary(rows)
    rows_all = [f for f in flights if f["wind_profile"] == profile and f["scenario"] in per]

    def mean_of(name):
        return float(np.mean([per[s][name] for s in per])) if per else float("nan")

    return {
        "profile": profile,
        "scenarios": list(per),
        "flights": len(rows_all),
        "position_rmse": mean_of("position_rmse"),
        "cross_track_rmse": mean_of("cross_track_rmse"),
        "completion_rate": mean_of("completion_rate"),
        "failure_rate": mean_of("failure_rate"),
        "completed_flights": int(sum(bool(r["completed"]) for r in rows_all)),
        "failed_flights": int(sum(bool(r["failed"]) for r in rows_all)),
        "failure_reasons": dict(Counter(r["failure_reason"] for r in rows_all if r["failure_reason"] is not None)),
        "max_deviation_mean_of_scenario_max": mean_of("max_deviation"),
        "max_deviation_overall_max": float(np.nanmax([r["max_deviation"] for r in rows_all])) if rows_all
        else float("nan"),
    }


def reproduction_check(flights, scenarios, readme=TEAM_README_PID):
    """Compare our PID-only flights with README section 2 rows (pre-registered pass range (a))."""
    out = {"pass_range": {"position_rmse_relative": RMSE_REL_TOL, "completed_and_failed_flights": COUNT_TOL,
                          "flights_per_profile": FLIGHTS_PER_PROFILE,
                          "max_deviation": "descriptive only (both aggregations)"},
           "profiles": {}}
    for profile, ref in readme.items():
        agg = profile_aggregate(flights, profile, scenarios)
        n = agg["flights"]
        complete = n == FLIGHTS_PER_PROFILE and len(agg["scenarios"]) == len(scenarios)
        rel = (agg["position_rmse"] - ref["position_rmse"]) / ref["position_rmse"]
        exp_completed = ref["completed_pct"] / 100.0 * FLIGHTS_PER_PROFILE
        exp_failed = ref["failed_pct"] / 100.0 * FLIGHTS_PER_PROFILE
        checks = {
            "complete": complete,
            "position_rmse": bool(abs(rel) <= RMSE_REL_TOL + _EPS),
            "completed_flights": bool(abs(agg["completed_flights"] - exp_completed) <= COUNT_TOL + _EPS),
            "failed_flights": bool(abs(agg["failed_flights"] - exp_failed) <= COUNT_TOL + _EPS),
        }
        out["profiles"][profile] = {
            "readme": ref,
            "ours": agg,
            "position_rmse_diff_m": agg["position_rmse"] - ref["position_rmse"],
            "position_rmse_rel_diff": rel,
            "completed_flights_expected": exp_completed,
            "failed_flights_expected": exp_failed,
            "max_deviation_diff_mean_of_scenario_max": agg["max_deviation_mean_of_scenario_max"] - ref["max_deviation"],
            "max_deviation_diff_overall_max": agg["max_deviation_overall_max"] - ref["max_deviation"],
            "checks": checks,
            "pass": all(checks.values()),
        }
    out["pass"] = all(p["pass"] for p in out["profiles"].values())
    return out


# ---------------------------------------------------------------- M2.3: consistency, still vs wind, H1 (pre-registered)

BOOT_RESAMPLES = 10_000
BOOT_SEED = 0          # each statistic's bootstrap starts its own np.random.default_rng(0)
H1_THRESHOLD = 0.5
HOVER_SANITY_MIN = 0.8
DIAGNOSTICS = ("accel_cap_binding_frac", "tilt_cap_binding_frac", "along_track_lag_mean", "along_track_lag_rms",
               "cross_track_component_rms")
PAIRED_METRICS = ("position_rmse", "cross_track_rmse", "max_deviation")


def index_flights(flights):
    out = {}
    for f in flights:
        key = (f["scenario"], f["wind_profile"], int(f["seed"]))
        if key in out:
            raise ValueError(f"duplicate flight {key}")
        out[key] = f
    return out


def consistency_check(flights, scenarios, profiles, seeds, still="still_air"):
    """M2.3 check: full grid; per (scenario, seed) identical route_seconds, max_decisions, initial observation across
    profiles; still-air flights have zero wind."""
    idx = index_flights(flights)
    problems = []
    missing = [(s, p, seed) for s in scenarios for p in profiles for seed in seeds if (s, p, seed) not in idx]
    if missing:
        problems.append({"missing_flights": missing})
    for s in scenarios:
        for seed in seeds:
            rows = {p: idx[(s, p, seed)] for p in profiles if (s, p, seed) in idx}
            for field in ("route_seconds", "max_decisions", "initial_obs_sha256"):
                values = {p: r[field] for p, r in rows.items()}
                if len(set(values.values())) > 1:
                    problems.append({"scenario": s, "seed": seed, "field": field, "values": values})
            if still in rows:
                r = rows[still]
                if r["wind_speed"] != 0.0 or any(v != 0.0 for v in r.get("wind_mean", [0.0])):
                    problems.append({"scenario": s, "seed": seed, "field": "still_air_wind",
                                     "wind_speed": r["wind_speed"], "wind_mean": r.get("wind_mean")})
    return {"pass": not problems, "flights": len(flights),
            "expected_flights": len(scenarios) * len(profiles) * len(seeds),
            "groups_checked": len(scenarios) * len(seeds),
            "fields": ["route_seconds", "max_decisions", "initial_obs_sha256", "still_air zero wind"],
            "problems": problems}


def _boot_idx(n, resamples=BOOT_RESAMPLES, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def bootstrap_ratio_of_sums(num, den):
    """Point = sum(num)/sum(den); 95% percentile CI resampling seeds (paired: same indices for num and den)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    if num.size == 0:
        return {"value": None, "ci95": [None, None], "n_seeds": 0}
    idx = _boot_idx(num.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return {"value": float(num.sum() / den.sum()), "ci95": [float(lo), float(hi)], "n_seeds": int(num.size),
            "resamples": BOOT_RESAMPLES, "rng_seed": BOOT_SEED}


def bootstrap_mean(diffs):
    """Mean of per-seed values; 95% percentile CI resampling seeds."""
    d = np.asarray(diffs, float)
    if d.size == 0:
        return {"value": None, "ci95": [None, None], "n_seeds": 0}
    vals = d[_boot_idx(d.size)].mean(axis=1)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"value": float(d.mean()), "ci95": [float(lo), float(hi)], "n_seeds": int(d.size),
            "resamples": BOOT_RESAMPLES, "rng_seed": BOOT_SEED}


def h1_verdict(ci):
    lo, hi = ci
    if lo is None:
        return "not computable"
    if hi < H1_THRESHOLD:
        return "supported"
    if lo > H1_THRESHOLD:
        return "rejected"
    return "inconclusive"


def per_seed_mean(idx, profile, seeds, scenarios, value):
    """M_p(s): per seed, mean over the scenarios of value(flight)."""
    return np.array([np.mean([value(idx[(sc, profile, s)]) for sc in scenarios]) for s in seeds], dtype=float)


def _mse(f):
    return float(f["position_rmse"]) ** 2


def _rmse(f):
    return float(f["position_rmse"])


def wind_share(idx, seeds, scenarios, value, still, wind):
    tr = per_seed_mean(idx, wind, seeds, scenarios, value)
    st = per_seed_mean(idx, still, seeds, scenarios, value)
    out = bootstrap_ratio_of_sums(tr - st, tr)
    out.update({"mean_wind": float(tr.mean()) if tr.size else None,
                "mean_still": float(st.mean()) if st.size else None})
    return out


def h1_analysis(flights, scenarios, seeds, still="still_air", wind="training_range"):
    """Pre-registered H1 test (plan, block above M2.3). Uses all flights; failed flights keep MSE over what was flown."""
    idx = index_flights(flights)
    seeds = list(seeds)
    primary = wind_share(idx, seeds, scenarios, _mse, still, wind)
    primary["verdict"] = h1_verdict(primary["ci95"])

    failed = [{"scenario": sc, "seed": s, "profile": p, "failure_reason": idx[(sc, p, s)]["failure_reason"],
               "decisions": idx[(sc, p, s)]["decisions"]}
              for p in (still, wind) for sc in scenarios for s in seeds if idx[(sc, p, s)]["failed"]]
    bad_seeds = sorted({f["seed"] for f in failed})
    keep = [s for s in seeds if s not in bad_seeds]
    if not failed:
        sensitivity = dict(primary, note="no failed flight in either profile: sensitivity set = primary set")
    elif keep:
        sensitivity = wind_share(idx, keep, scenarios, _mse, still, wind)
        sensitivity["verdict"] = h1_verdict(sensitivity["ci95"])
    else:
        sensitivity = {"value": None, "ci95": [None, None], "n_seeds": 0, "verdict": "not computable"}
    sensitivity["excluded_seeds"] = bad_seeds

    rmse_tr = per_seed_mean(idx, wind, seeds, scenarios, _rmse)
    rmse_st = per_seed_mean(idx, still, seeds, scenarios, _rmse)
    per_scenario = {}
    for sc in scenarios:
        per_scenario[sc] = wind_share(idx, seeds, (sc,), _mse, still, wind)

    hover = per_scenario.get("hover")
    hover_check = None
    if hover is not None:
        hover_check = {"still_air_mse_mean": hover["mean_still"], "training_range_mse_mean": hover["mean_wind"],
                       "mse_wind_share": hover["value"], "ci95": hover["ci95"], "threshold": HOVER_SANITY_MIN,
                       "suspect": bool(hover["value"] is None or hover["value"] < HOVER_SANITY_MIN)}
    interpretable = not (hover_check and hover_check["suspect"])
    return {
        "hypothesis": "H1: PID-only MSE in still air is close to training_range MSE (wind share < 0.5)",
        "profiles": {"still": still, "wind": wind}, "seeds": [seeds[0], seeds[-1], len(seeds)],
        "primary_wind_share_mse": primary,
        "decision_rule": "supported if upper 95% CI < 0.5; rejected if lower > 0.5; else inconclusive",
        "hover_sanity": hover_check,
        "h1_interpretable": interpretable,
        "verdict": primary["verdict"] if interpretable else "not interpreted (hover sanity check flagged)",
        "failed_flights": failed,
        "sensitivity_excluding_failed_seeds": sensitivity,
        "primary_and_sensitivity_agree": primary["verdict"] == sensitivity["verdict"],
        "secondary": {
            "wind_share_rmse": wind_share(idx, seeds, scenarios, _rmse, still, wind),
            "per_scenario_wind_share_mse": per_scenario,
            "paired_rmse_difference_m": bootstrap_mean(rmse_tr - rmse_st),
        },
    }


def still_vs_wind_table(flights, scenarios, profiles, seeds, still="still_air"):
    """Per scenario x profile: summaries (failed included) + diagnostics means; for wind profiles, paired per-seed
    differences wind - still_air with seed-bootstrap CI."""
    idx = index_flights(flights)
    rows = []
    for sc in scenarios:
        for p in profiles:
            cell = [idx[(sc, p, s)] for s in seeds]
            row = {"scenario": sc, "wind_profile": p, **cell_summary(cell)}
            for d in DIAGNOSTICS:
                vals = [f.get(d) for f in cell if f.get(d) is not None]
                row[f"{d}_mean"] = float(np.mean(vals)) if vals else None
                row[f"{d}_n_flights"] = len(vals)
            if p != still:
                for m in PAIRED_METRICS:
                    diff = np.array([idx[(sc, p, s)][m] - idx[(sc, still, s)][m] for s in seeds], float)
                    b = bootstrap_mean(diff)
                    row[f"{m}_diff_vs_still_mean"] = b["value"]
                    row[f"{m}_diff_vs_still_ci95_lo"] = b["ci95"][0]
                    row[f"{m}_diff_vs_still_ci95_hi"] = b["ci95"][1]
            rows.append(row)
    return rows


def reproduction_text(rep):
    lines = ["Team README section 2 reproduction (PID only, open scene, seeds 300000-300009, mean over 6 scenarios)",
             f"pass range: RMSE within +-{RMSE_REL_TOL:.0%} relative; completed and failed within +-{COUNT_TOL} of "
             f"{FLIGHTS_PER_PROFILE} flights; max deviation descriptive",
             f"{'profile':<16}{'RMSE ours':>10}{'README':>8}{'rel':>8}  {'compl ours/exp':>15}  {'fail ours/exp':>14}"
             f"  {'maxdev meanmax/overall':>23}{'README':>7}  verdict"]
    for profile, p in rep["profiles"].items():
        o = p["ours"]
        lines.append(
            f"{profile:<16}{o['position_rmse']:>10.3f}{p['readme']['position_rmse']:>8.3f}"
            f"{p['position_rmse_rel_diff']:>+8.1%}  {o['completed_flights']:>6}/{p['completed_flights_expected']:<8.1f}"
            f"  {o['failed_flights']:>5}/{p['failed_flights_expected']:<8.1f}"
            f"  {o['max_deviation_mean_of_scenario_max']:>11.2f}/{o['max_deviation_overall_max']:<11.2f}"
            f"{p['readme']['max_deviation']:>7.2f}  {'PASS' if p['pass'] else 'OUTSIDE RANGE'}"
            + ("" if p["pass"] else f" ({', '.join(k for k, v in p['checks'].items() if not v)})")
            + (f"; failures {o['failure_reasons']}" if o["failure_reasons"] else ""))
    lines.append(f"overall: {'PASS' if rep['pass'] else 'OUTSIDE RANGE -> investigate per plan (b) before M2.4'}")
    return "\n".join(lines)
