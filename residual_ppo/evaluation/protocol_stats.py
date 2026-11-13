"""Protocol statistics (docs/milestone2_plan.md, section E and M2.3). NumPy only, no Isaac.

Implemented now: per-cell summaries (team summarize() + median/IQR + failure reasons), README-style per-profile
aggregation, and the M2.3 team README section 2 reproduction check (pass range pre-registered in the plan).
Bootstrap / paired differences / H1 wind share come later (after the M2.3 consistency check).
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
