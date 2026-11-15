"""Independent numerical audit of stored tables against original flight records.

This module does not import the production statistics or archive loaders. Exact
tests use enumerated sign assignments and binomial coefficients; BCa intervals
use an explicit jackknife calculation rather than scipy.stats.bootstrap.
Only machine-readable audit data are written.
"""
import argparse
import csv
from decimal import Decimal
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from scipy.special import betaincinv, ndtr, ndtri


def read_csv(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def flag(value):
    if value not in ("True", "False"):
        raise ValueError("Invalid source flag: "+str(value))
    return value == "True"


def exact_binomial_tail(k, n):
    return min(1., 2*sum(math.comb(n, j) for j in range(min(k, n-k)+1))/2**n) if n else 1.


def signed_rank(difference):
    values = [float(v) for v in np.round(difference, 12) if abs(v)>1e-12]
    if not values:
        return 0., 1.
    absolute = sorted(abs(v) for v in values)
    ranks = [(absolute.index(abs(v))+1+len(absolute)-absolute[::-1].index(abs(v)))/2 for v in values]
    observed = sum(r for r, v in zip(ranks, values) if v>0)
    total = sum(ranks)
    distribution = [sum(r*b for r, b in zip(ranks, bits))
                    for bits in itertools.product((0, 1), repeat=len(values))]
    p = min(1., 2*min(sum(t<=observed for t in distribution),
                       sum(t>=observed for t in distribution))/len(distribution))
    return min(observed, total-observed), p


def bca_interval(difference, seed, resamples):
    n = len(difference)
    if n<2:
        return None, None
    mean = float(np.mean(difference))
    if np.ptp(difference)<=1e-12:
        return mean, mean
    rng = np.random.default_rng(seed)
    samples = np.concatenate([np.mean(difference[rng.integers(n, size=(min(2000, resamples-i), n))], axis=1)
                              for i in range(0, resamples, 2000)])
    bias = ndtri((np.sum(samples<mean)+.5*np.sum(samples==mean))/resamples)
    jackknife = np.array([np.mean(np.delete(difference, i)) for i in range(n)])
    centered = np.mean(jackknife)-jackknife
    acceleration = np.sum(centered**3)/(6*np.sum(centered**2)**1.5)
    z = ndtri(np.array([.025, .975]))
    levels = ndtr(bias+(bias+z)/(1-acceleration*(bias+z)))
    return tuple(float(v) for v in np.quantile(samples, levels))


def holm(pvalues):
    ordered = sorted(range(len(pvalues)), key=lambda i: 1. if pvalues[i] is None else pvalues[i])
    adjusted, running = [None]*len(pvalues), 0.
    for rank, i in enumerate(ordered):
        running = min(1., max(running, (len(pvalues)-rank)*(1. if pvalues[i] is None else pvalues[i])))
        if pvalues[i] is not None:
            adjusted[i] = running
    return adjusted


def audit(output, archive):
    output, archive = Path(output), Path(archive)
    with zipfile.ZipFile(archive) as zipped:
        raw = zipped.read("runs/flight_sac/flight1/evaluation/final_model/flights.csv")
        if raw != (output/"source_data/flights.csv").read_bytes():
            raise ValueError("Stored CSV differs from the original archive")
        model_hash = hashlib.sha256(zipped.read("runs/flight_sac/flight1/final_model.zip")).hexdigest()
    metadata = json.loads((output/"source_data/dataset_metadata.json").read_text())
    if model_hash != metadata["selected_model"]["model_sha256"]:
        raise ValueError("Stored checkpoint identity differs from the archive")
    for name, entry in metadata["source_files"].items():
        if hashlib.sha256((output/"source_data"/name).read_bytes()).hexdigest()!=entry["sha256"]:
            raise ValueError("Source file hash mismatch: "+name)
    rows = read_csv(raw)
    index = {}
    for row in rows:
        key = row["scenario"], row["wind_profile"], int(row["seed"])
        group = index.setdefault(key, {})
        if row["controller"] in group:
            raise ValueError("Duplicate source flight")
        group[row["controller"]] = row
    for members in index.values():
        if set(members)!={"pid_only", "pid_sac"}:
            raise ValueError("Incomplete pair")
        a, b = members["pid_only"], members["pid_sac"]
        for field in ("wind_seed", "wind_speed", "route_seconds"):
            if a[field]!=b[field]:
                raise ValueError("Paired source fields differ")
    tables = {p.name:read_csv(p.read_bytes()) for p in sorted((output/"tables").glob("*.csv"))}
    checks = 0

    def check(actual, expected):
        nonlocal checks
        checks += 1
        if expected is None:
            valid = actual == ""
        elif isinstance(expected, bool):
            valid = actual == str(expected)
        elif isinstance(expected, int):
            valid = Decimal(actual)==expected
        elif isinstance(expected, (float, np.floating)):
            valid = math.isfinite(float(actual)) and math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-10)
        else:
            valid = actual == expected
        if not valid:
            raise ValueError(f"Independent audit mismatch: actual={actual!r}, expected={expected!r}")

    def group_for(row):
        return [members for key, members in sorted(index.items()) if key[:2]==(row["scenario"], row["wind_profile"])]

    def continuous(table, vectors):
        raw_wilcoxon, raw_sign = [], []
        for row, (pid, sac) in zip(table, vectors):
            pid, sac = np.array(pid, float), np.array(sac, float)
            difference = sac-pid
            n = len(difference)
            check(row["n_pairs"], n)
            wins, losses = int(np.sum(difference < -1e-12)), int(np.sum(difference>1e-12))
            check(row["sac_lower"], wins); check(row["pid_lower"], losses)
            check(row["ties"], n-wins-losses)
            values = dict(pid_mean=float(pid.mean()) if n else None, sac_mean=float(sac.mean()) if n else None,
                          mean_difference=float(difference.mean()) if n else None,
                          median_difference=float(np.median(difference)) if n else None,
                          difference_sd=float(np.std(difference, ddof=1)) if n>1 else None,
                          relative_mean_reduction_percent=float(100*(pid.mean()-sac.mean())/pid.mean()) if n and pid.mean()>0 else None)
            w, wp = signed_rank(difference) if n else (None, None)
            sp = exact_binomial_tail(wins, wins+losses) if n else None
            values.update(wilcoxon_statistic=w, wilcoxon_p=wp, sign_test_p=sp)
            low, high = bca_interval(difference, int(row["bootstrap_seed"]), 50000)
            values.update(mean_difference_ci_low=low, mean_difference_ci_high=high)
            for field, expected in values.items():
                check(row[field], expected)
            raw_wilcoxon.append(wp); raw_sign.append(sp)
        for field, pvalues in (("wilcoxon", raw_wilcoxon), ("sign_test", raw_sign)):
            for row, corrected in zip(table, holm(pvalues)):
                check(row[field+"_p_holm"], corrected)
                check(row[field+"_reject_holm"], corrected is not None and corrected<=.05)

    for filename in ("primary_rmse.csv", "secondary_metrics.csv", "sensitivity_rmse.csv"):
        vectors = []
        for row in tables[filename]:
            group = group_for(row)
            if filename=="sensitivity_rmse.csv":
                group = [g for g in group if not flag(g["pid_only"]["failed"]) and not flag(g["pid_sac"]["failed"])
                         and g["pid_only"]["decisions"]==g["pid_sac"]["decisions"]]
                check(row["excluded_pairs"], 10-len(group))
            vectors.append(tuple([float(g[c][row["metric"]]) for g in group] for c in ("pid_only", "pid_sac")))
        continuous(tables[filename], vectors)

    seed_values = {}
    for row in tables["seed_cluster_values.csv"]:
        name, seed = row["benchmark"], int(row["seed"])
        group = [members for key, members in sorted(index.items()) if key[2]==seed
                 and ((key[0]=="dijkstra") if name=="dijkstra_all_profiles" else key[0]!="dijkstra")
                 and (not name.startswith("open_air_") or name=="open_air_all_profiles" or key[1]==name.removeprefix("open_air_"))]
        pid, sac = [float(np.mean([float(g[c]["position_rmse"]) for g in group])) for c in ("pid_only", "pid_sac")]
        check(row["cases"], len(group)); check(row["pid_mean_flight_rmse"], pid)
        check(row["sac_mean_flight_rmse"], sac); check(row["difference"], sac-pid)
        seed_values.setdefault(name, []).append((seed, pid, sac))
    continuous(tables["seed_cluster_overview.csv"],
               [tuple([t[i] for t in sorted(seed_values[r["benchmark"]])] for i in (1, 2))
                for r in tables["seed_cluster_overview.csv"]])

    for outcome in ("completed", "failed", "collided"):
        family = [r for r in tables["paired_outcomes.csv"] if r["outcome"]==outcome]
        pvalues = []
        for row in family:
            bits = [(flag(g["pid_only"][outcome]), flag(g["pid_sac"][outcome])) for g in group_for(row)]
            both, pid_only, sac_only, neither = [bits.count(pair) for pair in ((True, True), (True, False), (False, True), (False, False))]
            pid, sac, n = both+pid_only, both+sac_only, len(bits)
            p = exact_binomial_tail(sac_only, pid_only+sac_only)
            values = dict(n_pairs=n, both_true=both, pid_only_true=pid_only, sac_only_true=sac_only, both_false=neither,
                          discordant_pairs=pid_only+sac_only, pid_true=pid, sac_true=sac,
                          pid_rate=pid/n, sac_rate=sac/n, rate_difference=(sac-pid)/n, mcnemar_exact_p=p)
            for controller, count in (("pid", pid), ("sac", sac)):
                values[controller+"_rate_exact_ci_low"] = float(betaincinv(count, n-count+1, .025)) if count else 0.
                values[controller+"_rate_exact_ci_high"] = float(betaincinv(count+1, n-count, .975)) if count<n else 1.
            for field, expected in values.items():
                check(row[field], expected)
            pvalues.append(p)
        for row, p in zip(family, holm(pvalues)):
            check(row["mcnemar_p_holm"], p); check(row["mcnemar_reject_holm"], p<=.05)

    for row in tables["paired_flights.csv"]:
        key = row["scenario"], row["wind_profile"], int(row["seed"])
        group = index[key]
        for prefix, controller in (("pid", "pid_only"), ("sac", "pid_sac")):
            for field, actual in row.items():
                if field.startswith(prefix+"_"):
                    check(actual, group[controller][field[len(prefix)+1:]])
        for metric in ("position_rmse", "cross_track_rmse", "max_deviation", "cross_track_max", "saturation_rate"):
            check(row[metric+"_difference"], float(group["pid_sac"][metric])-float(group["pid_only"][metric]))
        for field in ("wind_seed", "wind_speed", "route_seconds"):
            check(row[field], group["pid_only"][field])
        same = group["pid_only"]["decisions"]==group["pid_sac"]["decisions"]
        failed = any(flag(g["failed"]) for g in group.values())
        check(row["same_recorded_decision_count"], same); check(row["either_failed"], failed)
        check(row["sensitivity_eligible"], same and not failed)
    for row in tables["observed_lengths.csv"]:
        group = group_for(row)
        values = dict(pairs=len(group),
                      different_decision_counts=sum(g["pid_only"]["decisions"]!=g["pid_sac"]["decisions"] for g in group),
                      either_failed=sum(any(flag(v["failed"]) for v in g.values()) for g in group),
                      both_failed=sum(all(flag(v["failed"]) for v in g.values()) for g in group),
                      both_nonfailed_equal_counts=sum(not any(flag(v["failed"]) for v in g.values()) and g["pid_only"]["decisions"]==g["pid_sac"]["decisions"] for g in group),
                      sac_failed_zero_mean_residual=sum(flag(g["pid_sac"]["failed"]) and float(g["pid_sac"]["residual_norm"])==0 for g in group))
        for field, expected in values.items():
            check(row[field], expected)
    for row in tables["control_effort_descriptive.csv"]:
        group = [g[row["controller"]] for g in group_for(row)]
        check(row["flights"], len(group))
        for field in ("residual_norm", "residual_change", "saturation_rate", "clearance_violation_rate"):
            expected = float(np.mean([float(g[field]) for g in group])) if group[0][field] else None
            check(row["mean_"+field], expected)
    result = dict(status="passed", source_csv_matches_original_archive=True,
                  original_checkpoint_hash_matches=True, numeric_field_checks=checks,
                  tables_checked={name:len(table) for name, table in tables.items()},
                  exact_tests="independent sign-assignment enumeration and binomial coefficients",
                  confidence_intervals="independent explicit BCa calculation and beta quantiles",
                  production_statistics_imported=False)
    (output/"numerical_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path, default=Path(__file__).parent/"inputs/runs.zip")
    args = parser.parse_args()
    result = audit(args.output, args.archive)
    print(f"Independent audit passed: {result['numeric_field_checks']} field checks across {len(result['tables_checked'])} tables.")


if __name__ == "__main__":
    main()
