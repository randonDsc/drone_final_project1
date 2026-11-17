"""Compare validated, matched original/adapted wind-event evaluations.

Usage: python -B drone_code/compare_wind_checkpoints.py COMPARISON_FOLDER
Expects hover/straight/waypoints_{original,adapted} evaluation folders.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from summarize_wind_events import analyze


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(root):
    pairs = []
    hashes = {}
    for route in ("hover", "straight", "waypoints"):
        folders = [root / f"{route}_{label}" for label in ("original", "adapted")]
        for folder in folders:
            analyze(folder)
        manifests = [read_json(folder / "manifest.json") for folder in folders]
        for key in ("seeds", "scenarios", "duration_s", "wind_profile", "mean_wind_scenarios", "route", "contract", "sampling_hz"):
            if manifests[0][key] != manifests[1][key]:
                raise ValueError(f"Unmatched {route} manifest: {key}")
        source_snapshots = [{p.name: p.read_bytes() for p in (folder / "source").glob("*.py")} for folder in folders]
        if not source_snapshots[0] or source_snapshots[0] != source_snapshots[1]:
            raise ValueError("Evaluation source snapshots differ or are missing")
        for label, manifest in zip(("original", "adapted"), manifests):
            previous = hashes.setdefault(label, manifest["model_sha256"])
            if previous != manifest["model_sha256"]:
                raise ValueError("Checkpoint changed between routes")
        for seed in manifests[0]["seeds"]:
            if read_json(folders[0] / f"reference_{seed}.json") != read_json(folders[1] / f"reference_{seed}.json"):
                raise ValueError("Reference trajectories differ")
        flights = [{(f["scenario"], f["seed"], f["controller"]): f for f in read_json(folder / "flights.json")} for folder in folders]
        if flights[0].keys() != flights[1].keys():
            raise ValueError("Flight grids differ")
        for (scenario, seed, mode), original in flights[0].items():
            adapted = flights[1][(scenario, seed, mode)]
            filename = f"{scenario}_{seed}_{mode}.csv"
            traces = []
            for folder in folders:
                with (folder / filename).open(encoding="utf-8") as handle:
                    traces.append(list(csv.DictReader(handle)))
            if mode == "pid_only":
                if original != adapted or traces[0] != traces[1]:
                    raise ValueError(f"Repeated PID baseline differs: {route}/{scenario}/{seed}")
                continue
            for key in ("wind_seed", "heading_rad", "event_s", "route", "reference_speed_at_event_m_s", "route_duration_s"):
                if original[key] != adapted[key]:
                    raise ValueError(f"Unmatched flight: {key}")
            for a, b in zip(*traces):
                for key in ("time_s", "target_x_m", "target_y_m", "target_z_m", "mean_wind_x_m_s", "mean_wind_y_m_s", "mean_wind_z_m_s"):
                    if a[key] != b[key]:
                        raise ValueError(f"Unmatched trajectory input: {key}")
            row = dict(route=route, scenario=scenario, seed=seed)
            for key in ("position_rmse_m", "post_event_peak_m", "failed", "recovered", "recovery_time_s"):
                row[f"original_{key}"] = original[key]
                row[f"adapted_{key}"] = adapted[key]
            row["adapted_minus_original_rmse_m"] = (adapted["position_rmse_m"] - original["position_rmse_m"] if original["position_rmse_m"] is not None and adapted["position_rmse_m"] is not None else None)
            pairs.append(row)
    if hashes["original"] == hashes["adapted"]:
        raise ValueError("Both evaluations used the same checkpoint")
    with (root / "checkpoint_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    groups = []
    rng = np.random.default_rng(20261116)
    for route in ("all", "hover", "straight", "waypoints"):
        for scenario in ("all", "control", "turn90", "turn180", "speed_up", "speed_down"):
            group = [p for p in pairs if (route == "all" or p["route"] == route) and (scenario == "all" or p["scenario"] == scenario)]
            valid = [p for p in group if p["adapted_minus_original_rmse_m"] is not None and not p["original_failed"] and not p["adapted_failed"]]
            result = dict(route=route, scenario=scenario, pairs=len(group), successful_pairs=len(valid), original_failures=sum(p["original_failed"] for p in group), adapted_failures=sum(p["adapted_failed"] for p in group))
            if valid:
                # Resample seed clusters: conditions using a shared seed are not independent.
                seeds = sorted({p["seed"] for p in valid})
                clusters = np.array([np.mean([p["adapted_minus_original_rmse_m"] for p in valid if p["seed"] == seed]) for seed in seeds])
                boot = rng.choice(clusters, size=(50000, len(seeds)), replace=True).mean(axis=1)
                result.update(original_rmse_m=float(np.mean([p["original_position_rmse_m"] for p in valid])), adapted_rmse_m=float(np.mean([p["adapted_position_rmse_m"] for p in valid])), mean_delta_m=float(clusters.mean()), seed_cluster_ci95_m=np.quantile(boot, [.025, .975]).tolist(), seed_count=len(seeds), adapted_lower_count=sum(p["adapted_minus_original_rmse_m"] < 0 for p in valid))
            groups.append(result)
    report = dict(model_sha256=hashes, pairing="Same seeds, references, mean wind, timing, contracts; repeated PID traces match exactly", limitations="Five environment seeds, one checkpoint per model; descriptive bootstrap intervals, no multiple-comparison correction. Fixed 20-second segments; control is steady mean wind, not full original training distribution. Failed pairs excluded from RMSE and counted separately.", groups=groups)
    (root / "checkpoint_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, route in zip(axes, ("hover", "straight", "waypoints")):
        subset = [g for g in groups if g["route"] == route and g["scenario"] != "all"]
        x = np.arange(len(subset))
        ax.bar(x - .18, [g.get("original_rmse_m", np.nan) for g in subset], .36, label="Original SAC")
        ax.bar(x + .18, [g.get("adapted_rmse_m", np.nan) for g in subset], .36, label="Adapted SAC")
        ax.set_xticks(x, [g["scenario"] for g in subset], rotation=35)
        ax.set_title(route)
        ax.set_ylabel("Mean flight RMSE (m), successful pairs")
        ax.legend()
    fig.suptitle("Matched checkpoint comparison — five environment seeds per condition")
    fig.tight_layout()
    fig.savefig(root / "checkpoint_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(groups[0], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    compare(parser.parse_args().folder)
