"""Validate task-2 CSVs and produce descriptive paired results (no simulator).

Usage: python drone_code/summarize_wind_events.py RUN_DIRECTORY [--plot]
Three evaluation seeds are a pilot, not evidence of statistical significance.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from wind_events import WindEvent, event_metrics


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(folder, plot=False):
    manifest = json.loads((folder / "manifest.json").read_text())
    if not manifest["completed"]:
        raise ValueError("Run is incomplete; do not summarize partial flights as a full evaluation")
    flights = json.loads((folder / "flights.json").read_text())
    modes = ["pid_only"] if manifest["model"] is None else ["pid_only", "pid_sac"]
    expected = set(itertools.product(manifest["scenarios"], manifest["seeds"], modes))
    by_key = {(r["scenario"], r["seed"], r["controller"]): r for r in flights}
    if len(by_key) != len(flights) or set(by_key) != expected:
        raise ValueError("Duplicate or missing scenario/seed/controller flight")
    traces = {}
    for key, row in by_key.items():
        scenario, seed, mode = key
        with (folder / f"{scenario}_{seed}_{mode}.csv").open(newline="") as f:
            samples = list(csv.DictReader(f))
        t = np.array([float(s["time_s"]) for s in samples])
        e = np.array([float(s["error_m"]) if s["error_m"] else np.nan for s in samples])
        if not row["failed"]:
            count = round(manifest["duration_s"] * manifest["sampling_hz"])
            np.testing.assert_allclose(t, np.arange(1, count + 1) / manifest["sampling_hz"], atol=1e-7)
        event_s = row.get("event_s", manifest["event_s"])
        if manifest.get("route", "hover") != "hover" and row["reference_speed_at_event_m_s"] <= .05:
            raise ValueError("Moving-route wind event was scheduled while stationary")
        if "target_x_m" in samples[0] and not row["invalid_samples"]:
            target = np.array([[float(s[f"target_{a}_m"]) for a in "xyz"] for s in samples])
            position = np.array([[float(s[f"position_{a}_m"]) for a in "xyz"] for s in samples])
            np.testing.assert_allclose(np.linalg.norm(target-position, axis=1), e, atol=1e-9)
        event = WindEvent(scenario, row["heading_rad"], event_s)
        # The CSV records the mean applied during the preceding physics step.
        physics_dt = 1 / manifest["contract"]["physics_rate_hz"]
        actual = np.array([[float(s[f"mean_wind_{a}_m_s"]) for a in "xyz"] for s in samples])
        predicted = np.array([event.at(time - physics_dt / 2) for time in t])
        np.testing.assert_allclose(actual, predicted, atol=1e-8)
        recomputed = event_metrics(t, e, event_s, failed=row["failure_reason"] is not None)
        for name, value in recomputed.items():
            recorded = row[name]
            if value is None or isinstance(value, bool):
                if recorded != value:
                    raise ValueError(f"Metric mismatch: {key} / {name}")
            else:
                np.testing.assert_allclose(recorded, value, atol=1e-9)
        moving = np.array([float(s.get("reference_speed_m_s", 0)) > .05 for s in samples])
        row["moving_rmse_m"] = float(np.sqrt(np.mean(e[moving]**2))) if moving.any() and np.isfinite(e[moving]).all() else None
        moving_post = moving & (t > event_s + 1e-9)
        row["post_event_moving_rmse_m"] = float(np.sqrt(np.mean(e[moving_post]**2))) if moving_post.any() and np.isfinite(e[moving_post]).all() else None
        post = e[t > event_s + 1e-9]
        row["stayed_below_threshold"] = bool(len(post) and not row["failed"] and np.all(post < .1))
        traces[key] = (t - event_s, e)
    summary = []
    for scenario, mode in itertools.product(manifest["scenarios"], modes):
        rows = [by_key[(scenario, seed, mode)] for seed in manifest["seeds"]]
        def mean(field):
            values = [r[field] for r in rows if r[field] is not None]
            return float(np.mean(values)) if values else None
        summary.append({"scenario": scenario, "controller": mode, "episodes": len(rows),
                        "failed": sum(r["failed"] for r in rows),
                        "invalid_flights": sum(r["invalid_samples"] > 0 for r in rows),
                        "mean_finite_episode_rmse_m": mean("position_rmse_m"),
                        "mean_observed_post_event_peak_m": mean("post_event_peak_m"),
                        "mean_moving_rmse_m": mean("moving_rmse_m"),
                        "mean_post_event_moving_rmse_m": mean("post_event_moving_rmse_m"),
                        "recovered": sum(r["recovered"] for r in rows),
                        "stayed_below_threshold": sum(r["stayed_below_threshold"] for r in rows),
                        "mean_post_peak_recovery_s_recovered_only": mean("recovery_time_s")})
    write_csv(folder / "summary.csv", summary)
    pairs = []
    if len(modes) == 2:
        for scenario, seed in itertools.product(manifest["scenarios"], manifest["seeds"]):
            pid, sac = [by_key[(scenario, seed, mode)] for mode in modes]
            delta = None
            if pid["position_rmse_m"] is not None and sac["position_rmse_m"] is not None:
                delta = sac["position_rmse_m"] - pid["position_rmse_m"]
            pairs.append({"scenario": scenario, "seed": seed,
                          "sac_minus_pid_rmse_m": delta,
                          "pid_failed": pid["failed"], "sac_failed": sac["failed"],
                          "pid_recovered": pid["recovered"], "sac_recovered": sac["recovered"],
                          "pid_stayed_below_threshold": pid["stayed_below_threshold"],
                          "sac_stayed_below_threshold": sac["stayed_below_threshold"],
                          "pid_recovery_s": pid["recovery_time_s"], "sac_recovery_s": sac["recovery_time_s"]})
        write_csv(folder / "paired.csv", pairs)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(manifest["scenarios"]), 1, figsize=(9, 2.7 * len(manifest["scenarios"])), squeeze=False)
        for scenario, ax in zip(manifest["scenarios"], axes[:, 0]):
            for mode, color in zip(modes, ["#be5b24", "#1769aa"]):
                group = [traces[(scenario, seed, mode)] for seed in manifest["seeds"]]
                # Only plot an across-seed mean when every trajectory is complete.
                complete = all(not by_key[(scenario, seed, mode)]["failed"] for seed in manifest["seeds"])
                for t, e in group:
                    ax.plot(t, e, color=color, alpha=.2, linewidth=.8)
                if complete:
                    grid = np.linspace(max(t[0] for t, _ in group), min(t[-1] for t, _ in group), 200)
                    ax.plot(grid, np.mean([np.interp(grid, t, e) for t, e in group], axis=0), color=color, label=mode)
                else:
                    ax.plot([], [], color=color, label=mode + " (individual; failures present)")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axhline(.1, color="gray", linestyle=":", linewidth=1)
            ax.set(title=scenario, ylabel="Position error (m)", xlabel="Time relative to wind event (s)")
            ax.legend()
            ax.grid(alpha=.2)
        fig.suptitle(f"{manifest.get('route', 'hover')} wind-step evaluation: {len(manifest['seeds'])} paired seeds; thin lines = individual flights")
        fig.tight_layout(rect=(0, 0, 1, .98))
        fig.savefig(folder / "error_over_time.png", dpi=150)
        plt.close(fig)
    print(f"Validated {len(flights)} flights: timing, mean wind, and recomputed metrics")
    for row in summary:
        print(row)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    analyze(args.folder, args.plot)
