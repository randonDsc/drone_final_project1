"""Strict paired comparisons; raw Isaac exports need a verified protocol key."""
import csv
from pathlib import Path

import numpy as np

from .experiments import json_write, write_csv


def compare(left, right, output):
    """Compare one mode per input CSV using exact protocol/seed matches."""
    def read(path):
        with Path(path).open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError("Comparison input is empty")
        required = {"protocol_hash", "seed", "simulator", "mode", "controller_family", "metric_version",
                    "position_rmse", "cross_track_rmse", "max_deviation", "altitude_rmse", "saturation_rate", "completed", "failed", "crashed"}
        if not required.issubset(rows[0]):
            raise ValueError("Missing verified experiment protocol or metric columns: "+", ".join(sorted(required-set(rows[0]))))
        result = {}
        for row in rows:
            key = (row["protocol_hash"], int(row["seed"]))
            if key in result:
                raise ValueError("Duplicate protocol/seed; select a single controller mode per input")
            result[key] = row
        return result
    a, b = read(left), read(right)
    if a.keys() != b.keys():
        raise ValueError("Protocols/seeds differ; no paired comparison is justified. Export equivalent task schedules, wind definitions, initial states and metrics first.")
    rows = []
    for key in sorted(a):
        x, y = a[key], b[key]
        if x["controller_family"] != y["controller_family"] or x["metric_version"] != y["metric_version"]:
            raise ValueError("Controller families or metric definitions differ")
        row = dict(protocol_hash=key[0], seed=key[1], left_simulator=x["simulator"],
                   right_simulator=y["simulator"], left_mode=x["mode"], right_mode=y["mode"])
        for metric in ("position_rmse", "cross_track_rmse", "max_deviation", "altitude_rmse", "saturation_rate"):
            values = np.array([float(x[metric]), float(y[metric])])
            if not np.all(np.isfinite(values)):
                raise ValueError("Non-finite comparison metric: "+metric)
            row[metric+"_right_minus_left"] = float(values[1]-values[0])
        for name in ("completed", "failed", "crashed"):
            for side, value in (("left", x), ("right", y)):
                raw = value[name].lower()
                if raw not in ("true", "false", "1", "0"):
                    raise ValueError("Invalid boolean metric: "+name)
                row[f"{name}_{side}"] = raw in ("true", "1")
        rows.append(row)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output/"paired.csv", rows)
    summary = dict(pairs=len(rows), difference="right minus left; negative error differences favor the right input",
                   interpretation="Measured differences only. Different airframes, model parameters or controllers prevent attribution solely to the simulator.")
    for key in rows[0]:
        if key.endswith("_right_minus_left"):
            values = [row[key] for row in rows]
            summary[key+"_mean"] = float(np.mean(values))
            summary[key+"_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    for name in ("completed", "failed", "crashed"):
        for side in ("left", "right"):
            summary[f"{name}_rate_{side}"] = float(np.mean([r[f"{name}_{side}"] for r in rows]))
    json_write(output/"comparison.json", summary)
    return summary
