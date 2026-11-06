"""Exportable experiment figures, with no interactive display requirement."""
import csv
import os
from pathlib import Path

import numpy as np


def plot_run(directory, output=None):
    directory = Path(directory)
    os.environ.setdefault("MPLCONFIGDIR", str(directory/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (directory/"flight.csv").open() as file:
        records = list(csv.DictReader(file))
    def values(key):
        return np.array([float(r[key]) for r in records])
    time = values("time_s")
    position = np.column_stack([values("truth_origin_position_m_"+axis) for axis in "xyz"])
    reference = np.column_stack([values("reference_position_m_"+axis) for axis in "xyz"])
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    ax = fig.add_subplot(221, projection="3d")
    ax.plot(*position.T, label="flight")
    ax.plot(*reference.T, "--", label="reference")
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", title="Trajectory")
    ax.legend()
    ax = fig.add_subplot(222)
    ax.plot(time, np.linalg.norm(position-reference, axis=1))
    ax.set(xlabel="Time (s)", ylabel="Position error (m)")
    ax = fig.add_subplot(223)
    for axis in "xyz":
        ax.plot(time, values("wind_area_mean_world_m_s_"+axis), label=axis)
    ax.set(xlabel="Time (s)", ylabel="Area-weighted air velocity (m/s)")
    ax.legend()
    ax = fig.add_subplot(224)
    for i in range(1, 5):
        ax.plot(time, values(f"motor_{i}_command_N"), label=str(i))
    ax.set(xlabel="Time (s)", ylabel="Rotor thrust (N)")
    ax.legend()
    output = Path(output) if output else directory/"flight.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_suite(directory, output=None):
    directory = Path(directory)
    os.environ.setdefault("MPLCONFIGDIR", str(directory/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (directory/"summary.csv").open() as file:
        rows = list(csv.DictReader(file))
    tasks = list(dict.fromkeys(r["task"] for r in rows))
    winds = list(dict.fromkeys(r["wind"] for r in rows))
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    fig, axes = plt.subplots(1, len(modes), figsize=(7*len(modes), 4), squeeze=False, layout="constrained")
    maximum = max(float(r["position_rmse_mean"]) for r in rows)
    for ax, mode in zip(axes[0], modes):
        data = np.full((len(tasks), len(winds)), np.nan)
        for row in rows:
            if row["mode"] == mode:
                data[tasks.index(row["task"]), winds.index(row["wind"])] = float(row["position_rmse_mean"])
        im = ax.imshow(data, vmin=0, vmax=maximum, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(winds)), winds, rotation=30, ha="right")
        ax.set_yticks(range(len(tasks)), tasks)
        ax.set_title(mode+" mean position RMSE")
        for i in range(len(tasks)):
            for j in range(len(winds)):
                ax.text(j, i, f"{data[i,j]:.3f}", ha="center", va="center", color="white" if data[i,j] < maximum*.6 else "black")
        fig.colorbar(im, ax=ax, label="RMSE (m)")
    output = Path(output) if output else directory/"summary.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
