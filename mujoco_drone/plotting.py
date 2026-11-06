"""Exportable experiment figures, with no interactive display requirement."""
import csv
import json
import os
from pathlib import Path

import numpy as np


def _cache(directory):
    cache = Path(directory)/".mplconfig"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    (Path(os.environ["XDG_CACHE_HOME"])/"fontconfig").mkdir(parents=True, exist_ok=True)


def plot_run(directory, output=None):
    directory = Path(directory)
    _cache(directory)
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
    if (directory/"navigation"/"plan.json").exists():
        plot_navigation(directory)
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        clearance = values("navigation_clearance_m")
        launch = values("navigation_launch_active") > 0
        margin = json.loads((directory/"navigation"/"plan.json").read_text())["tracking_reserve_m"]
        ax.plot(time, clearance, label="Conservative vehicle-envelope clearance")
        ax.axhline(margin, color="orange", linestyle="--", label="Tracking reserve")
        ax.axhline(0, color="red", linestyle=":", label="Envelope touches an obstacle")
        if np.any(launch):
            ax.axvspan(0, time[launch].max(), color="gray", alpha=.2, label="Validated launch departure")
        ax.set(xlabel="Time (s)", ylabel="Clearance (m)")
        ax.legend()
        fig.savefig(directory/"clearance.png", dpi=180)
        plt.close(fig)
    return output


def plot_suite(directory, output=None):
    directory = Path(directory)
    _cache(directory)
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


def plot_navigation(directory, output=None):
    directory = Path(directory)
    _cache(directory)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    nav = directory/"navigation"
    geometry = json.loads((nav/"collision_geometry.json").read_text())
    waypoints = np.loadtxt(nav/"waypoints_world.csv", delimiter=",", skiprows=1, ndmin=2)
    fig = plt.figure(figsize=(11, 8), layout="constrained")
    ax = fig.add_subplot(111, projection="3d")
    faces = []
    maximum_height = waypoints[:, 2].max()+.5
    shown = 0
    for box in geometry["boxes"]:
        low, high = np.array(box["lower"]), np.array(box["upper"])
        if np.prod(high-low) < .01 or low[2] > maximum_height:
            continue  # Display simplification only; planner/physics use every box.
        x0, y0, z0 = low
        x1, y1, z1 = high
        p = np.array([[x0,y0,z0], [x1,y0,z0], [x1,y1,z0], [x0,y1,z0],
                      [x0,y0,z1], [x1,y0,z1], [x1,y1,z1], [x0,y1,z1]])
        faces.extend(p[list(index)] for index in [(0,1,2,3), (4,5,6,7), (0,1,5,4),
                                                 (2,3,7,6), (0,3,7,4), (1,2,6,5)])
        shown += 1
    if faces:
        ax.add_collection3d(Poly3DCollection(faces, facecolors="gray", edgecolors="none", alpha=.13))
    ax.plot(*waypoints.T, "o--", linewidth=1.5, markersize=3, label="Shortened Dijkstra route")
    if (directory/"reference.npz").exists():
        with np.load(directory/"reference.npz") as reference:
            if reference["positions"].size:
                ax.plot(*reference["positions"].T, linewidth=1, alpha=.6, label="Timed reference")
    if (directory/"flight.csv").exists():
        with (directory/"flight.csv").open() as file:
            rows = list(csv.DictReader(file))
        position = np.array([[float(row["truth_origin_position_m_"+axis]) for axis in "xyz"] for row in rows])
        ax.plot(*position.T, linewidth=1.8, label="Flown trajectory")
    ax.scatter(*waypoints[0], color="green", s=55, label="Start")
    ax.scatter(*waypoints[-1], color="red", s=55, label="Goal")
    all_bounds = np.array([point for b in geometry["boxes"] for point in (b["lower"], b["upper"])]+waypoints.tolist())
    low, high = all_bounds.min(axis=0), all_bounds.max(axis=0)
    ax.set(xlim=(low[0]-.5, high[0]+.5), ylim=(low[1]-.5, high[1]+.5), zlim=(0, max(3, maximum_height)),
           xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)",
           title=f"Dijkstra navigation ({shown} larger boxes shown; high roof hidden)")
    ax.set_box_aspect([max(high[0]-low[0], 1), max(high[1]-low[1], 1), max(maximum_height, 1)])
    ax.legend(loc="upper left")
    output = Path(output) if output else directory/"navigation.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output
