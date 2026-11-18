r"""M9 Step 6: per-case figures, a 3x3 grid per scene and the summary table. NumPy + matplotlib. NOT eval-v1.

    # Directory: <repo root>\residual_ppo
    python demo\plot_results.py

Inputs: results/m9_demo/flights/flights_<scene>.json + the .npz series (sha256 verified before use),
results/m9_demo/geometry_<scene>.json. Outputs under results/m9_demo/plots/: cases/<case>.png (top-down, side view
= height over along-track distance, error over time), grid_<scene>.png, and results/m9_demo/summary_table.{csv,md}.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from demo import demo_config  # noqa: E402

OUT = RESIDUAL_DIR / "results" / "m9_demo"
PLOTS = OUT / "plots"
CTRL = {"pid": ("PID", "#e07b00"), "pid_ppo": ("PID + PPO residual", "#1f5fbf"), "ppo_direct": ("PPO direct", "#2a9d3c")}
WIND_TITLE = {"calm": "calm (0 m/s)", "w3p5": "3.5 m/s", "w7": "7 m/s"}
WIND_TAG = "crosswind toward obstacles (worst side)"
FOOT = dict(fc="#a0522d", ec="#5c2e0e", lw=0.3, alpha=0.6, zorder=1)


def banner(scene):
    return (f"{demo_config.LABEL.upper()}  |  {demo_config.NO_PERCEPTION_NOTE}  |  {WIND_TAG}"
            + (f"  |  indoor wind = {demo_config.SCENES['indoor']['wind_label']}" if scene == "indoor" else ""))


def load(scene):
    rec = json.loads((OUT / "flights" / f"flights_{scene}.json").read_text(encoding="utf-8"))
    series = {}
    for r in rec["rows"]:
        path = RESIDUAL_DIR / r["series_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != r["series_sha256"]:
            raise SystemExit(f"sha256 mismatch: {r['series_file']}")
        with np.load(path) as z:
            series[(r["track"], r["wind"], r["controller"])] = {k: z[k] for k in z.files}
    geo = json.loads((OUT / f"geometry_{scene}.json").read_text(encoding="utf-8"))["shapes"]
    return rec, series, geo


def footprints(ax, shapes, zlo, zhi, box):
    (x0, x1), (y0, y1) = box
    for s in shapes:
        lo, hi = np.array(s["aabb"][0]), np.array(s["aabb"][1])
        if hi[2] < zlo or lo[2] > zhi or hi[0] < x0 or lo[0] > x1 or hi[1] < y0 or lo[1] > y1:
            continue
        vertical = s["kind"] == "cylinder" and abs(abs(s["axis"][2]) - 1.0) < 1e-6
        if s["kind"] == "sphere" or vertical:
            ax.add_patch(Circle(s["center"][:2], s["radius"], **FOOT))
        else:
            ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], **FOOT))


def along_track(points, path):
    """Arc length of the nearest reference path point, per point (side view x-axis)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    out = np.empty(len(points))
    for i in range(0, len(points), 512):
        d = np.linalg.norm(points[i:i + 512, None, :] - path[None], axis=-1)
        out[i:i + 512] = s[np.argmin(d, axis=1)]
    return out


def view_box(path, margin):
    lo, hi = path[:, :2].min(0) - margin, path[:, :2].max(0) + margin
    span = (hi - lo).max()
    c = 0.5 * (lo + hi)
    return (c[0] - span / 2, c[0] + span / 2), (c[1] - span / 2, c[1] + span / 2)


def wind_arrow(ax, row, path, scale):
    if row["wind_speed"] == 0:
        return
    w = np.asarray(row["wind_direction_xy"])
    c = path[len(path) // 2, :2]
    ax.annotate("", xy=c + scale * w, xytext=c, arrowprops=dict(arrowstyle="simple", color="#555", alpha=0.5),
                zorder=2)


def case_figure(scene, track, wind, rows, series, shapes, meta):
    ref = series[(track, wind, "pid")]["reference_path"]
    zs = [series[(track, wind, c)]["rec60"][:, 3] for c in CTRL]
    zlo, zhi = min(ref[:, 2].min(), *(np.nanmin(z) for z in zs)) - 0.3, max(ref[:, 2].max(), *(np.nanmax(z) for z in zs)) + 0.3
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.6], hspace=0.35, wspace=0.18)
    ax_top, ax_side, ax_err = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    box = view_box(np.vstack([ref] + [series[(track, wind, c)]["rec60"][:, 1:4] for c in CTRL]), 1.8)
    footprints(ax_top, shapes, zlo, zhi, box)
    ax_top.plot(ref[:, 0], ref[:, 1], "k--", lw=1.3, label="reference", zorder=3)
    s_ref = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))])
    ax_side.plot(s_ref, ref[:, 2], "k--", lw=1.3, label="reference")
    for c, (label, color) in CTRL.items():
        r = rows[(track, wind, c)]
        rec = series[(track, wind, c)]["rec60"]
        p, pref = rec[:, 1:4], rec[:, 14:17]
        txt = (f"{label}: RMSE {r['position_rmse']:.3f} m, max {r['max_deviation']:.2f} m, "
               f"{'FAILED ' + r['failure_reason'] if r['failed'] else 'no failure'}, "
               f"{'completed' if r['completed'] else 'not completed'}, "
               f"{'CONTACT' if r['obstacle_contact'] else 'no contact'} (min clear {r['min_clearance_m']:.2f} m)")
        ax_top.plot(p[:, 0], p[:, 1], color=color, lw=1.6, label=txt, zorder=4)
        ax_side.plot(along_track(p, ref), p[:, 2], color=color, lw=1.4)
        ax_err.plot(rec[:, 0], np.linalg.norm(p - pref, axis=1), color=color, lw=1.4, label=label)
    ax_top.plot(*ref[0, :2], "k^", ms=9, zorder=5)
    wind_arrow(ax_top, rows[(track, wind, "pid")], ref, 1.2)
    ax_top.set_xlim(box[0]); ax_top.set_ylim(box[1]); ax_top.set_aspect("equal")
    ax_top.set_xlabel("x [m]"); ax_top.set_ylabel("y [m]")
    ax_top.set_title("top-down (obstacle footprints in brown; grey arrow = wind)", fontsize=9)
    ax_side.set_xlabel("along-track distance of the nearest reference point [m]"); ax_side.set_ylabel("z [m]")
    ax_side.set_title("side view: height along the track", fontsize=9)
    ax_err.set_xlabel("time [s]"); ax_err.set_ylabel("|p - p_ref(t)| [m]")
    ax_err.set_title("time-indexed position error", fontsize=9); ax_err.legend(fontsize=8, loc="upper left")
    for a in (ax_side, ax_err):
        a.grid(alpha=0.3)
    ax_top.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=7.5, frameon=False)
    t = meta["tracks"][track]
    fig.suptitle(f"M9 {scene} / {track} ({t['label']}) / wind {WIND_TITLE[wind]}, gust scale "
                 f"{demo_config.WIND_LEVELS[wind]['gust_scale']}, case seed {rows[(track, wind, 'pid')]['case_seed']}\n"
                 f"{banner(scene)}", fontsize=9.5)
    path = PLOTS / "cases" / f"{scene}_{track}_{wind}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def grid_figure(scene, rows, series, shapes, meta):
    tracks_ = list(meta["tracks"])
    fig, axes = plt.subplots(3, 3, figsize=(16, 16.5))
    for i, track in enumerate(tracks_):
        ref = series[(track, "calm", "pid")]["reference_path"]
        box = view_box(np.vstack([ref] + [series[(track, w, c)]["rec60"][:, 1:4] for w in WIND_TITLE for c in CTRL]), 1.5)
        zlo, zhi = ref[:, 2].min() - 0.8, ref[:, 2].max() + 0.8
        for j, wind in enumerate(WIND_TITLE):
            ax = axes[i, j]
            footprints(ax, shapes, zlo, zhi, box)
            ax.plot(ref[:, 0], ref[:, 1], "k--", lw=1.1, zorder=3)
            lines = []
            for c, (label, color) in CTRL.items():
                r = rows[(track, wind, c)]
                p = series[(track, wind, c)]["rec60"][:, 1:4]
                ax.plot(p[:, 0], p[:, 1], color=color, lw=1.4, zorder=4)
                lines.append(f"{label}: {r['position_rmse']:.3f} m"
                             + (" FAIL" if r["failed"] else "") + (" CONTACT" if r["obstacle_contact"] else ""))
            ax.plot(*ref[0, :2], "k^", ms=7, zorder=5)
            wind_arrow(ax, rows[(track, wind, "pid")], ref, 1.2)
            ax.text(0.02, 0.02, "RMSE\n" + "\n".join(lines), transform=ax.transAxes, fontsize=8, va="bottom",
                    bbox=dict(fc="white", alpha=0.85, ec="0.7"), zorder=6)
            ax.set_xlim(box[0]); ax.set_ylim(box[1]); ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"wind {WIND_TITLE[wind]}", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{track}\ny [m]", fontsize=10)
    handles = [plt.Line2D([], [], color="k", ls="--", label="reference")] + \
              [plt.Line2D([], [], color=col, lw=2, label=lab) for lab, col in CTRL.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle(f"M9 {scene}: 3 tracks x 3 wind levels (top-down; grey arrow = wind)\n{banner(scene)}", fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    path = PLOTS / f"grid_{scene}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def main():
    (PLOTS / "cases").mkdir(parents=True, exist_ok=True)
    table = []
    for scene in ("indoor", "forest"):
        rec, series, shapes = load(scene)
        meta = rec["meta"]
        if not all(v["pass"] for v in meta["pairing"].values()) or len(meta["pairing"]) != 9:
            raise SystemExit(f"{scene}: pairing record incomplete")
        rows = {(r["track"], r["wind"], r["controller"]): r for r in rec["rows"]}
        for track in meta["tracks"]:
            for wind in WIND_TITLE:
                p = case_figure(scene, track, wind, rows, series, shapes, meta)
                print(f"written: {p.relative_to(RESIDUAL_DIR).as_posix()}")
        print(f"written: {grid_figure(scene, rows, series, shapes, meta).relative_to(RESIDUAL_DIR).as_posix()}")
        table += rec["rows"]

    keys = ["scene", "track", "wind", "controller", "position_rmse", "cross_track_rmse", "max_deviation", "completed",
            "failed", "failure_reason", "obstacle_contact", "min_clearance_m", "saturation_rate", "action_norm_mean",
            "decisions", "case_seed"]
    with open(OUT / "summary_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(table)
    idx = {(r["scene"], r["track"], r["wind"], r["controller"]): r for r in table}
    md = [f"# M9 summary - {demo_config.LABEL}", "",
          f"{demo_config.NO_PERCEPTION_NOTE}; {WIND_TAG}; indoor wind = "
          f"{demo_config.SCENES['indoor']['wind_label']}. One flight per cell (n = 1): no CIs, no tests.", "",
          "| Scene | Track | Wind | RMSE PID / PID+PPO / direct (m) | Max dev PID / PID+PPO / direct (m) "
          "| Completed P/R/D | Failures | Contacts | Min clearance P/R/D (m) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        if r["controller"] != "pid":
            continue
        trio = [idx[(r["scene"], r["track"], r["wind"], c)] for c in CTRL]
        md.append(f"| {r['scene']} | {r['track']} | {WIND_TITLE[r['wind']]} | "
                  + " / ".join(f"{x['position_rmse']:.3f}" for x in trio) + " | "
                  + " / ".join(f"{x['max_deviation']:.2f}" for x in trio) + " | "
                  + "/".join("Y" if x["completed"] else "n" for x in trio) + " | "
                  + (", ".join(f"{c}: {x['failure_reason']}" for c, x in zip(CTRL, trio) if x["failed"]) or "none")
                  + " | " + (", ".join(c for c, x in zip(CTRL, trio) if x["obstacle_contact"]) or "none") + " | "
                  + " / ".join(f"{x['min_clearance_m']:.2f}" for x in trio) + " |")
    md += ["", "Mean RMSE over the 3 tracks of a scene (m):", "",
           "| Scene | Wind | PID | PID + PPO residual | PPO direct |", "|---|---|---|---|---|"]
    for scene in ("indoor", "forest"):
        for wind in WIND_TITLE:
            vals = [np.mean([r["position_rmse"] for r in table
                             if r["scene"] == scene and r["wind"] == wind and r["controller"] == c]) for c in CTRL]
            md.append(f"| {scene} | {WIND_TITLE[wind]} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    (OUT / "summary_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("written: results/m9_demo/summary_table.csv, summary_table.md")
    print("RESULT: DONE")


if __name__ == "__main__":
    main()
