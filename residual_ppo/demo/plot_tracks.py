r"""M9 Step 2: top-down plot of the demo tracks over the obstacle footprints and the clearance map. NumPy + matplotlib.

    # Directory: <repo root>\residual_ppo
    python demo\plot_tracks.py

Inputs: results/m9_demo/geometry_<scene>.json (written by demo/check_scene.py in Isaac).
Outputs: results/m9_demo/tracks.json and results/m9_demo/tracks_<scene>.png. Prints RESULT: PASS only if every
track meets its clearance requirement and has no in-distribution flag.
"""
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

from demo import demo_config, tracks  # noqa: E402
from demo.geometry import Obstacles  # noqa: E402

OUT = RESIDUAL_DIR / "results" / "m9_demo"
VIEW = {"indoor": ((-11.5, 11.5), (-13.5, 18.5)), "forest": ((-10.0, 26.0), (-17.0, 17.0))}
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]


def draw_footprints(ax, shapes, zlo, zhi):
    for s in shapes:
        lo, hi = np.array(s["aabb"][0]), np.array(s["aabb"][1])
        if hi[2] < zlo or lo[2] > zhi:
            continue
        vertical = s["kind"] == "cylinder" and abs(abs(s["axis"][2]) - 1.0) < 1e-6
        style = dict(fc="#a0522d", ec="#5c2e0e", lw=0.3, alpha=0.75, zorder=2)
        if s["kind"] == "sphere" or vertical:
            ax.add_patch(Circle(s["center"][:2], s["radius"], **style))
        else:
            ax.add_patch(Rectangle(lo[:2], *(hi - lo)[:2], **style))


def main():
    record = {"label": demo_config.LABEL, "note": demo_config.NO_PERCEPTION_NOTE,
              "track_limits": demo_config.TRACK_LIMITS, "full_stage_ranges": demo_config.FULL_RANGES, "scenes": {}}
    ok_all = True
    for scene in ("indoor", "forest"):
        geo = OUT / f"geometry_{scene}.json"
        obstacles = Obstacles.load(geo)
        checked = tracks.check_tracks(scene, obstacles)
        record["scenes"][scene] = {"geometry_file": geo.name, "wind_label": demo_config.SCENES[scene]["wind_label"],
                                   "tracks": checked}

        z_all = np.concatenate([np.asarray(t["points"])[:, 2] for t in tracks.TRACKS[scene].values()])
        zs = np.unique(np.append(np.arange(z_all.min(), z_all.max(), 0.5), z_all.max()))
        (x0, x1), (y0, y1) = VIEW[scene]
        xs, ys = np.arange(x0, x1, 0.15), np.arange(y0, y1, 0.15)
        X, Y = np.meshgrid(xs, ys)
        D = None
        for z in zs:
            d = obstacles.clearance(np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)]), chunk=1024)[0]
            D = d if D is None else np.minimum(D, d)
        D = D.reshape(X.shape)

        fig, ax = plt.subplots(figsize=(11, 11 * (y1 - y0) / (x1 - x0) * 0.92))
        mesh = ax.pcolormesh(xs, ys, np.minimum(D, 5.0), cmap="Greys_r", vmin=0, vmax=5, shading="auto", zorder=1)
        ax.contour(xs, ys, D, levels=[1.0, 2.0], colors=["#ff9900", "#cc00cc"], linewidths=0.9, zorder=3)
        draw_footprints(ax, obstacles.shapes, zs.min() - 0.5, zs.max() + 0.5)
        for (name, t), color in zip(checked.items(), COLORS):
            ref = tracks.build_reference(tracks.TRACKS[scene][name])
            p = ref.path
            ax.plot(p[:, 0], p[:, 1], color=color, lw=2.4, zorder=5,
                    label=f"{name}: {t['label']}\n  min clearance {t['min_clearance_m']:.2f} m "
                          f"(need {t['min_clearance_required_m']:.0f}), {t['polyline_length_m']:.1f} m, "
                          f"z {t['height_range_m'][0]:.1f}-{t['height_range_m'][1]:.1f} m, {t['duration_s']:.1f} s")
            pts = np.asarray(t["points"])
            ax.plot(pts[:, 0], pts[:, 1], "o", color=color, ms=5, mfc="white", zorder=6)
            ax.plot(*p[0, :2], marker="^", color=color, ms=11, zorder=7)
            k = len(p) // 2
            ax.annotate("", xy=p[k + 3, :2], xytext=p[k, :2],
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2), zorder=7)
            mid = 0.5 * (pts[0] + pts[-1]) if tracks.TRACKS[scene][name]["main_leg"] is None else \
                0.5 * (pts[tracks.TRACKS[scene][name]["main_leg"]] + pts[tracks.TRACKS[scene][name]["main_leg"] + 1])
            w = np.asarray(t["wind_direction_xy"])
            ax.annotate("", xy=mid[:2] + 1.8 * w, xytext=mid[:2] - 0.2 * w,
                        arrowprops=dict(arrowstyle="simple", color=color, alpha=0.6), zorder=6)
            ok_all &= t["clearance_ok"] and not t["in_distribution_flags"]
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xlabel("x [m] (world)"); ax.set_ylabel("y [m] (world)")
        ax.set_title(f"M9 {scene}: demo tracks over obstacle footprints and clearance "
                     f"(min over z {zs.min():.1f}-{zs.max():.1f} m)\n"
                     f"{demo_config.LABEL.upper()}  |  {demo_config.NO_PERCEPTION_NOTE}  |  "
                     f"wind: {demo_config.SCENES[scene]['wind_label']}", fontsize=10)
        fig.colorbar(mesh, ax=ax, shrink=0.5, label="3D clearance to nearest obstacle [m] (capped at 5)")
        ax.plot([], [], color="#ff9900", lw=1, label="1 m clearance contour")
        ax.plot([], [], color="#cc00cc", lw=1, label="2 m clearance contour")
        ax.add_patch(Rectangle((0, 0), 0, 0, fc="#a0522d", ec="#5c2e0e", alpha=0.75,
                               label=f"obstacle footprints overlapping z {zs.min() - 0.5:.1f}-{zs.max() + 0.5:.1f} m"))
        ax.plot([], [], "^", color="k", label="start (triangle); waypoints (circles); arrow = direction")
        ax.plot([], [], color="k", alpha=0.5, lw=6, label="wide arrow = crosswind direction (all levels)")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), fontsize=8, ncol=1, frameon=False)
        png = OUT / f"tracks_{scene}.png"
        fig.savefig(png, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"written: {png.relative_to(RESIDUAL_DIR).as_posix()}")
        for name, t in checked.items():
            print(f"  {name:14s} clearance {t['min_clearance_m']:.2f} m (need {t['min_clearance_required_m']}) "
                  f"{'OK' if t['clearance_ok'] else 'FAIL'}; length {t['polyline_length_m']:.2f} m; "
                  f"turns {[round(x) for x in t['turns_deg']]}; z {t['height_range_m']}; {t['duration_s']:.1f} s; "
                  f"wind dir {np.round(t['wind_direction_xy'], 2).tolist()} ({t['wind_rule']['side']}); "
                  f"flags {t['in_distribution_flags'] or 'none'}")
    out = OUT / "tracks.json"
    out.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    print(f"written: {out.relative_to(RESIDUAL_DIR).as_posix()}")
    print(f"RESULT: {'PASS' if ok_all else 'FAIL'}")


if __name__ == "__main__":
    main()
