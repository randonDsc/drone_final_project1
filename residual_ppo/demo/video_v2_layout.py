r"""M9 Step 8 (video v2): camera, framing and overlay geometry for the demo videos. NumPy only. NOT eval-v1.

    # Directory: <repo root>\residual_ppo
    & <venv>\python.exe demo\video_v2_layout.py --case indoor_I1_aisle_w7      (plan + asserts, no Isaac)
    & <venv>\python.exe demo\video_v2_layout.py --all-w7

Everything here is computed from the COMMITTED 60 Hz series (results/m9_demo/flights/<scene>/*.npz, sha256-checked
against flights_<scene>.json) and the committed geometry_<scene>.json; no flight is re-run.

v1 framing bug (fixed here): make_replay.camera_pose clamped the focal length at 8 mm, so in indoor_I1_aisle_w7 the
shared start point projected below the frame. v2 searches camera positions (inside the warehouse and under its 9 m
ceiling indoors; above the ground outdoors), orientations and focal lengths >= 7 mm, and keeps the camera that makes a
1 m wind-direction displacement largest on screen, subject to HARD constraints: every reference point, every flown
point of all three controllers, the start ring and the 3D wind arrow project inside the frame and outside the overlay
panels (PANELS, with PANEL_GAP px clearance). Occlusion of the drones by obstacles (camera->drone rays against the exact
obstacle shapes, 4 Hz samples) is minimised and reported (0 is preferred; it is not part of the framing assert).

Camera model = USD pinhole camera (horizontal aperture 20.955 mm, vertical = horizontal / (16/9), no aperture offset),
camera frame rows (r, u, -f) as in make_replay: pixel x = W/2 + s (d.r)/(d.f), pixel y = H/2 - s (d.u)/(d.f), with
s = focal / aperture * W. render_replay_v2 checks this projection against the rendered drones (pixel offsets).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))

import numpy as np  # noqa: E402

from demo import demo_config  # noqa: E402
from demo.geometry import Obstacles  # noqa: E402
from demo.make_replay import CONTROLLERS as V1_CONTROLLERS  # noqa: E402  (single source of the colours)

OUT = RESIDUAL_DIR / "results" / "m9_demo" / "video_v2"
W, H = 1920, 1080
H_APERTURE = 20.955                    # mm, USD default (as make_replay)
ASPECT = W / H
FOCAL_MIN_MM = 7.0                     # v1 clamped at 8 mm; v2 may go slightly wider and moves the camera instead
HZ = 60

ORDER = ("pid", "pid_ppo", "ppo_direct")
LABELS = {"pid": "PID", "pid_ppo": "PID+PPO", "ppo_direct": "PPO only"}
COLORS_SRGB = {c: tuple(V1_CONTROLLERS[c][1]) for c in ORDER}      # legend / trail / tag colour (display sRGB)
WIND_ARROW_SRGB = (0.55, 0.80, 1.00)                                # light blue

# Overlay panels (x0, y0, x1, y1) in pixels. Each touches the frame margin, so the allowed region is star-shaped
# around the image centre (needed by max_scale; asserted in tests).
EDGE = 24
PANEL_GAP = 12
PANELS = {
    "legend": (EDGE, EDGE, 560, 310),   # 222 before the clearance lines (moved the F3 / F2 cameras; see M9 notes)
    "wind": (1296, EDGE, W - EDGE, 196),
    "inset": (W - EDGE - 480, 600, W - EDGE, 960),
    "caption": (0, 984, W, H),
}
INSET = PANELS["inset"]

SCENE_BOUNDS = {   # camera position bounds (x, y, z) lo / hi
    "indoor": (np.array([-11.3, -17.3, 3.0]), np.array([11.3, 17.3, 8.3])),     # walls +-12 / +-18, ceiling 9 m
    "forest": (np.array([-40.0, -40.0, 2.0]), np.array([40.0, 40.0, 14.0])),
}
CAMERA_CLEARANCE = 1.0                 # m from any obstacle surface
START_MARGIN_PX = 160                  # start ring this far from panels / edges: room for the tags and hold caption
MIN_DOWN_PITCH_DEG = 20.0              # elevated 3/4 view: the floor plane and the sideways drift stay readable
MAX_DOWN_PITCH_DEG = 50.0
FOCAL_MAX_MM = 35.0                    # without a cap the search prefers distant telephoto views (flat, no depth cue)
RING_RADIUS = 0.60
ARROW_LENGTH = 1.40
ARROW_SHAFT_RADIUS = 0.035
ARROW_HEAD_RADIUS = 0.10
ARROW_HEAD_LENGTH = 0.28


def srgb_to_linear(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def bgr255(rgb):
    return tuple(int(round(255 * v)) for v in reversed(rgb))


# ---------------------------------------------------------------- data

def load_case(case):
    """The three controllers' committed 60 Hz series for one case, sha256-checked, plus its wind and track."""
    from demo import tracks

    scene, rest = case.split("_", 1)
    track, wind = rest.rsplit("_", 1)
    rec = json.loads((RESIDUAL_DIR / "results/m9_demo/flights" / f"flights_{scene}.json").read_text("utf-8"))
    rows = {r["controller"]: r for r in rec["rows"] if r["track"] == track and r["wind"] == wind}
    if set(rows) != set(ORDER):
        raise SystemExit(f"{case}: controllers {sorted(rows)}")
    series, sources = {}, {}
    for c in ORDER:
        src = RESIDUAL_DIR / rows[c]["series_file"]
        if hashlib.sha256(src.read_bytes()).hexdigest() != rows[c]["series_sha256"]:
            raise SystemExit(f"sha256 mismatch {src}")
        with np.load(src) as z:
            series[c] = {k: z[k] for k in z.files}
        sources[c] = {"file": rows[c]["series_file"], "sha256": rows[c]["series_sha256"]}
    wdir = np.asarray(rows["pid"]["wind_direction_xy"], dtype=float)
    for c in ORDER:
        if not np.array_equal(np.asarray(rows[c]["wind_direction_xy"], dtype=float), wdir):
            raise SystemExit(f"{case}: wind direction differs between controllers")
    # The overlay indexes every controller by the same 60 Hz row; the replay runs to the longest series.
    if len({len(series[c]["rec60"]) for c in ORDER}) != 1:
        raise SystemExit(f"{case}: series lengths differ {[len(series[c]['rec60']) for c in ORDER]}")
    start = series["pid"]["rec60"][0, 1:4]
    same_start = all(np.array_equal(series[c]["rec60"][0, 1:4], start) for c in ORDER)
    clearance = load_clearance(scene, track, wind, rows)
    heading, leg_mid = tracks.main_heading(tracks.TRACKS[scene][track])
    spec = demo_config.WIND_LEVELS[wind]
    return {"case": case, "scene": scene, "track": track, "wind": wind, "rows": rows, "series": series,
            "sources": sources, "ref_path": series["pid"]["reference_path"], "wind_dir": wdir / np.linalg.norm(wdir),
            "wind_speed": float(spec["speed"]), "gust_scale": float(spec["gust_scale"]), "start": start,
            "same_start": bool(same_start), "heading": heading, "leg_mid": leg_mid, "clearance": clearance}


SUMMARY_CSV = RESIDUAL_DIR / "results/m9_demo/summary_table.csv"


def load_clearance(scene, track, wind, rows):
    """Per controller, from summary_table.csv: min_clearance_m (whole flight; drone CENTRE to the nearest obstacle
    surface, demo/geometry.py) and obstacle_contact (body box, team collision_half_size). Cross-checked against the
    same fields in flights_<scene>.json (the rows the series come from)."""
    import csv

    with open(SUMMARY_CSV, newline="", encoding="utf-8") as fh:
        csv_rows = {r["controller"]: r for r in csv.DictReader(fh)
                    if (r["scene"], r["track"], r["wind"]) == (scene, track, wind)}
    if set(csv_rows) != set(ORDER):
        raise SystemExit(f"{scene}_{track}_{wind}: summary_table.csv controllers {sorted(csv_rows)}")
    out = {}
    for c in ORDER:
        m = float(csv_rows[c]["min_clearance_m"])
        contact = {"True": True, "False": False}[csv_rows[c]["obstacle_contact"]]
        if m != rows[c]["min_clearance_m"] or contact != rows[c]["obstacle_contact"]:
            raise SystemExit(f"{scene}_{track}_{wind} {c}: summary_table.csv != flights json")
        out[c] = {"min_clearance_m": m, "obstacle_contact": contact}
    return out


def obstacles_for(scene):
    return Obstacles.load(RESIDUAL_DIR / "results/m9_demo" / f"geometry_{scene}.json")


# ---------------------------------------------------------------- markers

def ring_points(center, radius=RING_RADIUS, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(center) + radius * np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1)


def arrow_samples(tail, tip, n=15):
    return tail + np.linspace(0, 1, n)[:, None] * (tip - tail)


def place_wind_arrow(data, obstacles):
    """3D wind arrow (non-physical): horizontal, along wind_direction_xy, near the track midpoint at flight height,
    on the UPWIND side so its tip points at the path. First candidate with >= 0.25 m obstacle clearance and >= 0.4 m
    from the reference path and every flown point. Returns (tail, tip, info)."""
    w = np.array([data["wind_dir"][0], data["wind_dir"][1], 0.0])
    ref = data["ref_path"]
    flown = np.vstack([data["series"][c]["rec60"][:, 1:4] for c in ORDER])
    k_mid = int(np.argmin(np.linalg.norm(ref[:, :2] - data["leg_mid"][:2], axis=1)))
    tried = []
    for dk in (0, -20, 20, -40, 40, -60, 60):
        k = int(np.clip(k_mid + dk, 0, len(ref) - 1))
        anchor = ref[k]
        for gap in (0.8, 0.6, 1.0, 1.2, 0.5):
            for dz in (0.0, 0.5, 1.0):
                tip = anchor - gap * w + np.array([0, 0, dz])
                tail = tip - ARROW_LENGTH * w
                s = arrow_samples(tail, tip)
                clear = float(obstacles.clearance(s)[0].min())
                d_ref = float(np.min(np.linalg.norm(s[:, None] - ref[None], axis=-1)))
                d_fl = float(np.min(np.linalg.norm(s[:, None] - flown[None, ::6], axis=-1)))
                tried.append((dk, gap, dz))
                if clear >= 0.25 and d_ref >= 0.4 and d_fl >= 0.4:
                    return tail, tip, {"anchor_ref_index": k, "anchor": anchor.tolist(), "upwind_gap_m": gap,
                                       "raised_m": dz, "obstacle_clearance_m": clear, "min_dist_reference_m": d_ref,
                                       "min_dist_flown_m": d_fl, "candidates_tried": len(tried)}
    raise SystemExit(f"{data['case']}: no wind-arrow placement found ({len(tried)} tried)")


# ---------------------------------------------------------------- camera

def look_basis(fwd):
    f = fwd / np.linalg.norm(fwd)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    return r, np.cross(r, f), f


def project(points, cam):
    """Pixel coordinates (n, 2) and depth (n,) of world points through cam = {pos, r, u, f, s}."""
    d = np.asarray(points, dtype=float) - cam["pos"]
    depth = d @ cam["f"]
    x = W / 2 + cam["s"] * (d @ cam["r"]) / depth
    y = H / 2 - cam["s"] * (d @ cam["u"]) / depth
    return np.stack([x, y], 1), depth


def _forbidden_rects(gap=PANEL_GAP):
    return np.array([(x0 - gap, y0 - gap, x1 + gap, y1 + gap) for x0, y0, x1, y1 in PANELS.values()], dtype=float)


def max_scale(nx, ny):
    """Largest s such that every point (W/2 + s nx, H/2 - s ny) stays inside [EDGE, W-EDGE] x [EDGE, H-EDGE] and outside
    every (gap-expanded) panel. Exact per point (ray from the image centre vs rectangles); global s = min over points,
    valid because the allowed region is star-shaped around the centre. Works on arrays of any shape (last axis =
    points)."""
    dx, dy = nx, -ny
    big = 1e12
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = np.where(dx > 0, (W - EDGE - W / 2) / dx, np.where(dx < 0, (EDGE - W / 2) / dx, big))
        sy = np.where(dy > 0, (H - EDGE - H / 2) / dy, np.where(dy < 0, (EDGE - H / 2) / dy, big))
        s = np.minimum(sx, sy)
        for x0, y0, x1, y1 in _forbidden_rects():
            tx0, tx1 = (x0 - W / 2) / dx, (x1 - W / 2) / dx
            ty0, ty1 = (y0 - H / 2) / dy, (y1 - H / 2) / dy
            txa, txb = np.minimum(tx0, tx1), np.maximum(tx0, tx1)
            tya, tyb = np.minimum(ty0, ty1), np.maximum(ty0, ty1)
            # a zero direction component: inside the slab for all t, or never
            txa = np.where(dx == 0, np.where((x0 <= W / 2) & (W / 2 <= x1), -big, big), txa)
            txb = np.where(dx == 0, np.where((x0 <= W / 2) & (W / 2 <= x1), big, -big), txb)
            tya = np.where(dy == 0, np.where((y0 <= H / 2) & (H / 2 <= y1), -big, big), tya)
            tyb = np.where(dy == 0, np.where((y0 <= H / 2) & (H / 2 <= y1), big, -big), tyb)
            t_in, t_out = np.maximum(txa, tya), np.minimum(txb, tyb)
            hit = (t_in <= t_out) & (t_out > 0)
            s = np.where(hit, np.minimum(s, np.maximum(t_in, 0.0)), s)
    return np.min(s, axis=-1)


def pixel_margin(px):
    """Signed clearance (px) of each point from the forbidden region (frame margin + gap-expanded panels): > 0 = ok."""
    m = np.minimum.reduce([px[:, 0] - EDGE, W - EDGE - px[:, 0], px[:, 1] - EDGE, H - EDGE - px[:, 1]])
    for x0, y0, x1, y1 in _forbidden_rects():
        ddx = np.maximum.reduce([x0 - px[:, 0], px[:, 0] - x1, np.zeros(len(px))])
        ddy = np.maximum.reduce([y0 - px[:, 1], px[:, 1] - y1, np.zeros(len(px))])
        outside = np.hypot(ddx, ddy)
        inside = (px[:, 0] >= x0) & (px[:, 0] <= x1) & (px[:, 1] >= y0) & (px[:, 1] <= y1)
        depth_in = np.minimum.reduce([px[:, 0] - x0, x1 - px[:, 0], px[:, 1] - y0, y1 - px[:, 1]])
        m = np.minimum(m, np.where(inside, -depth_in, outside))
    return m


def _orient(base_fwd, yaw, pitch):
    """Rotate base_fwd by yaw (about z) and pitch (about the camera right axis), radians; arrays broadcast."""
    yaw0 = np.arctan2(base_fwd[1], base_fwd[0])
    pitch0 = np.arcsin(np.clip(base_fwd[2] / np.linalg.norm(base_fwd), -1, 1))
    a, b = yaw0 + yaw, pitch0 + pitch
    return np.stack([np.cos(b) * np.cos(a), np.cos(b) * np.sin(a), np.sin(b)], -1)


def _best_orientation(pos, pts, center, wind_pts, grid):
    """Over yaw/pitch offsets: the orientation with the largest feasible scale; returns (scale, fwd)."""
    base = center - pos
    yy, pp = np.meshgrid(grid, grid, indexing="ij")
    fwd = _orient(base, yy.ravel(), pp.ravel())                       # (G, 3)
    down = np.degrees(-np.arcsin(fwd[:, 2]))
    ok = (down >= MIN_DOWN_PITCH_DEG) & (down <= MAX_DOWN_PITCH_DEG)
    r = np.cross(fwd, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    u = np.cross(r, fwd)
    d = pts - pos                                                     # (n, 3)
    depth = fwd @ d.T                                                 # (G, n)
    nx = (r @ d.T) / depth
    ny = (u @ d.T) / depth
    s = np.where(np.all(depth > 0.3, axis=1) & ok, max_scale(nx, ny), 0.0)
    g = int(np.argmax(s))
    return float(s[g]), fwd[g]


SCORE_PERCENTILE = 10


def wind_visibility(cam, ref_pts, w, q=50):
    """Percentile q (default median) of the on-screen length (px) of a 1 m wind-direction displacement at the
    reference points."""
    a, _ = project(ref_pts, cam)
    b, _ = project(ref_pts + w, cam)
    return float(np.percentile(np.linalg.norm(b - a, axis=1), q))


def path_visibility(cam, ref_pts, q=50):
    """Percentile q of the on-screen length (px) per metre of path, per reference segment (progress along the
    track)."""
    px, _ = project(ref_pts, cam)
    l3 = np.linalg.norm(np.diff(ref_pts, axis=0), axis=1)
    l2 = np.linalg.norm(np.diff(px, axis=0), axis=1)
    keep = l3 > 1e-6
    return float(np.percentile(l2[keep] / l3[keep], q))


def view_score(cam, ref_pts, w):
    """Camera score: geometric mean of the wind-deviation and along-path visibilities (px per m), each at its 10th
    percentile over the reference path. Rewarding only the wind deviation picked telephoto views straight down the
    path; the median picked a wide lens 2.7 m from the track end (far start tiny); the 10th percentile makes the
    worst-resolved part of the track count."""
    q = SCORE_PERCENTILE
    return float(np.sqrt(wind_visibility(cam, ref_pts, w, q) * path_visibility(cam, ref_pts, q)))


def occluded(cam_pos, targets, obstacles, stop_short=0.35):
    """Boolean per target: does the segment camera -> (target - stop_short along the ray) cross an obstacle?
    Boxes: exact slab test in each box frame; cylinders / spheres: dense samples (2 cm) tested for inside."""
    t = np.asarray(targets, dtype=float)
    v = t - cam_pos
    L = np.linalg.norm(v, axis=1)
    dirs = v / L[:, None]
    seg = np.maximum(L - stop_short, 0.0)
    hit = np.zeros(len(t), dtype=bool)
    if len(obstacles.box_c):
        o = np.einsum("mjk,k->mj", obstacles.box_u, cam_pos - 0) - np.einsum("mjk,mk->mj", obstacles.box_u,
                                                                            obstacles.box_c)   # (m, 3) local origin
        dl = np.einsum("mjk,nk->nmj", obstacles.box_u, dirs)                                    # (n, m, 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-obstacles.box_h[None] - o[None]) / dl
            t2 = (obstacles.box_h[None] - o[None]) / dl
        inside_slab = np.abs(o[None]) <= obstacles.box_h[None]
        tmin = np.where(dl == 0, np.where(inside_slab, -np.inf, np.inf), np.minimum(t1, t2))
        tmax = np.where(dl == 0, np.where(inside_slab, np.inf, -np.inf), np.maximum(t1, t2))
        t_in, t_out = tmin.max(-1), tmax.min(-1)                                                # (n, m)
        hit |= np.any((t_in <= t_out) & (t_out >= 0) & (t_in <= seg[:, None]), axis=1)
    if len(obstacles.cyl_c) or len(obstacles.sph_c):
        for i in range(len(t)):
            n = max(int(seg[i] / 0.02), 2)
            p = cam_pos + np.linspace(0, seg[i], n)[:, None] * dirs[i]
            if len(obstacles.cyl_c):
                d = p[:, None] - obstacles.cyl_c[None]
                ax = np.einsum("nmk,mk->nm", d, obstacles.cyl_a)
                rad = np.linalg.norm(d - ax[..., None] * obstacles.cyl_a[None], axis=-1)
                if np.any((np.abs(ax) <= obstacles.cyl_h[None]) & (rad <= obstacles.cyl_r[None])):
                    hit[i] = True
                    continue
            if len(obstacles.sph_c):
                if np.any(np.linalg.norm(p[:, None] - obstacles.sph_c[None], axis=-1) <= obstacles.sph_r[None]):
                    hit[i] = True
    return hit


def scene_points(data, arrow):
    """Every point that must be framed: reference path, all flown 60 Hz points, start ring, wind-arrow samples."""
    flown = np.vstack([data["series"][c]["rec60"][:, 1:4] for c in ORDER])
    pts = np.vstack([data["ref_path"], flown, ring_points(data["start"]), arrow_samples(*arrow[:2], n=25)])
    return pts[np.all(np.isfinite(pts), axis=1)]


def occlusion_targets(data):
    """Drone positions at 4 Hz for all three controllers (+ the start)."""
    return np.vstack([data["series"][c]["rec60"][::15, 1:4] for c in ORDER])


def search_camera(data, obstacles, arrow, n_occlusion=40, verbose=False):
    pts_all = scene_points(data, arrow)
    rng = np.random.default_rng(0)
    pts = pts_all[rng.choice(len(pts_all), size=min(240, len(pts_all)), replace=False)]
    lo_b, hi_b = pts_all.min(0), pts_all.max(0)
    center = 0.5 * (lo_b + hi_b)
    w = np.array([data["wind_dir"][0], data["wind_dir"][1], 0.0])
    ref_sub = data["ref_path"][::8]
    blo, bhi = SCENE_BOUNDS[data["scene"]]
    cands = []
    for th in np.radians(np.arange(0, 360, 15)):
        for D in np.arange(4.0, 24.1, 1.5):
            for hz in np.arange(1.0, 12.1, 0.75):
                p = np.array([center[0] + D * np.cos(th), center[1] + D * np.sin(th), center[2] + hz])
                if np.all(p >= blo) and np.all(p <= bhi):
                    cands.append(p)
    cands = np.array(cands)
    clear = obstacles.clearance(cands)[0]
    cands = cands[clear >= CAMERA_CLEARANCE]
    s_min = FOCAL_MIN_MM / H_APERTURE * W
    s_cap = FOCAL_MAX_MM / H_APERTURE * W
    coarse = np.radians(np.arange(-24, 24.1, 3.0))
    fine = np.radians(np.arange(-3, 3.01, 0.75))
    scored = []
    for p in cands:
        s, fwd = _best_orientation(p, pts, center, None, coarse)
        if s < s_min:
            continue
        s2, fwd2 = _best_orientation(p, pts, p + fwd, None, fine)
        if s2 > s:
            s, fwd = s2, fwd2
        r, u, f = look_basis(fwd)
        cam = {"pos": p, "r": r, "u": u, "f": f, "s": min(s, s_cap)}    # a smaller scale stays feasible
        if pixel_margin(project(data["start"][None], cam)[0])[0] < START_MARGIN_PX:
            continue
        scored.append((view_score(cam, ref_sub, w), cam))
    if not scored:
        raise SystemExit(f"{data['case']}: no feasible camera")
    scored.sort(key=lambda x: -x[0])
    targets = occlusion_targets(data)
    best = None
    for score, cam in scored[:n_occlusion]:
        occ = int(occluded(cam["pos"], targets, obstacles).sum())
        if best is None or occ < best[1] or (occ == best[1] and score > best[0]):
            best = (score, occ, cam)
        if occ == 0:
            break
    score, occ, cam = best
    # Final scale from ALL points (the search used a subsample), then the hard assert.
    d = pts_all - cam["pos"]
    depth = d @ cam["f"]
    s_all = float(max_scale((d @ cam["r"]) / depth, (d @ cam["u"]) / depth))
    cam["s"] = min(cam["s"], s_all) * 0.995
    if cam["s"] < s_min:
        raise SystemExit(f"{data['case']}: focal below {FOCAL_MIN_MM} mm after the full-point check")
    return cam, {"candidates_feasible": len(scored), "candidate_positions": int(len(cands)),
                 "wind_visibility_px_per_m": wind_visibility(cam, ref_sub, w),
                 "path_visibility_px_per_m": path_visibility(cam, ref_sub),
                 "wind_visibility_p10_px_per_m": wind_visibility(cam, ref_sub, w, SCORE_PERCENTILE),
                 "path_visibility_p10_px_per_m": path_visibility(cam, ref_sub, SCORE_PERCENTILE),
                 "start_margin_px": float(pixel_margin(project(data["start"][None], cam)[0])[0]),
                 "score": view_score(cam, ref_sub, w),
                 "occluded_targets": occ,
                 "occlusion_targets": int(len(targets)), "occlusion_checked": min(n_occlusion, len(scored))}


def framing_report(cam, data, arrow):
    """HARD framing assert: every reference / flown / ring / arrow point in frame, clear of the panels, in front."""
    groups = {"reference": data["ref_path"], "ring": ring_points(data["start"]),
              "wind_arrow": arrow_samples(*arrow[:2], n=25)}
    for c in ORDER:
        groups[f"flown_{c}"] = data["series"][c]["rec60"][:, 1:4]
    out, ok = {}, True
    for name, p in groups.items():
        p = p[np.all(np.isfinite(p), axis=1)]
        px, depth = project(p, cam)
        m = pixel_margin(px)
        g_ok = bool(np.all(depth > 0.3) and np.all(m > 0))
        ok &= g_ok
        out[name] = {"points": int(len(p)), "min_margin_px": float(m.min()), "min_depth_m": float(depth.min()),
                     "x_px": [float(px[:, 0].min()), float(px[:, 0].max())],
                     "y_px": [float(px[:, 1].min()), float(px[:, 1].max())], "ok": g_ok}
    return ok, out


def focal_mm(cam):
    return cam["s"] * H_APERTURE / W


def plan_case(case, verbose=True):
    data = load_case(case)
    obstacles = obstacles_for(data["scene"])
    tail, tip, arrow_info = place_wind_arrow(data, obstacles)
    cam, search = search_camera(data, obstacles, (tail, tip))
    ok, framing = framing_report(cam, data, (tail, tip))
    tf = inset_transform(data, (tail, tip))
    foot = inset_footprints(data, obstacles, tf)
    plan = {
        "case": case, "label": demo_config.LABEL, "size": [W, H], "panels": PANELS, "panel_gap_px": PANEL_GAP,
        "edge_px": EDGE,
        "camera": {"position": cam["pos"].tolist(), "right": cam["r"].tolist(), "up": cam["u"].tolist(),
                   "forward": cam["f"].tolist(), "scale_px": cam["s"], "focal_mm": focal_mm(cam),
                   "horizontal_aperture_mm": H_APERTURE, "vertical_aperture_mm": H_APERTURE / ASPECT,
                   "down_pitch_deg": float(np.degrees(-np.arcsin(cam["f"][2]))),
                   "obstacle_clearance_m": float(obstacles_for(data["scene"]).clearance(cam["pos"][None])[0][0])},
        "search": search, "framing_ok": ok, "framing": framing,
        "wind_arrow": {"tail": tail.tolist(), "tip": tip.tolist(), "length_m": ARROW_LENGTH,
                       "direction_xy": data["wind_dir"].tolist(), **arrow_info},
        "start": data["start"].tolist(), "same_start_all_controllers": data["same_start"],
        "clearance_panel": {"source": "results/m9_demo/summary_table.csv (== flights_<scene>.json)",
                            "measured_from": "drone centre to nearest obstacle surface", **data["clearance"]},
        "wind": {"speed_mps": data["wind_speed"], "gust_scale": data["gust_scale"]},
        "inset": {"transform": tf, "footprint_band_z_m": foot["band_z_m"], "footprint_counts": foot["counts"]},
        "sources": data["sources"],
    }
    if verbose:
        print(f"[M9.v2.plan] {case}: camera {np.round(cam['pos'], 2).tolist()} focal {focal_mm(cam):.2f} mm, "
              f"pitch {plan['camera']['down_pitch_deg']:.1f} deg down, wind {search['wind_visibility_px_per_m']:.0f} "
              f"px/m (p10 {search['wind_visibility_p10_px_per_m']:.0f}), path {search['path_visibility_px_per_m']:.0f} "
              f"px/m (p10 {search['path_visibility_p10_px_per_m']:.0f}), start margin "
              f"{search['start_margin_px']:.0f} px, occluded {search['occluded_targets']}/{search['occlusion_targets']}, min margin "
              f"{min(v['min_margin_px'] for v in framing.values()):.1f} px -> framing {'PASS' if ok else 'FAIL'}",
              flush=True)
    if not ok:
        raise SystemExit(f"{case}: framing assert FAILED {json.dumps(framing)}")
    data["footprints"] = foot
    data["arrow"] = (tail, tip)
    return plan, data, cam


def camera_from_plan(plan):
    c = plan["camera"]
    return {"pos": np.array(c["position"]), "r": np.array(c["right"]), "u": np.array(c["up"]),
            "f": np.array(c["forward"]), "s": float(c["scale_px"])}


# ---------------------------------------------------------------- inset map (top view, north up)

INSET_PAD = 18
INSET_TITLE = 30


def inset_transform(data, arrow, pad_m=0.8):
    """World xy -> inset pixel (absolute frame coordinates), equal scale, north up."""
    x0, y0, x1, y1 = INSET
    pts = scene_points(data, arrow)[:, :2]
    lo, hi = pts.min(0) - pad_m, pts.max(0) + pad_m
    avail = np.array([x1 - x0 - 2 * INSET_PAD, y1 - y0 - 2 * INSET_PAD - INSET_TITLE], dtype=float)
    scale = float(np.min(avail / (hi - lo)))
    c_world = 0.5 * (lo + hi)
    c_px = np.array([0.5 * (x0 + x1), y0 + INSET_TITLE + INSET_PAD + 0.5 * avail[1]])
    return {"scale_px_per_m": scale, "world_center": c_world.tolist(), "px_center": c_px.tolist(),
            "world_lo": (c_world - 0.5 * avail / scale).tolist(), "world_hi": (c_world + 0.5 * avail / scale).tolist()}


def to_inset(xy, tf):
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    c, p, s = np.array(tf["world_center"]), np.array(tf["px_center"]), tf["scale_px_per_m"]
    return np.stack([p[0] + s * (xy[:, 0] - c[0]), p[1] - s * (xy[:, 1] - c[1])], 1)


def inset_footprints(data, obstacles, tf, band_pad=0.3):
    """Obstacle footprints at flight height for the inset: shapes whose vertical extent overlaps the flown height band
    (min..max flown z, +-band_pad) and whose xy extent overlaps the inset window. Boxes -> rectangles (all exported
    indoor boxes are axis-aligned; asserted), vertical cylinders -> circles, other cylinders -> thick segments (xy
    projection of the axis, width = diameter), spheres -> circles of the largest cross-section inside the band."""
    flown = np.vstack([data["series"][c]["rec60"][:, 1:4] for c in ORDER])
    zlo, zhi = float(flown[:, 2].min() - band_pad), float(flown[:, 2].max() + band_pad)
    wlo, whi = np.array(tf["world_lo"]), np.array(tf["world_hi"])
    out = {"band_z_m": [zlo, zhi], "rects": [], "circles": [], "segments": []}
    for s in obstacles.shapes:
        a = np.asarray(s["aabb"], dtype=float)
        if a[1, 2] < zlo or a[0, 2] > zhi or np.any(a[1, :2] < wlo) or np.any(a[0, :2] > whi):
            continue
        if s["kind"] == "box":
            if not np.allclose(s["axes"], np.eye(3)):
                raise SystemExit(f"non axis-aligned box {s['path']}: footprint not implemented")
            out["rects"].append([a[0, 0], a[0, 1], a[1, 0], a[1, 1]])
        elif s["kind"] == "cylinder":
            ax = np.asarray(s["axis"], dtype=float)
            c = np.asarray(s["center"], dtype=float)
            if abs(ax[2]) > 0.99:
                out["circles"].append([c[0], c[1], s["radius"]])
            else:
                e = s["half_height"] * ax
                out["segments"].append([c[0] - e[0], c[1] - e[1], c[0] + e[0], c[1] + e[1], 2 * s["radius"]])
        else:
            c = np.asarray(s["center"], dtype=float)
            dz = 0.0 if zlo <= c[2] <= zhi else min(abs(c[2] - zlo), abs(c[2] - zhi))
            if dz < s["radius"]:
                out["circles"].append([c[0], c[1], float(np.sqrt(s["radius"] ** 2 - dz ** 2))])
    out["counts"] = {k: len(out[k]) for k in ("rects", "circles", "segments")}
    return out


W7_CASES = ("indoor_I1_aisle_w7", "forest_F3_weave_w7", "indoor_I2_uturn_w7", "indoor_I3_climb_w7",
            "forest_F1_north_edge_w7", "forest_F2_sw_corner_w7")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    ap.add_argument("--all-w7", action="store_true")
    ap.add_argument("--write", action="store_true", help="write video_v2/<case>_layout.json")
    args = ap.parse_args()
    cases = W7_CASES if args.all_w7 else (args.case,)
    for case in cases:
        plan, _, _ = plan_case(case)
        if args.write:
            OUT.mkdir(parents=True, exist_ok=True)
            (OUT / f"{case}_layout.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    print("[M9.v2.plan] RESULT: PASS", flush=True)


if __name__ == "__main__":
    main()
