"""M9 demo tracks: 3 fixed, hand-placed waypoint tracks per scene. NumPy only. DESCRIPTIVE DEMO, NOT eval-v1.

Each track is a polyline of hand-placed waypoints, corners rounded with path_reference.round_corners (radius 0.8 m),
flown as a path_reference.TimedReference with the `full` stage's limits (demo_config.TRACK_LIMITS), a 0.5 s start
hold and a 3 s settle, exactly like a training route. Waypoints were placed by hand on the exact-geometry clearance
map. For the forest weave, a grid search showed that the deeper corridor (south-east, y ~ -6) is only 1.0-1.15 m
wide, too tight for 1 m after corner rounding; the weave therefore uses the wider entry corridor (y ~ -1), with the
corner waypoints chosen from a small grid of hand-picked candidates by largest clearance.

Clearance requirement on the REFERENCE path (exact 3D distance to the nearest obstacle surface, demo/geometry.py):
indoor >= 2.0 m; forest edge tracks >= 2.0 m; forest weave >= 1.0 m. Checked by check_tracks().

Wind direction (decided: crosswind to the track's main leg): perpendicular to the main leg's horizontal heading,
pointing toward the side where the nearest obstacle lies along that perpendicular (ray from the main leg's midpoint),
i.e. the wind pushes the drone toward obstacles. If neither side hits within 30 m: the left side (+90 deg).
"""
import numpy as np

from path_reference import TimedReference, round_corners

from demo import demo_config

TRACKS = {
    "indoor": {
        "I1_aisle": {
            "label": "straight down the west aisle, slight climb",
            "points": [(-5.0, -6.0, 2.0), (-5.0, 9.8, 2.4)],
            "main_leg": 0, "min_clearance": 2.0,
        },
        "I2_uturn": {
            "label": "U-turn through the south cross-aisle (two 90 deg turns)",
            "points": [(-4.5, -4.0, 2.0), (-4.5, -7.5, 2.0), (4.5, -7.5, 2.0), (4.5, -4.0, 2.0)],
            "main_leg": 1, "min_clearance": 2.0,
        },
        "I3_climb": {
            "label": "zig-zag climb 1.5 -> 3.5 m in the east aisle",
            "points": [(3.3, -3.0, 1.5), (6.0, 1.0, 2.5), (3.3, 5.0, 3.5), (6.0, 9.0, 3.5)],
            "main_leg": None, "min_clearance": 2.0,            # None: overall start -> end heading
        },
    },
    "forest": {
        "F1_north_edge": {
            "label": "straight along the north edge of the forest",
            "points": [(0.0, 14.5, 2.0), (15.8, 14.5, 2.4)],
            "main_leg": 0, "min_clearance": 2.0,
        },
        "F2_sw_corner": {
            "label": "around the south-west corner of the forest (two ~45 deg turns)",
            "points": [(-8.0, -4.5, 2.0), (-8.0, -9.5, 2.1), (-4.5, -13.5, 2.2), (0.8, -13.5, 2.3)],
            "main_leg": 0, "min_clearance": 2.0,
        },
        "F3_weave": {
            "label": "S-curve weaving ~6 m into the forest between trunks and branches at 1.5 m (ends inside)",
            "points": [(-4.6, -4.0, 1.5), (-1.5, -1.2, 1.5), (3.0, -1.2, 1.5), (5.0, 1.2, 1.5)],
            "main_leg": 1, "min_clearance": 1.0,
        },
    },
}


def build_reference(track):
    L = demo_config.TRACK_LIMITS
    points = round_corners(np.asarray(track["points"], dtype=float), L["corner_radius"], L["spacing"])
    return TimedReference(points, L["speed"], L["acceleration"], L["lateral_acceleration"], L["vertical_speed"],
                          start_hold=L["start_hold"], end_hold=L["end_hold"], spacing=L["spacing"])


def main_heading(track):
    p = np.asarray(track["points"], dtype=float)
    k = track["main_leg"]
    a, b = (p[0], p[-1]) if k is None else (p[k], p[k + 1])
    d = (b - a)[:2]
    return d / np.linalg.norm(d), 0.5 * (a + b)


def _ray_hit(obstacles, origin, direction, max_range=30.0):
    """Sphere tracing along a horizontal ray: distance to the first obstacle surface, or None."""
    s = 0.0
    while s < max_range:
        p = origin + s * np.array([direction[0], direction[1], 0.0])
        d = obstacles.clearance(p[None])[0][0]
        if d < 0.02:
            return s
        s += max(d, 0.02)
    return None


def wind_direction(track, obstacles):
    heading, mid = main_heading(track)
    left = np.array([-heading[1], heading[0]])
    hits = {"left": _ray_hit(obstacles, mid, left), "right": _ray_hit(obstacles, mid, -left)}
    if hits["left"] is None and hits["right"] is None:
        side = "left"
    elif hits["right"] is None or (hits["left"] is not None and hits["left"] <= hits["right"]):
        side = "left"
    else:
        side = "right"
    return (left if side == "left" else -left), {"side": side, "hit_distance_m": hits}


def describe(track):
    """In-distribution numbers vs the `full` stage (demo_config.FULL_RANGES)."""
    p = np.asarray(track["points"], dtype=float)
    seg = np.diff(p, axis=0)
    turns = []
    for a, b in zip(seg[:-1, :2], seg[1:, :2]):
        c = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        turns.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    ref = build_reference(track)
    speeds = np.linalg.norm(ref.velocities, axis=1)
    acc = np.linalg.norm(ref.accelerations, axis=1)
    return {
        "segments": int(len(seg)),
        "polyline_length_m": float(np.linalg.norm(seg, axis=1).sum()),
        "path_length_m": float(np.linalg.norm(np.diff(ref.path, axis=0), axis=1).sum()),
        "turns_deg": turns,
        "height_range_m": [float(p[:, 2].min()), float(p[:, 2].max())],
        "height_change_m": float(p[-1, 2] - p[0, 2]),
        "duration_s": float(ref.duration),
        "max_ref_speed_mps": float(speeds.max()),
        "max_ref_accel_mps2": float(acc.max()),
    }


def in_distribution_flags(desc):
    F = demo_config.FULL_RANGES
    flags = []
    if not (F["length_range"][0] <= desc["polyline_length_m"] <= F["length_range"][1]):
        flags.append(f"length {desc['polyline_length_m']:.2f} m outside {F['length_range']}")
    if abs(desc["height_change_m"]) > F["height_change_range"][1]:
        flags.append(f"height change {desc['height_change_m']:.2f} m > {F['height_change_range'][1]}")
    lo, hi = F["height_range"]
    if desc["height_range_m"][0] < lo or desc["height_range_m"][1] > hi:
        flags.append(f"height {desc['height_range_m']} outside {F['height_range']}")
    for t in desc["turns_deg"]:
        if not (F["turn_degrees_range"][0] <= t <= F["turn_degrees_range"][1]):
            flags.append(f"turn {t:.1f} deg outside {F['turn_degrees_range']}")
    if desc["segments"] > 1 and not (F["waypoint_count_range"][0] <= desc["segments"] <= F["waypoint_count_range"][1]):
        flags.append(f"{desc['segments']} segments outside waypoint_count_range {F['waypoint_count_range']}")
    return flags


def check_tracks(scene, obstacles):
    """Per track: reference, description, clearance along the reference path, wind direction."""
    out = {}
    for name, track in TRACKS[scene].items():
        ref = build_reference(track)
        clear, who = obstacles.clearance(ref.path)
        k = int(np.argmin(clear))
        wdir, wmeta = wind_direction(track, obstacles)
        desc = describe(track)
        out[name] = {
            "label": track["label"], "points": [list(map(float, q)) for q in track["points"]],
            "min_clearance_required_m": track["min_clearance"],
            "min_clearance_m": float(clear[k]), "closest_obstacle": obstacles.paths[who[k]],
            "closest_at": ref.path[k].tolist(), "clearance_ok": bool(clear[k] >= track["min_clearance"]),
            "wind_direction_xy": wdir.tolist(), "wind_rule": wmeta, **desc,
            "in_distribution_flags": in_distribution_flags(desc),
        }
    return out
