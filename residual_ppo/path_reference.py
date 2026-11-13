"""Timed flight references: routes, speed profiles, and path geometry.

NumPy only. Training, evaluation, and navigation all fly a TimedReference, so they
share one convention:

    p_ref(t), v_ref(t), a_ref(t), all world frame, from one speed profile

A route is a polyline. Its speed profile respects a speed limit, a tangential
acceleration limit, a lateral acceleration limit from curvature, and a vertical
speed limit, starting and ending at rest. v_ref and a_ref are computed from that
profile, so they are the true derivatives of p_ref along the path, not estimates.

The polyline's own points are kept exactly. A reference never cuts inside a route,
which is what makes checking a route's polyline enough for Dijkstra paths.
"""
from collections import namedtuple

import numpy as np


# finished: the motion is over and the reference is holding its last point.
ReferenceSample = namedtuple(
    "ReferenceSample",
    "position velocity acceleration finished",
)


def densify(points, spacing):
    """Add points along each segment; every original point stays exactly."""
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    dense = [points[0]]

    for start, end in zip(points[:-1], points[1:]):
        length = np.linalg.norm(end - start)

        if length < 1e-9:
            continue

        count = max(1, int(np.ceil(length / spacing)))

        for k in range(1, count + 1):
            dense.append(start + (end - start) * k / count)

    return np.array(dense)


class TimedReference:
    def __init__(
        self,
        points,
        speed,
        acceleration,
        lateral_acceleration,
        vertical_speed,
        start_hold=0.0,
        end_hold=0.0,
        spacing=0.05,
        min_speed=0.05,
    ):
        """speed, acceleration, ...: limits [m/s, m/s^2]. Holds are in seconds."""
        self.path = densify(points, spacing)
        count = len(self.path)

        if count < 2:
            self.times = np.array([float(start_hold)])
            self.velocities = np.zeros((1, 3))
            self.accelerations = np.zeros((1, 3))
            self.motion_end = float(start_hold)
            self.duration = self.motion_end + float(end_hold)
            return

        segments = np.diff(self.path, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        units = segments / lengths[:, None]

        # Tangent and curvature vector at each point. Curvature is the change of
        # direction per meter, so a sharp corner gives a large value there.
        tangents = np.empty_like(self.path)
        tangents[0] = units[0]
        tangents[-1] = units[-1]

        between = units[:-1] + units[1:]
        sizes = np.linalg.norm(between, axis=1, keepdims=True)
        tangents[1:-1] = np.where(
            sizes > 1e-6, between / np.maximum(sizes, 1e-6), units[1:]
        )

        bends = np.zeros_like(self.path)
        bends[1:-1] = (units[1:] - units[:-1]) / (
            0.5 * (lengths[:-1] + lengths[1:])
        )[:, None]

        curvature = np.linalg.norm(bends, axis=1)

        limit = np.minimum.reduce([
            np.full(count, float(speed)),
            np.sqrt(lateral_acceleration / np.maximum(curvature, 1e-9)),
            vertical_speed / np.maximum(np.abs(tangents[:, 2]), 1e-9),
        ])
        limit = np.maximum(limit, min_speed)

        # Forward then backward pass: v^2 changes by at most 2 a ds per segment.
        speeds = limit.copy()
        speeds[0] = 0.0
        speeds[-1] = 0.0

        for i in range(count - 1):
            speeds[i + 1] = min(
                speeds[i + 1],
                np.sqrt(speeds[i]**2 + 2.0 * acceleration * lengths[i]),
            )

        for i in range(count - 2, -1, -1):
            speeds[i] = min(
                speeds[i],
                np.sqrt(speeds[i + 1]**2 + 2.0 * acceleration * lengths[i]),
            )

        durations = 2.0 * lengths / (speeds[:-1] + speeds[1:])

        self.times = float(start_hold) + np.concatenate(
            [[0.0], np.cumsum(durations)]
        )

        tangential = (speeds[1:]**2 - speeds[:-1]**2) / (2.0 * lengths)
        along = np.zeros(count)
        along[1:-1] = 0.5 * (tangential[:-1] + tangential[1:])

        self.velocities = tangents * speeds[:, None]
        self.accelerations = (
            tangents * along[:, None]
            + bends * (speeds**2)[:, None]
        )

        self.motion_end = float(self.times[-1])
        self.duration = self.motion_end + float(end_hold)

    @property
    def final_position(self):
        return self.path[-1]

    def sample(self, time):
        """The reference at a time [s]. Before and after the motion it holds."""
        times = self.times

        if len(times) == 1 or time <= times[0]:
            return ReferenceSample(
                self.path[0].copy(), np.zeros(3), np.zeros(3), time >= times[-1]
            )

        if time >= times[-1]:
            return ReferenceSample(
                self.path[-1].copy(), np.zeros(3), np.zeros(3), True
            )

        i = int(np.searchsorted(times, time, side="right")) - 1
        share = (time - times[i]) / (times[i + 1] - times[i])

        def between_points(values):
            return values[i] + share * (values[i + 1] - values[i])

        return ReferenceSample(
            between_points(self.path),
            between_points(self.velocities),
            between_points(self.accelerations),
            False,
        )

    def time_at_point(self, point):
        """Reference time at the path point nearest to point [s]."""
        distances = np.linalg.norm(self.path - np.asarray(point), axis=1)

        return float(self.times[min(int(np.argmin(distances)), len(self.times) - 1)])


def distances_to_path(positions, path):
    """Geometric distance from each position to a polyline [m]."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 3)
    path = np.asarray(path, dtype=float).reshape(-1, 3)

    if len(path) < 2:
        return np.linalg.norm(positions - path[0], axis=1)

    starts = path[:-1]
    segments = path[1:] - starts
    squared = np.maximum(np.sum(segments**2, axis=1), 1e-12)
    result = np.empty(len(positions))

    for index, position in enumerate(positions):
        share = np.clip(
            np.sum((position - starts) * segments, axis=1) / squared,
            0.0,
            1.0,
        )
        closest = starts + share[:, None] * segments
        result[index] = np.sqrt(np.min(np.sum((position - closest)**2, axis=1)))

    return result


def round_corners(points, radius, spacing=0.05, is_safe=None, locked=()):
    """Replace corners with quadratic Bezier arcs, where that is safe.

    radius: largest distance from a corner where the arc begins [m]. It is also
        limited to 45% of either neighboring segment, so arcs never overlap.
    spacing: arcs get chords about this long [m]. Longer chords would bunch the
        curvature at their joints, and the speed profile would pulse.
    is_safe(start, end): checks one chord. Each corner tries its full radius, then
        half of that twice, then stays sharp. The arc is only ever accepted whole.
    locked: corner indices that always stay sharp.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)

    if len(points) < 3 or radius <= 0.0:
        return points.copy()

    result = [points[0]]

    for i in range(1, len(points) - 1):
        corner = points[i]
        before = points[i - 1] - corner
        after = points[i + 1] - corner
        before_length = np.linalg.norm(before)
        after_length = np.linalg.norm(after)

        arc = None

        if i not in locked and min(before_length, after_length) > 1e-9:
            cut = min(radius, 0.45 * before_length, 0.45 * after_length)

            for _ in range(3):
                entry = corner + before / before_length * cut
                leave = corner + after / after_length * cut

                chords = max(4, int(np.ceil(2.0 * cut / spacing)))
                shares = np.linspace(0.0, 1.0, chords + 1)[:, None]
                candidate = (
                    (1.0 - shares)**2 * entry
                    + 2.0 * (1.0 - shares) * shares * corner
                    + shares**2 * leave
                )

                if is_safe is None or all(
                    is_safe(a, b)
                    for a, b in zip(candidate[:-1], candidate[1:])
                ):
                    arc = candidate
                    break

                cut *= 0.5

        if arc is None:
            result.append(corner)
        else:
            result.extend(arc)

    result.append(points[-1])

    return np.array(result)


def _route_points(rng, kind, start, heading, route, spacing):
    """Polyline for one route type, starting at start."""
    length = float(rng.uniform(*route["length_range"]))
    height_change = float(rng.uniform(*route["height_change_range"]))
    height_change *= 1.0 if rng.random() < 0.5 else -1.0

    def direction(angle):
        return np.array([np.cos(angle), np.sin(angle), 0.0])

    if kind == "hover":
        return start[None, :]

    if kind == "straight":
        # Mostly level; altitude routes cover real climbs.
        return np.array([
            start,
            start + length * direction(heading)
            + np.array([0.0, 0.0, 0.25 * height_change]),
        ])

    if kind == "altitude":
        return np.array([
            start,
            start + 0.3 * length * direction(heading)
            + np.array([0.0, 0.0, height_change]),
        ])

    if kind == "waypoints":
        low, high = route["waypoint_count_range"]
        count = int(rng.integers(low, high + 1))
        points = [start]
        segment = length / count

        for _ in range(count):
            step = segment * direction(heading)
            step[2] = height_change / count
            points.append(points[-1] + step)

            turn = np.deg2rad(rng.uniform(*route["turn_degrees_range"]))
            heading += turn if rng.random() < 0.5 else -turn

        return round_corners(
            np.array(points),
            float(rng.uniform(*route["corner_radius_range"])),
            spacing,
        )

    radius = float(rng.uniform(*route["loop_radius_range"]))
    sign = 1.0 if rng.random() < 0.5 else -1.0

    # Dense enough that chords stay below spacing on the longest loop, a figure
    # eight of about 12 radii, so curvature stays even along it.
    angles = np.linspace(
        0.0, 2.0 * np.pi, int(np.ceil(12.0 * radius / spacing)) + 1
    )

    if kind == "circle":
        # Starts on the circle, moving along it. A slight climb makes it a helix.
        local = np.column_stack([
            radius * np.sin(angles),
            sign * radius * (1.0 - np.cos(angles)),
        ])
    elif kind == "figure_eight":
        local = np.column_stack([
            1.5 * radius * np.sin(angles),
            sign * 0.75 * radius * np.sin(2.0 * angles),
        ])
    else:
        raise ValueError(f"Unknown route type: {kind}")

    c, s = np.cos(heading), np.sin(heading)

    return np.column_stack([
        start[0] + c * local[:, 0] - s * local[:, 1],
        start[1] + s * local[:, 0] + c * local[:, 1],
        start[2] + 0.25 * height_change * angles / (2.0 * np.pi),
    ])


def sample_reference(rng, route, episode, route_type=None):
    """Draw one route from a curriculum entry; return (kind, TimedReference).

    route: a stage's "route" settings. episode: path_sac_config.EPISODE.
    route_type: a name to always use, for evaluation. It draws nothing from
    weights, so a seed still fixes everything else.
    """
    kinds = list(route["weights"])
    weights = np.array([route["weights"][kind] for kind in kinds], dtype=float)
    chosen = str(rng.choice(kinds, p=weights / weights.sum()))
    kind = chosen if route_type is None else route_type

    low, high = episode["height_range"]
    limit = float(episode["start_xy"])

    start = np.array([
        rng.uniform(-limit, limit),
        rng.uniform(-limit, limit),
        rng.uniform(low, high),
    ])
    heading = float(rng.uniform(-np.pi, np.pi))

    speed = float(rng.uniform(*route["speed_range"]))
    acceleration = float(rng.uniform(*route["acceleration_range"]))
    hover_seconds = float(rng.uniform(*route["hover_seconds_range"]))

    points = _route_points(
        rng, kind, start, heading, route, episode["spacing"]
    )

    # Keep the whole route inside the height range, moving it, not squashing it.
    points = points.copy()
    points[:, 2] += max(low - points[:, 2].min(), 0.0)
    points[:, 2] -= max(points[:, 2].max() - high, 0.0)

    return kind, TimedReference(
        points,
        speed,
        acceleration,
        route["lateral_acceleration"],
        route["vertical_speed"],
        start_hold=hover_seconds if kind == "hover" else 0.5,
        end_hold=episode["settle_seconds"],
        spacing=episode["spacing"],
    )
