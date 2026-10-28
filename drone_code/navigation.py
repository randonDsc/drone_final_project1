"""
3D Dijkstra, collision-aware path shortening, and waypoint following.

Search graph
------------
Each free cell connects to its 26 neighbors: every offset in {-1, 0, 1}^3
except (0, 0, 0). Edge cost is the Euclidean distance between cell centers:

    face move   (1 axis changes):  cell_size
    edge move   (2 axes change):   sqrt(2) * cell_size
    corner move (3 axes change):   sqrt(3) * cell_size

Corner cutting
--------------
The closed straight segment between the centers of cell c and c + d crosses
every shared boundary exactly at its midpoint, where all cells
c + e with e_i in {0, d_i} meet. Those 2^k cells (k = number of changed
axes) are exactly the voxels the segment touches, boundary contact included.
A move is allowed only when ALL of them are free. Checking only the
destination cell would let a diagonal slip between two blocked cells that
meet at an edge or corner.

Because a free cell's closed volume never intersects any expanded obstacle
(grid_map.rasterize blocks boundary-touching cells), every allowed edge is
also collision-free in continuous space. plan_flight still re-checks the
reconstructed route with grid_map.segment_free as an independent safeguard.

Path shortening
---------------
After Dijkstra, a greedy pass keeps only the waypoints needed: from each
retained point it jumps to the FARTHEST later route point that has a clear
straight-line connection. The connection is checked continuously against
the expanded obstacle geometry and against every touched occupancy-grid cell.
Both endpoints must lie inside the finite map, which is a box, so the whole
segment does too.

This produces a shorter collision-free route, but it is NOT guaranteed to be
the globally shortest path in continuous space. Shortcuts can only connect
points that already lie on the Dijkstra route, so a better route through a
different homotopy class or around a different side of an obstacle can be
missed. Any-angle planners (e.g. Theta*) or visibility graphs would be
needed for stronger guarantees.
"""

import heapq
import itertools
import time

from collections import deque
from dataclasses import dataclass

import numpy as np

import sim_config

from mapping import find_launch_entry


# All 26 neighbor offsets, in a fixed order shared by every lookup table.
NEIGHBOR_OFFSETS = tuple(
    offset
    for offset in itertools.product((-1, 0, 1), repeat=3)
    if offset != (0, 0, 0)
)


@dataclass
class Plan:
    # World-coordinate commands given to the follower, after shortening.
    waypoints: np.ndarray

    # Complete cell-by-cell Dijkstra route.
    cells: np.ndarray

    # Length of the shortened route including the connection from the
    # actual start (launch departure included).
    distance: float

    # Number of cells expanded during search.
    expanded: int

    # Dijkstra cost between entry cell and goal cell centers, in meters.
    # Useful for comparing against the shortened route length.
    grid_distance: float


def edge_footprint(offset):
    """
    Cell offsets touched by the closed segment between the centers of a
    cell and its neighbor at `offset`.

    For each axis: {0} if the move does not change that axis,
    otherwise {0, d}. The result includes the source (all zeros) and the
    destination (the offset itself).
    """

    return list(
        itertools.product(*[
            (0,) if d == 0 else (0, d)
            for d in offset
        ])
    )


def allowed_moves(free):
    """
    Precompute which of the 26 moves leave each cell without corner cutting.

    Returns a (26, total_cells) boolean array. allowed[k, i] is True when
    the move NEIGHBOR_OFFSETS[k] from flat cell i stays inside the map and
    every cell in its footprint is free.

    Uses about 26 bytes per grid cell (52 MB at MAX_GRID_CELLS = 2,000,000),
    but turns the per-expansion corner check into one table lookup.
    """

    shape = free.shape

    allowed = np.zeros(
        (len(NEIGHBOR_OFFSETS),) + shape,
        dtype=bool,
    )

    for k, offset in enumerate(NEIGHBOR_OFFSETS):
        # Source cells whose neighbor at `offset` is inside the map.
        source = tuple(
            slice(1 if d < 0 else 0, n - 1 if d > 0 else n)
            for d, n in zip(offset, shape)
        )

        ok = np.ones(free[source].shape, dtype=bool)

        # Shift the free mask by each footprint offset and AND them.
        for touched in edge_footprint(offset):
            shifted = tuple(
                slice(s.start + e, s.stop + e)
                for s, e in zip(source, touched)
            )

            ok &= free[shifted]

        allowed[k][source] = ok

    return allowed.reshape(len(NEIGHBOR_OFFSETS), -1)


def dijkstra(
    grid_map,
    start,
    goal,
    max_expanded=None,
    progress=None,
):
    """
    Find a shortest path on the 26-connected occupancy grid.

    Returns:
        cells: array of integer [ix, iy, iz] indices
        distance: grid-path length in meters
        expanded: number of expanded cells
    """

    start_cell = grid_map.require_free(
        start,
        "Search entry",
    )

    goal_cell = grid_map.require_free(
        goal,
        "Destination",
    )

    shape = grid_map.grid.shape
    total = grid_map.grid.size

    # For a flattened C-order array:
    # moving in X changes the flat index by Ny*Nz,
    # moving in Y changes it by Nz,
    # moving in Z changes it by 1.
    strides = (
        shape[1] * shape[2],
        shape[2],
        1,
    )

    # Flat-index change and Euclidean cost of each of the 26 moves.
    flat_offsets = [
        int(np.dot(offset, strides))
        for offset in NEIGHBOR_OFFSETS
    ]

    step_costs = [
        float(np.linalg.norm(offset)) * grid_map.resolution
        for offset in NEIGHBOR_OFFSETS
    ]

    # In-bounds, footprint-free moves for every cell.
    # This replaces both the bounds test and the blocked-neighbor test.
    allowed = allowed_moves(grid_map.grid == 0)

    start_id = int(
        np.ravel_multi_index(start_cell, shape)
    )

    goal_id = int(
        np.ravel_multi_index(goal_cell, shape)
    )

    distance = np.full(total, np.inf)

    parent = np.full(
        total,
        -1,
        dtype=np.int64,
    )

    closed = np.zeros(
        total,
        dtype=bool,
    )

    distance[start_id] = 0.0

    # Heap entries are (distance from start, flat cell index).
    heap = [(0.0, start_id)]

    expanded = 0

    budget = (
        sim_config.MAX_EXPANDED_NODES
        if max_expanded is None
        else max_expanded
    )

    last_progress = time.monotonic()

    while heap:
        cost, current = heapq.heappop(heap)

        if closed[current]:
            continue

        closed[current] = True
        expanded += 1

        if current == goal_id:
            # Follow parent pointers backward, then reverse the chain.
            chain = deque([current])

            while current != start_id:
                current = int(parent[current])
                chain.appendleft(current)

            ids = np.asarray(
                chain,
                dtype=np.int64,
            )

            cells = np.column_stack(
                np.unravel_index(ids, shape)
            )

            return cells, float(cost), expanded

        if expanded >= budget:
            raise RuntimeError(
                f"Dijkstra stopped at {budget:,} expanded cells. "
                "Increase the budget or reduce map size/resolution."
            )

        # Only moves that stay in bounds and touch no blocked cell.
        for k in np.flatnonzero(allowed[:, current]).tolist():
            neighbor = current + flat_offsets[k]

            if closed[neighbor]:
                continue

            candidate = cost + step_costs[k]

            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                parent[neighbor] = current

                heapq.heappush(
                    heap,
                    (candidate, neighbor),
                )

        # Let the caller update the GUI while physics stays paused.
        if (
            progress is not None
            and expanded % 2048 == 0
            and time.monotonic() - last_progress > 0.1
        ):
            progress(expanded)
            last_progress = time.monotonic()

    raise RuntimeError(
        "No path exists within this occupancy grid. "
        "Check obstacle bounds, clearance, and the planning domain."
    )


def line_of_sight(grid_map, start, end):
    """
    Whether the whole straight segment is safe for the drone origin.

    Both checks are conservative and both must pass:
      - continuous test against expanded boxes and planes
      - every occupancy-grid cell touched by the segment is free
    Each raises ValueError if an endpoint lies outside the finite map.
    """

    return (
        grid_map.segment_free(start, end)
        and grid_map.segment_cells_free(start, end)
    )


def shorten_path(grid_map, points):
    """
    Greedy farthest-visible shortcutting of an already valid route.

    From each retained point, connect to the farthest later point with a
    clear straight line. The first and last points are kept exactly.

    Consecutive input points are accepted without re-testing: the caller
    must already have validated them (plan_flight does). This keeps the
    route intact when a connector to an off-center start or goal only
    touches a cell boundary.

    The result is never longer than the input (triangle inequality), but
    it is not necessarily the shortest continuous-space path.
    """

    points = [
        np.asarray(point, dtype=float).reshape(3).copy()
        for point in points
    ]

    if len(points) <= 2:
        return points

    kept = [points[0]]
    last = len(points) - 1
    i = 0

    while i < last:
        # Try the farthest point first and step back until visible.
        j = last

        while (
            j > i + 1
            and not line_of_sight(grid_map, points[i], points[j])
        ):
            j -= 1

        kept.append(points[j])
        i = j

    return kept


def path_length(points):
    return float(sum(
        np.linalg.norm(end_point - start_point)
        for start_point, end_point in zip(
            points[:-1],
            points[1:],
        )
    ))


def plan_flight(
    grid_map,
    geometry,
    initial_position,
    goal,
    progress=None,
):
    """
    Plan from the actual USD starting pose to the exact destination.

    A launch-clearance connection is added only when the initial
    cell is blocked by clearance around the support/floor.

    No configured takeoff altitude is used.

    The launch departure (initial pose -> entry) is kept as its own
    segment. Shortening starts at the entry point, never at the actual
    launch pose, so no shortcut can pass through the platform clearance.
    """

    initial_position = np.asarray(
        initial_position,
        dtype=float,
    )

    goal = np.asarray(
        goal,
        dtype=float,
    )

    entry = find_launch_entry(
        grid_map,
        geometry,
        initial_position,
    )

    cells, grid_distance, expanded = dijkstra(
        grid_map,
        entry,
        goal,
        progress=progress,
    )

    # Convert grid indices to world-coordinate cell centers.
    centers = (
        grid_map.origin
        + (cells.astype(float) + 0.5)
        * grid_map.resolution
    )

    # The exact start/goal usually differ from voxel centers.
    # Check the continuous connectors rather than assuming them safe.
    if not grid_map.segment_free(
        entry,
        centers[0],
    ):
        raise ValueError(
            "Launch-entry-to-grid connector intersects an obstacle."
        )

    if not grid_map.segment_free(
        centers[-1],
        goal,
    ):
        raise ValueError(
            "Grid-to-destination connector intersects an obstacle."
        )

    # Full route: entry, every Dijkstra cell center, exact goal.
    # Keeping every center (not only turns) gives shortening more
    # candidate points to connect to.
    route = []

    for point in [entry, *centers, goal]:
        if (
            not route
            or np.linalg.norm(
                point - route[-1]
            ) > 1e-8
        ):
            route.append(point.copy())

    # Independent continuous check of every Dijkstra edge.
    for start_point, end_point in zip(
        route[:-1],
        route[1:],
    ):
        if not grid_map.segment_free(
            start_point,
            end_point,
        ):
            raise ValueError(
                "Dijkstra route segment intersects expanded obstacle geometry."
            )

    waypoints = shorten_path(
        grid_map,
        route,
    )

    # Validate every ordinary segment against the expanded geometry.
    for start_point, end_point in zip(
        waypoints[:-1],
        waypoints[1:],
    ):
        if not grid_map.segment_free(
            start_point,
            end_point,
        ):
            raise ValueError(
                "Planned segment intersects expanded obstacle geometry."
            )

    # The special launch departure was checked separately by
    # find_launch_entry. Include the actual initial pose as the first
    # follower command.
    if np.linalg.norm(
        waypoints[0] - initial_position
    ) > 1e-8:
        waypoints.insert(
            0,
            initial_position.copy(),
        )

    return Plan(
        waypoints=np.asarray(waypoints),
        cells=cells,
        distance=path_length(waypoints),
        expanded=expanded,
        grid_distance=float(grid_distance),
    )


class WaypointFollower:
    """
    Hold each waypoint until position and velocity have settled.

    The final waypoint stays active indefinitely for hover.
    Timers use simulation time, so pausing does not consume time.

    Segments that both climb/descend and translate are tracked with
    targets that slide along the line (see sim_config TRACK_* settings).
    Settling is still judged against the retained waypoint itself.
    Purely horizontal, purely vertical (including the launch departure)
    and short segments are commanded point-to-point, as before.
    """

    def __init__(self, waypoints, guided=True):
        self.waypoints = np.asarray(
            waypoints,
            dtype=float,
        )

        if (
            self.waypoints.ndim != 2
            or self.waypoints.shape[1] != 3
            or len(self.waypoints) == 0
            or not np.all(np.isfinite(self.waypoints))
        ):
            raise ValueError(
                "Waypoints must be a nonempty finite (N, 3) array."
            )

        # guided=False restores pure point-to-point commands.
        self.guided = guided

        self.index = 0

        self.settled_time = 0.0
        self.active_time = 0.0

        self.done = False

    def command(self, position):
        """
        Controller target for the active segment.

        Returns [x, y, z]. The XY and height components can come from
        different points on the segment, because the controller's
        horizontal and height loops have different speeds.
        """

        target = self.waypoints[self.index]

        if not self.guided or self.index == 0:
            return target.copy()

        start = self.waypoints[self.index - 1]
        delta = target - start

        length = float(np.linalg.norm(delta))
        horizontal = float(np.linalg.norm(delta[:2]))
        vertical = abs(float(delta[2]))

        # Flying point-to-point keeps each axis between its start and end
        # value, so the path stays inside the box spanned by the segment.
        # That box's farthest corner is horizontal*vertical/length from
        # the line. If that is within tolerance, no guidance is needed.
        if (
            length < 1e-9
            or horizontal * vertical / length
            <= sim_config.TRACK_TOLERANCE
        ):
            return target.copy()

        # Progress of each axis along the segment, 0 at start, 1 at end.
        xy_progress = float(np.clip(
            (position[:2] - start[:2]) @ delta[:2] / horizontal**2,
            0.0,
            1.0,
        ))

        z_progress = float(np.clip(
            (position[2] - start[2]) / delta[2],
            0.0,
            1.0,
        ))

        # The lagging axis sets progress, so the faster axis cannot
        # run ahead of the line.
        progress = min(xy_progress, z_progress)

        # Distance from the drone to the segment.
        closest = float(np.clip(
            (position - start) @ delta / length**2,
            0.0,
            1.0,
        ))

        cross_track = np.linalg.norm(
            position - (start + closest * delta)
        )

        # Full lead within TRACK_TOLERANCE, shrinking linearly to zero at
        # TRACK_STOP_DISTANCE. The lead point lies on the line, so aiming
        # at it already steers back; shrinking the lead only slows progress
        # while the drone recovers, instead of halting it at a hard switch.
        lead_scale = float(np.clip(
            (sim_config.TRACK_STOP_DISTANCE - cross_track)
            / (sim_config.TRACK_STOP_DISTANCE - sim_config.TRACK_TOLERANCE),
            0.0,
            1.0,
        ))

        xy_lead = sim_config.TRACK_XY_LEAD * lead_scale
        z_lead = sim_config.TRACK_Z_LEAD * lead_scale

        xy_point = start + min(1.0, progress + xy_lead / length) * delta
        z_point = start + min(1.0, progress + z_lead / length) * delta

        return np.array([
            xy_point[0],
            xy_point[1],
            z_point[2],
        ])

    def update(
        self,
        position,
        velocity,
        dt,
    ):
        position = np.asarray(
            position,
            dtype=float,
        )

        velocity = np.asarray(
            velocity,
            dtype=float,
        )

        self.active_time += dt

        position_error = np.linalg.norm(
            self.waypoints[self.index]
            - position
        )

        speed = np.linalg.norm(velocity)

        settled = (
            position_error <= sim_config.WAYPOINT_RADIUS
            and speed <= sim_config.WAYPOINT_SPEED
        )

        if settled:
            self.settled_time += dt
        else:
            self.settled_time = 0.0

        if self.settled_time >= sim_config.WAYPOINT_DWELL:
            if self.index + 1 < len(self.waypoints):
                self.index += 1

                self.settled_time = 0.0
                self.active_time = 0.0

            else:
                self.done = True

        if (
            not self.done
            and self.active_time > sim_config.WAYPOINT_TIMEOUT
        ):
            raise RuntimeError(
                f"Waypoint {self.index + 1} timed out. "
                "Inspect flight_log.txt and controller tracking."
            )

        return self.command(position)
