"""Simulator-independent voxel map and conservative segment collision checks.

This module needs only NumPy. USD reading lives separately in mapping.py so
Dijkstra, coordinate conversions and waypoint logic can be tested without Sim.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    """World-axis-aligned obstacle bounds, in meters, plus the source USD path."""
    lower: np.ndarray
    upper: np.ndarray
    path: str


@dataclass
class Plane:
    """Infinite solid half-space: normal dot position <= offset is blocked.

    The unit normal points toward free space. mapping.py rejects tilted planes
    for now, rather than incorrectly turning them into horizontal floors.
    """
    normal: np.ndarray
    offset: float
    path: str


@dataclass
class GridMap:
    """grid[ix, iy, iz] == 1 means the entire cell is unavailable to the drone.

    origin: minimum world corner; resolution: cube edge in meters.
    Obstacles stored here are ALREADY expanded for the vehicle and clearance.
    A cell is blocked whenever it intersects an expanded obstacle. This makes
    discretization conservative, at the expense of some additional lost space.
    """
    grid: np.ndarray
    origin: np.ndarray
    resolution: float
    boxes: list
    planes: list

    @property
    def upper(self):
        return self.origin + np.asarray(self.grid.shape) * self.resolution

    def index(self, position):
        """World meters -> integer cell. Never clip an out-of-map destination."""
        p = np.asarray(position, dtype=float).reshape(3)
        if not np.all(np.isfinite(p)):
            raise ValueError('Position must contain three finite coordinates.')
        ijk = np.floor((p - self.origin) / self.resolution).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= self.grid.shape):
            raise ValueError(f'Position outside map: {p}; bounds {self.origin}, {self.upper}')
        return tuple(int(v) for v in ijk)

    def center(self, index):
        """Cell -> center in world meters, not its lower corner."""
        return self.origin + (np.asarray(index, dtype=float) + 0.5) * self.resolution

    def require_free(self, position, label):
        ijk = self.index(position)
        if self.grid[ijk]:
            raise ValueError(
                f'{label} {np.asarray(position)} maps to blocked cell {ijk}. '
                'Inspect occupancy_obstacles.txt and change the target/scene; '
                'the planner will not clear cells or move the target automatically.'
            )
        return ijk

    def segment_free(self, start, end):
        """Check an entire straight segment against expanded obstacle geometry.

        Endpoint checks alone miss thin walls. The slab algorithm below tests
        the continuous line against every AABB; a plane uses its signed distance.
        Also require both endpoints inside this rectangular planning domain.
        """
        a, b = np.asarray(start, float), np.asarray(end, float)
        self.index(a)
        self.index(b)
        for box in self.boxes:
            if segment_hits_box(a, b, box.lower, box.upper):
                return False
        for plane in self.planes:
            if min(a @ plane.normal, b @ plane.normal) <= plane.offset:
                return False
        return True


    def segment_cells_free(self, start, end):
        """Check a straight segment against the occupancy grid itself.

        Every blocked voxel whose CLOSED volume touches the segment counts,
        including cells that only share a face, edge or corner with it. Path
        shortening uses this together with segment_free(), so a shortcut can
        never cross a blocked cell, even one with no matching box (e.g.
        cells blocked directly in tests).
        """
        a, b = np.asarray(start, float).reshape(3), np.asarray(end, float).reshape(3)
        self.index(a)
        self.index(b)
        shape = np.asarray(self.grid.shape)

        # Candidate window: the segment's cell bounding box, widened by one
        # cell wherever the segment lies exactly on a voxel boundary.
        low = (np.minimum(a, b) - self.origin) / self.resolution
        high = (np.maximum(a, b) - self.origin) / self.resolution
        first = np.clip(np.floor(low - 1e-9).astype(int), 0, shape - 1)
        last = np.clip(np.floor(high + 1e-9).astype(int), 0, shape - 1)
        window = self.grid[tuple(slice(int(f), int(l) + 1) for f, l in zip(first, last))]

        blocked = np.argwhere(window)
        if len(blocked) == 0:
            return True

        # Exact closed segment / closed box test against each blocked cell,
        # with a tiny tolerance so floating-point error cannot hide a touch.
        lower = self.origin + (blocked + first) * self.resolution - 1e-9
        upper = lower + self.resolution + 2e-9
        return not np.any(segment_hits_boxes(a, b, lower, upper))


def segment_hits_boxes(a, b, lower, upper):
    """Vectorized segment_hits_box over (M, 3) arrays of box bounds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    direction = b - a
    count = len(lower)
    t_enter, t_exit = np.zeros(count), np.ones(count)
    hit = np.ones(count, dtype=bool)
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            hit &= (a[axis] >= lower[:, axis]) & (a[axis] <= upper[:, axis])
        else:
            t1 = (lower[:, axis] - a[axis]) / direction[axis]
            t2 = (upper[:, axis] - a[axis]) / direction[axis]
            t_enter = np.maximum(t_enter, np.minimum(t1, t2))
            t_exit = np.minimum(t_exit, np.maximum(t1, t2))
    return hit & (t_enter <= t_exit)


def segment_hits_box(a, b, lower, upper):
    """Closed segment / closed AABB intersection; touching counts as blocked."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    direction = b - a
    t_enter, t_exit = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if a[axis] < lower[axis] or a[axis] > upper[axis]:
                return False
        else:
            t1 = (lower[axis] - a[axis]) / direction[axis]
            t2 = (upper[axis] - a[axis]) / direction[axis]
            t_enter = max(t_enter, min(t1, t2))
            t_exit = min(t_exit, max(t1, t2))
            if t_enter > t_exit:
                return False
    return True


def rasterize(origin, upper, resolution, boxes, planes, max_cells):
    """Create a uint8 occupancy array from ALREADY expanded obstacles."""
    origin, upper = np.asarray(origin, float), np.asarray(upper, float)
    if not np.isfinite(resolution) or resolution <= 0:
        raise ValueError('CELL_SIZE must be positive and finite.')
    if not np.all(np.isfinite([origin, upper])) or np.any(upper <= origin):
        raise ValueError('Map bounds must be finite with upper > origin.')
    shape = np.ceil((upper - origin) / resolution).astype(int)
    count = int(shape[0]) * int(shape[1]) * int(shape[2])
    if count > max_cells:
        raise ValueError(
            f'Map would contain {count:,} cells, shape {tuple(shape)}. '
            'Inspect scene scale/large geometry, increase CELL_SIZE, '
            'or provide explicit map bounds; no silent coarsening is applied.'
        )
    grid = np.zeros(tuple(shape), dtype=np.uint8)
    for box in boxes:
        # Inclusive boundaries: a zero-thickness obstacle on a voxel face must
        # block cells on both sides too. nextafter handles exact face equality.
        lower_index = np.floor(np.nextafter(
            (box.lower - origin) / resolution, -np.inf)).astype(int)
        upper_index = np.floor((box.upper - origin) / resolution).astype(int) + 1
        start = np.clip(lower_index, 0, shape)
        stop = np.clip(upper_index, 0, shape)
        grid[tuple(slice(int(a), int(b)) for a, b in zip(start, stop))] = 1

    # Planes are axis aligned. Mark all cells whose closed volume intersects
    # the solid side; do this per axis without allocating an XYZ meshgrid.
    for plane in planes:
        axis = int(np.argmax(np.abs(plane.normal)))
        sign = plane.normal[axis]
        low = origin[axis] + np.arange(shape[axis]) * resolution
        high = low + resolution
        minimum = np.minimum(sign * low, sign * high)
        blocked = np.flatnonzero(minimum <= plane.offset)
        slices = [slice(None)] * 3
        slices[axis] = blocked
        grid[tuple(slices)] = 1
    return GridMap(grid, origin, float(resolution), boxes, planes)
