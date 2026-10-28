"""
Read a static USD scene into a conservative 3D occupancy grid.

Finite obstacles use world-axis-aligned bounding boxes.
Infinite axis-aligned planes use solid half-spaces.

The drone is excluded from obstacles. Its dimensions are used
to expand obstacles so the planner can treat the drone origin
as a point.

Bounding boxes are conservative:
a complex mesh can block empty space inside its bounding box.
"""

from dataclasses import dataclass

import numpy as np

import sim_config

from grid_map import (
    Box,
    Plane,
    rasterize,
    segment_hits_box,
)


@dataclass
class SceneGeometry:
    """Raw scene geometry before obstacle inflation."""

    boxes: list
    planes: list

    # Conservative half-size around the drone's body origin.
    drone_half_size: np.ndarray

    # Lowest geometry point relative to body origin.
    drone_bottom_offset: float

    # Human-readable obstacle paths and bounds.
    report: list


def _under(path, root):
    """Whether a USD path is the root or one of its descendants."""
    return (
        path == root
        or path.startswith(root.rstrip("/") + "/")
    )


def read_scene_geometry(stage, body_position):
    """
    Read obstacle bounds and estimate the drone's dimensions.

    Call after loading/resetting the stage, before advancing physics.
    At that point the USD geometry and initial simulator pose should agree.
    """

    # These imports require USD, which Isaac Sim provides.
    from pxr import Usd, UsdGeom, UsdPhysics, Gf

    if not np.isclose(
        UsdGeom.GetStageMetersPerUnit(stage),
        1.0,
    ):
        raise ValueError(
            "This controller requires a meter-based USD scene."
        )

    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError(
            "This controller requires a Z-up USD scene."
        )

    time = Usd.TimeCode.Default()

    cache = UsdGeom.BBoxCache(
        time,
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=False,
    )

    transforms = UsdGeom.XformCache(time)

    def bounds(prim):
        """Return this prim's world-axis-aligned minimum/maximum."""

        result = (
            cache
            .ComputeWorldBound(prim)
            .ComputeAlignedRange()
        )

        if result.IsEmpty():
            raise ValueError(
                f"Cannot determine geometry bounds: {prim.GetPath()}"
            )

        lower = np.array(result.GetMin(), dtype=float)
        upper = np.array(result.GetMax(), dtype=float)

        if not np.all(np.isfinite([lower, upper])):
            raise ValueError(
                f"Nonfinite geometry bounds: {prim.GetPath()}"
            )

        return lower, upper

    # ---------- Estimate drone envelope ----------

    drone_root = stage.GetPrimAtPath(sim_config.DRONE_ROOT)

    if not drone_root.IsValid():
        raise ValueError(
            f"Drone root missing: {sim_config.DRONE_ROOT}"
        )

    drone_lower, drone_upper = bounds(drone_root)

    body_position = np.asarray(
        body_position,
        dtype=float,
    ).reshape(3)

    relative_lower = drone_lower - body_position
    relative_upper = drone_upper - body_position

    if (
        np.any(relative_lower > 0)
        or np.any(relative_upper < 0)
    ):
        raise ValueError(
            "Drone body position is outside its USD geometry bounds. "
            "Check root/body transforms before planning."
        )

    # Maximum extent on each side of the body origin.
    extent = np.maximum(
        np.abs(relative_lower),
        np.abs(relative_upper),
    )

    # This horizontal radius encloses the XY bounding rectangle,
    # allowing the vehicle to yaw without losing clearance.
    radius = float(np.hypot(extent[0], extent[1]))

    tilt = np.deg2rad(sim_config.MAX_TILT_DEGREES)

    # Reserve additional space for the controller's allowed tilt.
    half_size = np.array([
        radius + extent[2] * np.sin(tilt),
        radius + extent[2] * np.sin(tilt),
        extent[2] + radius * np.sin(tilt),
    ])

    if (
        not np.all(np.isfinite(half_size))
        or np.any(half_size <= 0)
    ):
        raise ValueError(
            "Cannot estimate drone dimensions from its geometry."
        )

    # ---------- Read scene obstacles ----------

    boxes = []
    planes = []
    report = []

    # Include descendants of ordinary USD instances.
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())

        # Never map the drone itself as an obstacle.
        if any(
            _under(path, root)
            for root in sim_config.EXCLUDE_ROOTS
        ):
            continue

        # Point instancers need per-instance geometry handling.
        # Refuse them instead of silently producing an incomplete map.
        if prim.IsA(UsdGeom.PointInstancer):
            raise ValueError(
                f"PointInstancer unsupported by AABB mapper: {path}. "
                "Convert environment instances to ordinary USD prims."
            )

        has_collision = prim.HasAPI(UsdPhysics.CollisionAPI)

        collision_enabled = has_collision and (
            UsdPhysics.CollisionAPI(prim)
            .GetCollisionEnabledAttr()
            .Get()
            is not False
        )

        explicitly_included = any(
            _under(path, root)
            for root in sim_config.EXTRA_OBSTACLE_ROOTS
        )

        if not collision_enabled:
            if not (
                explicitly_included
                and prim.IsA(UsdGeom.Boundable)
            ):
                continue

        if prim.GetTypeName() == "Plane":
            # Infinite planes cannot be represented with finite boxes.
            # Extract their world normal and a point on the plane.
            axis_token = prim.GetAttribute("axis").Get() or "Z"

            axis = {
                "X": 0,
                "Y": 1,
                "Z": 2,
            }[str(axis_token).upper()]

            local_normal = np.zeros(3)
            local_normal[axis] = 1.0

            matrix = transforms.GetLocalToWorldTransform(prim)

            # Normals transform using inverse transpose.
            normal = np.array(
                matrix
                .GetInverse()
                .GetTranspose()
                .TransformDir(Gf.Vec3d(*local_normal)),
                dtype=float,
            )

            normal /= np.linalg.norm(normal)

            dominant_axis = int(
                np.argmax(np.abs(normal))
            )

            # This rasterizer currently supports axis-aligned planes.
            if not np.isclose(
                abs(normal[dominant_axis]),
                1.0,
                atol=1e-6,
            ):
                raise ValueError(
                    f"Tilted infinite plane unsupported: {path}"
                )

            # Remove small floating-point components.
            normal = (
                np.eye(3)[dominant_axis]
                * np.sign(normal[dominant_axis])
            )

            point = np.array(
                matrix.Transform(Gf.Vec3d(0, 0, 0)),
                dtype=float,
            )

            offset = float(normal @ point)

            planes.append(
                Plane(normal, offset, path)
            )

            report.append(
                f"PLANE {path}: "
                f"normal={normal}, offset={offset}"
            )

        else:
            if not prim.IsA(UsdGeom.Boundable):
                raise ValueError(
                    "Collision prim has no supported boundable geometry: "
                    f"{path}"
                )

            lower, upper = bounds(prim)

            boxes.append(
                Box(lower, upper, path)
            )

            report.append(
                f"BOX {path}: min={lower}, max={upper}"
            )

    if not boxes and not planes:
        raise ValueError(
            "No obstacle collision geometry found. "
            "Check collision APIs or EXTRA_OBSTACLE_ROOTS."
        )

    return SceneGeometry(
        boxes=boxes,
        planes=planes,
        drone_half_size=half_size,
        drone_bottom_offset=float(relative_lower[2]),
        report=report,
    )


def build_occupancy_map(
    geometry,
    start,
    goal,
    bounds=None,
):
    """
    Choose finite planning bounds and build the 3D NumPy grid.

    Obstacles are expanded by:
        drone envelope + safety margin

    The resulting map describes where the drone's body origin may go.
    """

    points = [
        np.asarray(start, dtype=float),
        np.asarray(goal, dtype=float),
    ]

    if not np.all(np.isfinite(points)):
        raise ValueError(
            "Start and goal must be finite XYZ coordinates."
        )

    if bounds is None:
        for box in geometry.boxes:
            size = box.upper - box.lower

            large_floor = (
                size[2] < 1.0
                and max(size[0], size[1])
                >= sim_config.LARGE_FLOOR_WIDTH
            )

            # Large floors remain obstacles. They simply do not force
            # the search area to be hundreds of meters wide.
            if not large_floor:
                points.extend([
                    box.lower,
                    box.upper,
                ])

        origin = (
            np.min(points, axis=0)
            - sim_config.MAP_PADDING
        )

        upper = (
            np.max(points, axis=0)
            + sim_config.MAP_PADDING
        )

        upper[2] += sim_config.MAP_HEADROOM

    else:
        origin, upper = [
            np.asarray(value, dtype=float)
            for value in bounds
        ]

    # Align map edges to global voxel boundaries.
    origin = (
        np.floor(origin / sim_config.CELL_SIZE)
        * sim_config.CELL_SIZE
    )

    upper = (
        np.ceil(upper / sim_config.CELL_SIZE)
        * sim_config.CELL_SIZE
    )

    clearance = (
        geometry.drone_half_size
        + sim_config.SAFETY_MARGIN
    )

    expanded_boxes = [
        Box(
            box.lower - clearance,
            box.upper + clearance,
            box.path,
        )
        for box in geometry.boxes
    ]

    expanded_planes = [
        Plane(
            plane.normal,
            plane.offset
            + float(np.abs(plane.normal) @ clearance),
            plane.path,
        )
        for plane in geometry.planes
    ]

    return rasterize(
        origin,
        upper,
        sim_config.CELL_SIZE,
        expanded_boxes,
        expanded_planes,
        sim_config.MAX_GRID_CELLS,
    )


def validate_launch_departure(
    geometry,
    start,
    entry,
):
    """
    Check an initial upward exit from launch clearance.

    A drone resting on its stand starts inside that stand's
    expanded clearance region. Allow upward departure from the
    confirmed stand, while retaining checks against other obstacles.

    This does not remove obstacles from the map.
    """

    start = np.asarray(start, dtype=float)
    entry = np.asarray(entry, dtype=float)

    if (
        not np.allclose(start[:2], entry[:2])
        or entry[2] <= start[2]
    ):
        raise ValueError(
            "Initial departure from blocked clearance must be upward."
        )

    bottom = (
        start[2]
        + geometry.drone_bottom_offset
    )

    clearance = (
        geometry.drone_half_size
        + sim_config.SAFETY_MARGIN
    )

    for box in geometry.boxes:
        supports_xy = (
            np.all(start[:2] >= box.lower[:2])
            and np.all(start[:2] <= box.upper[:2])
        )

        named_support = any(
            _under(box.path, root)
            for root in sim_config.LAUNCH_SUPPORT_ROOTS
        )

        # For the confirmed small launch stand, compare its top
        # against body origin. The vehicle-wide geometry bottom
        # need not be the stand's contact location.
        #
        # For other supports, preserve the stricter bottom check.
        support_top_limit = (
            start[2]
            if named_support
            else bottom + 0.02
        )

        departure_support = (
            supports_xy
            and box.upper[2] <= support_top_limit
        )

        if departure_support:
            continue

        if segment_hits_box(
            start,
            entry,
            box.lower - clearance,
            box.upper + clearance,
        ):
            raise ValueError(
                f"Initial departure blocked by {box.path}"
            )

    for plane in geometry.planes:
        # Permit upward departure from a horizontal floor beneath
        # body origin, including while starting on the small stand.
        departure_support = (
            plane.normal[2] > 0.99
            and plane.offset <= start[2]
        )

        if departure_support:
            continue

        expanded_offset = (
            plane.offset
            + float(np.abs(plane.normal) @ clearance)
        )

        minimum_projection = min(
            start @ plane.normal,
            entry @ plane.normal,
        )

        if minimum_projection <= expanded_offset:
            raise ValueError(
                f"Initial departure blocked by {plane.path}"
            )


def find_launch_entry(
    grid_map,
    geometry,
    start,
):
    """
    Connect the actual initial pose to the free voxel graph.

    If the initial cell is free, return the actual position.

    Otherwise, find the lowest free cell directly above it and
    validate that upward departure. Its height is computed from
    the map, not specified as a takeoff target.
    """

    start = np.asarray(
        start,
        dtype=float,
    ).copy()

    ix, iy, iz = grid_map.index(start)

    if grid_map.grid[ix, iy, iz] == 0:
        return start

    for k in range(
        iz + 1,
        grid_map.grid.shape[2],
    ):
        if grid_map.grid[ix, iy, k] != 0:
            continue

        entry = start.copy()

        # Keep the exact starting XY. Only Z changes.
        entry[2] = grid_map.center((ix, iy, k))[2]

        validate_launch_departure(
            geometry,
            start,
            entry,
        )

        return entry

    raise ValueError(
        "No free upward exit above initial position within map bounds. "
        "Inspect overhead geometry or increase MAP_HEADROOM."
    )