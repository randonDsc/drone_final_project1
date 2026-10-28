"""Rebuild dij3d.usd's warehouse as plain collision boxes.

Run with Isaac Sim's python (it needs Kit to download the warehouse asset):

    C:\\isaacsim\\python.bat sim_scene\\build_box_scene.py

Input:  sim_scene/dij3d.usd
Output: sim_scene/dij3d_boxes.usd

Everything except /World/warehouse_multiple_shelves is kept exactly as
authored: the iris drone reference, the launch pad /World/Cube, the ground
plane, the physics scene, and the light.

Each collision-enabled warehouse prim becomes UsdGeom.Cube colliders under
/World/BoxWarehouse, with no materials or visual detail:

- Most prims: one cube equal to the prim's world-axis-aligned box. This is
  exactly what mapping.read_scene_geometry already used for them.
- Prims whose box would hide drone-sized open space (the 9 m pillar frames,
  L-shaped corner walls, ...): their real triangles are voxelized at
  FINE_RES, the interior filled, and the solid voxels merged into a few
  boxes. They keep their true shape, slightly thickened (<= FINE_RES).

Collision then matches the planner's map exactly: both see the same boxes.

Lights under /World/Lights are visual only (no collider, so the mapper
ignores them): one sphere light under each ceiling lamp, plus a dim dome
light for fill, because the roof blocks the scene's distant light.
"""
from pathlib import Path
import re
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402

SCENE_DIR = Path(__file__).resolve().parent
SOURCE = SCENE_DIR / "dij3d.usd"
OUTPUT = SCENE_DIR / "dij3d_boxes.usd"
WAREHOUSE = "/World/warehouse_multiple_shelves"
BOX_ROOT = "/World/BoxWarehouse"

# Hollowness test, matching the planner: 0.25 m cells, open space counts as
# hidden when it is ~0.5 m from every surface (about the drone half-size).
CHECK_RES = 0.25
NEAR_CELLS = 2
HOLLOW_LIMIT = 0.25  # m^3 of hidden open space before a prim is rebuilt

FINE_RES = 0.10  # voxel size used to rebuild hollow prims [m]
MIN_THICKNESS = 0.02  # zero-thickness wall sheets get this thickness [m]

COLORS = {
    "Shelf": (0.85, 0.55, 0.20),
    "Pillar": (0.35, 0.35, 0.40),
    "Wall": (0.75, 0.75, 0.75),
    "Ceiling": (0.60, 0.60, 0.60),
}
DEFAULT_COLOR = (0.65, 0.50, 0.35)

# Lighting (visual only). Brightness is a starting guess: raise or lower
# the intensities if the view is too dark or washed out.
LIGHT_ROOT = "/World/Lights"
LAMP_RADIUS = 0.3  # [m]
LAMP_INTENSITY = 5.0e4
LAMP_DROP = 0.3  # light sits this far below the lamp's lowest point [m]
DOME_INTENSITY = 300.0


# ---------- Mesh sampling and voxels ----------

def mesh_triangles(mesh, xforms):
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        return None
    matrix = np.array(
        xforms.GetLocalToWorldTransform(mesh.GetPrim()), dtype=float
    )
    world = np.asarray(points, dtype=float) @ matrix[:3, :3] + matrix[3, :3]
    counts = np.asarray(counts)
    indices = np.asarray(indices)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    fans = []
    for count in np.unique(counts):
        if count < 3:
            continue
        first = starts[counts == count]
        for j in range(1, count - 1):
            fans.append(np.stack(
                [indices[first], indices[first + j], indices[first + j + 1]], 1
            ))
    if not fans:
        return None
    return world[np.concatenate(fans)]


def surface_points(triangles, spacing):
    edges = np.linalg.norm(
        triangles - np.roll(triangles, 1, axis=1), axis=2
    ).max(axis=1)
    steps = np.clip(np.ceil(edges / spacing).astype(int), 1, 400)
    out = [triangles.reshape(-1, 3)]
    for n in np.unique(steps):
        group = triangles[steps == n]
        i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1))
        keep = i + j <= n
        a, b = i[keep] / n, j[keep] / n
        weights = np.stack([1 - a - b, a, b], 1)
        for chunk in range(0, len(group), 1000):
            out.append(np.einsum(
                "pk,tkd->tpd", weights, group[chunk:chunk + 1000]
            ).reshape(-1, 3))
    return np.concatenate(out)


def shift(a, d):
    out = np.zeros_like(a)
    src = tuple(slice(max(-s, 0), n - max(s, 0)) for s, n in zip(d, a.shape))
    dst = tuple(slice(max(s, 0), n - max(-s, 0)) for s, n in zip(d, a.shape))
    out[dst] = a[src]
    return out


FACE_STEPS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
CUBE_STEPS = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)]


def voxelize(points, lower, upper, res):
    """Surface voxels plus a one-cell empty border. Returns (grid, origin)."""
    origin = lower - res
    shape = np.ceil((upper - lower) / res).astype(int) + 2
    grid = np.zeros(shape, bool)
    ijk = np.clip(np.floor((points - origin) / res).astype(int), 0, shape - 1)
    grid[tuple(ijk.T)] = True
    return grid, origin


def outside_region(surface):
    """Empty cells connected to the border (6-connected flood fill)."""
    empty = ~surface
    outside = np.zeros_like(surface)
    outside[[0, -1]] = outside[:, [0, -1]] = outside[:, :, [0, -1]] = True
    outside &= empty
    while True:
        grown = outside.copy()
        for step in FACE_STEPS:
            grown |= shift(outside, step)
        grown &= empty
        if grown.sum() == outside.sum():
            return outside
        outside = grown


def hidden_open_volume(points, lower, upper):
    surface, _ = voxelize(points, lower, upper, CHECK_RES)
    if surface.size > 4_000_000:
        return np.inf
    near = surface.copy()
    for _ in range(NEAR_CELLS):
        grown = near.copy()
        for step in CUBE_STEPS:
            grown |= shift(near, step)
        near = grown
    hidden = outside_region(surface) & ~near
    hidden[[0, -1]] = hidden[:, [0, -1]] = hidden[:, :, [0, -1]] = False
    return hidden.sum() * CHECK_RES**3


def merge_voxels(solid):
    """Greedy x -> y -> z merge of solid voxels into index boxes."""
    free = solid.copy()
    boxes = []
    for i, j, k in np.argwhere(solid):
        if not free[i, j, k]:
            continue
        i2 = i
        while i2 + 1 < solid.shape[0] and free[i2 + 1, j, k]:
            i2 += 1
        j2 = j
        while j2 + 1 < solid.shape[1] and free[i:i2 + 1, j2 + 1, k].all():
            j2 += 1
        k2 = k
        while k2 + 1 < solid.shape[2] and free[i:i2 + 1, j:j2 + 1, k2 + 1].all():
            k2 += 1
        free[i:i2 + 1, j:j2 + 1, k:k2 + 1] = False
        boxes.append((np.array([i, j, k]), np.array([i2 + 1, j2 + 1, k2 + 1])))
    return boxes


def rebuild_solid(points, lower, upper):
    surface, origin = voxelize(points, lower, upper, FINE_RES)
    solid = ~outside_region(surface)
    return [
        (origin + a * FINE_RES, origin + b * FINE_RES)
        for a, b in merge_voxels(solid)
    ]


# ---------- Read the warehouse ----------

def collect_boxes(stage):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
    )
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    root = stage.GetPrimAtPath(WAREHOUSE)
    if not root.IsValid():
        raise RuntimeError(f"Missing {WAREHOUSE} in {SOURCE}")

    boxes, rebuilt, skipped = [], [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        path = str(prim.GetPath())
        if prim.GetTypeName() == "Plane":
            skipped.append((path, "plane (kept /World/GroundPlane instead)"))
            continue
        if not prim.IsA(UsdGeom.Boundable):
            continue
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            skipped.append((path, "empty bounds"))
            continue
        lower = np.array(box.GetMin(), dtype=float)
        upper = np.array(box.GetMax(), dtype=float)

        # Flat sheets lying on the floor (decals) have no physical volume.
        if upper[2] <= 0.005:
            skipped.append((path, "zero-height floor sheet"))
            continue

        triangles = [
            mesh_triangles(UsdGeom.Mesh(p), xforms)
            for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
            if p.IsA(UsdGeom.Mesh)
        ]
        triangles = [t for t in triangles if t is not None]

        if triangles:
            tris = np.concatenate(triangles)
            hidden = hidden_open_volume(
                surface_points(tris, CHECK_RES / 3), lower, upper
            )
            if hidden >= HOLLOW_LIMIT:
                parts = rebuild_solid(
                    surface_points(tris, FINE_RES / 3), lower, upper
                )
                rebuilt.append((path, hidden, len(parts)))
                boxes.extend((lo, hi, path) for lo, hi in parts)
                continue

        boxes.append((lower, upper, path))

    return boxes, rebuilt, skipped


# ---------- Write the box scene ----------

def color_for(path):
    for key, color in COLORS.items():
        if key in path:
            return color
    return DEFAULT_COLOR


def write_scene(boxes):
    if OUTPUT.exists():
        OUTPUT.unlink()
    Sdf.Layer.FindOrOpen(str(SOURCE)).Export(str(OUTPUT))

    stage = Usd.Stage.Open(str(OUTPUT))
    stage.RemovePrim(WAREHOUSE)
    UsdGeom.Xform.Define(stage, BOX_ROOT)

    used = set()
    for index, (lower, upper, source) in enumerate(boxes):
        size = np.maximum(upper - lower, MIN_THICKNESS)
        center = 0.5 * (lower + upper)

        base = re.sub(r"[^A-Za-z0-9_]", "_", source.split("/")[-2])[:40]
        name = f"{base}_{index}"
        while name in used:
            name += "_"
        used.add(name)

        cube = UsdGeom.Cube.Define(stage, f"{BOX_ROOT}/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color_for(source))])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        cube.GetPrim().SetCustomDataByKey("sourcePrim", source)

    add_lights(stage, boxes)
    stage.GetRootLayer().Save()
    return stage


def add_lights(stage, boxes):
    """One sphere light below each ceiling lamp, plus a dome fill light."""
    lamps = {}
    for lower, upper, source in boxes:
        match = re.search(r"SM_LampCeiling\w*?_\d+", source)
        if not match:
            continue
        lo, hi = lamps.get(match.group(0), (lower, upper))
        lamps[match.group(0)] = (np.minimum(lo, lower), np.maximum(hi, upper))

    UsdGeom.Xform.Define(stage, LIGHT_ROOT)
    for name, (lower, upper) in sorted(lamps.items()):
        light = UsdLux.SphereLight.Define(stage, f"{LIGHT_ROOT}/{name}")
        light.CreateRadiusAttr(LAMP_RADIUS)
        light.CreateIntensityAttr(LAMP_INTENSITY)
        center = 0.5 * (lower + upper)
        light.AddTranslateOp().Set(
            Gf.Vec3d(center[0], center[1], lower[2] - LAMP_DROP)
        )

    UsdLux.DomeLight.Define(stage, f"{LIGHT_ROOT}/Fill").CreateIntensityAttr(
        DOME_INTENSITY
    )
    print(f"Lights: {len(lamps)} lamp lights + 1 dome fill")


def verify(stage):
    """Map and plan on the written stage with the project's own code."""
    sys.path.insert(0, str(SCENE_DIR.parent / "drone_code"))
    from mapping import read_scene_geometry, build_occupancy_map
    from navigation import plan_flight

    body = stage.GetPrimAtPath("/World/iris/body")
    start = np.array(
        UsdGeom.Xformable(body)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation(),
        dtype=float,
    )
    geometry = read_scene_geometry(stage, start)
    print(f"Mapper reads {len(geometry.boxes)} boxes, "
          f"{len(geometry.planes)} planes; start={start.round(3)}")

    goals = {
        "old goal under rack": (5.93, 1.12, 1.12),
        "far end of west aisle": (-2.7, 11.0, 1.5),
        "far east aisle": (6.8, 11.0, 1.0),
        "between ceiling lamps": (5.7, 3.8, 7.0),
    }
    for label, (dx, dy, z) in goals.items():
        goal = np.array([start[0] + dx, start[1] + dy, z])
        try:
            plan = plan_flight(
                build_occupancy_map(geometry, start, goal),
                geometry, start, goal,
            )
            top = max(p[2] for p in plan.waypoints)
            print(f"  {label:24s} route={plan.distance:6.2f} m "
                  f"direct={np.linalg.norm(goal - start):5.2f} m "
                  f"max z={top:4.2f} waypoints={len(plan.waypoints)}")
        except Exception as error:
            print(f"  {label:24s} FAILED: {error}")


def main():
    source = Usd.Stage.Open(str(SOURCE))
    boxes, rebuilt, skipped = collect_boxes(source)

    print(f"Boxes written: {len(boxes)}")
    print(f"Rebuilt from real shape ({len(rebuilt)} prims):")
    for path, hidden, parts in sorted(rebuilt, key=lambda r: -r[1]):
        print(f"  {path.split('/')[-3:]}  hidden {hidden:5.1f} m^3 "
              f"-> {parts} boxes")
    print(f"Skipped {len(skipped)}: "
          + ", ".join(sorted({reason for _, reason in skipped})))

    stage = write_scene(boxes)
    print(f"Saved {OUTPUT}")
    verify(stage)


try:
    main()
finally:
    app.close()
