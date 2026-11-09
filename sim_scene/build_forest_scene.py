"""Build a simple outdoor forest scene from dij3d_boxes.usd.

Run with Isaac Sim's python:

    C:\\isaacsim\\python.bat sim_scene\\build_forest_scene.py

Input:  sim_scene/dij3d_boxes.usd
Output: sim_scene/forest.usd

Kept exactly as authored: the iris drone, the launch pad /World/Cube, the
flat ground plane, the physics scene, and /Environment/defaultLight (sun).
Removed: /World/BoxWarehouse and its lamp lights /World/Lights.

Added under /World/Forest, all static colliders (CollisionAPI, no rigid
body), so PhysX gives real contacts with the drone:

- Trees: a vertical trunk cylinder, 1-4 horizontal branch cylinders, and a
  sphere canopy. Branches point along +-X or +-Y so their axis-aligned
  bounds are tight: mapping.read_scene_geometry reads every collider as an
  AABB, and a diagonal branch would block a much larger box.
- Fallen logs lying on the ground (also axis-aligned) and half-buried rocks.

The layout is random but fixed by SEED. A clear disc is kept around the
launch pad so takeoff is never blocked.
"""
from pathlib import Path
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

SCENE_DIR = Path(__file__).resolve().parent
SOURCE = SCENE_DIR / "dij3d_boxes.usd"
OUTPUT = SCENE_DIR / "forest.usd"
FOREST = "/World/Forest"
REMOVE = ("/World/BoxWarehouse", "/World/Lights")

SEED = 7

# Forest area in world XY [m]. The drone starts at about (-2.3, 0).
AREA_X = (-1.0, 22.0)
AREA_Y = (-10.0, 10.0)
CLEAR_RADIUS = 2.5          # no trees, logs or rocks this close to the pad [m]
TREE_COUNT = 40
TREE_MIN_SPACING = 2.6      # trunk center to trunk center [m]

TRUNK_RADIUS = (0.12, 0.25)
TRUNK_HEIGHT = (4.5, 8.0)
CANOPY_RADIUS = (1.2, 2.0)
BRANCH_COUNT = (1, 4)       # inclusive
BRANCH_HEIGHT = (1.0, 3.8)  # low branches cross normal flight heights [m]
BRANCH_LENGTH = (0.7, 1.6)
BRANCH_RADIUS = (0.035, 0.06)

LOG_COUNT = 6
LOG_RADIUS = (0.15, 0.25)
LOG_LENGTH = (2.0, 4.0)
ROCK_COUNT = 10
ROCK_RADIUS = (0.2, 0.5)

FRICTION = (0.8, 0.6)       # static, dynamic: bark and stone
COLORS = {
    "trunk": (0.36, 0.25, 0.15),
    "branch": (0.42, 0.30, 0.18),
    "canopy": (0.20, 0.45, 0.18),
    "log": (0.40, 0.28, 0.16),
    "rock": (0.50, 0.50, 0.48),
}
DOME_INTENSITY = 600.0


# ---------- Layout ----------

def uniform(rng, bounds):
    return float(rng.uniform(*bounds))


def place_points(rng, count, spacing, start, taken=()):
    """Dart throwing: points in the area, away from the pad and each other."""
    points = list(taken)
    placed = []
    for _ in range(count * 200):
        if len(placed) == count:
            break
        p = np.array([uniform(rng, AREA_X), uniform(rng, AREA_Y)])
        if np.linalg.norm(p - start) < CLEAR_RADIUS + spacing / 2:
            continue
        if any(np.linalg.norm(p - q) < spacing for q in points):
            continue
        points.append(p)
        placed.append(p)
    return placed


# ---------- Geometry ----------

def collider(prim, color):
    UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
    prim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def cylinder(stage, path, center, axis, radius, height, color):
    geom = UsdGeom.Cylinder.Define(stage, path)
    geom.CreateAxisAttr(axis)
    geom.CreateRadiusAttr(radius)
    geom.CreateHeightAttr(height)
    half = np.full(3, radius)
    half["XYZ".index(axis)] = height / 2
    geom.CreateExtentAttr([Gf.Vec3f(*-half), Gf.Vec3f(*half)])
    geom.AddTranslateOp().Set(Gf.Vec3d(*center))
    collider(geom, color)
    return geom


def sphere(stage, path, center, radius, color):
    geom = UsdGeom.Sphere.Define(stage, path)
    geom.CreateRadiusAttr(radius)
    geom.CreateExtentAttr([Gf.Vec3f(-radius), Gf.Vec3f(radius)])
    geom.AddTranslateOp().Set(Gf.Vec3d(*center))
    collider(geom, color)
    return geom


def add_tree(stage, rng, index, xy):
    root = f"{FOREST}/Trees/Tree_{index:02d}"
    UsdGeom.Xform.Define(stage, root)
    r = uniform(rng, TRUNK_RADIUS)
    h = uniform(rng, TRUNK_HEIGHT)
    cylinder(stage, f"{root}/Trunk", (xy[0], xy[1], h / 2), "Z", r, h,
             COLORS["trunk"])

    directions = [((1, 0), "X"), ((-1, 0), "X"), ((0, 1), "Y"), ((0, -1), "Y")]
    n = int(rng.integers(BRANCH_COUNT[0], BRANCH_COUNT[1] + 1))
    for j, k in enumerate(rng.permutation(4)[:n]):
        (dx, dy), axis = directions[k]
        length = uniform(rng, BRANCH_LENGTH)
        z = min(uniform(rng, BRANCH_HEIGHT), h - 1.0)
        offset = r + length / 2 - 0.02  # root slightly inside the trunk
        cylinder(stage, f"{root}/Branch_{j}",
                 (xy[0] + dx * offset, xy[1] + dy * offset, z), axis,
                 uniform(rng, BRANCH_RADIUS), length, COLORS["branch"])

    sphere(stage, f"{root}/Canopy", (xy[0], xy[1], h),
           uniform(rng, CANOPY_RADIUS), COLORS["canopy"])


def add_physics_material(stage):
    material = UsdShade.Material.Define(stage, f"{FOREST}/PhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(FRICTION[0])
    api.CreateDynamicFrictionAttr(FRICTION[1])
    api.CreateRestitutionAttr(0.1)
    binding = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(FOREST))
    binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def build():
    if OUTPUT.exists():
        OUTPUT.unlink()
    Sdf.Layer.FindOrOpen(str(SOURCE)).Export(str(OUTPUT))
    stage = Usd.Stage.Open(str(OUTPUT))
    for path in REMOVE:
        stage.RemovePrim(path)

    body = stage.GetPrimAtPath("/World/iris/body")
    start = np.array(
        UsdGeom.Xformable(body)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation(), dtype=float,
    )

    rng = np.random.default_rng(SEED)
    UsdGeom.Xform.Define(stage, FOREST)
    for group in ("Trees", "Logs", "Rocks"):
        UsdGeom.Xform.Define(stage, f"{FOREST}/{group}")

    trees = place_points(rng, TREE_COUNT, TREE_MIN_SPACING, start[:2])
    for i, xy in enumerate(trees):
        add_tree(stage, rng, i, xy)

    logs = place_points(rng, LOG_COUNT, 1.8, start[:2], trees)
    for i, xy in enumerate(logs):
        radius = uniform(rng, LOG_RADIUS)
        axis = "X" if rng.random() < 0.5 else "Y"
        cylinder(stage, f"{FOREST}/Logs/Log_{i}", (xy[0], xy[1], radius),
                 axis, radius, uniform(rng, LOG_LENGTH), COLORS["log"])

    rocks = place_points(rng, ROCK_COUNT, 1.0, start[:2], trees + logs)
    for i, xy in enumerate(rocks):
        radius = uniform(rng, ROCK_RADIUS)
        sphere(stage, f"{FOREST}/Rocks/Rock_{i}",
               (xy[0], xy[1], 0.3 * radius), radius, COLORS["rock"])

    add_physics_material(stage)
    UsdLux.DomeLight.Define(stage, "/World/SkyLight").CreateIntensityAttr(
        DOME_INTENSITY
    )
    stage.GetRootLayer().Save()
    print(f"Saved {OUTPUT}: {len(trees)} trees, {len(logs)} logs, "
          f"{len(rocks)} rocks; start={start.round(3)}")
    return stage, start


# ---------- Check with the project's own mapper and planner ----------

def verify(stage, start):
    sys.path.insert(0, str(SCENE_DIR.parent / "drone_code"))
    from mapping import read_scene_geometry, build_occupancy_map
    from navigation import plan_flight

    geometry = read_scene_geometry(stage, start)
    print(f"Mapper reads {len(geometry.boxes)} boxes, "
          f"{len(geometry.planes)} planes")

    goals = {
        "straight through": (22.0, 0.0, 1.5),
        "far left corner": (20.0, 8.0, 1.5),
        "far right corner": (18.0, -8.0, 2.0),
        "low, under branches": (12.0, 2.0, 1.0),
    }
    for label, (dx, dy, z) in goals.items():
        goal = np.array([start[0] + dx, start[1] + dy, z])
        try:
            plan = plan_flight(
                build_occupancy_map(geometry, start, goal),
                geometry, start, goal,
            )
            top = max(p[2] for p in plan.waypoints)
            print(f"  {label:22s} route={plan.distance:6.2f} m "
                  f"direct={np.linalg.norm(goal - start):5.2f} m "
                  f"max z={top:4.2f} waypoints={len(plan.waypoints)}")
        except Exception as error:
            print(f"  {label:22s} FAILED: {error}")


if __name__ == "__main__":
    verify(*build())
