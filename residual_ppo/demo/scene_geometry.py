"""Export the obstacle colliders of a loaded stage as exact primitive shapes (for demo/geometry.py). Isaac side.

Only UsdGeom Cube / Cylinder / Sphere colliders are supported (the two demo scenes contain nothing else under their
obstacle roots; anything else raises). Each shape is written in world coordinates:
  box       center, 3 unit axes, 3 half extents         (Cube: size x local-to-world; no shear allowed)
  cylinder  center, unit axis, half height, radius      (radius = largest scale of the two cross axes)
  sphere    center, radius                              (radius = largest axis scale)
plus the world axis-aligned bounding box. Inactive prims (pruned subtrees) are skipped by Usd.PrimRange.
"""
import numpy as np

SHEAR_TOL = 1e-6


def _matrix(prim, Usd, UsdGeom):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array([[m[i][j] for j in range(4)] for i in range(4)], dtype=float)   # row vectors: p' = p M


def export_obstacles(stage, root_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"obstacle root not found: {root_path}")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    shapes = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        kind = prim.GetTypeName()
        M = _matrix(prim, Usd, UsdGeom)
        rows, center = M[:3, :3], M[3, :3]
        scales = np.linalg.norm(rows, axis=1)
        units = rows / scales[:, None]
        if np.max(np.abs(units @ units.T - np.eye(3))) > SHEAR_TOL:
            raise RuntimeError(f"{prim.GetPath()}: sheared transform not supported")
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        rec = {
            "path": str(prim.GetPath()),
            "kind": None,
            "aabb": [list(map(float, rng.GetMin())), list(map(float, rng.GetMax()))],
            "rigid_body": bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
        }
        if kind == "Cube":
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            rec.update(kind="box", center=center.tolist(), axes=units.tolist(),
                       half=(0.5 * size * scales).tolist())
        elif kind == "Cylinder":
            g = UsdGeom.Cylinder(prim)
            k = "XYZ".index(g.GetAxisAttr().Get())
            others = [i for i in range(3) if i != k]
            rec.update(kind="cylinder", center=center.tolist(), axis=units[k].tolist(),
                       half_height=float(0.5 * g.GetHeightAttr().Get() * scales[k]),
                       radius=float(g.GetRadiusAttr().Get() * max(scales[i] for i in others)))
        elif kind == "Sphere":
            rec.update(kind="sphere", center=center.tolist(),
                       radius=float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() * scales.max()))
        else:
            raise RuntimeError(f"{prim.GetPath()}: collider type {kind!r} not supported")
        shapes.append(rec)
    return shapes
