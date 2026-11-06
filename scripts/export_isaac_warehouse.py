"""Export flattened Isaac warehouse colliders with OpenUSD; Isaac is not needed.

python scripts/export_isaac_warehouse.py --output /tmp/warehouse.json
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def export(source, initial_npz, output, freeze_supports=()):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    source, initial_npz, output = map(Path, (source, initial_npz, output))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    stage = Usd.Stage.Open(str(source.resolve()), load=Usd.Stage.LoadNone)
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1) or UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise ValueError("Warehouse conversion requires a metre-based, Z-up USD")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=False)
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    boxes, planes, frozen = [], [], set()
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path == "/World/iris" or path.startswith("/World/iris/"):
            continue
        if prim.IsA(UsdGeom.PointInstancer):
            raise ValueError(f"Unconverted point instancer: {path}")
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                api = UsdPhysics.RigidBodyAPI(ancestor)
                if api.GetRigidBodyEnabledAttr().Get() is not False and api.GetKinematicEnabledAttr().Get() is not True:
                    if str(ancestor.GetPath()) not in freeze_supports:
                        raise ValueError(f"Dynamic obstacle cannot be exported as static: {path}; explicitly freeze a known launch support if appropriate")
                    frozen.add(str(ancestor.GetPath()))
            ancestor = ancestor.GetParent()
        if prim.GetTypeName() == "Plane":
            axis = {"X": 0, "Y": 1, "Z": 2}[str(prim.GetAttribute("axis").Get() or "Z").upper()]
            matrix = transforms.GetLocalToWorldTransform(prim)
            normal = np.array(matrix.GetInverse().GetTranspose().TransformDir(Gf.Vec3d(*np.eye(3)[axis])), dtype=float)
            normal /= np.linalg.norm(normal)
            dominant = int(np.argmax(abs(normal)))
            if not np.isclose(abs(normal[dominant]), 1, atol=1e-6):
                raise ValueError(f"Tilted infinite plane unsupported: {path}")
            normal = np.eye(3)[dominant]*np.sign(normal[dominant])
            point = np.array(matrix.Transform(Gf.Vec3d(0, 0, 0)))
            planes.append(dict(path=path, normal=normal.tolist(), offset=float(normal@point)))
            continue
        support_mesh = prim.IsA(UsdGeom.Mesh) and path in frozen
        if not prim.IsA(UsdGeom.Cube) and not support_mesh:
            raise ValueError(f"Use the already flattened box scene; found {prim.GetTypeName()} at {path}")
        matrix = transforms.GetLocalToWorldTransform(prim)
        axes = np.array([matrix.TransformDir(Gf.Vec3d(*axis)) for axis in np.eye(3)])
        normalized = axes/np.linalg.norm(axes, axis=1, keepdims=True)
        if not np.allclose(np.sort(abs(normalized), axis=1), [0, 0, 1], atol=1e-6):
            raise ValueError(f"A rotated source cube would need an approximate AABB: {path}")
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lower, upper = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        if not np.all(np.isfinite([lower, upper])) or np.any(upper <= lower):
            raise ValueError(f"Invalid collider bounds: {path}")
        if support_mesh:
            mesh = UsdGeom.Mesh(prim)
            points = np.array([matrix.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])
            corner_coordinates = np.isclose(points, lower, atol=1e-7) | np.isclose(points, upper, atol=1e-7)
            if not np.all(corner_coordinates) or len(np.unique(np.round(points, 7), axis=0)) != 8:
                raise ValueError("The frozen launch mesh is not a verified box")
            indices, counts = list(mesh.GetFaceVertexIndicesAttr().Get()), list(mesh.GetFaceVertexCountsAttr().Get())
            areas, offset = np.zeros((3, 2)), 0
            for count in counts:
                face = points[indices[offset:offset+count]]
                offset += count
                sides = [(axis, side) for axis in range(3) for side, bound in enumerate((lower, upper))
                         if np.allclose(face[:, axis], bound[axis], atol=1e-7)]
                if len(sides) != 1:
                    raise ValueError("Launch mesh contains a non-box face")
                area = sum(np.linalg.norm(np.cross(face[j]-face[0], face[j+1]-face[0]))/2 for j in range(1, count-1))
                areas[sides[0]] += area
            lengths = upper-lower
            expected = np.array([np.prod(np.delete(lengths, i)) for i in range(3)])
            if not np.allclose(areas, expected[:, None], rtol=1e-5, atol=1e-9):
                raise ValueError("Launch mesh does not cover six complete box faces")
        colors = prim.GetAttribute("primvars:displayColor").Get()
        rgb = [float(x) for x in colors[0]] if colors else [.6, .5, .35]
        boxes.append(dict(path=path, lower=lower.tolist(), upper=upper.tolist(), rgba=rgb+[1.0]))
    if not boxes or not planes:
        raise ValueError("Expected collision boxes and an infinite ground plane")
    with np.load(initial_npz) as saved:
        mission = dict(start=saved["start"].tolist(), goal=saved["goal"].tolist(), orientation=[1, 0, 0, 0])
    value = dict(schema_version=1, name="Isaac flattened warehouse", units="metres", up_axis="Z",
                 source=dict(usd="sim_scene/"+source.name, usd_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                             initial_pose_source="navigation_output/"+initial_npz.name,
                             initial_pose_sha256=hashlib.sha256(initial_npz.read_bytes()).hexdigest(),
                             usd_version=list(Usd.GetVersion()),
                             frozen_dynamic_launch_supports=sorted(frozen),
                             conversion="Static axis-aligned collision cubes and planes; Iris, lights and visual-only assets excluded."),
                 mission=mission, boxes=boxes, planes=planes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    print(f"Exported {len(boxes)} boxes and {len(planes)} planes to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="sim_scene/dij3d_boxes.usd")
    parser.add_argument("--initial-state", default="navigation_output/occupancy_grid.npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--freeze-support", action="append", default=[], help="Explicit USD rigid-body path to freeze as a launch support")
    args = parser.parse_args()
    export(args.source, args.initial_state, args.output, args.freeze_support)
