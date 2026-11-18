r"""M9 Step 7a: one replay .usda per case (18), from the committed 60 Hz series. DESCRIPTIVE DEMO, NOT eval-v1.

    # Directory: <repo root>\residual_ppo
    & <isaac>\python.bat demo\make_replay.py            (pxr only; a headless SimulationApp provides it)

Each results/m9_demo/replay/<case>.usda:
- sublayers the case's scene (sim_scene/dij3d_boxes.usd or forest.usd, repo-relative); deactivates the physical
  drone /World/iris and the launch pad /World/Cube (as in the demo), so pressing Play simulates nothing that moves;
- /World/M9Replay/<controller>/drone: a copy of sim_asset/iris.usd made NON-PHYSICAL (every Physics*/Physx* applied
  schema removed with list-op deletes, joints deactivated; verified after authoring), tinted with a
  UsdPreviewSurface (PID orange, PID + PPO blue, PPO direct green), and animated with time-sampled
  xformOp:translate (double3) + xformOp:orient (quatd) = the recorded body pose. iris.usd's body is the identity
  child of its root, so root pose == body pose exactly;
- the reference path as a thin white curve and each controller's full flown path as a faint (lightened) curve;
- /World/M9Replay/OverviewCamera: the fixed elevated camera the MP4s use (frames every point of the case);
- timeline 60 time codes per second, 60 fps, real time: time code k = 60 Hz row k (t = k / 60 s).
A manifest results/m9_demo/replay/replay_manifest.json lists every file with its sha256 and its source .npz.
"""
import hashlib
import json
import sys
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
sys.path.insert(0, str(RESIDUAL_DIR))

import numpy as np  # noqa: E402

OUT = RESIDUAL_DIR / "results" / "m9_demo" / "replay"
CONTROLLERS = {  # name: (label, rgb)
    "pid": ("PID", (1.0, 0.45, 0.0)),
    "pid_ppo": ("PID + PPO residual", (0.10, 0.35, 0.95)),
    "ppo_direct": ("PPO direct", (0.10, 0.75, 0.20)),
}
HZ = 60
ASPECT = 16 / 9
H_APERTURE = 20.955          # mm, USD default
CAMERA = {   # behind the start along the main heading, elevated; indoor stays under the 9 m ceiling
    "indoor": {"back": 9.0, "z_abs": 8.2},
    "forest": {"back": 11.0, "z_above": 7.0},
}


def rel(path):
    return Path(path).resolve().relative_to(REPO_DIR.resolve())


def camera_pose(points, heading, scene):
    """Position, target and focal length so every point fits a 16:9 frame with a 12% margin."""
    lo, hi = points.min(0), points.max(0)
    c = 0.5 * (lo + hi)
    spec = CAMERA[scene]
    h = np.array([heading[0], heading[1], 0.0])
    pos = c - spec["back"] * h
    pos[2] = spec["z_abs"] if "z_abs" in spec else c[2] + spec["z_above"]
    f = c - pos
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    d = points - pos
    depth = d @ f
    tan_needed = max(np.max(np.abs(d @ r) / depth), np.max(np.abs(d @ u) / depth) * ASPECT) * 1.12
    focal = max(H_APERTURE / 2 / tan_needed, 8.0)
    return pos, c, r, u, f, focal


def build_case(case, rows, series, scene, heading):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    path = OUT / f"{case}.usda"
    if path.exists():
        path.unlink()
    stage = Usd.Stage.CreateNew(str(path))
    root = stage.GetRootLayer()
    scene_usd = REPO_DIR / ("sim_scene/dij3d_boxes.usd" if scene == "indoor" else "sim_scene/forest.usd")
    root.subLayerPaths.append(Path("../../../../" + rel(scene_usd).as_posix()).as_posix())
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    n = min(len(series[c]["rec60"]) for c in CONTROLLERS)
    n_max = max(len(series[c]["rec60"]) for c in CONTROLLERS)
    stage.SetTimeCodesPerSecond(HZ)
    stage.SetFramesPerSecond(HZ)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(n_max - 1)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    root.customLayerData = {"m9": f"{case}; descriptive demo, not eval-v1; no controller perceives obstacles; "
                                  f"crosswind toward obstacles (worst side)"}
    for p in ("/World/iris", "/World/Cube"):
        stage.OverridePrim(p).SetActive(False)

    base = UsdGeom.Xform.Define(stage, "/World/M9Replay")
    looks = UsdGeom.Scope.Define(stage, "/World/M9Replay/Looks")

    def material(name, rgb, emissive=0.35):
        mat = UsdShade.Material.Define(stage, f"{looks.GetPath()}/{name}")
        sh = UsdShade.Shader.Define(stage, f"{mat.GetPath()}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(emissive * np.asarray(rgb))))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def curve(prim_path, points, rgb, width, mat):
        cv = UsdGeom.BasisCurves.Define(stage, prim_path)
        pts = points[np.all(np.isfinite(points), axis=1)]
        cv.CreateTypeAttr(UsdGeom.Tokens.linear)
        cv.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, q)) for q in pts]))
        cv.CreateCurveVertexCountsAttr([len(pts)])
        cv.CreateWidthsAttr([width])
        cv.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        cv.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        UsdShade.MaterialBindingAPI.Apply(cv.GetPrim()).Bind(mat)

    ref_path = series["pid"]["reference_path"]
    curve("/World/M9Replay/reference", ref_path, (1, 1, 1), 0.02, material("white", (1, 1, 1), emissive=0.8))

    removed = {}
    for name, (label, rgb) in CONTROLLERS.items():
        rec = series[name]["rec60"]
        t = rec[:, 0]
        k = np.arange(len(rec))
        if np.max(np.abs(t - k / HZ)) > 1e-9:
            raise SystemExit(f"{case}/{name}: 60 Hz rows are not at k/60 s")
        group = UsdGeom.Xform.Define(stage, f"/World/M9Replay/{name}")
        group.GetPrim().SetCustomDataByKey("controller", label)
        drone = UsdGeom.Xform.Define(stage, f"{group.GetPath()}/drone")
        drone.GetPrim().GetReferences().AddReference(Path("../../../../sim_asset/iris.usd").as_posix())
        # iris.usd's /vehicle carries identity translate / rotateXYZ / scale ops; replace the op ORDER (not the asset)
        # with our two animated ops, suffixed "m9" so they never collide with the asset's attributes.
        drone.ClearXformOpOrder()
        tr = drone.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "m9")
        orient = drone.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "m9")
        for i in range(len(rec)):
            p, q = rec[i, 1:4], rec[i, 4:8]                       # q = (w, x, y, z), read_drone_state convention
            if not (np.all(np.isfinite(p)) and np.all(np.isfinite(q))):
                continue
            tr.Set(Gf.Vec3d(*map(float, p)), Usd.TimeCode(i))
            orient.Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*map(float, q[1:]))), Usd.TimeCode(i))
        n_removed = 0
        for prim in Usd.PrimRange(drone.GetPrim()):
            for schema in list(prim.GetAppliedSchemas()):
                if schema.startswith(("Physics", "Physx")):
                    prim.RemoveAppliedSchema(schema)
                    n_removed += 1
            if prim.GetTypeName().startswith("Physics") and prim.GetTypeName().endswith("Joint"):
                prim.SetActive(False)
        removed[name] = n_removed
        mat = material(f"tint_{name}", rgb, emissive=0.9)
        binding = UsdShade.MaterialBindingAPI.Apply(drone.GetPrim())
        binding.Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        faint = tuple(0.7 * np.asarray(rgb) + 0.3)
        curve(f"{group.GetPath()}/flown_path", rec[:, 1:4], faint, 0.03, material(f"faint_{name}", faint, 0.5))

    pts = np.vstack([ref_path] + [series[c]["rec60"][:, 1:4] for c in CONTROLLERS])
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    pos, target, r, u, f, focal = camera_pose(pts, heading, scene)
    cam = UsdGeom.Camera.Define(stage, "/World/M9Replay/OverviewCamera")
    m = Gf.Matrix4d(*r, 0, *u, 0, *(-f), 0, *pos, 1)
    cam.AddTransformOp().Set(m)
    cam.CreateFocalLengthAttr(float(focal))
    cam.CreateHorizontalApertureAttr(H_APERTURE)
    cam.CreateVerticalApertureAttr(H_APERTURE / ASPECT)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
    root.Save()

    # Verify: no physics schema and no active joint left under any copy.
    check = Usd.Stage.Open(str(path), load=Usd.Stage.LoadNone)
    left = [str(p.GetPath()) for p in Usd.PrimRange(check.GetPrimAtPath("/World/M9Replay"))
            if any(s.startswith(("Physics", "Physx")) for s in p.GetAppliedSchemas())
            or (p.IsActive() and p.GetTypeName().startswith("Physics"))]
    if left:
        raise SystemExit(f"{case}: physics left on {left[:5]}")
    return path, {"frames": int(n_max), "seconds": (n_max - 1) / HZ, "schemas_removed": removed,
                  "camera": {"position": pos.tolist(), "target": target.tolist(), "focal_mm": float(focal)}}


def main():
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from demo import tracks

        OUT.mkdir(parents=True, exist_ok=True)
        manifest = {"label": "descriptive demo, not eval-v1", "hz": HZ, "colors": {k: v[1] for k, v in CONTROLLERS.items()},
                    "files": []}
        for scene in ("indoor", "forest"):
            rec = json.loads((RESIDUAL_DIR / "results/m9_demo/flights" / f"flights_{scene}.json").read_text("utf-8"))
            by_case = {}
            for r in rec["rows"]:
                by_case.setdefault((r["track"], r["wind"]), {})[r["controller"]] = r
            for (track, wind), rows in by_case.items():
                series = {}
                for c, r in rows.items():
                    src = RESIDUAL_DIR / r["series_file"]
                    if hashlib.sha256(src.read_bytes()).hexdigest() != r["series_sha256"]:
                        raise SystemExit(f"sha256 mismatch {src}")
                    with np.load(src) as z:
                        series[c] = {k: z[k] for k in z.files}
                heading, _ = tracks.main_heading(tracks.TRACKS[scene][track])
                case = f"{scene}_{track}_{wind}"
                path, info = build_case(case, rows, series, scene, heading)
                manifest["files"].append({"case": case, "file": path.name,
                                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                          "size_bytes": path.stat().st_size,
                                          "sources": {c: rows[c]["series_file"] for c in CONTROLLERS}, **info})
                print(f"[M9.replay] {path.name}: {info['frames']} frames ({info['seconds']:.2f} s), "
                      f"{path.stat().st_size / 1e6:.2f} MB, schemas removed {info['schemas_removed']}", flush=True)
        (OUT / "replay_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        print("[M9.replay] RESULT: DONE", flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()          # app.close() ends the process before Python could print it
        print("[M9.replay] RESULT: FAIL", flush=True)
        sys.stdout.flush()
    finally:
        app.close()


if __name__ == "__main__":
    main()
