"""Static checks of the indoor sensor assets. Isaac Sim is not started.

    C:\\isaacsim\\python.bat drone_code\\check_sensor_assets.py

Reads the USD files with Isaac's bundled USD library only, so it takes seconds.
Confirms the sensors are authored as intended and that adding them changed
nothing about the drone's physics. It says nothing about whether a sensor
produces data: inspect_sensors.py checks that, in Isaac Sim.

The XT32 is fetched from a URL at run time, so its contents are not visible here
and a warning about opening that payload is expected.
"""
import glob
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Isaac's USD library, without Kit.
_libs = glob.glob(r"C:\isaacsim\extscache\omni.usd.libs-*")[0]
sys.path.insert(0, _libs)
os.environ["PATH"] = os.pathsep.join(
    [os.path.join(_libs, "bin"), _libs, os.environ["PATH"]]
)

for _folder in (os.path.join(_libs, "bin"), _libs):
    if os.path.isdir(_folder):
        os.add_dll_directory(_folder)

import numpy as np  # noqa: E402

# Usd must come first: importing Sdf alone leaves its file formats unloaded.
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import sensor_config  # noqa: E402

ORIGINAL = ROOT / "sim_scene" / "dij3d_boxes.usd"
INDOOR = ROOT / "sim_scene" / "dij3d_boxes_indoor.usda"
UNCHANGED_FILES = ("sim_asset/iris.usd", "sim_scene/dij3d_boxes.usd")

DRONE = "/World/iris"
BODY = "/World/iris/body"
LIDAR_MOUNT = BODY + "/HESAI_XT32_SD10"
XT32_LOWEST_CHANNEL = -16.0     # [degrees]
HOUSING_RADIUS = 0.06           # [m], airframe closer than this is the sensor


def physics_summary(stage):
    """Everything that decides mass, inertia, joints, and collisions."""
    summary = {}

    for prim in Usd.PrimRange(stage.GetPrimAtPath(DRONE)):
        apis = sorted(
            api for api in prim.GetAppliedSchemas()
            if "Physics" in api or "Physx" in api
        )
        is_joint = prim.IsA(UsdPhysics.Joint)

        if not apis and not is_joint:
            continue

        values = {}

        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim)
            values = {
                "mass": mass.GetMassAttr().Get(),
                "inertia": mass.GetDiagonalInertiaAttr().Get(),
                "com": mass.GetCenterOfMassAttr().Get(),
            }

        summary[prim.GetPath().pathString] = (
            prim.GetTypeName(),
            apis,
            str(values),
        )

    return summary


def body_translation(stage, path):
    """Authored mount; the payload may be unavailable, so no type is assumed."""
    return np.array(
        stage.GetPrimAtPath(path).GetAttribute("xformOp:translate").Get(),
        dtype=float,
    )


def highest_airframe_elevation(stage, origin):
    """Highest elevation of airframe vertices seen from origin [degrees]."""
    cache = UsdGeom.XformCache()
    to_body = cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(BODY)
    ).GetInverse()

    highest = -90.0

    for prim in Usd.PrimRange(stage.GetPrimAtPath(DRONE)):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        matrix = np.array(cache.GetLocalToWorldTransform(prim) * to_body)
        offset = (
            np.c_[points, np.ones(len(points))] @ matrix
        )[:, :3] - origin

        distance = np.hypot(offset[:, 0], offset[:, 1])
        outside = distance > HOUSING_RADIUS

        if outside.any():
            highest = max(highest, float(np.degrees(np.max(
                np.arctan2(offset[outside, 2], distance[outside])
            ))))

    return highest


def git_unchanged(path):
    return subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", path],
        cwd=ROOT,
    ).returncode == 0


def main():
    original = Usd.Stage.Open(str(ORIGINAL), Usd.Stage.LoadNone)
    indoor = Usd.Stage.Open(str(INDOOR), Usd.Stage.LoadNone)
    results = []

    def check(name, passed, detail=""):
        results.append(bool(passed))
        print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}", flush=True)

    for path in UNCHANGED_FILES:
        check("original file unchanged", git_unchanged(path), path)

    asset = Usd.Stage.Open(str(ROOT / "sim_asset" / "iris.usd"), Usd.Stage.LoadNone)
    names = [prim.GetName() for prim in asset.Traverse()]
    check(
        "iris.usd has no sensors of its own",
        "imu" not in names and "rangefinder" not in names,
    )

    before, after = physics_summary(original), physics_summary(indoor)
    check(
        "physics prims, mass, inertia, and joints identical",
        before == after,
        f"{len(after)} prims compared",
    )

    bodies = sum(
        UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get() is not False
        for prim in Usd.PrimRange(indoor.GetPrimAtPath(DRONE))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    check("rigid body count", bodies == 5, str(bodies))

    for path in (
        LIDAR_MOUNT,
        sensor_config.IMU_PATH,
        sensor_config.RANGEFINDER["path"],
    ):
        prim = indoor.GetPrimAtPath(path)
        physical = [
            child.GetPath().pathString
            for child in Usd.PrimRange(prim)
            if child.HasAPI(UsdPhysics.RigidBodyAPI)
            or child.HasAPI(UsdPhysics.CollisionAPI)
            or child.HasAPI(UsdPhysics.MassAPI)
        ]
        check(f"no physics on {path}", prim.IsValid() and not physical)

    imu = indoor.GetPrimAtPath(sensor_config.IMU_PATH)
    check(
        "IMU type and filters",
        imu.GetTypeName() == "IsaacImuSensor"
        and imu.GetAttribute("linearAccelerationFilterWidth").Get() == 1
        and imu.GetAttribute("angularVelocityFilterWidth").Get() == 1,
    )

    settings = sensor_config.RANGEFINDER
    ranger = indoor.GetPrimAtPath(settings["path"])
    get = lambda name: ranger.GetAttribute(name).Get()

    check(
        "rangefinder matches sensor_config.py",
        ranger.GetTypeName() == "IsaacRaycastSensor"
        and get("enabled")
        and get("numRays") == 1
        and abs(get("minRange") - settings["min_range"]) < 1e-6
        and abs(get("maxRange") - settings["max_range"]) < 1e-6,
        f"{get('minRange'):.2f} to {get('maxRange'):.1f} m",
    )

    direction = np.array(get("rayDirections")[0], dtype=float)
    check(
        "rangefinder points straight down",
        np.allclose(direction, (0, 0, -1)),
        str(direction),
    )
    check(
        "only depths are read: hit prim reporting off",
        get("reportHitPrimPaths") is False,
    )

    hull_bottom = -0.067
    mount = body_translation(indoor, settings["path"])
    check(
        "rangefinder below the hull",
        mount[2] < hull_bottom,
        f"z = {mount[2]:.3f} m, hull bottom {hull_bottom} m",
    )

    origin = body_translation(indoor, LIDAR_MOUNT)
    highest = highest_airframe_elevation(indoor, origin)
    check(
        "airframe below all LiDAR channels",
        highest < XT32_LOWEST_CHANNEL,
        f"mount z = {origin[2]:.3f} m, airframe peaks at {highest:.1f} deg, "
        f"lowest channel {XT32_LOWEST_CHANNEL:.0f} deg",
    )

    print(f"\n{sum(results)}/{len(results)} checks passed.", flush=True)

    return all(results)


sys.exit(0 if main() else 1)
