"""Three simple local scene presets. All geometry is authored IN MEMORY.

No asset downloads, permanent USD edits, or team repository modifications.
These are geometric placeholders for navigation tests, not realistic assets.
"""
import random
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics


def _path(stage, name):
    return f"/World/SensorMission/{name}"


def _color(geom, rgb):
    geom.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgb))])


def _cube(stage, name, pos, scale, rgb, collision=True):
    obj = UsdGeom.Cube.Define(stage, _path(stage, name))
    obj.CreateSizeAttr(1.0)
    _color(obj, rgb)
    x = UsdGeom.XformCommonAPI(obj.GetPrim())
    x.SetTranslate(Gf.Vec3d(*map(float, pos)))
    x.SetScale(Gf.Vec3f(*map(float, scale)))
    if collision:
        UsdPhysics.CollisionAPI.Apply(obj.GetPrim())
    return obj.GetPrim()


def _cylinder(stage, name, pos, radius, height, rgb):
    obj = UsdGeom.Cylinder.Define(stage, _path(stage, name))
    obj.CreateRadiusAttr(float(radius))
    obj.CreateHeightAttr(float(height))
    _color(obj, rgb)
    UsdGeom.XformCommonAPI(obj.GetPrim()).SetTranslate(Gf.Vec3d(*map(float, pos)))
    UsdPhysics.CollisionAPI.Apply(obj.GetPrim())


def _cone(stage, name, pos, radius, height, rgb):
    obj = UsdGeom.Cone.Define(stage, _path(stage, name))
    obj.CreateRadiusAttr(float(radius))
    obj.CreateHeightAttr(float(height))
    _color(obj, rgb)
    UsdGeom.XformCommonAPI(obj.GetPrim()).SetTranslate(Gf.Vec3d(*map(float, pos)))
    UsdPhysics.CollisionAPI.Apply(obj.GetPrim())


def _sphere(stage, name, pos, radius, rgb):
    obj = UsdGeom.Sphere.Define(stage, _path(stage, name))
    obj.CreateRadiusAttr(float(radius))
    _color(obj, rgb)
    UsdGeom.XformCommonAPI(obj.GetPrim()).SetTranslate(Gf.Vec3d(*map(float, pos)))
    UsdPhysics.CollisionAPI.Apply(obj.GetPrim())


def _lights(stage, ambient, sun):
    sun_prim = UsdLux.DistantLight.Define(stage, _path(stage, "Sun"))
    sun_prim.CreateIntensityAttr(float(sun))
    sun_prim.CreateAngleAttr(0.53)
    UsdGeom.XformCommonAPI(sun_prim.GetPrim()).SetRotate(Gf.Vec3f(-35, -25, 20))
    dome = UsdLux.DomeLight.Define(stage, _path(stage, "Ambient"))
    dome.CreateIntensityAttr(float(ambient))


def build(stage, config):
    name = config["environment"]
    base = "/World/SensorMission"
    if stage.GetPrimAtPath(base).IsValid():
        raise RuntimeError("SensorMission scene group already exists. Load the USD afresh for each experiment.")
    UsdGeom.Xform.Define(stage, base)
    w = config["world"]
    _lights(stage, w["ambient_intensity"], w["sun_intensity"])
    rng = random.Random(int(w["seed"]))
    if name == "empty":
        # Keep the scene's existing ground plane. Extend visible ground across 100m.
        _cube(stage, "Ground", (55, 0, -0.07), (145, 50, 0.1), (0.40, 0.46, 0.49))
        return
    if name == "warehouse":
        p = w["warehouse"]
        length, width, height = float(p["length_m"]), float(p["width_m"]), float(p["height_m"])
        _cube(stage, "WarehouseFloor", (55, 0, -0.07), (length, width, 0.1), (0.42, 0.44, 0.46))
        # Back and side walls, roof; avoid obstructing camera's forward corridor.
        _cube(stage, "WallLeft", (55, -width/2, height/2), (length, 0.4, height), (0.71,0.73,0.76))
        _cube(stage, "WallRight", (55, width/2, height/2), (length, 0.4, height), (0.71,0.73,0.76))
        _cube(stage, "FarWall", (55+length/2, 0, height/2), (0.4, width, height), (0.68,0.71,0.74))
        _cube(stage, "Roof", (55, 0, height+0.1), (length, width, 0.3), (0.86,0.87,0.88))
        # Simple shelving; central +/-5m corridor kept empty for baseline flight.
        for i in range(int(p["rack_rows_per_side"])):
            x = 7.5 + 10.5*i
            for label, sign in (("L",-1),("R",1)):
                y = sign*10.5
                _cube(stage, f"Rack_{label}_{i}", (x,y,2.1), (5,3,4.2), (0.19,0.28,0.38))
                _cube(stage, f"Box_{label}_{i}", (x+0.4,y,4.45), (1.4,2,0.5), (0.71,0.46,0.25))
        # Functional fixtures: warehouse roof blocks outside skylight.
        for i in range(12):
            lamp = UsdLux.SphereLight.Define(stage, _path(stage, f"WarehouseLamp_{i}"))
            lamp.CreateIntensityAttr(3000.0)
            lamp.CreateRadiusAttr(1.0)
            UsdGeom.XformCommonAPI(lamp.GetPrim()).SetTranslate(Gf.Vec3d(-1.0 + i*10.2,0,13.5))
        return
    if name == "outdoor":
        p = w["outdoor"]
        _cube(stage, "LandscapeGround", (55, 0, -0.08),
              (float(p["length_m"]),float(p["width_m"]),0.1), (0.30,0.49,0.28))
        for i in range(int(p["trees_per_side"])):
            for side, sign in (("L",-1),("R",1)):
                x = 2.0 + 6.0*i + rng.uniform(-1.5,1.5)
                y = sign*rng.uniform(10.0, 28.0)
                trunk_h = rng.uniform(2.0, 3.5)
                foliage_h = rng.uniform(4, 6)
                _cylinder(stage, f"Trunk_{side}_{i}",(x,y,trunk_h/2), 0.55,trunk_h,(0.42,0.30,0.16))
                _cone(stage, f"Foliage_{side}_{i}", (x,y,trunk_h+foliage_h/2),
                      rng.uniform(1.8,2.5), foliage_h,(0.10,0.37+0.05*rng.random(),0.15))
        for i in range(int(p["rocks_per_side"])):
            for side, sign in (("L",-1),("R",1)):
                x = rng.uniform(5, 110)
                y = sign*rng.uniform(8.0,31.0)
                _sphere(stage, f"Rock_{side}_{i}",(x,y,0.0),rng.uniform(0.7,2.2),(0.46,0.47,0.43))
        return
    raise ValueError(f"Unknown environment preset: {name}")
