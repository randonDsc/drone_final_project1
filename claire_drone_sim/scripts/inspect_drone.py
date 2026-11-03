
from isaacsim import SimulationApp

# Start Isaac Sim first
simulation_app = SimulationApp({"headless": False})

from pathlib import Path
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

# Locate our saved scene
project_dir = Path(__file__).resolve().parents[1]
scene_path = project_dir / "scenes" / "my_world_with_iris_drone.usd"

if not scene_path.exists():
    raise FileNotFoundError(scene_path)

# Open the USD scene
omni.usd.get_context().open_stage(str(scene_path))
simulation_app.update()

stage = omni.usd.get_context().get_stage()

# Locate the drone body
body_path = "/iris/body"
body = stage.GetPrimAtPath(body_path)

if not body.IsValid():
    raise RuntimeError(f"Drone body not found: {body_path}")

# Get the drone body's world position
cache = UsdGeom.XformCache(Usd.TimeCode.Default())
transform = cache.GetLocalToWorldTransform(body)
position = transform.ExtractTranslation()

print("Drone body found!")
print(f"Body path: {body_path}")
print(f"World position: {position}")

# Check for rigid-body physics
has_physics = body.HasAPI(UsdPhysics.RigidBodyAPI)
print(f"Rigid-body physics enabled on prim: {has_physics}")

# Print its immediate child prims
print("\nBody components:")
for child in body.GetChildren():
    print(" -", child.GetName())

print("\nInspection complete. Close Isaac Sim when finished.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
