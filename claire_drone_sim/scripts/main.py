from isaacsim import SimulationApp
from pathlib import Path


# 1. Start Isaac Sim with a visible window
simulation_app = SimulationApp({"headless": False})

# 2. Import Isaac Sim libraries AFTER startup
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import save_stage

# 3. Create our simulation world
world = World(stage_units_in_meters=1.0)

# 4. Add a ground plane
world.scene.add_default_ground_plane()

# 5. Add a blue box above the ground
box = world.scene.add(
    DynamicCuboid(
        prim_path="/World/MyBox",
        name="my_box",
        position=np.array([0.0, 0.0, 3.0]),
        scale=np.array([1.0, 1.0, 1.0]),
        color=np.array([0.2, 0.5, 1.0]),
        mass=1.0,
    )
)

world.reset()
# Save our environment before the physics simulation begins
scene_path = (
    Path(__file__).resolve().parents[1]
    / "scenes"
    / "my_world.usd"
)

saved = save_stage(
    str(scene_path),
    save_and_reload_in_place=False
)
if not saved:
    raise RuntimeError("Could not save the USD scene.")

print(f"Environment saved to: {scene_path}")

print("My first 3D simulation is running!")
# 7. Keep updating until the window is closed
while simulation_app.is_running():
    world.step(render=True)

# 8. Clean up
simulation_app.close()
