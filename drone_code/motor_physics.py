from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# Isaac Sim imports must come AFTER SimulationApp starts.
import numpy as np
import omni.usd

from pathlib import Path
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim

# Keep this import if your file is motor.py.
# If it is motor_model.py, use: from motor_model import MotorModel
from motor import MotorModel


def main():
    # ---------- Load the existing scene ----------

    project_dir = Path(__file__).resolve().parents[1]
    scene_path = project_dir / "sim_scene" / "drone1.usd"

    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    print(f"Loading: {scene_path}", flush=True)
    omni.usd.get_context().open_stage(str(scene_path))
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    # Find Physics Scenes BEFORE constructing World.
    physics_scene_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Scene)
    ]

    print(
        f"Physics Scenes in loaded USD: {physics_scene_paths}",
        flush=True,
    )

    if len(physics_scene_paths) > 1:
        raise RuntimeError(
            "The loaded USD already contains multiple Physics Scenes:\n"
            + "\n".join(physics_scene_paths)
            + "\nRemove the duplicate from drone1.usd, save it, "
              "and launch this script again."
        )

    # Reuse the existing Physics Scene; create one only if none exists.
    physics_scene_path = (
        physics_scene_paths[0]
        if physics_scene_paths
        else "/World/PhysicsScene"
    )

    dt = 1.0 / 120.0

    world = World(
        physics_dt=dt,
        rendering_dt=dt,
        stage_units_in_meters=1.0,
        physics_prim_path=physics_scene_path,
    )

    body_path = "/World/iris/body"

    if not stage.GetPrimAtPath(body_path).IsValid():
        raise RuntimeError(f"Drone body not found: {body_path}")

    # Register the body so world.reset() also resets its state.
    body = world.scene.add(
        RigidPrim(
            prim_paths_expr=body_path,
            name="iris_body",
        )
    )

    world.reset()

    # Store the initial pose and reset to zero velocity each time.
    initial_positions, initial_orientations = body.get_world_poses()

    body.set_default_state(
        positions=initial_positions.copy(),
        orientations=initial_orientations.copy(),
        linear_velocities=np.zeros((1, 3), dtype=np.float32),
        angular_velocities=np.zeros((1, 3), dtype=np.float32),
    )
    body.post_reset()

    # ---------- Motor configuration ----------

    motors = MotorModel(
        thrust_coefficient=5.84e-6,
        torque_coefficient=0.06 * 5.84e-6,
        max_speed=1100.0,
        reaction_signs=[-1, -1, +1, +1],
    )

    # Rotor positions relative to the body's center of mass.
    rotor_positions = np.array([
        [ 0.13798545, -0.20671639, 0.023],
        [-0.12511168,  0.21875414, 0.023],
        [ 0.13800000,  0.20257577, 0.023],
        [-0.12415360, -0.22234580, 0.023],
    ])

    def apply_motor_commands(speeds):
        thrusts, reaction_torques = motors.calculate(speeds)

        # Each rotor pushes along the drone's local +Z.
        rotor_forces = np.zeros((4, 3))
        rotor_forces[:, 2] = thrusts

        total_force = rotor_forces.sum(axis=0)

        # Torque from thrust applied away from the center of mass.
        total_torque = np.cross(
            rotor_positions, rotor_forces
        ).sum(axis=0)

        # Add yaw reaction torque.
        total_torque[2] += reaction_torques.sum()

        body.apply_forces_and_torques_at_pos(
            forces=total_force.astype(np.float32).reshape(1, 3),
            torques=total_torque.astype(np.float32).reshape(1, 3),
            is_global=False,
        )

    # ---------- Run and support Stop → Play ----------

    elapsed = 0.0
    reset_requested = False

    print(
        "Running thrust test. Click Stop, then Play to repeat.",
        flush=True,
    )

    while simulation_app.is_running():
        if world.is_stopped():
            # Remember the stop until the user presses Play.
            reset_requested = True
            simulation_app.update()
            continue

        if not world.is_playing():
            # Pause preserves the current flight and timer.
            simulation_app.update()
            continue

        if reset_requested:
            # Reset registered objects and rebuild physics handles.
            world.reset()
            elapsed = 0.0
            reset_requested = False
            print("Drone reset. Repeating thrust test.", flush=True)

        # Apply thrust for one simulated second, then turn motors off.
        speeds = [850.0] * 4 if elapsed < 1.0 else [0.0] * 4

        apply_motor_commands(speeds)
        world.step(render=True)
        elapsed += dt


try:
    main()
finally:
    simulation_app.close()