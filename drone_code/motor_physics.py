from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# Isaac Sim imports must come AFTER SimulationApp starts.
import numpy as np
import omni.usd

from pathlib import Path
from pxr import UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim

from motor import MotorModel
from controller import controller
from drone_diagnostics import (
    diagnose_thrust_offset,
    read_total_com_in_body_frame,
)


def main(log_file):
    # Write to the text file and display in the terminal.
    def log(message):
        print(message, flush=True)
        print(message, file=log_file, flush=True)

    # ---------- Load scene ----------

    project_dir = Path(__file__).resolve().parents[1]
    scene_path = project_dir / "sim_scene" / "drone1.usd"

    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)

    log(f"Running script: {Path(__file__).resolve()}")
    log(f"Loading: {scene_path}")

    omni.usd.get_context().open_stage(str(scene_path))
    simulation_app.update()

    stage = omni.usd.get_context().get_stage()

    physics_scene_paths = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.Scene)
    ]

    log(f"Physics Scenes: {physics_scene_paths}")

    if len(physics_scene_paths) > 1:
        raise RuntimeError(
            "Multiple Physics Scenes found:\n"
            + "\n".join(physics_scene_paths)
            + "\nRemove the duplicate, save the USD, and try again."
        )

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

    # ---------- Initialize body ----------

    body_path = "/World/iris/body"

    if not stage.GetPrimAtPath(body_path).IsValid():
        raise RuntimeError(f"Drone body not found: {body_path}")

    body = world.scene.add(
        RigidPrim(
            prim_paths_expr=body_path,
            name="iris_body",
        )
    )

    world.reset()

    initial_positions, initial_orientations = body.get_world_poses()
    # initial_positions = initial_positions.copy()
    # initial_positions[:, 2] += 0.5

    body.set_default_state(
        positions=initial_positions.copy(),
        orientations=initial_orientations.copy(),
        linear_velocities=np.zeros((1, 3), dtype=np.float32),
        angular_velocities=np.zeros((1, 3), dtype=np.float32),
    )
    body.post_reset()

    # ---------- Read simulated mass ----------

    total_mass = 0.0
    mass_views = []

    log("\nDrone rigid-body masses:")

    for prim in stage.Traverse():
        path = str(prim.GetPath())

        if not path.startswith("/World/iris/"):
            continue

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)

        if not rigid_body_api.GetRigidBodyEnabledAttr().Get():
            continue

        view = RigidPrim(
            prim_paths_expr=path,
            name=f"mass_check_{len(mass_views)}",
            reset_xform_properties=False,
        )
        view.initialize()
        mass_views.append(view)

        mass = float(
            np.asarray(view.get_masses()).reshape(-1)[0]
        )

        log(f"{path}: {mass:.6f} kg")
        total_mass += mass

    if not np.isfinite(total_mass) or total_mass <= 0:
        raise RuntimeError(f"Invalid total drone mass: {total_mass}")

    log(f"Total drone rigid-body mass: {total_mass:.6f} kg\n")

    # ---------- Motor model ----------

    motors = MotorModel(
        thrust_coefficient=5.84e-6,
        torque_coefficient=0.06 * 5.84e-6,
        max_speed=1100.0,
        reaction_signs=[-1, -1, +1, +1],
    )

    # Body-frame positions relative to the authored body COM.
    rotor_positions = np.array([
        [ 0.13798545, -0.20671639, 0.023],
        [-0.12511168,  0.21875414, 0.023],
        [ 0.13800000,  0.20257577, 0.023],
        [-0.12415360, -0.22234580, 0.023],
    ])

    # ---------- Flight target ----------

    target = {
        "height": 1.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "mass": total_mass,
    }

    # ---------- Motor mixer ----------

    # Rows: thrust, roll torque, pitch torque, yaw torque.
    # Columns: rotor0, rotor1, rotor2, rotor3.
    allocation_matrix = np.vstack([
        np.ones(4),
        rotor_positions[:, 1],
        -rotor_positions[:, 0],
        motors.reaction_signs * motors.km / motors.kf,
    ])

    max_rotor_thrust = motors.kf * motors.max_speed**2

    def mix_motor_commands(
        total_thrust,
        roll_torque,
        pitch_torque,
        yaw_torque,
    ):
        desired_wrench = np.array([
            total_thrust,
            roll_torque,
            pitch_torque,
            yaw_torque,
        ], dtype=float)

        # Controller torques are desired about the whole vehicle COM.
        # The existing allocation matrix uses the body origin.
        com_offset = read_total_com_in_body_frame(body, mass_views)

        force = np.array([0.0, 0.0, total_thrust])

        # Convert the requested torque to the body-origin reference.
        desired_wrench[1:] += np.cross(com_offset, force)

        if not np.all(np.isfinite(desired_wrench)):
            raise ValueError(
                f"Controller returned invalid values: {desired_wrench}"
            )

        rotor_thrusts = np.linalg.solve(
            allocation_matrix, desired_wrench
        )

        saturated = bool(
            np.any(rotor_thrusts < 0.0)
            or np.any(rotor_thrusts > max_rotor_thrust)
        )

        rotor_thrusts = np.clip(
            rotor_thrusts, 0.0, max_rotor_thrust
        )

        speeds = np.sqrt(rotor_thrusts / motors.kf)

        return speeds, saturated

    # ---------- Apply motor forces ----------

    def apply_motor_commands(speeds):
        thrusts, reaction_torques = motors.calculate(speeds)

        rotor_forces = np.zeros((4, 3))
        rotor_forces[:, 2] = thrusts

        total_force = rotor_forces.sum(axis=0)

        total_torque = np.cross(
            rotor_positions, rotor_forces
        ).sum(axis=0)

        total_torque[2] += reaction_torques.sum()

        body.apply_forces_and_torques_at_pos(
            forces=total_force.astype(np.float32).reshape(1, 3),
            torques=total_torque.astype(np.float32).reshape(1, 3),
            is_global=False,
        )

    # ---------- Read and record state ----------

    def read_drone_state():
        positions, orientations = body.get_world_poses()
        linear_velocities = body.get_linear_velocities()
        angular_velocities = body.get_angular_velocities()

        return {
            "position": positions[0].copy(),
            "orientation": orientations[0].copy(),
            "linear_velocity": linear_velocities[0].copy(),
            "angular_velocity": angular_velocities[0].copy(),
        }

    def log_drone_state(state, elapsed):
        log(
            f"\nDrone state at t={elapsed:.3f} s\n"
            f"Position [m]: {state['position']}\n"
            f"Orientation [w, x, y, z]: {state['orientation']}\n"
            f"Linear velocity [m/s]: {state['linear_velocity']}\n"
            f"Angular velocity [rad/s]: {state['angular_velocity']}"
        )

    # ---------- Feedback control loop ----------

    elapsed = 0.0
    step_count = 0
    reset_requested = False

    log(
        "Feedback control running.\n"
        "Stop -> Play resets. Pause -> Play resumes."
    )

    while simulation_app.is_running():
        if world.is_stopped():
            reset_requested = True
            simulation_app.update()
            continue

        if not world.is_playing():
            simulation_app.update()
            continue

        if reset_requested:
            world.reset()

            elapsed = 0.0
            step_count = 0
            reset_requested = False

            log("\nSimulation reset.")

        # 1. Read current state.
        state = read_drone_state()

        # 2. Calculate corrective thrust and torques.
        total_thrust, roll_torque, pitch_torque, yaw_torque = controller(
            positions=state["position"],
            orientations=state["orientation"],
            velocities=state["linear_velocity"],
            angular_velocities=state["angular_velocity"],
            target=target,
        )

        # 3. Calculate four motor speeds.
        speeds, saturated = mix_motor_commands(
            total_thrust,
            roll_torque,
            pitch_torque,
            yaw_torque,
        )

        # 4. Apply forces and advance physics.
        apply_motor_commands(speeds)
        world.step(render=True)

        elapsed += dt
        step_count += 1

        # 5. Record the first step, then every 30 steps.
        if step_count == 1 or step_count % 30 == 0:
            updated_state = read_drone_state()

            log_drone_state(updated_state, elapsed)

            com_offset, offset_torque = diagnose_thrust_offset(
                body, mass_views, motors, speeds
            )

            log(
                f"Vehicle COM relative to body origin [m]: "
                f"{com_offset}\n"
                f"Torque from thrust offset [N m]: "
                f"{offset_torque}"
            )

            log(
                f"Target height [m]: {target['height']:.3f}\n"
                f"Height error [m]: "
                f"{target['height'] - updated_state['position'][2]:.3f}\n"
                f"Motor commands [rad/s]: {np.round(speeds, 1)}\n"
                f"Requested thrust [N]: {total_thrust:.3f}\n"
                f"Requested torques [N m]: "
                f"{roll_torque:.3f}, "
                f"{pitch_torque:.3f}, "
                f"{yaw_torque:.3f}\n"
                f"Motor limits reached: {saturated}"
            )


try:
    log_path = Path(__file__).resolve().parents[1] / "flight_log.txt"

    print(f"Saving flight log to: {log_path}", flush=True)

    with open(log_path, "w", encoding="utf-8") as log_file:
        main(log_file)

finally:
    simulation_app.close()