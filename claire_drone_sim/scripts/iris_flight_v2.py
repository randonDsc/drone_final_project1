"""DSC 180A: Iris point-to-point flight + experiment timestamps + simple wind.

Runs with Isaac Sim's python.bat, not a normal virtual environment.
Your scene must contain /iris/body and its rotor rigid bodies.
Controller, motor model, and COM helper are based on the team's working Iris baseline.
This script does not modify or save the USD scene.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# Isaac Sim imports must happen AFTER SimulationApp is constructed.
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim

from controller import controller
from motor import MotorModel
from drone_diagnostics import read_total_com_in_body_frame, rotate_vector

PROJECT = Path(__file__).resolve().parents[1]
SCENE_PATH = PROJECT / "scenes" / "my_world_with_iris_drone.usd"
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# Each trial gets its own ID and outputs. Use UTC to avoid time zone ambiguity.
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
CSV_PATH = RESULTS_DIR / f"iris_{RUN_ID}.csv"
METADATA_PATH = RESULTS_DIR / f"iris_{RUN_ID}.json"

# --- Experiment parameters: change ONE at a time to make fair comparisons ---
START_HEIGHT_M = 1.5
TARGET_OFFSET_XY_M = np.array([8.0, 0.0])  # longer: 8 m along world X
TARGET_HEIGHT_M = 1.5
PHYSICS_HZ = 120
MAX_SIM_SECONDS = 40.0

# 'calm' or 'steady' or 'gust'. Start with 'calm', then try 'steady'.
WIND_MODE = "gust"
WIND_START_S = 5.0
STEADY_WIND_MPS = np.array([0.0, 3.0, 0.0])  # crosswind along world +Y
LINEAR_DRAG_KG_PER_S = 0.25


def wind_velocity_world(sim_time_s):
    """Desired air velocity in world frame, m/s. No fluid flow is simulated."""
    if WIND_MODE == "calm" or sim_time_s < WIND_START_S:
        return np.zeros(3, dtype=float)
    if WIND_MODE == "steady":
        return STEADY_WIND_MPS.copy()
    if WIND_MODE == "gust":
        return np.array([0.0, 3.0 + 1.5 * np.sin(2 * np.pi * 0.25 * sim_time_s), 0.0])
    raise ValueError(f"Unsupported WIND_MODE: {WIND_MODE}")

try:
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"Scene not found: {SCENE_PATH}")

    print(f"Opening personal scene: {SCENE_PATH}", flush=True)
    omni.usd.get_context().open_stage(str(SCENE_PATH))
    simulation_app.update()
    stage = omni.usd.get_context().get_stage()

    # Your imported Iris is under /iris, not /World/iris.
    iris_prim = stage.GetPrimAtPath("/iris")
    body_prim = stage.GetPrimAtPath("/iris/body")
    if not iris_prim.IsValid() or not body_prim.IsValid():
        raise RuntimeError("Expected /iris and /iris/body in your personal USD scene.")
    if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("/iris/body has no RigidBodyAPI.")

    # Move the entire drone root up as a group so it doesn't start inside the floor.
    # This changes only the loaded stage in memory. It does NOT save the USD file.
    UsdGeom.XformCommonAPI(iris_prim).SetTranslate(Gf.Vec3d(0, 0, START_HEIGHT_M))

    physics_scenes = [str(p.GetPath()) for p in stage.Traverse()
                      if p.IsA(UsdPhysics.Scene)]
    if len(physics_scenes) > 1:
        raise RuntimeError(f"Expected at most one physics scene: {physics_scenes}")
    physics_path = physics_scenes[0] if physics_scenes else "/World/PhysicsScene"

    dt = 1.0 / PHYSICS_HZ
    world = World(
        physics_dt=dt,
        rendering_dt=dt,
        stage_units_in_meters=1.0,
        physics_prim_path=physics_path,
    )
    body = world.scene.add(RigidPrim(prim_paths_expr="/iris/body", name="personal_iris_body"))
    world.reset()

    positions, orientations = body.get_world_poses()
    body.set_default_state(
        positions=positions.copy(),
        orientations=orientations.copy(),
        linear_velocities=np.zeros((1, 3), dtype=np.float32),
        angular_velocities=np.zeros((1, 3), dtype=np.float32),
    )
    body.post_reset()

    # Read the mass of all enabled Iris rigid bodies, just as in the working baseline.
    total_mass = 0.0
    mass_views = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/iris/") or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        if rigid_api.GetRigidBodyEnabledAttr().Get() is False:
            continue
        view = RigidPrim(prim_paths_expr=path,
                         name=f"mass_view_{len(mass_views)}",
                         reset_xform_properties=False)
        view.initialize()
        mass_views.append(view)
        part_mass = float(np.asarray(view.get_masses()).reshape(-1)[0])
        total_mass += part_mass
        print(f"Mass {path}: {part_mass:.5f} kg")
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise RuntimeError(f"Bad Iris total mass: {total_mass}")

    motors = MotorModel(
        thrust_coefficient=5.84e-6,
        torque_coefficient=0.06 * 5.84e-6,
        max_speed=1100.0,
        reaction_signs=[-1, -1, +1, +1],
    )

    # These rotor coordinates and constants are inherited from the team's Iris model.
    # They are not generic quadcopter parameters.
    rotor_positions = np.array([
        [ 0.13798545, -0.20671639, 0.023],
        [-0.12511168,  0.21875414, 0.023],
        [ 0.13800000,  0.20257577, 0.023],
        [-0.12415360, -0.22234580, 0.023],
    ])
    allocation_matrix = np.vstack([
        np.ones(4),
        rotor_positions[:, 1],
        -rotor_positions[:, 0],
        motors.reaction_signs * motors.km / motors.kf,
    ])
    max_rotor_thrust = motors.kf * motors.max_speed**2

    start_xy = body.get_world_poses()[0][0, :2].astype(float).copy()
    target_xy = start_xy + TARGET_OFFSET_XY_M
    target = dict(height=TARGET_HEIGHT_M, xy=target_xy,
                  roll=0.0, pitch=0.0, yaw=0.0, mass=total_mass)
    print(f"START XY: {start_xy}; TARGET XY: {target_xy}; Z: {TARGET_HEIGHT_M}")
    print(f"Total mass: {total_mass:.4f} kg; writing: {CSV_PATH}")
    print(f"RUN_ID: {RUN_ID}; WIND_MODE: {WIND_MODE}")

    # Save the settings associated with this trial, in a separate file.
    metadata = {
        "run_id": RUN_ID,
        "start_wall_time_utc": datetime.now(timezone.utc).isoformat(),
        "scene": str(SCENE_PATH),
        "controller": "team-derived PD controller",
        "simulation_hz": PHYSICS_HZ,
        "max_sim_seconds": MAX_SIM_SECONDS,
        "start_height_m": START_HEIGHT_M,
        "target_offset_xy_m": TARGET_OFFSET_XY_M.tolist(),
        "target_xy_m": target_xy.tolist(),
        "target_height_m": TARGET_HEIGHT_M,
        "total_mass_kg": total_mass,
        "wind_mode": WIND_MODE,
        "wind_start_s": WIND_START_S,
        "steady_wind_mps": STEADY_WIND_MPS.tolist(),
        "linear_drag_kg_per_s": LINEAR_DRAG_KG_PER_S,
        "data_type": "physics-engine ground truth (not IMU/LiDAR)",
        "wind_model": "horizontal force = drag_coefficient * (wind_velocity - drone_velocity); no aerodynamic CFD",
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "wall_time_utc", "time_s",
                         "x", "y", "z", "vx", "vy", "vz",
                         "target_x", "target_y", "target_z", "error_m",
                         "motor_0_rad_s", "motor_1_rad_s", "motor_2_rad_s",
                         "motor_3_rad_s", "saturated", "wind_mode",
                         "wind_vx_mps", "wind_vy_mps", "wind_vz_mps",
                         "air_force_x_n", "air_force_y_n", "air_force_z_n"])

        world.play()
        for step in range(round(MAX_SIM_SECONDS * PHYSICS_HZ)):
            if not simulation_app.is_running():
                break
            if not world.is_playing():
                simulation_app.update()
                continue

            pos, quat = body.get_world_poses()
            velocity = body.get_linear_velocities()
            angular_velocity = body.get_angular_velocities()

            total_thrust, roll_torque, pitch_torque, yaw_torque = controller(
                positions=pos[0], orientations=quat[0],
                velocities=velocity[0], angular_velocities=angular_velocity[0],
                target=target,
            )

            desired_wrench = np.array([
                total_thrust, roll_torque, pitch_torque, yaw_torque
            ], dtype=float)
            com_offset = read_total_com_in_body_frame(body, mass_views)
            desired_wrench[1:] += np.cross(
                com_offset, [0.0, 0.0, total_thrust]
            )
            if not np.all(np.isfinite(desired_wrench)):
                raise ValueError(f"Invalid controller wrench: {desired_wrench}")

            rotor_thrusts = np.linalg.solve(allocation_matrix, desired_wrench)
            saturated = bool(np.any((rotor_thrusts < 0) |
                                    (rotor_thrusts > max_rotor_thrust)))
            rotor_thrusts = np.clip(rotor_thrusts, 0.0, max_rotor_thrust)
            rotor_speeds = np.sqrt(rotor_thrusts / motors.kf)

            thrusts, reaction_torques = motors.calculate(rotor_speeds)
            rotor_forces = np.zeros((4, 3))
            rotor_forces[:, 2] = thrusts
            force = rotor_forces.sum(axis=0)
            torque = np.cross(rotor_positions, rotor_forces).sum(axis=0)
            torque[2] += reaction_torques.sum()
            # A simplified wind / drag model in world coordinates. This is not CFD.
            sim_time_s = step * dt
            v_wind = wind_velocity_world(sim_time_s)
            air_force_world = LINEAR_DRAG_KG_PER_S * (v_wind - velocity[0])
            air_force_world[2] = 0.0  # horizontal drag only

            # Do NOT make two independent force API calls on this body per step:
            # RigidPrim tensor APIs may overwrite earlier calls within the step.
            # Instead, transform the wind force to the drone's local frame,
            # combine it with thrust, and issue ONE force/torque command.
            q_inv = np.asarray(quat[0], dtype=float).copy()  # quaternion wxyz
            q_inv[1:] *= -1.0
            air_force_body = rotate_vector(q_inv, air_force_world)
            combined_force_body = force + air_force_body

            # Model aerodynamic force through the center of mass, rather than
            # generating an extra torque from an arbitrary application point.
            combined_torque_body = torque + np.cross(com_offset, air_force_body)

            body.apply_forces_and_torques_at_pos(
                forces=combined_force_body.astype(np.float32).reshape(1, 3),
                torques=combined_torque_body.astype(np.float32).reshape(1, 3),
                is_global=False,
            )
            world.step(render=True)

            # Record every 4th physics step (30 measurements/simulated second).
            if step % 4 == 0:
                pos_now = body.get_world_poses()[0][0]
                v_now = body.get_linear_velocities()[0]
                err = float(np.linalg.norm(pos_now - np.array([
                    target_xy[0], target_xy[1], TARGET_HEIGHT_M
                ])))
                writer.writerow([RUN_ID,
                                 datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                 round((step + 1) * dt, 4),
                                 *map(float, pos_now), *map(float, v_now),
                                 float(target_xy[0]), float(target_xy[1]),
                                 TARGET_HEIGHT_M, err,
                                 *map(float, rotor_speeds), int(saturated), WIND_MODE,
                                 *map(float, v_wind), *map(float, air_force_world)])
                if step % (PHYSICS_HZ * 2) == 0:
                    print(f"t={(step + 1)*dt:5.1f}s position={np.round(pos_now, 2)} "
                          f"target={np.round(target_xy, 2)} error={err:.2f}m",
                          flush=True)

        print(f"Flight loop finished. Data: {CSV_PATH}; metadata: {METADATA_PATH}", flush=True)
        print("Close the Isaac Sim window to finish.", flush=True)
        world.pause()
        while simulation_app.is_running():
            simulation_app.update()

finally:
    simulation_app.close()
