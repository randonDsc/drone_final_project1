"""
Simulation entry point:

USD scene
    -> occupancy grid
    -> 26-connected 3D Dijkstra
    -> collision-checked path shortening
    -> world-coordinate waypoints
    -> flight controller
    -> motor mixer
    -> simulated forces (+ optional gusting-wind drag, wind.py)

Run with Isaac Sim's python.bat.

The starting pose comes from the USD.
The final destination comes from the target dictionary below.

Stop -> Play resets and replans.
Pause -> Play resumes.
"""

from pathlib import Path
import traceback

from isaacsim import SimulationApp


simulation_app = SimulationApp({
    "headless": False,
})


def main(log_file):
    # Isaac Sim-dependent imports must follow SimulationApp startup.
    import numpy as np
    import omni.usd

    from pxr import UsdPhysics

    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation, RigidPrim

    import sim_config

    from mapping import (
        read_scene_geometry,
        build_occupancy_map,
    )

    from navigation import (
        plan_flight,
        WaypointFollower,
    )

    from sensors import SensorSuite
    from motor import MotorModel
    from controller import controller
    from wind import GustingWind

    from drone_diagnostics import (
        diagnose_thrust_offset,
        read_total_com_in_body_frame,
    )

    def log(message):
        """Print to the terminal and save the same message to the log."""
        print(
            message,
            flush=True,
        )

        print(
            message,
            file=log_file,
            flush=True,
        )

    # ---------- Load scene ----------

    project_dir = Path(__file__).resolve().parents[1]

    scene_path = (
        project_dir
        / "sim_scene"
        / "dij3d_boxes.usd"
    )

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

    # Physics/controller frequency: 120 Hz.
    dt = 1.0 / 120.0

    world = World(
        physics_dt=dt,
        rendering_dt=dt,
        stage_units_in_meters=1.0,
        physics_prim_path=physics_scene_path,
    )

    # ---------- Initialize the physical drone body ----------

    body_path = sim_config.BODY_PATH

    if not stage.GetPrimAtPath(body_path).IsValid():
        raise RuntimeError(
            f"Drone body not found: {body_path}"
        )

    body = world.scene.add(
        RigidPrim(
            prim_paths_expr=body_path,
            name="iris_body",
        )
    )

    # ---------- Propeller spin drives (visual only) ----------

    # The rotors are separate rigid bodies on free revolute joints with no
    # drive, so nothing turns them. Add a velocity drive to each joint in
    # the loaded stage. This changes only the in-memory stage, never the
    # saved USD files, unless you save the stage from the GUI yourself.
    propellers = None
    propeller_dofs = None

    if sim_config.PROPELLER_SPIN:
        for joint_path in sim_config.ROTOR_JOINT_PATHS:
            joint_prim = stage.GetPrimAtPath(joint_path)

            if not joint_prim.IsA(UsdPhysics.RevoluteJoint):
                raise RuntimeError(
                    f"Rotor joint missing or not revolute: {joint_path}"
                )

            # Pure velocity drive: no stiffness, high damping, capped torque.
            drive = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            drive.CreateDampingAttr(1.0e3)
            drive.CreateMaxForceAttr(
                sim_config.PROPELLER_DRIVE_MAX_TORQUE
            )
            drive.CreateTargetVelocityAttr(0.0)

        # Must be added before world.reset() so it gets initialized.
        propellers = world.scene.add(
            Articulation(
                prim_paths_expr=sim_config.DRONE_ROOT,
                name="iris_propellers",
            )
        )

    world.reset()

    if propellers is not None:
        propeller_dofs = np.array([
            propellers.get_dof_index(
                joint_path.rsplit("/", 1)[-1]
            )
            for joint_path in sim_config.ROTOR_JOINT_PATHS
        ])

    initial_positions, initial_orientations = (
        body.get_world_poses()
    )

    initial_positions = initial_positions.copy()
    initial_orientations = initial_orientations.copy()

    # Save the pose read from the scene for later resets.
    # No separate manually chosen starting position is introduced.
    body.set_default_state(
        positions=initial_positions.copy(),
        orientations=initial_orientations.copy(),
        linear_velocities=np.zeros(
            (1, 3),
            dtype=np.float32,
        ),
        angular_velocities=np.zeros(
            (1, 3),
            dtype=np.float32,
        ),
    )

    body.post_reset()

    # Keep physics paused while mapping and Dijkstra run.
    world.pause()

    # ---------- Read total vehicle mass ----------

    total_mass = 0.0
    mass_views = []

    log("\nDrone rigid-body masses:")

    drone_prefix = (
        sim_config.DRONE_ROOT.rstrip("/")
        + "/"
    )

    for prim in stage.Traverse():
        path = str(prim.GetPath())

        if not path.startswith(drone_prefix):
            continue

        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue

        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)

        if (
            rigid_body_api
            .GetRigidBodyEnabledAttr()
            .Get()
            is False
        ):
            continue

        # These views are also used to calculate the full vehicle COM.
        view = RigidPrim(
            prim_paths_expr=path,
            name=f"mass_check_{len(mass_views)}",
            reset_xform_properties=False,
        )

        view.initialize()
        mass_views.append(view)

        mass = float(
            np.asarray(
                view.get_masses()
            ).reshape(-1)[0]
        )

        log(f"{path}: {mass:.6f} kg")

        total_mass += mass

    if (
        not np.isfinite(total_mass)
        or total_mass <= 0
    ):
        raise RuntimeError(
            f"Invalid total drone mass: {total_mass}"
        )

    log(
        "Total drone rigid-body mass: "
        f"{total_mass:.6f} kg\n"
    )

    # ---------- Motor model ----------

    # Preserve the motor parameters and reaction-sign ordering
    # from your uploaded working project.
    motors = MotorModel(
        thrust_coefficient=5.84e-6,
        torque_coefficient=0.06 * 5.84e-6,
        max_speed=1100.0,
        reaction_signs=[-1, -1, +1, +1],
    )

    # Rotor application positions in body coordinates [m].
    rotor_positions = np.array([
        [0.13798545, -0.20671639, 0.023],
        [-0.12511168, 0.21875414, 0.023],
        [0.13800000, 0.20257577, 0.023],
        [-0.12415360, -0.22234580, 0.023],
    ])

    # ---------- Final destination: edit this block ----------

    target = {
        # Absolute world Z, not height above the initial pose.
        "height": 1.12,

        # Final world XY, relative to the initial USD pose:
        # initial X - 5 m, initial Y + 0 m.
        "xy": (
            initial_positions[0, :2].astype(float).copy()
            + np.array([5.93, 1.12])
        ),

        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "mass": total_mass,
    }

    # Shared controller limits.
    target["max_xy_speed"] = sim_config.MAX_XY_SPEED
    target["max_z_speed"] = sim_config.MAX_Z_SPEED

    target["max_tilt_degrees"] = (
        sim_config.MAX_TILT_DEGREES
    )

    target["max_total_thrust"] = (
        4
        * motors.kf
        * motors.max_speed**2
    )

    if total_mass * 9.81 >= target["max_total_thrust"]:
        raise RuntimeError(
            "Configured motors cannot support "
            "the measured drone mass."
        )

    # The mission destination stays unchanged while intermediate
    # waypoint targets are passed to the controller.
    goal = np.array([
        target["xy"][0],
        target["xy"][1],
        target["height"],
    ], dtype=float)

    output_dir = (
        project_dir
        / "navigation_output"
    )

    output_dir.mkdir(exist_ok=True)

    # ---------- Build map and plan route ----------

    def make_navigation():
        """
        Read the actual current/reset pose, map the scene,
        and compute the route before flight starts.
        """

        world.pause()

        positions, _ = body.get_world_poses()

        start = np.asarray(
            positions[0],
            dtype=float,
        ).copy()

        log(
            "Planning from USD pose: "
            f"start={start}, destination={goal}"
        )

        geometry = read_scene_geometry(
            stage,
            start,
        )

        log(
            "Drone envelope half-size [m]: "
            f"{geometry.drone_half_size}"
        )

        # Save source geometry before map creation.
        # This report remains useful if a map-size check fails.
        obstacle_report_path = (
            output_dir
            / "occupancy_obstacles.txt"
        )

        obstacle_report_path.write_text(
            "\n".join(geometry.report),
            encoding="utf-8",
        )

        occupancy = build_occupancy_map(
            geometry,
            start,
            goal,
        )

        log(
            f"Map origin={occupancy.origin}; "
            f"upper={occupancy.upper}; "
            f"shape={occupancy.grid.shape}; "
            "blocked cells="
            f"{np.count_nonzero(occupancy.grid):,}"
        )

        # grid[ix, iy, iz]:
        #   0 = free
        #   1 = blocked after vehicle-clearance inflation
        np.savez_compressed(
            output_dir / "occupancy_grid.npz",
            grid=occupancy.grid,
            origin=occupancy.origin,
            cell_size=occupancy.resolution,
            drone_half_size=geometry.drone_half_size,
            safety_margin=sim_config.SAFETY_MARGIN,
            goal=goal,
            start=start,
        )

        def progress(expanded):
            # Render/update the application during search,
            # while keeping physics paused.
            world.pause()
            simulation_app.update()

            if not simulation_app.is_running():
                raise RuntimeError(
                    "Simulation closed during planning."
                )

        plan = plan_flight(
            occupancy,
            geometry,
            start,
            goal,
            progress=progress,
        )

        np.savetxt(
            output_dir / "path_cells.csv",
            plan.cells,
            fmt="%d",
            delimiter=",",
            header="ix,iy,iz",
            comments="",
        )

        np.savetxt(
            output_dir / "waypoints_world.csv",
            plan.waypoints,
            delimiter=",",
            header="x_m,y_m,z_m",
            comments="",
        )

        # grid_distance: Dijkstra cell-center path (entry cell to goal cell).
        # distance: shortened route actually flown, including departure.
        log(
            f"Dijkstra expanded {plan.expanded:,} cells. "
            f"Grid path length={plan.grid_distance:.3f} m "
            f"over {len(plan.cells)} cells; "
            f"shortened route length={plan.distance:.3f} m; "
            f"waypoints={len(plan.waypoints)}"
        )

        for index, waypoint in enumerate(plan.waypoints):
            log(
                f"  Waypoint {index + 1}: {waypoint}"
            )

        log(f"Navigation outputs: {output_dir}")

        return (
            occupancy,
            WaypointFollower(plan.waypoints),
        )

    occupancy, follower = make_navigation()

    # ---------- Motor mixer ----------

    # Each column maps one rotor's thrust into:
    #
    # [total thrust, roll torque, pitch torque, yaw torque]
    #
    # For body +Z thrust at position [x, y, z]:
    #   roll torque  =  y * thrust
    #   pitch torque = -x * thrust
    #
    # Reaction torque contributes yaw.
    allocation_matrix = np.vstack([
        np.ones(4),
        rotor_positions[:, 1],
        -rotor_positions[:, 0],
        motors.reaction_signs * motors.km / motors.kf,
    ])

    max_rotor_thrust = (
        motors.kf
        * motors.max_speed**2
    )

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

        com_offset = read_total_com_in_body_frame(
            body,
            mass_views,
        )

        force = np.array([
            0.0,
            0.0,
            total_thrust,
        ])

        # Shift torque reference from full-vehicle COM to body origin.
        desired_wrench[1:] += np.cross(
            com_offset,
            force,
        )

        if not np.all(np.isfinite(desired_wrench)):
            raise ValueError(
                "Controller returned invalid values: "
                f"{desired_wrench}"
            )

        rotor_thrusts = np.linalg.solve(
            allocation_matrix,
            desired_wrench,
        )

        saturated = bool(
            np.any(rotor_thrusts < 0.0)
            or np.any(rotor_thrusts > max_rotor_thrust)
        )

        # Clipping means the exact requested wrench may not
        # be achievable. Keep logging the saturation flag.
        rotor_thrusts = np.clip(
            rotor_thrusts,
            0.0,
            max_rotor_thrust,
        )

        # T = kf * omega^2  ->  omega = sqrt(T / kf).
        speeds = np.sqrt(
            rotor_thrusts / motors.kf
        )

        return speeds, saturated

    # ---------- Gusting wind ----------

    wind = GustingWind(**sim_config.WIND)

    log(
        f"Wind enabled={wind.enabled}; mean={wind.mean_wind} m/s; "
        f"gust std={wind.gust_std} m/s over {wind.gust_timescale} s; "
        f"local std={wind.local_gust_std} m/s over "
        f"{wind.local_gust_timescale} s; "
        f"correlation length={wind.correlation_length} m; "
        f"Cd*A={wind.drag_coefficient * wind.effective_area:.4f} m^2 "
        f"over {len(wind.offsets)} points; seed={wind.seed}"
    )

    def read_body_com():
        """Body rigid-body COM relative to its origin, body frame [m]."""
        local_coms, _ = body.get_coms()
        return np.asarray(local_coms, dtype=float).reshape(3)

    body_com = read_body_com()

    def wind_wrench(state, step_dt):
        """Advance the gusts by one physics step; return body-frame drag.

        The point forces are reduced to the exact equivalent force and
        torque about the body origin, which is where the motor wrench is
        applied (no position_data). Both go out in one API call: the
        force-at-position API takes one point per body per call, and
        whether repeated calls in a step accumulate is undocumented.
        """
        if not wind.enabled:
            return np.zeros(3), np.zeros(3)

        wind.step(step_dt)

        # PhysX reports the velocity of the body's center of mass.
        return wind.body_wrench(
            state["position"],
            state["orientation"],
            state["linear_velocity"],
            state["angular_velocity"],
            body_com,
        )

    # ---------- Apply simulated motor forces ----------

    def apply_motor_commands(
        speeds,
        extra_force=np.zeros(3),
        extra_torque=np.zeros(3),
    ):
        """Apply motor thrust plus an extra body-frame wrench.

        extra_torque must be about the body origin (here: wind drag).
        """
        thrusts, reaction_torques = motors.calculate(
            speeds
        )

        rotor_forces = np.zeros((4, 3))
        rotor_forces[:, 2] = thrusts

        total_force = rotor_forces.sum(axis=0)

        # Sum moment-arm torques r x F.
        total_torque = np.cross(
            rotor_positions,
            rotor_forces,
        ).sum(axis=0)

        # Add yaw reaction torque.
        total_torque[2] += reaction_torques.sum()

        total_force = total_force + extra_force
        total_torque = total_torque + extra_torque

        # Force and torque are expressed in body coordinates.
        body.apply_forces_and_torques_at_pos(
            forces=total_force.astype(
                np.float32
            ).reshape(1, 3),
            torques=total_torque.astype(
                np.float32
            ).reshape(1, 3),
            is_global=False,
        )

    # ---------- Animate propellers ----------

    # A prop turns opposite to the reaction torque it puts on the body:
    # reaction sign -1 -> counter-clockwise (+Z), +1 -> clockwise.
    # Rotors 0, 1 are the iris_prop_ccw meshes; rotors 2, 3 iris_prop_cw.
    spin_directions = -np.asarray(
        motors.reaction_signs,
        dtype=float,
    )

    def spin_propellers(speeds):
        if propellers is None:
            return

        # Displayed speed follows each motor command, scaled down.
        targets = (
            spin_directions
            * np.asarray(speeds, dtype=float)
            * sim_config.PROPELLER_SPIN_SCALE
        )

        propellers.set_joint_velocity_targets(
            targets.reshape(1, -1),
            joint_indices=propeller_dofs,
        )

    # ---------- Read simulator ground truth ----------

    def read_drone_state():
        positions, orientations = body.get_world_poses()

        linear_velocities = (
            body.get_linear_velocities()
        )

        angular_velocities = (
            body.get_angular_velocities()
        )

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
            "Orientation [w, x, y, z]: "
            f"{state['orientation']}\n"
            "Linear velocity [m/s]: "
            f"{state['linear_velocity']}\n"
            "Angular velocity [rad/s]: "
            f"{state['angular_velocity']}"
        )

    # Sensors are optional logging devices.
    # They do not currently supply the planner's map/state.
    sensors = SensorSuite(
        stage,
        enabled=sim_config.ENABLE_SENSORS,
        log=log,
    )

    try:
        world.play()

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

                # Refresh views used for COM/mass calculations.
                for view in mass_views:
                    view.initialize()

                body_com = read_body_com()

                # Same seed -> the same gust sequence after every reset.
                wind.reset()

                elapsed = 0.0
                step_count = 0
                reset_requested = False

                occupancy, follower = make_navigation()

                world.play()

                log(
                    "\nSimulation reset and route replanned."
                )

            # 1. Read the current state.
            state = read_drone_state()

            # 2. Select the active waypoint.
            previous_index = follower.index
            was_done = follower.done

            waypoint = follower.update(
                state["position"],
                state["linear_velocity"],
                dt,
            )

            # Keep final destination separate from temporary commands.
            active_target = dict(target)

            active_target["xy"] = (
                waypoint[:2].copy()
            )

            active_target["height"] = float(
                waypoint[2]
            )

            if follower.index != previous_index:
                # Log the retained waypoint, not the sliding command.
                log(
                    "Advancing to waypoint "
                    f"{follower.index + 1}: "
                    f"{follower.waypoints[follower.index]}"
                )

            if follower.done and not was_done:
                log(
                    "Destination reached and settled. "
                    "Holding final position."
                )

            # 3. Controller: state + waypoint -> thrust and torques.
            (
                total_thrust,
                roll_torque,
                pitch_torque,
                yaw_torque,
            ) = controller(
                positions=state["position"],
                orientations=state["orientation"],
                velocities=state["linear_velocity"],
                angular_velocities=state["angular_velocity"],
                target=active_target,
            )

            # 4. Mixer: requested wrench -> four rotor speeds.
            speeds, saturated = mix_motor_commands(
                total_thrust,
                roll_torque,
                pitch_torque,
                yaw_torque,
            )

            # 5. Wind drag from the same pre-step state, using the
            #    simulator's actual physics timestep.
            wind_force, wind_torque = wind_wrench(
                state,
                world.get_physics_dt(),
            )

            # 6. Apply forces and advance simulation.
            apply_motor_commands(
                speeds,
                wind_force,
                wind_torque,
            )
            spin_propellers(speeds)
            world.step(render=True)

            elapsed += dt
            step_count += 1

            # Log the first step, then every 30 steps.
            if (
                step_count == 1
                or step_count % 30 == 0
            ):
                updated_state = read_drone_state()

                log_drone_state(
                    updated_state,
                    elapsed,
                )

                sensors.log_readings()

                # On sloped segments the command slides along the line,
                # so it can differ from the retained waypoint.
                log(
                    f"Waypoint {follower.index + 1}/"
                    f"{len(follower.waypoints)}: "
                    f"{follower.waypoints[follower.index]}; "
                    f"command={waypoint}; "
                    f"final goal={goal}; "
                    f"arrived={follower.done}"
                )

                com_offset, offset_torque = (
                    diagnose_thrust_offset(
                        body,
                        mass_views,
                        motors,
                        speeds,
                    )
                )

                log(
                    "Vehicle COM relative to body origin [m]: "
                    f"{com_offset}\n"
                    "Torque from thrust offset [N m]: "
                    f"{offset_torque}"
                )

                height_error = (
                    active_target["height"]
                    - updated_state["position"][2]
                )

                xy_error = (
                    active_target["xy"]
                    - updated_state["position"][:2]
                )

                horizontal_speed = np.linalg.norm(
                    updated_state["linear_velocity"][:2]
                )

                log(
                    "Active target height [m]: "
                    f"{active_target['height']:.3f}\n"
                    f"Height error [m]: {height_error:.3f}\n"
                    "Motor commands [rad/s]: "
                    f"{np.round(speeds, 1)}\n"
                    f"Requested thrust [N]: {total_thrust:.3f}\n"
                    "Requested torques [N m]: "
                    f"{roll_torque:.3f}, "
                    f"{pitch_torque:.3f}, "
                    f"{yaw_torque:.3f}\n"
                    f"Motor limits reached: {saturated}\n"
                    f"Target XY [m]: {active_target['xy']}\n"
                    f"XY error [m]: {xy_error}\n"
                    "Horizontal speed [m/s]: "
                    f"{horizontal_speed:.3f}"
                )

                if wind.enabled:
                    log(
                        "Wind at hull [m/s]: "
                        f"{np.round(wind.point_winds[0], 3)}; "
                        "point spread [m/s]: "
                        f"{np.round(np.ptp(wind.point_winds, axis=0), 3)}\n"
                        "Wind force, body [N]: "
                        f"{np.round(wind_force, 4)}\n"
                        "Wind torque about body origin, body [N m]: "
                        f"{np.round(wind_torque, 5)}"
                    )

    finally:
        world.pause()
        sensors.close()


try:
    log_path = (
        Path(__file__).resolve().parents[1]
        / "flight_log.txt"
    )

    print(
        f"Saving flight log to: {log_path}",
        flush=True,
    )

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as log_file:
        try:
            main(log_file)

        except Exception:
            # Print/save the traceback BEFORE closing Isaac Sim.
            error = traceback.format_exc()

            print(
                "\nFLIGHT FAILED:\n" + error,
                flush=True,
            )

            print(
                "\nFLIGHT FAILED:\n" + error,
                file=log_file,
                flush=True,
            )

finally:
    simulation_app.close()