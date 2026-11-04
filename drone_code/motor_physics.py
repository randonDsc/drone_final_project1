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

sim_config.CONTROL_MODE selects "pid" or "pid_sac". See CONTROL.md.
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
    from controller import PIDController
    from drone_sim import DroneSim

    from drone_diagnostics import diagnose_thrust_offset

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

    # ---------- Load scene, drone, motors, and wind ----------

    project_dir = Path(__file__).resolve().parents[1]

    scene_path = (
        project_dir
        / "sim_scene"
        / "dij3d_boxes.usd"
    )

    log(f"Running script: {Path(__file__).resolve()}")

    sim = DroneSim(
        simulation_app,
        scene_path,
        log=log,
    )

    world = sim.world
    stage = sim.stage
    body = sim.body
    wind = sim.wind
    dt = sim.dt

    initial_positions = sim.initial_positions

    # ---------- Final destination: edit this block ----------

    target = sim.base_target()

    # Absolute world Z, not height above the initial pose.
    target["height"] = 1.12

    # Final world XY, relative to the initial USD pose:
    # initial X - 5 m, initial Y + 0 m.
    target["xy"] = (
        initial_positions[0, :2].astype(float).copy()
        + np.array([5.93, 1.12])
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

    # ---------- Flight controller ----------

    pid = PIDController()

    if sim_config.CONTROL_MODE not in ("pid", "pid_sac"):
        raise ValueError(f"Unknown control mode: {sim_config.CONTROL_MODE}")

    # None flies PID only. The SAC code is imported only when selected.
    residual = None

    if sim_config.CONTROL_MODE == "pid_sac":
        from sac_policy import load_flight_residual

        residual = load_flight_residual(project_dir, sim, log)

    log(
        "Control mode: "
        + ("PID + SAC correction" if residual is not None else "PID only")
    )

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

        pid.reset()

        if residual is not None:
            residual.reset()

        return (
            occupancy,
            WaypointFollower(plan.waypoints),
        )

    occupancy, follower = make_navigation()

    # ---------- Read simulator ground truth ----------

    read_drone_state = sim.read_drone_state

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

        # The mixer's result from the last step, for PID anti-windup.
        saturated = False

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
                sim.reset_after_stop()
                elapsed = 0.0
                step_count = 0
                reset_requested = False
                saturated = False

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
            pid_command = np.array(
                pid.update(
                    positions=state["position"],
                    orientations=state["orientation"],
                    velocities=state["linear_velocity"],
                    angular_velocities=state["angular_velocity"],
                    target=active_target,
                    dt=dt,
                    actuators_saturated=saturated,
                ),
                dtype=float,
            )

            # final command = PID command + SAC correction, added once here.
            if residual is not None:
                pid_command = pid_command + residual.update(
                    state,
                    waypoint,
                    pid_command,
                    active_target.get("yaw", 0.0),
                )

            (
                total_thrust,
                roll_torque,
                pitch_torque,
                yaw_torque,
            ) = (float(value) for value in pid_command)

            # 4. Mixer, wind drag from the same pre-step state, forces,
            #    and one physics step.
            speeds, saturated, wind_force, wind_torque = sim.step_command(
                state,
                total_thrust,
                roll_torque,
                pitch_torque,
                yaw_torque,
            )

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
                        sim.vehicle_bodies,
                        sim.motors,
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