"""Reusable Isaac Sim drone backend, shared by motor_physics.py and hover_env.py.

Owns the parts of the simulation that do not depend on what the drone is being
asked to do: scene loading, the World, the iris views, the motor model and
mixer (including COM compensation), wind drag, force application, propeller
spin, and ground-truth state.

Import this module only after SimulationApp has started: it imports omni, pxr
and isaacsim.core, which need a running Kit application.
"""
from pathlib import Path

import numpy as np
import omni.usd

from pxr import UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim

import sim_config

from drone_diagnostics import cross, read_total_com_in_body_frame
from motor import MotorModel
from wind import GustingWind


# Physics/controller frequency: 120 Hz.
PHYSICS_DT = 1.0 / 120.0


class DroneSim:
    def __init__(
        self,
        simulation_app,
        scene_path,
        log=print,
        wind_config=None,
        deactivate_prims=(),
    ):
        """Load the scene and leave the World paused at the USD pose.

        wind_config: GustingWind arguments; defaults to sim_config.WIND.
        deactivate_prims: prims disabled in the loaded stage only, before
            physics starts. The saved USD files are not changed.
        """
        self.simulation_app = simulation_app
        self.log = log

        # ---------- Load scene ----------

        scene_path = Path(scene_path)

        if not scene_path.is_file():
            raise FileNotFoundError(scene_path)

        log(f"Loading: {scene_path}")

        omni.usd.get_context().open_stage(str(scene_path))
        simulation_app.update()

        self.stage = omni.usd.get_context().get_stage()
        stage = self.stage

        for path in deactivate_prims:
            prim = stage.GetPrimAtPath(path)

            if not prim.IsValid():
                raise RuntimeError(f"Cannot deactivate missing prim: {path}")

            prim.SetActive(False)
            log(f"Deactivated in loaded stage: {path}")

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

        self.dt = PHYSICS_DT
### set up PhysX scene
        self.world = World(
            physics_dt=self.dt,
            rendering_dt=self.dt,
            stage_units_in_meters=1.0,
            physics_prim_path=physics_scene_path,
        )
        world = self.world

        # ---------- Initialize the physical drone body ----------

        body_path = sim_config.BODY_PATH

        if not stage.GetPrimAtPath(body_path).IsValid():
            raise RuntimeError(
                f"Drone body not found: {body_path}"
            )

        self.body = world.scene.add(
            RigidPrim(
                prim_paths_expr=body_path,
                name="iris_body",
            )
        )
        body = self.body

        # ---------- Propeller spin drives (visual only) ----------

        # The rotors are separate rigid bodies on free revolute joints with no
        # drive, so nothing turns them. Add a velocity drive to each joint in
        # the loaded stage. This changes only the in-memory stage, never the
        # saved USD files, unless you save the stage from the GUI yourself.
        self.spin_enabled = bool(sim_config.PROPELLER_SPIN)

        if self.spin_enabled:
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
        # Also used to reset the whole vehicle, rotors included, in midair.
        self.articulation = world.scene.add(
            Articulation(
                prim_paths_expr=sim_config.DRONE_ROOT,
                name="iris_propellers",
            )
        )

        world.reset()

        # The simulator's actual timestep; querying it each step is slow.
        self.physics_dt = world.get_physics_dt()

        self.propeller_dofs = np.array([
            self.articulation.get_dof_index(
                joint_path.rsplit("/", 1)[-1]
            )
            for joint_path in sim_config.ROTOR_JOINT_PATHS
        ])

        initial_positions, initial_orientations = (
            body.get_world_poses()
        )

        self.initial_positions = initial_positions.copy()
        self.initial_orientations = initial_orientations.copy()

        # Save the pose read from the scene for later resets.
        # No separate manually chosen starting position is introduced.
        body.set_default_state(
            positions=self.initial_positions.copy(),
            orientations=self.initial_orientations.copy(),
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

        # Keep physics paused until the caller is ready to fly.
        world.pause()

        # ---------- Read total vehicle mass ----------

        vehicle_paths = []

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

            vehicle_paths.append(path)

        # This view is also used to calculate the full vehicle COM.
        self.vehicle_bodies = RigidPrim(
            prim_paths_expr=vehicle_paths,
            name="vehicle_bodies",
            reset_xform_properties=False,
        )

        self.vehicle_bodies.initialize()

        masses = np.asarray(
            self.vehicle_bodies.get_masses()
        ).reshape(-1)

        log("\nDrone rigid-body masses:")

        for path, mass in zip(self.vehicle_bodies.prim_paths, masses):
            log(f"{path}: {float(mass):.6f} kg")

        self.total_mass = float(masses.sum())

        if (
            not np.isfinite(self.total_mass)
            or self.total_mass <= 0
        ):
            raise RuntimeError(
                f"Invalid total drone mass: {self.total_mass}"
            )

        log(
            "Total drone rigid-body mass: "
            f"{self.total_mass:.6f} kg\n"
        )

        # ---------- Motor model ----------

        self.motors = MotorModel(**sim_config.MOTOR)
        motors = self.motors

        # Rotor application positions in body coordinates [m].
        self.rotor_positions = np.array(sim_config.ROTOR_POSITIONS)

        self.max_rotor_thrust = (
            motors.kf
            * motors.max_speed**2
        )

        self.max_total_thrust = 4 * self.max_rotor_thrust

        if self.total_mass * 9.81 >= self.max_total_thrust:
            raise RuntimeError(
                "Configured motors cannot support "
                "the measured drone mass."
            )

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
        self.allocation_matrix = np.vstack([
            np.ones(4),
            self.rotor_positions[:, 1],
            -self.rotor_positions[:, 0],
            motors.reaction_signs * motors.km / motors.kf,
        ])

        # ---------- Gusting wind ----------

        self.wind = GustingWind(
            **(sim_config.WIND if wind_config is None else wind_config)
        )
        wind = self.wind

        log(
            f"Wind enabled={wind.enabled}; mean={wind.mean_wind} m/s; "
            f"gust std={wind.gust_std} m/s over {wind.gust_timescale} s; "
            f"local std={wind.local_gust_std} m/s over "
            f"{wind.local_gust_timescale} s; "
            f"correlation length={wind.correlation_length} m; "
            f"Cd*A={wind.drag_coefficient * wind.effective_area:.4f} m^2 "
            f"over {len(wind.offsets)} points; seed={wind.seed}"
        )

        self.body_com = self.read_body_com()

        # Set by connect_imu; until then the mixer reads the COM every step.
        self.imu = None
        self.calibrated_com_offset = None

        # ---------- Animate propellers ----------

        # A prop turns opposite to the reaction torque it puts on the body:
        # reaction sign -1 -> counter-clockwise (+Z), +1 -> clockwise.
        # Rotors 0, 1 are the iris_prop_ccw meshes; rotors 2, 3 iris_prop_cw.
        self.spin_directions = -np.asarray(
            motors.reaction_signs,
            dtype=float,
        )

    def connect_imu(self, imu_path):
        """Attach the scene's IMU; raises if it cannot supply readings.

        Also records fixed calibration, like a bench measurement before
        flight: the sensor's mounting and the vehicle's COM. Both come from
        the geometry once here, so no flight state reaches the control path.
        Call while paused, before world.play().
        """
        import omni.kit.app

        from pxr import Usd, UsdGeom

        if not self.stage.GetPrimAtPath(imu_path).IsValid():
            raise RuntimeError(f"IMU prim not found: {imu_path}")

        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate(
            "isaacsim.sensors.experimental.physics", True
        )

        from isaacsim.sensors.experimental.physics import IMUSensor

        self.imu = IMUSensor(imu_path)

        self.calibrated_com_offset = np.array(
            read_total_com_in_body_frame(self.body, self.vehicle_bodies),
            dtype=float,
        )

        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        relative = (
            cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(imu_path))
            * cache.GetLocalToWorldTransform(
                self.stage.GetPrimAtPath(sim_config.BODY_PATH)
            ).GetInverse()
        )

        # USD matrices act on row vectors, hence the transpose.
        self.imu_mount_rotation = np.array(
            relative.RemoveScaleShear().ExtractRotationMatrix(),
            dtype=float,
        ).T
        self.imu_mount_offset = (
            np.array(relative.ExtractTranslation(), dtype=float)
            - self.calibrated_com_offset
        )

        self.log(
            f"IMU connected: {imu_path}; offset from COM "
            f"{np.round(self.imu_mount_offset, 4)} m; rotation to body\n"
            f"{np.round(self.imu_mount_rotation, 4)}"
        )

    def read_imu(self):
        """Latest (gyro, accelerometer) in sensor axes, or None if not ready.

        The accelerometer gives specific force, +g on the up axis at rest. The
        reading's orientation field is simulator ground truth and is not read.
        """
        reading = self.imu.get_sensor_reading()

        if not reading.is_valid:
            return None

        return (
            np.array([
                reading.angular_velocity_x,
                reading.angular_velocity_y,
                reading.angular_velocity_z,
            ], dtype=float),
            np.array([
                reading.linear_acceleration_x,
                reading.linear_acceleration_y,
                reading.linear_acceleration_z,
            ], dtype=float),
        )

    def base_target(self):
        """Controller limits shared by every task; add height/xy/yaw to fly."""
        return {
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
            "mass": self.total_mass,
            "max_xy_speed": sim_config.MAX_XY_SPEED,
            "max_z_speed": sim_config.MAX_Z_SPEED,
            "max_tilt_degrees": sim_config.MAX_TILT_DEGREES,
            "max_total_thrust": self.max_total_thrust,
        }

    def reset_after_stop(self):
        """World reset after the GUI's Stop: back to the USD pose."""
        self.world.reset()

        # Refresh the view used for COM/mass calculations.
        self.vehicle_bodies.initialize()

        self.body_com = self.read_body_com()

        # Same seed -> the same gust sequence after every reset.
        self.wind.reset()

    def mix_motor_commands(
        self,
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

        com_offset = self.calibrated_com_offset

        if com_offset is None:
            com_offset = read_total_com_in_body_frame(
                self.body,
                self.vehicle_bodies,
            )

        force = np.array([
            0.0,
            0.0,
            total_thrust,
        ])

        # Shift torque reference from full-vehicle COM to body origin.
        desired_wrench[1:] += cross(
            com_offset,
            force,
        )

        if not np.all(np.isfinite(desired_wrench)):
            raise ValueError(
                "Controller returned invalid values: "
                f"{desired_wrench}"
            )

        rotor_thrusts = np.linalg.solve(
            self.allocation_matrix,
            desired_wrench,
        )

        saturated = bool(
            np.any(rotor_thrusts < 0.0)
            or np.any(rotor_thrusts > self.max_rotor_thrust)
        )

        # Clipping means the exact requested wrench may not
        # be achievable. Keep logging the saturation flag.
        rotor_thrusts = np.clip(
            rotor_thrusts,
            0.0,
            self.max_rotor_thrust,
        )

        # T = kf * omega^2  ->  omega = sqrt(T / kf).
        speeds = np.sqrt(
            rotor_thrusts / self.motors.kf
        )

        return speeds, saturated

    def read_body_com(self):
        """Body rigid-body COM relative to its origin, body frame [m]."""
        local_coms, _ = self.body.get_coms()
        return np.asarray(local_coms, dtype=float).reshape(3)

    def wind_wrench(self, state, step_dt):
        """Advance the gusts by one physics step; return body-frame drag.

        The point forces are reduced to the exact equivalent force and
        torque about the body origin, which is where the motor wrench is
        applied (no position_data). Both go out in one API call: the
        force-at-position API takes one point per body per call, and
        whether repeated calls in a step accumulate is undocumented.
        """
        if not self.wind.enabled:
            return np.zeros(3), np.zeros(3)

        self.wind.step(step_dt)

        # PhysX reports the velocity of the body's center of mass.
        return self.wind.body_wrench(
            state["position"],
            state["orientation"],
            state["linear_velocity"],
            state["angular_velocity"],
            self.body_com,
        )

    def apply_motor_commands(
        self,
        speeds,
        extra_force=np.zeros(3),
        extra_torque=np.zeros(3),
    ):
        """Apply motor thrust plus an extra body-frame wrench.

        extra_torque must be about the body origin (wind drag).
        """
        thrusts, reaction_torques = self.motors.calculate(
            speeds
        )

        rotor_forces = np.zeros((4, 3))
        rotor_forces[:, 2] = thrusts

        total_force = rotor_forces.sum(axis=0)

        # Sum moment-arm torques r x F.
        total_torque = np.cross(
            self.rotor_positions,
            rotor_forces,
        ).sum(axis=0)

        # Add yaw reaction torque.
        total_torque[2] += reaction_torques.sum()

        total_force = total_force + extra_force
        total_torque = total_torque + extra_torque

        # Force and torque are expressed in body coordinates, actually push the
        # drone in physX
        self.body.apply_forces_and_torques_at_pos(
            forces=total_force.astype(
                np.float32
            ).reshape(1, 3),
            torques=total_torque.astype(
                np.float32
            ).reshape(1, 3),
            is_global=False,
        )

    def spin_targets(self, speeds):
        """Displayed rotor joint speeds [rad/s] for motor speeds [rad/s]."""
        # Displayed speed follows each motor command, scaled down.
        return (
            self.spin_directions
            * np.asarray(speeds, dtype=float)
            * sim_config.PROPELLER_SPIN_SCALE
        )

    def spin_propellers(self, speeds):
        if not self.spin_enabled:
            return

        self.articulation.set_joint_velocity_targets(
            self.spin_targets(speeds).reshape(1, -1),
            joint_indices=self.propeller_dofs,
        )


    # physics scene makes PhysX integrate F = m·a, the rotation equations, 
    # gravity and contacts for one
    def step_command(
        self,
        state,
        total_thrust,
        roll_torque,
        pitch_torque,
        yaw_torque,
        render=True,
    ):
        """Mix one wrench command, add wind, and advance one physics step.

        state is the pre-step state the command was computed from; wind drag
        uses it too. Returns (speeds, saturated, wind_force, wind_torque).
        """
        # Mixer: requested wrench -> four rotor speeds.
        speeds, saturated = self.mix_motor_commands(
            total_thrust,
            roll_torque,
            pitch_torque,
            yaw_torque,
        )

        # Wind drag from the same pre-step state, using the
        # simulator's actual physics timestep.
        wind_force, wind_torque = self.wind_wrench(
            state,
            self.physics_dt,
        )

        # Apply forces and advance simulation.
        self.apply_motor_commands(
            speeds,
            wind_force,
            wind_torque,
        )
        self.spin_propellers(speeds)
        self.world.step(render=render)

        return speeds, saturated, wind_force, wind_torque

    def read_drone_state(self):
        """Ground truth; quaternion is [w, x, y, z], velocities are world."""
        positions, orientations = self.body.get_world_poses()

        linear_velocities = (
            self.body.get_linear_velocities()
        )

        angular_velocities = (
            self.body.get_angular_velocities()
        )

        return {
            "position": positions[0].copy(),
            "orientation": orientations[0].copy(),
            "linear_velocity": linear_velocities[0].copy(),
            "angular_velocity": angular_velocities[0].copy(),
        }

    def clear_applied_wrench(self):
        """Overwrite any wrench queued on the body with zero."""
        self.body.apply_forces_and_torques_at_pos(
            forces=np.zeros((1, 3), dtype=np.float32),
            torques=np.zeros((1, 3), dtype=np.float32),
            is_global=False,
        )

    def teleport(
        self,
        position,
        orientation,
        linear_velocity,
        angular_velocity,
        motor_speeds,
    ):
        """Reset the whole articulation in midair without stopping physics.

        World.reset() stops the timeline and rebuilds every physics view, so
        it is reserved for GUI Stop -> Play. Here the root pose and velocity,
        rotor angles, rotor velocities, and spin targets are all overwritten
        together. Rotors restart at the speed they really settle at: starting
        above PROPELLER_SETTLED_SPEED gets clamped on the first step, which
        knocks the body to about 0.15 rad/s.
        """
        self.clear_applied_wrench()

        positions = np.asarray(
            position, dtype=np.float32
        ).reshape(1, 3)
        orientations = np.asarray(
            orientation, dtype=np.float32
        ).reshape(1, 4)
        velocities = np.hstack([
            np.asarray(linear_velocity, dtype=np.float32).reshape(1, 3),
            np.asarray(angular_velocity, dtype=np.float32).reshape(1, 3),
        ])

        rotor_velocities = np.zeros((1, 4), dtype=np.float32)

        if self.spin_enabled:
            spin_targets = self.spin_targets(motor_speeds)

            rotor_velocities[0] = np.clip(
                spin_targets,
                -sim_config.PROPELLER_SETTLED_SPEED,
                sim_config.PROPELLER_SETTLED_SPEED,
            )

        articulation = self.articulation

        articulation.set_world_poses(
            positions=positions,
            orientations=orientations,
        )
        articulation.set_velocities(velocities)
        articulation.set_joint_positions(
            np.zeros((1, 4), dtype=np.float32),
            joint_indices=self.propeller_dofs,
        )
        articulation.set_joint_velocities(
            rotor_velocities,
            joint_indices=self.propeller_dofs,
        )

        if self.spin_enabled:
            articulation.set_joint_velocity_targets(
                spin_targets.reshape(1, -1),
                joint_indices=self.propeller_dofs,
            )

    def is_running(self):
        return self.simulation_app.is_running()

    def close(self):
        self.world.stop()
