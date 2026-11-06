"""MuJoCo physics, SI actuator allocation, and explicit coordinate transforms."""
from pathlib import Path
import hashlib

import mujoco
import numpy as np

from drone_code.wind import quaternion_to_matrix
from .config import PhysicsConfig, WindConfig
from .weather import ScenarioWind


MODEL_DIR = Path(__file__).parent / "models" / "skydio_x2"


class DroneSimulation:
    def __init__(self, physics=None, wind=None, seed=0):
        self.config = physics or PhysicsConfig()
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "scene.xml"))
        self.model.opt.timestep = self.config.timestep
        # Only our explicit distributed drag is active, including in still air.
        self.model.opt.density = 0
        self.model.opt.viscosity = 0
        self.model.opt.wind[:] = 0
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        self.body_id = self.model.body("x2").id
        self.dt = float(self.model.opt.timestep)
        self.total_mass = float(self.model.body_mass[self.body_id])
        self.body_com = self.model.body_ipos[self.body_id].copy()
        self.rotor_ids = np.array([self.model.actuator(f"thrust{i}").id for i in range(1, 5)])
        self.site_ids = self.model.actuator_trnid[self.rotor_ids, 0]
        self.rotor_positions = self.model.site_pos[self.site_ids].copy()
        self.control_range = self.model.actuator_ctrlrange[self.rotor_ids].copy()
        self.max_rotor_thrust = float(self.control_range[:, 1].min())
        self.max_total_thrust = float(self.control_range[:, 1].sum())
        self.allocation_matrix = self._allocation()
        self.wind = ScenarioWind(wind or WindConfig(), self.rotor_positions, seed)
        self.closed = False
        self.last = {}
        self.reset()

    def _allocation(self):
        model = self.model
        if model.nv != 6 or model.nq != 7 or model.nu != 4 or model.nbody != 2:
            raise ValueError("Backend expects the audited single free-body X2 model")
        gear = model.actuator_gear[self.rotor_ids]
        if (not np.all(model.actuator_ctrllimited[self.rotor_ids])
                or not np.all(model.actuator_trntype[self.rotor_ids] == mujoco.mjtTrn.mjTRN_SITE)
                or not np.all(model.actuator_dyntype[self.rotor_ids] == mujoco.mjtDyn.mjDYN_NONE)
                or not np.allclose(model.site_quat[self.site_ids], [1, 0, 0, 0])
                or not np.allclose(gear[:, :3], [0, 0, 1])
                or not np.allclose(model.actuator_gainprm[self.rotor_ids, 0], 1)):
            raise ValueError("Actuator units changed; re-audit before using the mixer")
        moments = np.cross(self.rotor_positions - self.body_com, gear[:, :3]) + gear[:, 3:]
        allocation = np.vstack([gear[:, 2], moments.T])
        if np.linalg.matrix_rank(allocation) != 4:
            raise ValueError("Rotor allocation is singular")
        if self.total_mass <= 0 or np.any(model.body_inertia[self.body_id] <= 0):
            raise ValueError("Invalid mass or principal inertia")
        return allocation

    def base_target(self):
        return dict(mass=self.total_mass, yaw=0.0, roll=0.0, pitch=0.0,
                    max_xy_speed=self.config.max_xy_speed,
                    max_z_speed=self.config.max_z_speed,
                    max_tilt_degrees=self.config.max_tilt_degrees,
                    max_total_thrust=self.max_total_thrust)

    def reset(self, position=(0, 0, 1.5), orientation=(1, 0, 0, 0),
              linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0)):
        for value, count in ((position, 3), (orientation, 4), (linear_velocity, 3), (angular_velocity, 3)):
            if np.shape(value) != (count,) or not np.all(np.isfinite(value)):
                raise ValueError("Reset needs finite pose and velocity vectors")
        if np.linalg.norm(orientation) < 1e-8:
            raise ValueError("Invalid reset quaternion")
        mujoco.mj_resetData(self.model, self.data)
        q = np.array(orientation, dtype=float) / np.linalg.norm(orientation)
        self.data.qpos[:3] = position
        self.data.qpos[3:7] = q
        self.data.qvel[:3] = linear_velocity  # freejoint origin velocity, world
        self.data.qvel[3:6] = quaternion_to_matrix(q).T @ angular_velocity
        self.data.ctrl[:] = np.linalg.solve(self.allocation_matrix, [self.total_mass * 9.81, 0, 0, 0])
        self.wind.reset()
        self.wind.mean_wind = self.wind.programmed_mean(0.0)
        self.wind.point_winds = self.wind._compose()
        self.last = {}
        mujoco.mj_forward(self.model, self.data)

    def read_drone_state(self):
        d = self.data
        rotation = d.xmat[self.body_id].reshape(3, 3)
        # mj_objectVelocity returns (omega, v_COM) in world axes when flg_local=0.
        spatial = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_BODY,
                                self.body_id, spatial, 0)
        origin_acceleration = d.qacc[:3].copy()
        arm = rotation @ self.body_com
        alpha = rotation @ d.qacc[3:6]
        com_acceleration = origin_acceleration + np.cross(alpha, arm) + np.cross(spatial[:3], np.cross(spatial[:3], arm))
        return dict(position=d.xpos[self.body_id].copy(),
                    orientation=d.xquat[self.body_id].copy(),
                    linear_velocity=d.qvel[:3].copy(),
                    com_velocity=spatial[3:].copy(),
                    com_acceleration=com_acceleration,
                    angular_velocity=spatial[:3].copy(),
                    origin_velocity=d.qvel[:3].copy(),
                    origin_acceleration=d.qacc[:3].copy(),
                    com_position=d.xipos[self.body_id].copy(),
                    body_angular_velocity=rotation.T @ spatial[:3])

    def mix(self, command):
        command = np.asarray(command, dtype=float)
        if command.shape != (4,) or not np.all(np.isfinite(command)):
            raise ValueError("Command must contain finite thrust [N] and three torques [N m]")
        requested = np.linalg.solve(self.allocation_matrix, command)
        clipped = np.clip(requested, self.control_range[:, 0], self.control_range[:, 1])
        return requested, clipped, bool(np.any(np.abs(requested - clipped) > 1e-10))

    def step_command(self, state, thrust, roll, pitch, yaw, render=False):
        if self.closed:
            raise RuntimeError("Simulation is closed")
        command = np.array([thrust, roll, pitch, yaw])
        requested, controls, saturated = self.mix(command)
        self.data.ctrl[self.rotor_ids] = controls
        self.wind.step(self.dt)
        positions, forces, r_origin = self.wind.point_forces(
            state["position"], state["orientation"], state["com_velocity"],
            state["angular_velocity"], self.body_com)
        if not self.wind.enabled:
            forces[:] = 0
        rotation = quaternion_to_matrix(state["orientation"])
        r_com = r_origin - rotation @ self.body_com
        force = forces.sum(axis=0)
        torque = np.cross(r_com, forces).sum(axis=0)
        self.data.xfrc_applied[:] = 0
        self.data.xfrc_applied[self.body_id, :3] = force
        self.data.xfrc_applied[self.body_id, 3:] = torque
        start_time = float(self.data.time)
        mujoco.mj_step(self.model, self.data)
        if self.data.time <= start_time or not np.all(np.isfinite(self.data.qpos)):
            raise FloatingPointError("MuJoCo reset or produced invalid state")
        # mj_step's derived state/sensors can still describe its pre-integration
        # configuration. Refresh all of them before any post-step logging.
        mujoco.mj_forward(self.model, self.data)
        self.last = dict(command=command, requested=requested, controls=controls,
                         saturated=saturated, force=force, torque=torque,
                         wind=self.wind.point_winds.copy(), point_forces=forces,
                         interval_start=start_time, interval_end=float(self.data.time))
        return controls, saturated, force, torque

    def collided(self):
        for contact in self.data.contact[:self.data.ncon]:
            if self.model.geom_bodyid[contact.geom1] != self.model.geom_bodyid[contact.geom2]:
                return True
        return False

    def audit(self):
        axes = quaternion_to_matrix(self.model.body_iquat[self.body_id])
        inertia = axes @ np.diag(self.model.body_inertia[self.body_id]) @ axes.T
        hashes = {str(p.relative_to(MODEL_DIR)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(MODEL_DIR.rglob("*")) if p.is_file()}
        return dict(mass_kg=self.total_mass, com_body_m=self.body_com.tolist(),
                    inertia_body_kg_m2=inertia.tolist(),
                    principal_inertia_kg_m2=self.model.body_inertia[self.body_id].tolist(),
                    rotor_positions_body_m=self.rotor_positions.tolist(),
                    control_range_N=self.control_range.tolist(),
                    allocation_about_com=self.allocation_matrix.tolist(),
                    hover_controls_N=np.linalg.solve(self.allocation_matrix, [self.total_mass*9.81, 0, 0, 0]).tolist(),
                    timestep_s=self.dt, integrator="implicitfast",
                    builtin_density=0, builtin_viscosity=0, model_sha256=hashes,
                    cameras=[self.model.camera(i).name for i in range(self.model.ncam)],
                    sensors=[self.model.sensor(i).name for i in range(self.model.nsensor)])

    def is_running(self):
        return not self.closed

    def close(self):
        self.closed = True
