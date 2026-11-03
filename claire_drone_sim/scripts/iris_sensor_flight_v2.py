"""DSC180A Iris 100-m flight v2 (sensor warmup reset, staged waypoints, verbose failure diagnostics).

Requires Isaac Sim 6.2 standalone python.bat (not regular Python).
Works with the personal scene containing /iris and /iris/body.
No PPO, no wind, no team repository modifications, no USD saved.
"""
import argparse
import copy
import csv
import json
import math
import sqlite3
import sys
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "flight_100m_v2.json"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
parser.add_argument("--env", choices=["empty", "warehouse", "outdoor"], default=None)
parser.add_argument("--distance", type=float, default=None)
parser.add_argument("--duration", type=float, default=None)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--test", action="store_true", help="5-second hover and sensor test; not a full mission")
parser.add_argument("--keep-open", action="store_true", help="Keep GUI open for inspection after the flight until you close the window")
args = parser.parse_args()

cfg_path = args.config.resolve()
if not cfg_path.is_file():
    raise FileNotFoundError(f"Missing configuration: {cfg_path}")
CFG = json.loads(cfg_path.read_text(encoding="utf-8"))
if args.env:
    CFG["environment"] = args.env
if args.distance is not None:
    CFG["mission"]["distance_m"] = args.distance
if args.duration is not None:
    CFG["mission"]["max_duration_s"] = args.duration
if args.test:
    CFG["mission"]["max_duration_s"] = min(float(CFG["mission"]["max_duration_s"]), 5.0)
if args.headless:
    CFG["simulation"]["headless"] = True

PHYS_HZ = int(CFG["simulation"]["physics_hz"])
RENDER_HZ = int(CFG["simulation"]["render_hz"])
IMU_HZ = int(CFG["sensors"]["imu_hz"])
RGB_HZ = int(CFG["sensors"]["rgb_hz"])
TRUTH_HZ = IMU_HZ
if not (PHYS_HZ > 0 and RENDER_HZ > 0 and IMU_HZ > 0 and RGB_HZ > 0):
    raise ValueError("All sampling frequencies must be positive")
if any(PHYS_HZ % rate for rate in (RENDER_HZ, IMU_HZ, RGB_HZ)) or RENDER_HZ % RGB_HZ:
    raise ValueError("This starter requires all sample rates to divide physics_hz and rgb_hz to divide render_hz")
if CFG["environment"] not in ("empty", "warehouse", "outdoor"):
    raise ValueError("Unknown environment preset")
if CFG["simulation"].get("wind_enabled"):
    raise ValueError("Wind is intentionally disabled for this sensor-baseline experiment")
if float(CFG["mission"]["distance_m"]) <= 0 or float(CFG["mission"]["altitude_m"]) <= 0:
    raise ValueError("Mission distance and altitude must be positive")
if float(CFG["mission"]["max_duration_s"]) <= 0:
    raise ValueError("Flight duration must be positive")

# Start Isaac Sim before importing any Omniverse APIs.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": bool(CFG["simulation"]["headless"])})

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.experimental.physics import IMU, IMUSensor

from controller import controller
from motor import MotorModel
from drone_diagnostics import read_total_com_in_body_frame
from scene_presets import build as build_scene

DT = 1.0 / PHYS_HZ
IMU_DIV = PHYS_HZ // IMU_HZ
CAMERA_DIV = PHYS_HZ // RGB_HZ
RENDER_DIV = PHYS_HZ // RENDER_HZ
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
RUN_DIR = PROJECT_DIR / "results" / f"sensor_{CFG['environment']}_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=False)
SENSOR_DB = RUN_DIR / "sensors.sqlite"
TRUTH_CSV = RUN_DIR / "ground_truth.csv"


def now_utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def reading_tuple(reading, *names):
    return tuple(float(getattr(reading, name)) for name in names)


class SensorLogger:
    def __init__(self, filepath):
        self.db = sqlite3.connect(str(filepath))
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript("""
        CREATE TABLE run_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE imu (
          physics_step INTEGER PRIMARY KEY, sim_time_s REAL NOT NULL,
          sensor_time_s REAL, wall_time_utc TEXT NOT NULL, valid INTEGER NOT NULL,
          ax_mps2 REAL, ay_mps2 REAL, az_mps2 REAL,
          wx_rad_s REAL, wy_rad_s REAL, wz_rad_s REAL,
          qw REAL, qx REAL, qy REAL, qz REAL
        );
        CREATE TABLE camera (
          physics_step INTEGER PRIMARY KEY, sim_time_s REAL NOT NULL,
          wall_time_utc TEXT NOT NULL, width INTEGER NOT NULL,
          height INTEGER NOT NULL, channels INTEGER NOT NULL,
          codec TEXT NOT NULL, rgb_blob BLOB NOT NULL
        );
        CREATE INDEX camera_time ON camera(sim_time_s);
        """)
        self.imu_count = self.imu_valid_count = self.camera_count = 0
        self.pending = 0
        self.commit_every = int(CFG["sensors"].get("sqlite_commit_every_n_imu_samples", 120))
        self.set_info("run_id", RUN_ID)
        self.set_info("configuration", CFG)
        self.set_info("schema", "IMU: body/sensor-frame specific force + angular rate; camera: uint8 HWC RGB compressed with zlib; times: simulation seconds")
        self.db.commit()

    def set_info(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO run_info(key,value) VALUES (?,?)",
                        (key, value if isinstance(value, str) else json.dumps(value)))

    def insert_imu(self, step, sim_time, reading):
        valid = bool(getattr(reading, "is_valid", False))
        values = (None,) * 10
        sensor_time = None
        if valid:
            values = (
                *reading_tuple(reading, "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"),
                *reading_tuple(reading, "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"),
                *reading_tuple(reading, "orientation_w", "orientation_x", "orientation_y", "orientation_z"),
            )
            sensor_time = float(reading.time)
            self.imu_valid_count += 1
        self.db.execute("INSERT INTO imu VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (step, sim_time, sensor_time, now_utc(), int(valid), *values))
        self.imu_count += 1
        self.pending += 1
        if self.pending >= self.commit_every:
            self.db.commit()
            self.pending = 0

    def insert_rgb(self, step, sim_time, arr):
        rgb = np.asarray(arr)
        if rgb.ndim != 3 or rgb.shape[2] not in (3,4):
            raise ValueError(f"Camera delivered unexpected RGB shape: {rgb.shape}")
        rgb = rgb[:,:,:3]
        if np.issubdtype(rgb.dtype, np.floating):
            if np.max(rgb) <= 1.01:
                rgb = rgb * 255.0
            rgb = np.clip(rgb, 0, 255)
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        h,w,c = rgb.shape
        raw = zlib.compress(rgb.tobytes(), level=3)
        self.db.execute("INSERT INTO camera VALUES (?,?,?,?,?,?,?,?)",
                        (step, sim_time, now_utc(), w, h, c, "zlib-rgb8", sqlite3.Binary(raw)))
        self.camera_count += 1

    def close(self):
        self.db.commit()
        self.db.close()


def run():
    scene = (PROJECT_DIR / CFG["scene"]).resolve()
    if not scene.is_file():
        raise FileNotFoundError(f"Scene missing: {scene}\nKeep your personal USD at {CFG['scene']} or edit the config.")
    print(f"[RUN] {RUN_ID}, preset={CFG['environment']}, wind=OFF, scene={scene}", flush=True)
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError(f"Could not open USD scene {scene}")
    simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/iris")
    body_prim = stage.GetPrimAtPath("/iris/body")
    if not root.IsValid() or not body_prim.IsValid() or not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("The personal scene needs rigid body /iris/body and /iris root")
    # Only modify stage in memory; don't save USD.
    original_cube = stage.GetPrimAtPath("/World/MyBox")
    if original_cube.IsValid():
        print("[SCENE] Removed old MyBox from IN-MEMORY copy; original USD unchanged")
        stage.RemovePrim("/World/MyBox")
    # Move the WHOLE Iris root, including rotor bodies and attached sensors.
    # USD body geometry can have a local height offset; a root z of 5 is not
    # necessarily a rigid-body world z of 5. Correct it BEFORE PhysX initializes.
    desired_z = float(CFG["mission"]["start_height_m"])
    root_transform = UsdGeom.XformCommonAPI(root)
    root_transform.SetTranslate(Gf.Vec3d(0.0, 0.0, desired_z))
    initial_usd_body_z = float(UsdGeom.XformCache().GetLocalToWorldTransform(body_prim).ExtractTranslation()[2])
    correction_z = desired_z - initial_usd_body_z
    if abs(correction_z) > 1e-5:
        root_transform.SetTranslate(Gf.Vec3d(0.0, 0.0, desired_z + correction_z))
    final_usd_body_z = float(UsdGeom.XformCache().GetLocalToWorldTransform(body_prim).ExtractTranslation()[2])
    print(f"[SETUP] Requested body altitude {desired_z:.3f} m; USD before correction "
          f"{initial_usd_body_z:.3f} m; adjusted body altitude {final_usd_body_z:.3f} m", flush=True)
    build_scene(stage, CFG)

    # Parent both sensors under the *rigid body* so their transforms follow motion.
    sensor_cfg = CFG["sensors"]
    imu_path = "/iris/body/imu_sensor_v1"
    if stage.GetPrimAtPath(imu_path).IsValid():
        imu = IMUSensor(imu_path)
    else:
        imu = IMUSensor(IMU.create(
            imu_path,
            linear_acceleration_filter_size=int(sensor_cfg["imu_filter_width"]),
            angular_velocity_filter_size=int(sensor_cfg["imu_filter_width"]),
            orientation_filter_size=int(sensor_cfg["imu_filter_width"]),
            translations=np.array([sensor_cfg["imu_position_body_m"]],dtype=float),
            orientations=np.array([[1.0,0.0,0.0,0.0]],dtype=float)
        ))
    cam = Camera(prim_path="/iris/body/front_rgb_sensor_v1", name="mission_front_rgb",
                 resolution=(int(sensor_cfg["rgb_width"]),int(sensor_cfg["rgb_height"])),
                 frequency=RGB_HZ)
    # 'world' camera axes here mean +X forward, +Z up relative to body.
    cam.set_local_pose(translation=np.asarray(sensor_cfg["camera_position_body_m"],dtype=float),
                       orientation=np.array([1.0,0.0,0.0,0.0]), camera_axes="world")

    scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes)>1:
        raise RuntimeError(f"More than one physics scene found: {scenes}")
    physics_path = scenes[0] if scenes else "/World/PhysicsScene"
    world = World(physics_dt=DT, rendering_dt=1.0/RENDER_HZ,
                  stage_units_in_meters=1.0, physics_prim_path=physics_path)
    body = world.scene.add(RigidPrim(prim_paths_expr="/iris/body", name="iris_sensor_mission_body"))
    world.reset()
    # CRITICAL FIX: world.reset() may leave the timeline playing. In v1, three
    # renderer warm-up updates occurred without motor thrust; the log showed
    # z=4.52 instead of 5.0. Pause physics BEFORE warming the RGB camera.
    world.pause()
    cam.initialize()
    cam.add_rgb_to_frame()
    for _ in range(3):
        simulation_app.update()
    # Ensure all Iris rigid bodies/rotors are returned to their authored pose
    # after any possible warm-up movement. Do not reset again during flight.
    world.reset()
    world.pause()

    initial_p, initial_q = body.get_world_poses()
    initial_v = np.asarray(body.get_linear_velocities()[0], dtype=float)
    print(f"[RESET] after camera warm-up body XYZ={np.round(initial_p[0],4)} "
          f"velocity={np.round(initial_v,4)}", flush=True)
    if abs(float(initial_p[0][2]) - desired_z) > 0.15:
        raise RuntimeError(f"Post-reset Iris altitude is {float(initial_p[0][2]):.3f}m, "
                           f"expected {desired_z:.3f}m. Root/rigid-body transforms need inspection.")
    # Set a zero-velocity initial state for the body after the sensor warm-up;
    # world.reset above restores all other rigid bodies to the authored poses.
    body.set_default_state(positions=initial_p.copy(), orientations=initial_q.copy(),
                           linear_velocities=np.zeros((1,3),dtype=np.float32),
                           angular_velocities=np.zeros((1,3),dtype=np.float32))
    body.post_reset()
    if float(np.linalg.norm(body.get_linear_velocities()[0])) > 0.10:
        raise RuntimeError("Iris still has nonzero speed after reset; aborting instead of starting in free fall")

    mass_views = []
    mass = 0.0
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not p.startswith("/iris/") or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        api = UsdPhysics.RigidBodyAPI(prim)
        if api.GetRigidBodyEnabledAttr().Get() is False:
            continue
        view = RigidPrim(prim_paths_expr=p, name=f"mass_sensor_{len(mass_views)}", reset_xform_properties=False)
        view.initialize()
        mass_views.append(view)
        mass += float(np.asarray(view.get_masses()).reshape(-1)[0])
    if not math.isfinite(mass) or mass <= 0:
        raise RuntimeError(f"Bad Iris mass: {mass}")
    com_offset = np.asarray(read_total_com_in_body_frame(body, mass_views),dtype=float)
    motors = MotorModel(5.84e-6, 0.06*5.84e-6, 1100.0, [-1,-1,+1,+1])
    rotor_positions = np.array([
        [0.13798545,-0.20671639,0.023],
        [-0.12511168,0.21875414,0.023],
        [0.138,0.20257577,0.023],
        [-0.12415360,-0.22234580,0.023]
    ])
    allocation = np.vstack([
        np.ones(4), rotor_positions[:,1], -rotor_positions[:,0],
        motors.reaction_signs*motors.km/motors.kf
    ])
    max_f = motors.kf*motors.max_speed**2

    start = np.asarray(body.get_world_poses()[0][0],dtype=float)
    target_xyz = np.array([start[0]+float(CFG["mission"]["distance_m"]),
                           start[1],float(CFG["mission"]["altitude_m"])])
    target = dict(height=float(target_xyz[2]),xy=target_xyz[:2].copy(),
                  roll=0.0,pitch=0.0,yaw=0.0,mass=mass)
    hold_s = float(CFG["mission"].get("stabilization_s", 2.0))
    lookahead_m = float(CFG["mission"].get("lookahead_m", 6.0))
    if hold_s < 0 or lookahead_m <= 0:
        raise ValueError("stabilization_s must be >=0 and lookahead_m must be >0")
    print(f"[GUIDANCE] first hover {hold_s:.1f}s, then use {lookahead_m:.1f}m moving look-ahead "
          f"toward the actual 100m goal (or configured distance)", flush=True)
    print(f"[MISSION] start={start.round(2)} target={target_xyz.round(2)} mass={mass:.4f}kg",flush=True)
    print(f"[SENSORS] camera={RGB_HZ}fps @ {sensor_cfg['rgb_width']}x{sensor_cfg['rgb_height']}, IMU={IMU_HZ}Hz, truth={TRUTH_HZ}Hz",flush=True)
    if not bool(CFG["simulation"]["headless"]):
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            # Separate spectator viewport; does not modify the sensor camera.
            set_camera_view(eye=np.array([30.0,-85.0,42.0]),
                            target=np.array([start[0]+50.0,start[1],5.0]))
        except Exception as exc:
            print(f"[VIEWPORT] Wide camera positioning unavailable ({exc}); use Alt+drag to frame track.",flush=True)
    print(f"[DATA] sensor={SENSOR_DB}\n[DATA] truth ={TRUTH_CSV}",flush=True)

    logger = SensorLogger(SENSOR_DB)
    logger.set_info("world_scene", str(scene))
    logger.set_info("drone_mass_kg", mass)
    logger.set_info("start_xyz_m", start.tolist())
    logger.set_info("target_xyz_m", target_xyz.tolist())
    logger.set_info("start_wall_time_utc", now_utc())
    logger.db.commit()

    cfg_m = CFG["mission"]
    max_steps = round(float(cfg_m["max_duration_s"])*PHYS_HZ)
    goal_hold_steps = round(float(cfg_m["hold_at_goal_s"])*PHYS_HZ)
    at_goal_steps = 0
    finished = "timeout"
    step_count = 0
    warn_invalid_rgb = 0
    with TRUTH_CSV.open("w",newline="",encoding="utf-8") as output:
        writer=csv.writer(output)
        writer.writerow(["run_id","environment","wind_enabled","physics_step","sim_time_s","wall_time_utc",
                         "x_m","y_m","z_m","qw","qx","qy","qz",
                         "vx_mps","vy_mps","vz_mps",
                         "omega_world_x_rad_s","omega_world_y_rad_s","omega_world_z_rad_s",
                         "target_x_m","target_y_m","target_z_m","position_error_m",
                         "lateral_error_m","distance_progress_m","speed_mps",
                         "motor0_rad_s","motor1_rad_s","motor2_rad_s","motor3_rad_s",
                         "motor_saturated","active_waypoint_x_m","active_waypoint_y_m","thrust_command_n"])
        world.play()
        try:
            for physics_step in range(1,max_steps+1):
                if not simulation_app.is_running():
                    finished="app_closed"
                    break
                p,q = body.get_world_poses()
                v=body.get_linear_velocities()
                omega=body.get_angular_velocities()
                pos,qw,vel,ang = np.asarray(p[0],dtype=float),np.asarray(q[0],dtype=float),np.asarray(v[0],dtype=float),np.asarray(omega[0],dtype=float)
                if not all(np.all(np.isfinite(v)) for v in (pos,qw,vel,ang)):
                    finished = "invalid_nonfinite_physics_state"
                    print(f"[STOP] {finished} at t={(physics_step-1)*DT:.3f}s", flush=True)
                    break
                t_before = (physics_step - 1) * DT
                if args.test or t_before < hold_s:
                    # Test mode is a HOVER stability + sensor test, not a demand
                    # to fly 100m within 5 seconds.
                    waypoint = np.array([start[0],start[1]], dtype=float)
                else:
                    waypoint = np.array([min(target_xyz[0], pos[0] + lookahead_m),start[1]],dtype=float)
                target["xy"] = waypoint
                thrust,tx,ty,tz = controller(positions=pos,orientations=qw,
                                            velocities=vel,angular_velocities=ang,target=target)
                wrench = np.array([thrust,tx,ty,tz],dtype=float)
                wrench[1:] += np.cross(com_offset,np.array([0.,0.,thrust]))
                rotor_thrust = np.linalg.solve(allocation,wrench)
                saturated = bool(np.any((rotor_thrust<0)|(rotor_thrust>max_f)))
                rotor_thrust = np.clip(rotor_thrust,0.0,max_f)
                speeds = np.sqrt(rotor_thrust/motors.kf)
                f,react = motors.calculate(speeds)
                local_force = np.array([0.,0.,float(f.sum())],dtype=float)
                local_torque = np.cross(rotor_positions,np.column_stack([np.zeros(4),np.zeros(4),f])).sum(axis=0)
                local_torque[2] += float(react.sum())
                # One motor-generated physical wrench, NO wind or air-drag force.
                body.apply_forces_and_torques_at_pos(
                    forces=local_force.astype(np.float32).reshape(1,3),
                    torques=local_torque.astype(np.float32).reshape(1,3),
                    is_global=False)
                world.step(render=(physics_step % RENDER_DIV == 0))
                t = physics_step*DT
                step_count = physics_step

                if physics_step % IMU_DIV == 0:
                    actual_p, actual_q = body.get_world_poses()
                    actual_v = body.get_linear_velocities()
                    actual_omega=body.get_angular_velocities()
                    pos2=np.asarray(actual_p[0],dtype=float)
                    quat2=np.asarray(actual_q[0],dtype=float)
                    vel2=np.asarray(actual_v[0],dtype=float)
                    ang2=np.asarray(actual_omega[0],dtype=float)
                    error=float(np.linalg.norm(pos2-target_xyz))
                    writer.writerow([RUN_ID,CFG["environment"],0,physics_step,round(t,6),now_utc(),
                                     *map(float,pos2),*map(float,quat2),*map(float,vel2),
                                     *map(float,ang2),*map(float,target_xyz),error,
                                     float(pos2[1]-target_xyz[1]),float(pos2[0]-start[0]),
                                     float(np.linalg.norm(vel2)),*map(float,speeds),int(saturated),
                                     float(waypoint[0]),float(waypoint[1]),float(thrust)])
                    imu_read = imu.get_sensor_reading()
                    logger.insert_imu(physics_step,round(t,6),imu_read)

                if physics_step % CAMERA_DIV == 0:
                    rgb=cam.get_rgb(device="cpu")
                    if rgb is None or np.asarray(rgb).size==0:
                        warn_invalid_rgb+=1
                        if warn_invalid_rgb<=3:
                            print(f"[CAMERA] No frame yet at t={t:.2f}s; waiting for renderer",flush=True)
                    else:
                        logger.insert_rgb(physics_step,round(t,6),rgb)

                current_p = body.get_world_poses()[0][0]
                current_v = body.get_linear_velocities()[0]
                dist=float(np.linalg.norm(current_p-target_xyz))
                speed=float(np.linalg.norm(current_v))
                if dist<=float(cfg_m["arrival_radius_m"]) and speed<=float(cfg_m["arrival_max_speed_mps"]):
                    at_goal_steps+=1
                else:
                    at_goal_steps=0
                if at_goal_steps >= goal_hold_steps:
                    finished="arrived_and_held"
                    break
                if not (np.all(np.isfinite(current_p)) and np.all(np.isfinite(current_v))):
                    finished = "invalid_nonfinite_physics_state"
                elif float(current_p[2]) < float(cfg_m.get("min_safe_altitude_m",0.3)):
                    finished = "fell_below_min_safe_altitude"
                elif abs(float(current_p[1]-start[1])) > float(cfg_m["corridor_half_width_m"]):
                    finished = "left_lateral_corridor"
                else:
                    finished = "timeout"
                if finished != "timeout":
                    print(f"[STOP] {finished} t={t:.3f}s XYZ={np.round(current_p,3)} "
                          f"vel={np.round(current_v,3)} waypoint={np.round(waypoint,2)} "
                          f"thrust={thrust:.2f}N saturated={int(saturated)}",flush=True)
                    break
                log_interval = max(1,int(PHYS_HZ*(0.25 if t<=3.0 else 5.0)))
                if physics_step % log_interval == 0:
                    print(f"[FLIGHT] t={t:.2f}s XYZ={np.round(current_p,2)} "
                          f"V={np.round(current_v,2)} waypoint={np.round(waypoint,2)} "
                          f"goal_error={dist:.2f}m thrust={thrust:.1f}N "
                          f"sat={int(saturated)} RGB={logger.camera_count} IMU={logger.imu_valid_count}",flush=True)
        finally:
            logger.set_info("finish_status", finished)
            logger.set_info("elapsed_sim_s", step_count*DT)
            logger.set_info("end_wall_time_utc", now_utc())
            logger.set_info("valid_imu_samples", logger.imu_valid_count)
            logger.set_info("rgb_frames",logger.camera_count)
            logger.close()
            world.pause()
    print(f"[DONE] status={finished}, elapsed={step_count*DT:.2f}s, valid_imu={logger.imu_valid_count}/{logger.imu_count}, rgb_frames={logger.camera_count}",flush=True)
    print(f"[FILES] {SENSOR_DB}\n[FILES] {TRUTH_CSV}",flush=True)
    if logger.imu_valid_count==0:
        raise RuntimeError("No VALID IMU samples recorded. Check IMU prim and sensor initialization.")
    if logger.camera_count==0:
        raise RuntimeError("No RGB frames recorded. Check camera annotator, renderer, and headless settings.")
    if finished!="arrived_and_held" and not args.test:
        print(f"[NOTICE] Mission ended as '{finished}', not successful arrival. "
              "Inspect [STOP] or [FLIGHT] and ground_truth.csv to identify control instability.",flush=True)
    if args.keep_open and not bool(CFG["simulation"]["headless"]):
        print("[VIEW] Simulation finished. Scene will remain open until you close Isaac Sim.",flush=True)
        while simulation_app.is_running():
            simulation_app.update()

try:
    run()
except Exception:
    print("[ERROR] Flight or sensors failed; traceback below",flush=True)
    traceback.print_exc()
    sys.exit(1)
finally:
    simulation_app.close()
