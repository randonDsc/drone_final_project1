"""Iris physics-isolation + sensor recording, rebuilt from working iris_flight_v2.py.

The important principle: ONE World.reset(), NO update() warmup after reset,
NO physics pause/play/reset cycles during initialization. Sensors are optional.

Run with Isaac Sim 6.2 python.bat, from DSC180A:
  & $isaac ./my_drone_sim/scripts/iris_rebuild_v4.py --mode base --distance 0 --height 5 --duration 3
  & $isaac ./my_drone_sim/scripts/iris_rebuild_v4.py --mode imu --distance 0 --height 5 --duration 3
  & $isaac ./my_drone_sim/scripts/iris_rebuild_v4.py --mode camera --distance 0 --height 5 --duration 3
  & $isaac ./my_drone_sim/scripts/iris_rebuild_v4.py --mode both --distance 0 --height 5 --duration 5
  & $isaac ./my_drone_sim/scripts/iris_rebuild_v4.py --mode both --distance 10 --height 5 --duration 45

Only works with your personal USD scene containing /iris and rigid /iris/body.
Only edits an in-memory USD stage. No wind. Doesn't use PPO.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sqlite3
import traceback
import zlib
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_CONFIG = PROJECT / 'configs' / 'flight_rebuild_v4.json'

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
parser.add_argument('--mode', choices=['base','imu','camera','both'], default='base', help='Introduce sensors one at a time')
parser.add_argument('--env', choices=['empty','warehouse','outdoor'], default=None)
parser.add_argument('--height', type=float, default=None)
parser.add_argument('--distance', type=float, default=None, help='0 = hover; positive = fly in +X')
parser.add_argument('--duration', type=float, default=None)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--skip-scene', action='store_true', help='For isolation: never add preset geometry, even in warehouse/outdoor')
parser.add_argument('--fast-render', action='store_true', help='Render at configured 10Hz instead of 120Hz; only use after baseline works')
args = parser.parse_args()
if not args.config.is_file():
    raise FileNotFoundError(f'Missing config: {args.config}. Unzip configs/flight_rebuild_v4.json into my_drone_sim/configs.')
cfg = json.loads(args.config.read_text(encoding='utf-8'))
mission = cfg['mission']; sensors = cfg['sensors']; simcfg = cfg['simulation']
if args.env is not None: cfg['environment'] = args.env
if args.height is not None: mission['altitude_m'] = mission['start_height_m'] = args.height
if args.distance is not None: mission['distance_m'] = args.distance
if args.duration is not None: mission['max_duration_s'] = args.duration
if args.headless: simcfg['headless'] = True
assert cfg['environment'] in ('empty','warehouse','outdoor')
assert mission['distance_m'] >= 0 and mission['start_height_m'] > 0 and mission['max_duration_s'] > 0
assert not simcfg.get('wind_enabled', False), 'This experiment is wind OFF'
PHYS_HZ = int(simcfg['physics_hz']); CAMERA_HZ = int(sensors['rgb_hz']); IMU_HZ = int(sensors['imu_hz'])
RENDER_HZ = int(simcfg['render_hz']); TRUTH_HZ = int(sensors['truth_hz'])
assert all(PHYS_HZ % rate == 0 for rate in (CAMERA_HZ, IMU_HZ, RENDER_HZ, TRUTH_HZ))
DT = 1.0 / PHYS_HZ
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
out_dir = PROJECT / 'results' / f'rebuild_{args.mode}_{cfg["environment"]}_{stamp}'
out_dir.mkdir(parents=True, exist_ok=False)

# The Kit application must launch before any Omniverse imports.
from isaacsim import SimulationApp
app = SimulationApp({'headless': bool(simcfg['headless'])})

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim
from controller import controller
from motor import MotorModel
from drone_diagnostics import read_total_com_in_body_frame


def clock():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def read_body(body):
    p, q = body.get_world_poses()
    return (np.asarray(p[0], dtype=float).copy(), np.asarray(q[0], dtype=float).copy(),
            np.asarray(body.get_linear_velocities()[0], dtype=float).copy(),
            np.asarray(body.get_angular_velocities()[0], dtype=float).copy())


def tilt_deg(q):
    w,x,y,z = np.asarray(q,dtype=float)/np.linalg.norm(q)
    return math.degrees(math.acos(float(np.clip(1 - 2*(x*x+y*y),-1,1))))


class Recorder:
    """Only two user data outputs: sensors.sqlite and ground_truth.csv."""
    def __init__(self):
        self.db_path = out_dir/'sensors.sqlite'
        self.truth_path = out_dir/'ground_truth.csv'
        self.db = sqlite3.connect(str(self.db_path))
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.executescript('''
          CREATE TABLE run_info (key TEXT PRIMARY KEY, value TEXT);
          CREATE TABLE imu (
            physics_step INTEGER PRIMARY KEY, sim_time_s REAL, sensor_time_s REAL,
            wall_time_utc TEXT, valid INTEGER,
            ax_mps2 REAL, ay_mps2 REAL, az_mps2 REAL,
            wx_rad_s REAL, wy_rad_s REAL, wz_rad_s REAL,
            qw REAL, qx REAL, qy REAL, qz REAL);
          CREATE TABLE camera (
            physics_step INTEGER PRIMARY KEY, sim_time_s REAL, wall_time_utc TEXT,
            width INTEGER, height INTEGER, channels INTEGER,
            codec TEXT, rgb_blob BLOB);
        ''')
        self.db.execute('INSERT INTO run_info VALUES (?,?)',('configuration', json.dumps(cfg)))
        self.db.execute('INSERT INTO run_info VALUES (?,?)',('mode', args.mode))
        self.db.execute('INSERT INTO run_info VALUES (?,?)',('run_id', stamp))
        self.tf = self.truth_path.open('w', newline='', encoding='utf-8')
        self.writer = csv.writer(self.tf)
        self.writer.writerow(['run_id','physics_step','sim_time_s','x_m','y_m','z_m',
                             'qw','qx','qy','qz','vx_mps','vy_mps','vz_mps',
                             'omega_x','omega_y','omega_z','tilt_deg',
                             'target_x_m','target_y_m','target_z_m','goal_error_m',
                             'wp_x_m','wp_y_m','thrust_command_n','actual_thrust_n',
                             'saturated','wall_time_utc'])
        self.imu_n=0; self.imu_valid=0; self.rgb_n=0

    def truth(self, i, t, p, q, v, w, goal, wp, cmd, actual, saturated):
        self.writer.writerow([stamp,i,round(t,6),*p.tolist(),*q.tolist(),*v.tolist(),
                             *w.tolist(),tilt_deg(q),*goal.tolist(),float(np.linalg.norm(p-goal)),
                             *wp.tolist(),float(cmd),float(actual),int(saturated),clock()])
        if i % (PHYS_HZ*2) == 0: self.tf.flush()

    def imu(self, i,t,reading):
        valid = bool(getattr(reading,'is_valid',False))
        if valid:
            try:
                vec = [float(getattr(reading,n)) for n in (
                    'linear_acceleration_x','linear_acceleration_y','linear_acceleration_z',
                    'angular_velocity_x','angular_velocity_y','angular_velocity_z',
                    'orientation_w','orientation_x','orientation_y','orientation_z')]
                timestamp = float(reading.time)
            except Exception as exc:
                valid = False
                print(f'[IMU] Invalid field at t={t:.2f}s: {exc}', flush=True)
        if not valid:
            vec=[None]*10; timestamp=None
        self.db.execute('INSERT INTO imu VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (i,round(t,6),timestamp,clock(),int(valid),*vec))
        self.imu_n+=1; self.imu_valid+=int(valid)

    def rgb(self, i,t,arr):
        if arr is None:
            return False
        arr=np.asarray(arr)
        if arr.size==0 or arr.ndim!=3 or arr.shape[2] < 3:
            return False
        arr=arr[:,:,:3]
        if np.issubdtype(arr.dtype,np.floating):
            if float(np.nanmax(arr))<=1.01: arr=arr*255
            arr=np.clip(arr,0,255)
        arr=np.ascontiguousarray(arr,dtype=np.uint8)
        h,w,c=arr.shape
        self.db.execute('INSERT INTO camera VALUES (?,?,?,?,?,?,?,?)',
                        (i,round(t,6),clock(),w,h,c,'zlib-rgb8',sqlite3.Binary(zlib.compress(arr.tobytes(),3))))
        self.rgb_n+=1
        return True

    def close(self, status):
        self.db.execute('INSERT OR REPLACE INTO run_info VALUES (?,?)',('status',status))
        self.db.execute('INSERT OR REPLACE INTO run_info VALUES (?,?)',('valid_imu',str(self.imu_valid)))
        self.db.execute('INSERT OR REPLACE INTO run_info VALUES (?,?)',('rgb_frames',str(self.rgb_n)))
        self.db.commit(); self.db.close(); self.tf.close()


def main():
    scene=(PROJECT / cfg['scene']).resolve()
    if not scene.is_file():
        raise FileNotFoundError(f'Scene missing: {scene}')
    print(f'[START] mode={args.mode} env={cfg["environment"]} height={mission["start_height_m"]} '
          f'distance={mission["distance_m"]} duration={mission["max_duration_s"]} wind=OFF',flush=True)
    print('[PHASE] 1/4 load stage and move Iris root (no sensors)',flush=True)
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError('Cannot open USD stage')
    app.update()  # Deliberately follows old known-working script, BEFORE world creation.
    stage=omni.usd.get_context().get_stage()
    iris=stage.GetPrimAtPath('/iris'); bodyprim=stage.GetPrimAtPath('/iris/body')
    if not iris.IsValid() or not bodyprim.IsValid() or not bodyprim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError('Expected /iris and rigid /iris/body')
    if stage.GetPrimAtPath('/World/MyBox').IsValid():
        stage.RemovePrim('/World/MyBox')
    z=float(mission['start_height_m'])
    UsdGeom.XformCommonAPI(iris).SetTranslate(Gf.Vec3d(0,0,z))
    usd_before = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(bodyprim).ExtractTranslation(),dtype=float)
    print(f'[USD] body world pose before World.reset: {np.round(usd_before,5)}',flush=True)
    if abs(usd_before[2]-z)>0.25:
        raise RuntimeError(f'USD body at {usd_before[2]:.2f} m, not {z:.2f}. Inspect original stage transforms, aborting rather than force-correcting.')
    if cfg['environment']!='empty' and not args.skip_scene:
        from scene_presets import build
        build(stage,cfg)

    print('[PHASE] 2/4 add optional sensor prims BEFORE one physics reset',flush=True)
    imu=None; cam=None
    if args.mode in ('imu','both'):
        from isaacsim.sensors.experimental.physics import IMU,IMUSensor
        path='/iris/body/imu_rebuild_v4'
        if stage.GetPrimAtPath(path).IsValid():
            imu=IMUSensor(path)
        else:
            imu=IMUSensor(IMU.create(path,
               translations=np.array([sensors['imu_position_body_m']],dtype=float),
               orientations=np.array([[1.,0.,0.,0.]],dtype=float),
               linear_acceleration_filter_size=1,
               angular_velocity_filter_size=1,
               orientation_filter_size=1))
    if args.mode in ('camera','both'):
        from isaacsim.sensors.camera import Camera
        cam=Camera(prim_path='/iris/body/rgb_rebuild_v4',name='rgb_rebuild_v4',
                   resolution=(int(sensors['rgb_width']),int(sensors['rgb_height'])),
                   frequency=CAMERA_HZ)
        cam.set_local_pose(translation=np.asarray(sensors['camera_position_body_m'],dtype=float),
                           orientation=np.array([1.,0.,0.,0.],dtype=float),camera_axes='world')

    scenes=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes)>1: raise RuntimeError(f'Multiple physics scenes: {scenes}')
    physpath = scenes[0] if scenes else '/World/PhysicsScene'
    # Exactly the same world initialization pattern as the successful 8m test.
    world=World(physics_dt=DT,rendering_dt=DT,stage_units_in_meters=1.,physics_prim_path=physpath)
    body=world.scene.add(RigidPrim(prim_paths_expr='/iris/body',name='rebuild_body'))
    world.reset()                   # ONCE, not twice. Never pause or reset again.
    p,q,v,w=read_body(body)
    body.set_default_state(positions=p.reshape(1,3),orientations=q.reshape(1,4),
                           linear_velocities=np.zeros((1,3),dtype=np.float32),
                           angular_velocities=np.zeros((1,3),dtype=np.float32))
    body.post_reset()               # Same as known working iris_flight_v2.py
    p,q,v,w=read_body(body)
    print(f'[PRE] physics body p={np.round(p,5)} v={np.round(v,5)} w={np.round(w,5)}',flush=True)
    if abs(p[2]-z)>0.25: raise RuntimeError(f'PhysX start height mismatch {p[2]} vs {z}')
    # Camera initialize can attach a render product but MUST NOT simulate unpowered steps.
    if cam is not None:
        cam.initialize()            # No app.update(), no World.reset(), no warm-up sim steps.
        p2,q2,v2,w2=read_body(body)
        print(f'[POST-CAMERA-INIT] physics body p={np.round(p2,5)} v={np.round(v2,5)}',flush=True)
        if np.linalg.norm(p2-p)>0.05:
            raise RuntimeError('Camera initialization changed rigid-body pose: stage/tensor synchronization issue')

    print('[PHASE] 3/4 baseline controller & motor model',flush=True)
    mass=0.; mass_views=[]
    for pr in stage.Traverse():
        path=str(pr.GetPath())
        if not path.startswith('/iris/') or not pr.HasAPI(UsdPhysics.RigidBodyAPI): continue
        if UsdPhysics.RigidBodyAPI(pr).GetRigidBodyEnabledAttr().Get() is False: continue
        vprim=RigidPrim(prim_paths_expr=path,name=f'rebuild_mass_{len(mass_views)}',reset_xform_properties=False)
        vprim.initialize(); mass_views.append(vprim)
        mass += float(np.asarray(vprim.get_masses()).reshape(-1)[0])
    motors=MotorModel(5.84e-6,0.06*5.84e-6,1100.0,[-1,-1,+1,+1])
    rotor_pos=np.array([[0.13798545,-0.20671639,0.023],
                        [-0.12511168,0.21875414,0.023],
                        [0.138,0.20257577,0.023],
                        [-0.12415360,-0.22234580,0.023]],dtype=float)
    allocation=np.vstack([np.ones(4),rotor_pos[:,1],-rotor_pos[:,0],motors.reaction_signs*motors.km/motors.kf])
    max_rotor=motors.kf*motors.max_speed**2
    print(f'[MODEL] mass={mass:.5f}kg weight={mass*9.81:.3f}N max_total_thrust={4*max_rotor:.3f}N',flush=True)
    if not np.isfinite(mass) or mass<=0: raise RuntimeError('Invalid drone mass')
    start=read_body(body)[0]
    goal=np.array([start[0]+float(mission['distance_m']),start[1],float(mission['altitude_m'])],dtype=float)
    target=dict(height=float(goal[2]),xy=start[:2].copy(),roll=0.,pitch=0.,yaw=0.,mass=mass)
    rec=Recorder(); last=read_body(body)[0]; first_z=last[2]
    status='timeout'; nsteps=int(round(float(mission['max_duration_s'])*PHYS_HZ));
    stable_goal_count=0; hold_steps=int(round(float(mission.get('hold_at_goal_s',2.0))*PHYS_HZ))
    last_cmd=0.; last_actual=0.; last_sat=False; waypoint=start[:2].copy(); measured_steps=0
    print('[PHASE] 4/4 START PHYSICS (no warm-up without motor thrust)',flush=True)
    print(f'[OUTPUT] {rec.db_path}\n[OUTPUT] {rec.truth_path}',flush=True)
    world.play()
    try:
        for i in range(1,nsteps+1):
            if not app.is_running(): status='app_closed'; break
            before,q,v,w=read_body(body)
            if i<=15:
                print(f'[BEFORE] step={i} t={(i-1)*DT:.5f} p={np.round(before,5)} v={np.round(v,5)}',flush=True)
            if not np.all(np.isfinite(before)) or not np.all(np.isfinite(v)):
                status='nonfinite_body_state'; break
            t0=(i-1)*DT
            if float(mission['distance_m'])==0 or t0<float(mission.get('stabilization_s',2.0)):
                waypoint=start[:2].copy()
            else:
                # Limit target step: controller sees a nearby waypoint, not a 100m error.
                wp_x=min(goal[0], before[0]+float(mission.get('lookahead_m',6.)))
                waypoint=np.array([wp_x,start[1]],dtype=float)
            target['xy']=waypoint
            cmd,tx,ty,tz=controller(positions=before,orientations=q,
                                    velocities=v,angular_velocities=w,target=target)
            com=np.asarray(read_total_com_in_body_frame(body,mass_views),dtype=float)
            wrench=np.array([cmd,tx,ty,tz],dtype=float)
            wrench[1:] += np.cross(com,np.array([0.,0.,cmd]))
            raw=np.linalg.solve(allocation,wrench)
            saturated=bool(np.any((raw<0)|(raw>max_rotor)))
            thrusts=np.clip(raw,0.,max_rotor)
            speeds=np.sqrt(thrusts/motors.kf)
            f,react=motors.calculate(speeds)
            force=np.array([0.,0.,float(np.sum(f))],dtype=float)
            rotor_forces=np.zeros((4,3),dtype=float); rotor_forces[:,2]=f
            torque=np.cross(rotor_pos,rotor_forces).sum(axis=0)
            torque[2] += float(np.sum(react))
            # ONE local force+torque call; same as old successful script.
            body.apply_forces_and_torques_at_pos(
                forces=force.astype(np.float32).reshape(1,3),
                torques=torque.astype(np.float32).reshape(1,3),is_global=False)
            render_now = (not args.fast_render or i%(PHYS_HZ//RENDER_HZ)==0)
            world.step(render=render_now)
            p,q,v,w=read_body(body); t=i*DT; measured_steps=i
            if i<=15:
                print(f'[AFTER]  step={i} t={t:.5f} p={np.round(p,5)} v={np.round(v,5)} '
                      f'dz={p[2]-before[2]:.5f}m thrust={force[2]:.2f}N',flush=True)
            # Distinguish physics discontinuity/teleport from ordinary gravity.
            delta=float(np.linalg.norm(p-before))
            if delta>0.18 and i<PHYS_HZ:
                status='PHYSICS_POSE_JUMP'
                print(f'[ROOT-CAUSE] {status} at step={i}: pre={np.round(before,4)} post={np.round(p,4)} '
                      f'delta={delta:.4f}m within {DT:.5f}s; this is NOT normal gravitational fall. '
                      f'Check USD root xform / collision / physics view initialization.',flush=True)
            if i%(PHYS_HZ//TRUTH_HZ)==0 or status=='PHYSICS_POSE_JUMP':
                rec.truth(i,t,p,q,v,w,goal,waypoint,cmd,force[2],saturated)
            if imu is not None and i%(PHYS_HZ//IMU_HZ)==0:
                try: rec.imu(i,t,imu.get_sensor_reading())
                except Exception as exc: print(f'[IMU WARNING] {exc}',flush=True)
            if cam is not None and i%(PHYS_HZ//CAMERA_HZ)==0:
                try:
                    ok=rec.rgb(i,t,cam.get_rgb(device='cpu'))
                    if not ok and rec.rgb_n<3: print(f'[RGB] no valid image at t={t:.2f}s',flush=True)
                except Exception as exc: print(f'[RGB WARNING] t={t:.2f}s {exc}',flush=True)
            if i%max(1,int(PHYS_HZ*.5))==0:
                print(f'[FLIGHT] t={t:.2f}s xyz={np.round(p,3)} vel={np.round(v,3)} '
                      f'tilt={tilt_deg(q):.1f}deg thrust={force[2]:.2f}N sat={int(saturated)} '
                      f'IMU={rec.imu_valid}/{rec.imu_n} RGB={rec.rgb_n}',flush=True)
            if status=='PHYSICS_POSE_JUMP': break
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)):
                status='nonfinite_physics'; break
            if p[2]<float(mission.get('min_safe_altitude_m',0.3)):
                status='fell_below_safe_altitude'; break
            if abs(p[1]-start[1])>float(mission.get('corridor_half_width_m',5.)):
                status='left_corridor'; break
            if float(mission['distance_m'])>0:
                close=np.linalg.norm(p-goal)<=float(mission.get('arrival_radius_m',1.0))
                slow=np.linalg.norm(v)<=float(mission.get('arrival_max_speed_mps',0.5))
                stable_goal_count = (stable_goal_count+1) if (close and slow) else 0
                if stable_goal_count>=hold_steps:
                    status='arrived_and_held'; break
        else:
            if float(mission['distance_m'])==0:
                p,q,v,w=read_body(body)
                good=abs(p[2]-z)<0.6 and np.linalg.norm(v)<1.0 and abs(p[1]-start[1])<1.0
                status='hover_completed' if good else 'hover_unstable'
    finally:
        # Snapshot the dynamic state BEFORE pausing the PhysX timeline.
        try:
            last_state=read_body(body)
        except Exception:
            last_state=(np.full(3, np.nan),np.full(4,np.nan),np.full(3,np.nan),np.full(3,np.nan))
        rec.close(status)
        world.pause()
    p,q,v,w=last_state
    print(f'[END] status={status} elapsed={measured_steps*DT:.3f}s '
          f'xyz={np.round(p,3)} vel={np.round(v,3)} '
          f'valid_IMU={rec.imu_valid}/{rec.imu_n} RGB={rec.rgb_n}',flush=True)
    print(f'[FILES] {rec.db_path}\n[FILES] {rec.truth_path}',flush=True)
    if cam is not None and measured_steps>=PHYS_HZ and rec.rgb_n==0:
        print('[CAMERA ISSUE] Flight survived >=1s but camera provided 0 RGB frames. '
              'Check render-product warning and GUI/headless configuration.',flush=True)
    if status not in ('arrived_and_held','hover_completed'):
        print('[RESULT] Not a successful flight; do not advance to longer distance.',flush=True)

try:
    main()
except Exception:
    print('[EXCEPTION] Fatal script failure',flush=True)
    traceback.print_exc()
finally:
    app.close()
