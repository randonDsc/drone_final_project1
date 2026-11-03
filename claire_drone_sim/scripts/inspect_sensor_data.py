"""Review a completed Iris sensor run without opening Isaac Sim.

Usage:
  python inspect_sensor_data.py "C:\\...\\results\\sensor_empty_..."
  python inspect_sensor_data.py <run_directory> --plot
"""
import argparse
import csv
import json
import sqlite3
import zlib
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
parser.add_argument("--plot", action="store_true", help="Create summary PNG plots; requires matplotlib and numpy")
args = parser.parse_args()
folder = args.run_dir.resolve()
sensor_db = folder / "sensors.sqlite"
truth_csv = folder / "ground_truth.csv"
if not sensor_db.is_file() or not truth_csv.is_file():
    raise FileNotFoundError("Expected sensors.sqlite AND ground_truth.csv in the run folder")

conn = sqlite3.connect(str(sensor_db))
try:
    config = json.loads(conn.execute("SELECT value FROM run_info WHERE key='configuration'").fetchone()[0])
    status_row = conn.execute("SELECT value FROM run_info WHERE key='finish_status'").fetchone()
    imu_total, imu_valid = conn.execute("SELECT count(*), COALESCE(sum(valid),0) FROM imu").fetchone()
    frame_count = conn.execute("SELECT count(*) FROM camera").fetchone()[0]
    last_imu = conn.execute("SELECT physics_step,sim_time_s,ax_mps2,ay_mps2,az_mps2,wx_rad_s,wy_rad_s,wz_rad_s FROM imu WHERE valid=1 ORDER BY physics_step DESC LIMIT 1").fetchone()
    print("Environment:",config["environment"])
    print("Mission distance:",config["mission"]["distance_m"],"m")
    print("Run status:",status_row[0] if status_row else "unknown")
    print("IMU samples:",imu_total,"valid:",imu_valid)
    print("RGB frames:",frame_count)
    print("Most recent valid IMU reading:",last_imu)
    with truth_csv.open(newline="",encoding="utf-8") as f:
        r = csv.DictReader(f)
        truth_rows = list(r)
    print("Ground-truth samples:",len(truth_rows))
    if truth_rows:
        last = truth_rows[-1]
        print("Last truth position (m):",*(round(float(last[c]),3) for c in ("x_m","y_m","z_m")))
        print("Last target distance (m):",round(float(last["position_error_m"]),3))
    cam = conn.execute("SELECT physics_step,width,height,channels,codec,rgb_blob FROM camera ORDER BY physics_step LIMIT 1").fetchone()
    if cam:
        step,w,h,c,codec,blob = cam
        raw = zlib.decompress(blob)
        if codec!="zlib-rgb8" or len(raw)!=w*h*c:
            raise RuntimeError("Unexpected camera codec or corrupt frame")
        print(f"First image: step={step}, {w}x{h}x{c}, compressed bytes={len(blob)}")
    if args.plot:
        import matplotlib.pyplot as plt
        import numpy as np
        out = folder/"plots"
        out.mkdir(exist_ok=True)
        if truth_rows:
            t=np.array([float(r["sim_time_s"]) for r in truth_rows])
            x=np.array([float(r["x_m"]) for r in truth_rows])
            y=np.array([float(r["y_m"]) for r in truth_rows])
            z=np.array([float(r["z_m"]) for r in truth_rows])
            goal=float(config["mission"]["distance_m"])
            fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
            axes[0].plot(t,x,label="X position");axes[0].axhline(goal,linestyle="--",label="X target")
            axes[0].set_ylabel("Forward progress (m)");axes[0].legend();axes[0].grid(alpha=.25)
            axes[1].plot(t,y,label="Y drift");axes[1].plot(t,z,label="Z altitude")
            axes[1].axhline(float(config["mission"]["altitude_m"]),linestyle="--",label="Target altitude")
            axes[1].set_xlabel("Simulated time (s)");axes[1].set_ylabel("Position (m)");axes[1].legend();axes[1].grid(alpha=.25)
            fig.tight_layout();fig.savefig(out/"flight_ground_truth.png",dpi=160);plt.close(fig)
        imu_rows=conn.execute("SELECT sim_time_s,ax_mps2,ay_mps2,az_mps2 FROM imu WHERE valid=1 ORDER BY physics_step").fetchall()
        if imu_rows:
            a=np.array(imu_rows)
            fig,ax=plt.subplots(figsize=(10,4))
            for k,label in enumerate(("ax","ay","az"),1):
                ax.plot(a[:,0],a[:,k],label=label,linewidth=1)
            ax.set(xlabel="Simulated time (s)",ylabel="Specific force (m/s²)",title="IMU accelerometer (sensor frame)")
            ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/"imu_acceleration.png",dpi=160);plt.close(fig)
        # Compare gyro IMU readings with rigid-body WORLD angular velocity
        # rotated into the drone BODY frame. Exact matching is not expected.
        gyro_rows=conn.execute("SELECT physics_step,sim_time_s,wx_rad_s,wy_rad_s,wz_rad_s FROM imu WHERE valid=1").fetchall()
        if gyro_rows and truth_rows:
            truth_by_step={int(r["physics_step"]):r for r in truth_rows}
            def rotate_world_to_body(q,v):
                qw,qx,qy,qz=map(float,q)
                n=np.linalg.norm([qw,qx,qy,qz])
                if n<1.e-8: return np.array([np.nan]*3)
                qw,qx,qy,qz=np.array([qw,qx,qy,qz])/n
                # Rotation body-to-world, then transpose for world-to-body.
                rot=np.array([
                    [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
                    [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
                    [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]])
                return rot.T @ np.asarray(v,dtype=float)
            times=[]; sensor_gyro=[]; truth_gyro=[]
            for step,sim_t,wx,wy,wz in gyro_rows:
                r=truth_by_step.get(int(step))
                if r is None: continue
                quat=[r[k] for k in ("qw","qx","qy","qz")]
                ground_omega=[r[k] for k in ("omega_world_x_rad_s","omega_world_y_rad_s","omega_world_z_rad_s")]
                times.append(sim_t)
                sensor_gyro.append([wx,wy,wz])
                truth_gyro.append(rotate_world_to_body(quat,ground_omega))
            if times:
                sg=np.asarray(sensor_gyro);tg=np.asarray(truth_gyro)
                fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
                for j,ax in enumerate(axes):
                    ax.plot(times,sg[:,j],label="IMU gyro",linewidth=1)
                    ax.plot(times,tg[:,j],label="PhysX world gyro rotated to body",linewidth=1,alpha=.8)
                    ax.set_ylabel(f"{('X','Y','Z')[j]} (rad/s)")
                    ax.grid(alpha=.25);ax.legend(fontsize=8)
                axes[-1].set_xlabel("Simulated time (s)")
                fig.suptitle("IMU vs. physics ground truth (angular rates)")
                fig.tight_layout();fig.savefig(out/"imu_vs_truth_gyro.png",dpi=160);plt.close(fig)
            # Approximate truth specific force using differentiated world
            # velocity and gravity compensation; finite differences are noisy.
            # This approximation is for cross-checking, not a replacement IMU.
            alltruth=sorted(truth_rows,key=lambda r:int(r["physics_step"]))
            accel_truth_by_step={}
            for k in range(1,len(alltruth)-1):
                prev,cur,nxt=alltruth[k-1],alltruth[k],alltruth[k+1]
                delta=float(nxt["sim_time_s"])-float(prev["sim_time_s"])
                if delta<=0: continue
                vv0=np.array([float(prev[c]) for c in ("vx_mps","vy_mps","vz_mps")])
                vv1=np.array([float(nxt[c]) for c in ("vx_mps","vy_mps","vz_mps")])
                acc_world=(vv1-vv0)/delta
                specific_force_world=acc_world-np.array([0,0,-9.81])
                quat=[cur[c] for c in ("qw","qx","qy","qz")]
                accel_truth_by_step[int(cur["physics_step"])]=rotate_world_to_body(quat,specific_force_world)
            accel_rows=conn.execute("SELECT physics_step,sim_time_s,ax_mps2,ay_mps2,az_mps2 FROM imu WHERE valid=1").fetchall()
            a_times=[];a_sensor=[];a_truth=[]
            for step,t,ax,ay,az in accel_rows:
                if int(step) not in accel_truth_by_step: continue
                a_times.append(t);a_sensor.append([ax,ay,az]);a_truth.append(accel_truth_by_step[int(step)])
            if a_times:
                ats=np.asarray(a_sensor);att=np.asarray(a_truth)
                fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
                for j,axis in enumerate(axes):
                    axis.plot(a_times,ats[:,j],label="IMU accelerometer",linewidth=1)
                    axis.plot(a_times,att[:,j],label="Truth velocity-derived specific force (approx)",alpha=.8,linewidth=1)
                    axis.set_ylabel(f"{('X','Y','Z')[j]} (m/s²)")
                    axis.grid(alpha=.25);axis.legend(fontsize=8)
                axes[-1].set_xlabel("Simulated time (s)")
                fig.suptitle("IMU vs. approximate physics specific force")
                fig.tight_layout();fig.savefig(out/"imu_vs_truth_acceleration.png",dpi=160);plt.close(fig)
        if cam:
            frame=np.frombuffer(raw,dtype=np.uint8).reshape(h,w,c)
            plt.imsave(out/"first_camera_frame.png",frame)
        print("Graphs and RGB preview saved to:",out)
finally:
    conn.close()
