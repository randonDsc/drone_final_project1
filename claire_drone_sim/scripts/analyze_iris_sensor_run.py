#!/usr/bin/env python3
r"""Analyze an Iris Rebuild v4 run, without launching Isaac Sim.

Python dependencies: numpy and matplotlib.

Usage (from DSC180A, if saved in my_drone_sim/scripts):
  python .\\my_drone_sim\\scripts\\analyze_iris_sensor_run.py --list
  python .\\my_drone_sim\\scripts\\analyze_iris_sensor_run.py
  python .\\my_drone_sim\\scripts\\analyze_iris_sensor_run.py --run ".\my_drone_sim\results\rebuild_both_empty_..."

Default selects the newest completed *successful* Rebuild v4 run with BOTH sensors.
Only adds a new analysis/ folder; never changes sensors.sqlite or ground_truth.csv.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sqlite3
import sys
import zlib
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def metadata(folder):
    db = folder / 'sensors.sqlite'
    truth = folder / 'ground_truth.csv'
    if not db.is_file() or not truth.is_file():
        return None
    with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True) as c:
        info = dict(c.execute('SELECT key,value FROM run_info').fetchall())
        imu_count, imu_valid = c.execute('SELECT COUNT(*), COALESCE(SUM(valid),0) FROM imu').fetchone()
        rgb_count = c.execute('SELECT COUNT(*) FROM camera').fetchone()[0]
    conf = json.loads(info.get('configuration','{}'))
    return dict(folder=folder, info=info, config=conf, status=info.get('status',info.get('finish_status','unknown')),
                imu=int(imu_count),valid=int(imu_valid or 0),rgb=int(rgb_count),
                env=conf.get('environment','?'), distance=conf.get('mission',{}).get('distance_m','?'))


def list_runs(root):
    found=[]
    for folder in root.glob('rebuild_*'):
        if not folder.is_dir():
            continue
        try:
            m=metadata(folder)
            if m: found.append(m)
        except (sqlite3.DatabaseError, ValueError, OSError) as exc:
            print(f'[SKIP] {folder.name}: {exc}')
    return sorted(found, key=lambda x:x['folder'].stat().st_mtime, reverse=True)


def load_rows(path):
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def series(rows, col):
    return np.array([float(r[col]) for r in rows], dtype=float)


def quaternion_rotation(q):
    qw,qx,qy,qz=np.asarray(q,dtype=float)
    norm=np.linalg.norm([qw,qx,qy,qz])
    if norm<1e-9: return np.eye(3)*np.nan
    qw,qx,qy,qz=np.array([qw,qx,qy,qz])/norm
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)]
    ])


def stats(a,b):
    d=np.asarray(a)-np.asarray(b)
    return np.nanmean(d,axis=0), np.sqrt(np.nanmean(d*d,axis=0))


def make_plot_xyz(file, t, vals, ylabel, title, target=None):
    fig, ax = plt.subplots(figsize=(11,4.6))
    for i, c in enumerate('xyz'):
        ax.plot(t, vals[:,i], label=c.upper(),lw=1.25)
    if target is not None:
        ax.axhline(target,color='black',linestyle=':',linewidth=1,label='Altitude target')
    ax.set(title=title, xlabel='Simulated time (s)', ylabel=ylabel)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout();fig.savefig(file,dpi=150); plt.close(fig)


def paired_plot(file, t, measured, expected, title, unit):
    fig,axs=plt.subplots(3,1,figsize=(11,9),sharex=True)
    for j,axis in enumerate(axs):
        axis.plot(t,measured[:,j],label='IMU measurement',lw=1.2)
        axis.plot(t,expected[:,j],label='Physics-derived reference',lw=1.0,alpha=.78)
        axis.set_ylabel(f"{'XYZ'[j]} ({unit})")
        axis.grid(alpha=.25); axis.legend(loc='upper right',fontsize=8)
    axs[-1].set_xlabel('Simulated time (s)')
    fig.suptitle(title);fig.tight_layout();fig.savefig(file,dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent/'results',help='Path to my_drone_sim/results')
    ap.add_argument('--run',type=Path,help='Explicit results folder; otherwise choose most recent successful Rebuild v4 both-sensors run')
    ap.add_argument('--list',action='store_true',help='List recorded Rebuild v4 runs and exit')
    args=ap.parse_args()
    root=args.root.resolve()
    runs=list_runs(root)
    if args.list:
        print(f'Found {len(runs)} Rebuild v4 runs in {root}')
        for i,r in enumerate(runs,1):
            print(f'{i:>2}. {r["folder"].name}\n    status={r["status"]:<26} distance={r["distance"]}m  IMU={r["valid"]}/{r["imu"]}  RGB={r["rgb"]}')
        return
    if args.run:
        folder=args.run.resolve()
        m=metadata(folder)
        if m is None: raise SystemExit('Given folder does not have both sensors.sqlite and ground_truth.csv')
    else:
        possible=[r for r in runs if r['folder'].name.startswith('rebuild_both_') and r['status'] in ('arrived_and_held','hover_completed') and r['valid']>0 and r['rgb']>0]
        if not possible:
            raise SystemExit('No successful both-sensor Rebuild v4 run found. Use --list or specify --run.')
        m=possible[0];folder=m['folder']
    out=folder/'analysis';out.mkdir(exist_ok=True)
    print(f'\n[SELECTED] {folder}')
    print(f'[RUN] status={m["status"]} distance={m["distance"]}m mode={m["info"].get("mode")} IMU={m["valid"]}/{m["imu"]} camera={m["rgb"]}')

    truth=load_rows(folder/'ground_truth.csv')
    if len(truth)<3: raise SystemExit('Not enough ground-truth rows')
    truth_by_step={int(row['physics_step']):row for row in truth}
    t=series(truth,'sim_time_s')
    xyz=np.column_stack([series(truth,f'{c}_m') for c in 'xyz'])
    vel=np.column_stack([series(truth,f'v{c}_mps') for c in 'xyz'])
    omega=np.column_stack([series(truth,f'omega_{c}') for c in 'xyz'])
    quat=np.column_stack([series(truth,c) for c in ['qw','qx','qy','qz']])
    tgt=np.array([float(truth[-1][f'target_{c}_m']) for c in 'xyz'])
    dist=np.linalg.norm(xyz-tgt,axis=1)
    speed=np.linalg.norm(vel,axis=1)
    lateral=np.abs(xyz[:,1]-tgt[1]); alt=np.abs(xyz[:,2]-tgt[2])
    max_time=float(t[-1]); delta=np.diff(t)
    f_truth=1/np.median(delta)

    print(f'[TRUTH] {len(truth)} rows, approx {f_truth:.1f} Hz, duration={max_time:.3f}s')
    print(f'[FLIGHT] final xyz={np.round(xyz[-1],4)}, goal={np.round(tgt,4)}, 3D error={dist[-1]:.4f}m')
    print(f'[FLIGHT] mean |Y error|={np.mean(lateral):.5f}m, max={np.max(lateral):.5f}m; mean |altitude error|={np.mean(alt):.5f}m')

    with sqlite3.connect(f'file:{(folder/"sensors.sqlite").as_posix()}?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        imu=[dict(r) for r in c.execute('SELECT * FROM imu WHERE valid=1 ORDER BY physics_step')]
        cam_summary=[dict(r) for r in c.execute('SELECT physics_step, sim_time_s, width, height, channels, codec FROM camera ORDER BY physics_step')]
        selected=[]
        if cam_summary:
            ids=sorted(set([0,len(cam_summary)//2,len(cam_summary)-1]))
            for index in ids:
                step=cam_summary[index]['physics_step']
                row=c.execute('SELECT * FROM camera WHERE physics_step=?',(step,)).fetchone()
                selected.append(dict(row))

    if not imu: raise SystemExit('No valid IMU rows')
    imu_t=np.array([r['sim_time_s'] for r in imu],dtype=float)
    imua=np.array([[r[f'a{c}_mps2'] for c in 'xyz'] for r in imu],dtype=float)
    imug=np.array([[r[f'w{c}_rad_s'] for c in 'xyz'] for r in imu],dtype=float)
    aligned=[(i,truth_by_step[int(r['physics_step'])]) for i,r in enumerate(imu) if int(r['physics_step']) in truth_by_step]
    print(f'[ALIGNMENT] {len(aligned)} exact matches by physics_step out of {len(imu)} valid IMU samples; timestamps compared below')
    if not aligned: raise SystemExit('No shared physics steps between sensor and ground truth: check logging configuration.')
    ix=np.array([i for i,_ in aligned]); aligned_truth=[r for _,r in aligned]
    t2=imu_t[ix]
    time_offset=np.array([imu_t[i]-float(r['sim_time_s']) for i,r in aligned])
    print(f'[SYNC] max |IMU sim time - truth sim time|={np.max(np.abs(time_offset)):.9f}s')
    gyro_ref=[];quat_deg=[]
    for j,r in enumerate(aligned_truth):
        q=np.array([float(r[k]) for k in ('qw','qx','qy','qz')],dtype=float)
        R=quaternion_rotation(q)
        om=np.array([float(r[k]) for k in ('omega_x','omega_y','omega_z')])
        gyro_ref.append(R.T@om)  # Ground-truth omega assumed world frame.
        iq=np.array([float(imu[ix[j]][k]) for k in ('qw','qx','qy','qz')])
        nq=np.linalg.norm(q)*np.linalg.norm(iq)
        dot=abs(float(np.dot(q,iq)/nq)) if nq>1e-9 else np.nan
        quat_deg.append(2*np.degrees(np.arccos(np.clip(dot,0,1))))
    gyro_ref=np.asarray(gyro_ref)
    bg,rg=stats(imug[ix],gyro_ref)
    print(f'[GYRO] mean difference (IMU - rotated truth), xyz = {np.round(bg,6)} rad/s')
    print(f'[GYRO] RMSE xyz = {np.round(rg,6)} rad/s')
    print(f'[ORIENTATION] median absolute orientation difference = {np.nanmedian(quat_deg):.5f} degrees')

    # Central differences of physics world velocities, then subtract gravity and rotate into body.
    # Drop edge rows. Note this does not account for sensor offset/angular acceleration, although
    # configuration currently mounts IMU at body origin.
    a_world=np.gradient(vel,t,axis=0,edge_order=2)
    specific_world=a_world - np.array([0.,0.,-9.81])
    a_ref_by_step={}
    for k,r in enumerate(truth):
        if k==0 or k==len(truth)-1: continue
        rot=quaternion_rotation(quat[k])
        a_ref_by_step[int(r['physics_step'])]=rot.T@specific_world[k]
    accel_ix=np.array([i for i,r in enumerate(imu) if int(r['physics_step']) in a_ref_by_step],dtype=int)
    accel_ref=np.array([a_ref_by_step[int(imu[i]['physics_step'])] for i in accel_ix])
    ba,ra=stats(imua[accel_ix],accel_ref)
    print(f'[ACCEL] mean difference IMU - physics-derived specific force = {np.round(ba,5)} m/s^2')
    print(f'[ACCEL] RMSE xyz = {np.round(ra,5)} m/s^2 (numerical derivative approximation)')
    if cam_summary:
        ct=np.array([r['sim_time_s'] for r in cam_summary])
        rate=1/np.median(np.diff(ct)) if len(ct)>1 else float('nan')
        print(f'[CAMERA] {len(ct)} RGB frames, ~{rate:.2f} fps, size={cam_summary[0]["width"]}x{cam_summary[0]["height"]}')

    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    axes[0].plot(t,xyz[:,0],label='X position');axes[0].axhline(tgt[0],ls=':',color='black',label='X target');axes[0].set_ylabel('X (m)');axes[0].legend()
    axes[1].plot(t,xyz[:,1]);axes[1].axhline(tgt[1],ls=':',color='black');axes[1].set_ylabel('Y deviation (m)')
    axes[2].plot(t,xyz[:,2]);axes[2].axhline(tgt[2],ls=':',color='black');axes[2].set_ylabel('Z altitude (m)');axes[2].set_xlabel('Simulated time (s)')
    for a in axes:a.grid(alpha=.25)
    fig.suptitle('Drone flight path: ground truth');fig.tight_layout();fig.savefig(out/'01_flight_trajectory.png',dpi=150);plt.close(fig)
    make_plot_xyz(out/'02_ground_truth_velocity.png',t,vel,'m/s','Ground-truth velocities')
    paired_plot(out/'03_gyro_vs_truth.png',t2,imug[ix],gyro_ref,'IMU gyroscope vs. rotated ground-truth angular velocity','rad/s')
    paired_plot(out/'04_acceleration_vs_truth.png',imu_t[accel_ix],imua[accel_ix],accel_ref,'IMU accelerometer vs. estimated specific force','m/s²')

    if selected:
        fig,axs=plt.subplots(1,len(selected),figsize=(5*len(selected),4))
        axs=np.atleast_1d(axs)
        for ax,frame in zip(axs,selected):
            if frame['codec']!='zlib-rgb8':
                raise ValueError(f"Unexpected camera codec {frame['codec']}")
            arr=np.frombuffer(zlib.decompress(frame['rgb_blob']),dtype=np.uint8)
            expected=int(frame['width']*frame['height']*frame['channels'])
            if arr.size!=expected:raise ValueError(f'Camera data corrupt: expected {expected}, found {arr.size}')
            arr=arr.reshape(frame['height'],frame['width'],frame['channels'])
            ax.imshow(arr[:,:,:3]);ax.set_title(f't={frame["sim_time_s"]:.1f}s');ax.axis('off')
        fig.suptitle('Actual RGB frames: first / middle / last')
        fig.tight_layout();fig.savefig(out/'05_camera_examples.png',dpi=150);plt.close(fig)

    def vec(a):return ', '.join(f'{x:.5f}' for x in a)
    report=f'''# Iris sensor-vs-ground-truth report

Run folder: `{folder.name}`

## Run outcome

- Status: **{m["status"]}**
- Scene: `{m["env"]}`; flight target: `{m["distance"]}` meters
- Simulated duration: **{max_time:.3f} seconds**
- Final ground-truth XYZ: **({vec(xyz[-1])}) meters**
- Final 3D distance from target: **{dist[-1]:.5f} meters**
- Average absolute lateral error: **{np.mean(lateral):.5f} meters** (maximum {np.max(lateral):.5f})
- Average absolute altitude error: **{np.mean(alt):.5f} meters** (maximum {np.max(alt):.5f})
- Maximum speed: **{np.max(speed):.4f} meters/s**

## Recording and synchronization

- Ground-truth readings: {len(truth)} (approx. {f_truth:.2f} Hz)
- Valid IMU readings: {len(imu)} of {m['imu']}
- Matched IMU and ground-truth readings (exact physics step): {len(aligned)}
- Maximum simulated timestamp difference for matched rows: {np.max(np.abs(time_offset)):.9f} seconds
- Camera images: {len(cam_summary)} compressed RGB frames

## Sensor comparisons

- Gyroscope mean difference IMU minus ground-truth angular velocity rotated to body frame, X/Y/Z (rad/s): {vec(bg)}
- Gyroscope RMSE X/Y/Z (rad/s): {vec(rg)}
- Orientation median quaternion-angle difference (degrees): {np.nanmedian(quat_deg):.5f}
- Accelerometer mean difference from physics-derived specific force X/Y/Z (m/s²): {vec(ba)}
- Accelerometer RMSE X/Y/Z (m/s²): {vec(ra)}

## How to interpret carefully

1. IMU and physics truth are **not measurements of the same physical quantity** in every column. The IMU has acceleration and rotational velocity, but no global X/Y/Z position or global linear velocity. The camera provides images, not position numbers.
2. The physics angular velocity is *assumed* to be world-frame and transformed using the body quaternion. Validate this convention before interpreting gyroscope mismatch as sensor error.
3. The accelerometer reference is **estimated** by differentiating ground-truth velocity, subtracting gravity, and rotating to the body frame. This introduces numerical noise and may differ from the Isaac Sim IMU sensor's measurement convention or sensor-location effects. A hovering IMU may read about +9.81 m/s² on Z rather than 0.
4. No stochastic IMU noise was deliberately injected in the script. Both 'sensor' and 'ground truth' come from the same simulation: this is a consistency check, not evidence of real-world sensor accuracy.
5. RGB frames are checked by decoding a few examples. Visual review is still needed to confirm the camera faces forward and images are not black or blocked.
6. A successful flight **does not** show that the controller uses the sensors. The PD controller still uses PhysX ground truth for flight control.

## Generated plots

- `01_flight_trajectory.png`
- `02_ground_truth_velocity.png`
- `03_gyro_vs_truth.png`
- `04_acceleration_vs_truth.png`
- `05_camera_examples.png` (when RGB frames exist)
'''
    (out/'report.md').write_text(report,encoding='utf-8')
    (out/'summary_metrics.json').write_text(json.dumps({
        'run':folder.name,'status':m['status'],'duration_s':max_time,'final_xyz_m':xyz[-1].tolist(),
        'final_target_error_m':float(dist[-1]), 'mean_abs_lateral_error_m':float(np.mean(lateral)),
        'mean_abs_altitude_error_m':float(np.mean(alt)),
        'imu_count':len(imu),'truth_count':len(truth),'aligned_count':len(aligned),'camera_frames':len(cam_summary),
        'gyro_bias_imu_minus_truth_rad_s':bg.tolist(),'gyro_rmse_rad_s':rg.tolist(),
        'accel_bias_imu_minus_approx_truth_mps2':ba.tolist(),'accel_rmse_mps2':ra.tolist(),
        'orientation_median_angle_deg':float(np.nanmedian(quat_deg))
    },indent=2),encoding='utf-8')
    print(f'\n[DONE] Saved plots, report.md, and summary_metrics.json under {out}')

if __name__=='__main__':
    main()
