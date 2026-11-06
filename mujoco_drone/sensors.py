"""Ideal simulated IMU and optional synchronized RGB/depth capture."""
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


def read_imu(sim):
    return {"gyro_body_rad_s": sim.data.sensor("body_gyro").data.copy(),
            "specific_force_sensor_m_s2": sim.data.sensor("body_linacc").data.copy()}


class CameraCapture:
    def __init__(self, sim, config, output):
        self.sim, self.config = sim, config
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.renderer = None
        self.index_file = None
        try:
            camera_id = sim.model.camera(config.camera).id
            sim.model.vis.global_.offwidth = max(config.width, sim.model.vis.global_.offwidth)
            sim.model.vis.global_.offheight = max(config.height, sim.model.vis.global_.offheight)
            self.renderer = mujoco.Renderer(sim.model, height=config.height, width=config.width)
            self.index_file = (self.output/"frames.csv").open("w", newline="")
            self.index = csv.DictWriter(self.index_file, ["time_s", "file"])
            self.index.writeheader()
            self.frame = 0
            self.next_time = 0.0
            fovy = float(sim.model.cam_fovy[camera_id])
            focal = .5*config.height/np.tan(np.deg2rad(fovy)/2)
            metadata = dict(camera=config.camera, width=config.width, height=config.height,
                            fovy_degrees=fovy, fx_px=focal, fy_px=focal,
                            cx_px=(config.width-1)/2, cy_px=(config.height-1)/2,
                            mount_position_body_m=sim.model.cam_pos[camera_id].tolist(),
                            mount_quaternion_wxyz=sim.model.cam_quat[camera_id].tolist(),
                            depth="MuJoCo metric optical-axis depth in metres, not Euclidean range",
                            rgb="uint8 sRGB renderer output; ideal simulated camera",
                            clip_near_m=float(sim.model.vis.map.znear*sim.model.stat.extent),
                            clip_far_m=float(sim.model.vis.map.zfar*sim.model.stat.extent))
            (self.output/"camera.json").write_text(json.dumps(metadata, indent=2)+"\n")
        except Exception:
            self.close()
            raise

    def capture(self):
        time = float(self.sim.data.time)
        if time+1e-9 < self.next_time:
            return
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.sim.data, camera=self.config.camera)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.sim.data, camera=self.config.camera)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        if not np.all(np.isfinite(depth)):
            raise FloatingPointError("Non-finite camera depth")
        name = f"frame_{self.frame:06d}.npz"
        np.savez_compressed(self.output/name, time_s=time, rgb=rgb, depth_m=depth)
        self.index.writerow(dict(time_s=time, file=name))
        self.frame += 1
        self.next_time = self.frame/self.config.camera_hz

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.index_file is not None:
            self.index_file.close()
            self.index_file = None
