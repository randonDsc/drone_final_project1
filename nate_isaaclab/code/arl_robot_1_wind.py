# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to view ARL Robot 1.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    uv run python scripts/demos/arl_robot_1.py

    # Usage with Newton visualizer and default PhysX physics.
    uv run python scripts/demos/arl_robot_1.py --visualizer newton

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="View ARL Robot 1 with Lee Position Controller.",
    conflict_handler="resolve",
)
parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx"], help="Physics backend.")
parser.add_argument("--save", action="store_true", help="Save RGB, Depth, and LiDAR sensor data.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import torch
import types
from isaaclab.utils.math import quat_rotate_inverse

import isaaclab.sim as sim_utils

##
# Pre-defined configs
##
from isaaclab.physics import PhysicsCfg

from isaaclab_contrib.controllers.lee_position_control import LeePosController
from isaaclab_contrib.controllers.lee_position_control_cfg import LeePosControllerCfg

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG


def main():
    """Main function to spawn arl_robot_1."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # Create simulation context
        from isaaclab.sensors.camera import Camera, CameraCfg
        from isaaclab.sensors import Imu, ImuCfg
        from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
        import os
        import numpy as np
        import warp as wp
        import omni.kit.app
        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.replicator.core", True)
        import omni.replicator.core as rep
        from isaaclab.utils import convert_dict_to_backend
        # Set up camera output folder
        output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "drone_camera")
        # Folder for wind experiment results
        wind_output_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "output",
            "wind_experiment",
        )
        os.makedirs(wind_output_dir, exist_ok=True)
        import csv
        wind_records = []
        last_recorded_frame = -1


        rep_writer = rep.BasicWriter(
            output_dir=output_dir,
            frame_padding=0,
        )
        # ThrusterCfg is implemented in Isaac Lab and has no Newton-native execution path.
        sim_cfg = sim_utils.SimulationCfg(
            dt=0.01,
            device=args_cli.device,
            physics=physics_cfg,
            use_newton_actuators=False,
        )
        sim = sim_utils.SimulationContext(sim_cfg)

        # Create a dome light with light blue color
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.53, 0.81, 0.92))
        light_cfg.func("/World/DomeLight", light_cfg)

        # Spawn ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
        # Static mesh for LiDAR scanning
        lidar_ground_cfg = sim_utils.MeshCuboidCfg(
            size=(20.0, 20.0, 0.1),
        )
        lidar_ground_cfg.func(
            "/World/LidarGround",
            lidar_ground_cfg,
            translation=(0.0, 0.0, -0.05),
        )

        # Spawn robot
        robot_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Robot")
        robot_cfg.actuators["thrusters"].dt = sim_cfg.dt
        robot = robot_cfg.class_type(robot_cfg)
        # LiDAR sensor attached to the drone
        sim_utils.create_prim(
            "/World/Robot/base_link/LidarSensor",
            "Xform",
        )
        lidar_cfg = RayCasterCfg(
            prim_path="/World/Robot/base_link/LidarSensor",
            mesh_prim_paths=["/World/LidarGround/geometry/mesh"],
            pattern_cfg=patterns.LidarPatternCfg(
                channels=16,
                vertical_fov_range=(-15.0, 15.0),
                horizontal_fov_range=(-180.0, 180.0),
                horizontal_res=1.0,
            ),
            ray_alignment="base",
            max_distance=50.0,
            debug_vis=False,
        )
        lidar = RayCaster(cfg=lidar_cfg)
                # IMU attached to drone body
        imu_cfg = ImuCfg(
            prim_path="/World/Robot/base_link",
            update_period=0.0,
            debug_vis=False,
        )
        imu = Imu(cfg=imu_cfg)

        # Camera attached to the drone body
        camera_cfg = CameraCfg(
            prim_path="/World/Robot/base_link/CameraSensor",
            update_period=0,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            offset=CameraCfg.OffsetCfg(
                pos=(0.3, 0.0, 0.0),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 100.0),
            ),
        )
        camera = Camera(cfg=camera_cfg)

        # Play the simulator
        sim.reset()

        # Create Lee position controller
        controller_cfg = LeePosControllerCfg(
            K_pos_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
            K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
            max_inclination_angle_rad=1.0471975511965976,
            max_yaw_rate=1.0471975511965976,
        )
        controller = LeePosController(controller_cfg, robot, num_envs=1, device=str(sim.device))
        original_combine_thrusts = robot._combine_thrusts
        def combine_thrusts_with_wind(self):
            original_combine_thrusts()
            body_quat = self.data.root_quat_w.torch
            wind_body = quat_rotate_inverse(body_quat, wind_force[:, 0, :])
            self._internal_force_target_sim[:, 0, :] += wind_body
        robot._combine_thrusts = types.MethodType(combine_thrusts_with_wind, robot)

        # Get allocation matrix and compute pseudoinverse
        allocation_matrix = torch.tensor(robot_cfg.allocation_matrix, device=sim.device, dtype=torch.float32)
        # allocation_matrix is (6, num_thrusters), we need pseudoinverse for wrench -> thrust
        alloc_pinv = torch.linalg.pinv(allocation_matrix)  # Shape: (num_thrusters, 6)

        # Position command: hover in place (zero position, zero yaw)
        pos_command = torch.zeros((1, 4), device=sim.device)  # [x, y, z, yaw]
        pos_command[0, 2] = 1.0  # Hover at 1 meter height


        # Simple wind force in world X direction
        wind_force = torch.zeros((1, 5, 3), device=sim.device, dtype=torch.float32)
        wind_force[0, 0, 0] = 0.0
        # No extra rotation from wind
        wind_torque = torch.zeros_like(wind_force)
        # Simulation loop
        print("[INFO] Starting demo with Lee Position Controller. Press Ctrl+C to stop.")

        # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
        while sim.is_headless_or_exist_active_visualizer():
            # Compute wrench from position controller
            wrench = controller.compute(pos_command)  # Shape: (1, 6)

            # Allocate wrench to thrusters: thrust = pinv(A) @ wrench
            thrust_cmd = torch.matmul(wrench, alloc_pinv.T)  # Shape: (1, num_thrusters)
            thrust_cmd = thrust_cmd.clamp(min=0.0)  # Ensure non-negative thrust

            # Apply thrust
            robot.set_thrust_target(thrust_cmd)



            # Step simulation
            robot.write_data_to_sim()
            # Apply wind force directly through PhysX
            # robot.root_view.apply_forces_and_torques_at_position(
            #     force_data=wp.from_torch(wind_force.reshape(-1, 3).contiguous(), dtype=wp.float32),
            #     torque_data=wp.from_torch(wind_torque.reshape(-1, 3).contiguous(), dtype=wp.float32),
            #     position_data=None,
            #     indices=robot._ALL_INDICES,
            #     is_global=True,
            # )
            sim.step()

            # Update robot
            robot.update(sim_cfg.dt)
            # Update camera sensor
            camera.update(sim_cfg.dt)
            # Update LiDAR sensor
            lidar.update(sim_cfg.dt)
            # Update IMU sensor
            imu.update(sim_cfg.dt)
            print("IMU Acceleration:", wp.to_torch(imu.data.lin_acc_b).cpu().numpy())
            print("IMU Angular Velocity:", wp.to_torch(imu.data.ang_vel_b).cpu().numpy())
            print("LiDAR hits shape:", lidar.data.ray_hits_w.shape)
            print("LiDAR sample:", lidar.data.ray_hits_w[0, :10])
            print("RGB shape:", camera.data.output["rgb"].shape)
            print("Camera frame:", camera.frame[0])

            if not hasattr(main, "last_saved_frame"):
                main.last_saved_frame = -1
            if args_cli.save and camera.frame[0].item() > 0 and camera.frame[0].item() % 100 == 0 and camera.frame[0].item() != main.last_saved_frame and "rgb" in camera.data.output and "distance_to_image_plane" in camera.data.output:

                single_cam_data = convert_dict_to_backend(
                    {k: v[0] for k, v in camera.data.output.items()},
                    backend="numpy",
                )

                rep_output = {"annotators": {}}
                for key, data in single_cam_data.items():
                    info = camera.data.info.get(key)
                    if info is not None:
                        rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                    else:
                        rep_output["annotators"][key] = {"render_product": {"data": data}}

                rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}
                rep_writer.write(rep_output)
                print("[SUCCESS] Camera frame saved:", camera.frame[0])
                # Save LiDAR scan
                lidar_points = wp.to_torch(lidar.data.ray_hits_w)[0].cpu().numpy()
                np.save(os.path.join(output_dir, f"lidar_{camera.frame[0].item()}.npy"), lidar_points)
                print("[SUCCESS] LiDAR scan saved:", lidar_points.shape)
                # Save IMU acceleration
                imu_acc = wp.to_torch(imu.data.lin_acc_b)[0].cpu().numpy()
                np.save(os.path.join(output_dir, f"imu_acc_{camera.frame[0].item()}.npy"), imu_acc)
                # Save IMU angular velocity
                imu_ang = wp.to_torch(imu.data.ang_vel_b)[0].cpu().numpy()
                np.save(os.path.join(output_dir, f"imu_ang_{camera.frame[0].item()}.npy"), imu_ang)
                main.last_saved_frame = camera.frame[0].item()
            import warp as wp
            print("Drone Position:", wp.to_torch(robot.data.root_pos_w).cpu().numpy())
            print("Drone Velocity:", wp.to_torch(robot.data.root_lin_vel_w).cpu().numpy())

            # Record drone position every 100 camera frames
            frame = camera.frame[0].item()

            if frame > 0 and frame % 100 == 0 and frame != last_recorded_frame:
                position = wp.to_torch(robot.data.root_pos_w)[0].cpu().numpy()

                wind_records.append({
                    "frame": frame,
                    "wind_force_x": float(wind_force[0, 0, 0].item()),
                    "x": float(position[0]),
                    "y": float(position[1]),
                    "z": float(position[2]),
                })

                last_recorded_frame = frame

                csv_path = os.path.join(
                    wind_output_dir,
                    f"wind_{wind_force[0, 0, 0].item():.1f}N.csv"
                )

                with open(csv_path, "w", newline="") as file:
                    writer = csv.DictWriter(
                        file,
                        fieldnames=["frame", "wind_force_x", "x", "y", "z"]
                    )
                    writer.writeheader()
                    writer.writerows(wind_records)

                print("[SUCCESS] Wind data saved:", frame)

if __name__ == "__main__":
    main()
