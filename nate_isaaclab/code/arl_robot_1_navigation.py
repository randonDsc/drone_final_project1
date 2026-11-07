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
import csv
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="View ARL Robot 1 with Lee Position Controller.",
    conflict_handler="resolve",
)
parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx"], help="Physics backend.")
parser.add_argument("--save", action="store_true", help="Save navigation data to CSV.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import torch

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

        # Spawn robot
        robot_cfg = ARL_ROBOT_1_CFG.replace(prim_path="/World/Robot")
        robot_cfg.actuators["thrusters"].dt = sim_cfg.dt
        robot = robot_cfg.class_type(robot_cfg)

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

        # Get allocation matrix and compute pseudoinverse
        allocation_matrix = torch.tensor(robot_cfg.allocation_matrix, device=sim.device, dtype=torch.float32)
        # allocation_matrix is (6, num_thrusters), we need pseudoinverse for wrench -> thrust
        alloc_pinv = torch.linalg.pinv(allocation_matrix)  # Shape: (num_thrusters, 6)

        # Position command: hover in place (zero position, zero yaw)
        pos_command = torch.zeros((1, 4), device=sim.device)  # [x, y, z, yaw]
        pos_command[0, 2] = 1.0  # Hover at 1 meter height
        # Navigation settings
        navigation_start_time = 3.0
        navigation_duration = 30.0
        navigation_distance = 50.0
        simulation_time = 0.0
        # CSV logging settings
        if args_cli.save:
            output_dir = os.path.join(os.path.dirname(__file__), "output", "navigation")
            os.makedirs(output_dir, exist_ok=True)

            csv_path = os.path.join(output_dir, "navigation_50m.csv")

            # Create CSV file and write column headers
            with open(csv_path, "w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    "time", "target_x",
                    "position_x", "position_y", "position_z",
                    "velocity_x", "velocity_y", "velocity_z"
                ])
        # Simulation loop
        print("[INFO] Starting demo with Lee Position Controller. Press Ctrl+C to stop.")

        # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
        while sim.is_headless_or_exist_active_visualizer():
            # Smoothly move the target from X=0 to X=2
            elapsed = simulation_time - navigation_start_time

            if elapsed <= 0:
                target_x = 0.0
            elif elapsed >= navigation_duration:
                target_x = navigation_distance
            else:
                progress = elapsed / navigation_duration
                smooth_progress = progress * progress * (3.0 - 2.0 * progress)
                target_x = navigation_distance * smooth_progress

            pos_command[0, 0] = target_x
            # Compute wrench from position controller
            wrench = controller.compute(pos_command)  # Shape: (1, 6)

            # Allocate wrench to thrusters: thrust = pinv(A) @ wrench
            thrust_cmd = torch.matmul(wrench, alloc_pinv.T)  # Shape: (1, num_thrusters)
            thrust_cmd = thrust_cmd.clamp(min=0.0)  # Ensure non-negative thrust

            # Apply thrust
            robot.set_thrust_target(thrust_cmd)

            # Step simulation
            robot.write_data_to_sim()
            sim.step()
            simulation_time += sim_cfg.dt

            # Update robot
            robot.update(sim_cfg.dt)
            import warp as wp
            # Record navigation data every 100 simulation steps
            if args_cli.save and round(simulation_time / sim_cfg.dt) % 100 == 0:
                position = wp.to_torch(robot.data.root_pos_w)[0].cpu().numpy()
                velocity = wp.to_torch(robot.data.root_lin_vel_w)[0].cpu().numpy()

                with open(csv_path, "a", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow([
                        simulation_time - sim_cfg.dt,
                        target_x,
                        position[0], position[1], position[2],
                        velocity[0], velocity[1], velocity[2]
                    ])

                print(f"[NAV] Time: {simulation_time:.1f}s | Target X: {target_x:.2f}m | Actual X: {position[0]:.2f}m")


if __name__ == "__main__":
    main()
