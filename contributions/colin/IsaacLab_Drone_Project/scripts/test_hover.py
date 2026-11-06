
"""Test altitude PD control without PPO."""

import math

import argparse

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.cli import load_external_tasks
from isaaclab_tasks.utils import resolve_task_config

from IsaacLab_Drone_Project.controllers.hover_controller import (
    HoverController,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=["kit"])
    args = parser.parse_args()

    load_external_tasks()

    task = "MyDrone-TrackPosition-v0"
    env_cfg, _ = resolve_task_config(task, "")
    env_cfg.scene.num_envs = 1
    args.device = env_cfg.sim.device

    # Override action processing for this test only
    action_cfg = env_cfg.actions.thrust_command
    action_cfg.scale = 3.0
    action_cfg.offset = 3.0411
    action_cfg.use_default_offset = False

    # Disable sensor recording for this diagnostic test
    if hasattr(env_cfg.events, "record_sensors"):
        env_cfg.events.record_sensors = None

    controller = HoverController()

    with launch_simulation(env_cfg, args):
        env = gym.make(task, cfg=env_cfg)
        env.reset()

        robot = env.unwrapped.scene["robot"]

        print("\n=== DRONE MOTOR CONFIGURATION ===")
        print("Motor order:", robot.thruster_names)
        print("Allocation matrix:")
        print(robot.allocation_matrix)
        print("=================================\n")

        # Hold the drone at its initial altitude
        target_z = robot.data.root_pos_w.torch[0, 2].item()

        print(f"Target altitude: {target_z:.3f} m")

        for step in range(args.steps):
            current_z = robot.data.root_pos_w.torch[0, 2].item()
            velocity_z = robot.data.root_lin_vel_w.torch[0, 2].item()

            thrusts = controller.compute(
                current_z=current_z,
                target_z=target_z,
                velocity_z=velocity_z,
            )

            # Convert desired thrust into raw policy actions
            actions = torch.tensor(
                [(t - action_cfg.offset) / action_cfg.scale
                 for t in thrusts],
                dtype=torch.float32,
                device=env.unwrapped.device,
            ).unsqueeze(0)

            _, _, terminated, truncated, _ = env.step(actions)


            if step % 5 == 0:
                pos = robot.data.root_pos_w.torch[0].tolist()
                vel = robot.data.root_lin_vel_w.torch[0].tolist()
                quat = robot.data.root_quat_w.torch[0].tolist()

                # Quaternion uses (w, x, y, z)
                w, qx, qy, qz = quat

                # Angle between drone's vertical axis
                # and the world's vertical axis
                cos_tilt = 1.0 - 2.0 * (qx**2 + qy**2)
                cos_tilt = max(-1.0, min(1.0, cos_tilt))
                tilt_deg = math.degrees(math.acos(cos_tilt))

                applied = robot.data.applied_thrust
                if hasattr(applied, "torch"):
                    applied = applied.torch
                print(
                    "Individual motor thrusts:",
                    applied[0].detach().cpu().flatten().tolist()
                )

                print(
                    f"Step {step:03d} | "
                    f"X: {pos[0]:.3f} | "
                    f"Y: {pos[1]:.3f} | "
                    f"Z: {pos[2]:.3f} | "
                    f"Vz: {vel[2]:.3f} | "
                    f"Tilt: {tilt_deg:.1f} deg | "
                    f"Total thrust: {applied[0].sum().item():.3f} N"
                )


            if bool((terminated | truncated).any().item()):
                print("Episode ended; stopping test.")
                break

        env.close()


if __name__ == "__main__":
    main()
