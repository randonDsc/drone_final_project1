
"""Inspect the ARL drone's mass and estimated hovering thrust."""

import argparse

import gymnasium as gym
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.cli import load_external_tasks
from isaaclab_tasks.utils import resolve_task_config


def main():
    parser = argparse.ArgumentParser()
    add_launcher_args(parser)
    parser.set_defaults(
        device=None,
        visualizer=["kit"],
    )
    args = parser.parse_args()

    load_external_tasks()

    task = "MyDrone-TrackPosition-v0"
    env_cfg, _ = resolve_task_config(task, "")
    env_cfg.scene.num_envs = 1
    args.device = env_cfg.sim.device

    with launch_simulation(env_cfg, args):
        env = gym.make(task, cfg=env_cfg)
        env.reset()

        robot = env.unwrapped.scene["robot"]

        # Read physical body masses from PhysX
        masses = robot.root_physx_view.get_masses()
        masses = torch.as_tensor(masses)

        total_mass = masses[0].sum().item()

        gravity = 9.81
        total_hover_thrust = total_mass * gravity
        per_motor = total_hover_thrust / 4.0

        print("\n========== DRONE HOVER DATA ==========")
        print(f"Total mass: {total_mass:.4f} kg")
        print(f"Gravity: {gravity:.2f} m/s^2")
        print(f"Required total thrust: {total_hover_thrust:.4f} N")
        print(f"Estimated thrust per motor: {per_motor:.4f} N")
        print("======================================\n")

        env.close()


if __name__ == "__main__":
    main()
