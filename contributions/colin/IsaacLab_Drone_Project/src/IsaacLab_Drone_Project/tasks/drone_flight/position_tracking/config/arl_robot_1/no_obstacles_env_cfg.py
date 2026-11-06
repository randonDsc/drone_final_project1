# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass, replace
from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg

from IsaacLab_Drone_Project.navigation import configure_target

from .position_tracking_env_cfg import TrackPositionNoObstaclesEnvCfg


@configclass
class NoObstacleEnvCfg(TrackPositionNoObstaclesEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Use the scene background instead of solid blue.
        if self.sim.default_visualizer_cfg is None:
            self.sim.default_visualizer_cfg = KitVisualizerCfg(
                background_color=None,
            )
        else:
            self.sim.default_visualizer_cfg.background_color = None

        for visualizer_cfg in self.sim.visualizer_cfgs or []:
            visualizer_cfg.background_color = None

        # Outside camera: close view toward the drone's starting position.
        self.sim.default_visualizer_cfg.eye = (2.0, 2.0, 2.2)
        self.sim.default_visualizer_cfg.lookat = (0.0, 0.0, 1.5)

        for visualizer_cfg in self.sim.visualizer_cfgs or []:
            visualizer_cfg.eye = (2.0, 2.0, 2.2)
            visualizer_cfg.lookat = (0.0, 0.0, 1.5)

        # Select the drone model.
        self.scene.robot = replace(
            ARL_ROBOT_1_CFG,
            prim_path="{ENV_REGEX_NS}/Robot",
        )
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt

        # Fixed starting position for inspecting the room.
        self.scene.robot.init_state.pos = (0.0, 0.0, 1.5)

        # Remove random position, rotation, and velocity on reset.
        self.events.reset_base.params["pose_range"] = {
            name: (0.0, 0.0)
            for name in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.events.reset_base.params["velocity_range"] = {
            name: (0.0, 0.0)
            for name in ("x", "y", "z", "roll", "pitch", "yaw")
        }

        # Configure the drone's target position.
        configure_target(self)

    def play_mode(self):
        super().play_mode()

        self.events.base_external_force_torque = None
        self.events.push_robot = None