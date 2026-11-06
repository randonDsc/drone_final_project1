
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils

from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.assets import MultirotorCfg

import IsaacLab_Drone_Project.tasks.drone_flight.mdp as mdp

from IsaacLab_Drone_Project.sensor_recorder import capture_sensors

from IsaacLab_Drone_Project.tasks.drone_flight.mdp.commands import (
    DroneUniformPoseCommandCfg,
)

from IsaacLab_Drone_Project.tasks.drone_flight.mdp.rewards import (
    ang_vel_xyz_exp,
    distance_to_goal_exp,
    lin_vel_xyz_exp,
    yaw_aligned,
)


##
# Scene Definition
##


@configclass
class ArlTrackPositionStateBasedSceneCfg(InteractiveSceneCfg):
    """Scene containing the ARL drone and its sensors."""

    # Drone robot
    robot: MultirotorCfg = MISSING


    # Grid floor environment.
    floor_environment = AssetBaseCfg(
        prim_path="/World/FloorEnvironment",
        spawn=sim_utils.UsdFileCfg(
            usd_path=(
                "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
                "Assets/Isaac/6.1/Isaac/Environments/Grid/default_environment.usd"
            ),
        ),
    )

    # Environment lighting
        # Sunflower panorama and environment lighting.
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
                "Assets/Isaac/6.1/Isaac/Environments/Skies/Clear/sunflowers_4k.hdr"
            ),
        ),
    )

    # --------------------------------------
    # RGB + Depth Camera
    # --------------------------------------

    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_camera",
        update_period=0.1,
        height=240,
        width=320,
        data_types=[
            "rgb",
            "distance_to_image_plane",
        ],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.0),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    # --------------------------------------
    # IMU Sensor
    # --------------------------------------

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
    )

    # Paste camera_housing = AssetBaseCfg(...) here.
    # Paste imu_housing = AssetBaseCfg(...) here.
    # Blue camera housing centered at the camera's mounting position.
    camera_housing = None  # AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base_link/camera_housing",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.04, 0.05, 0.04),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.05, 0.25, 0.9),
    #         ),
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(0.15, 0.0, 0.0),
    #     ),
    # )

    # Yellow marker at the IMU's measurement origin.
    imu_housing = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/imu_housing",
        spawn=sim_utils.CuboidCfg(
            size=(0.025, 0.025, 0.012),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.8, 0.05),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )


##
# MDP Settings
##

##
# MDP Settings
##


@configclass
class CommandsCfg:
    """Target-position commands for the drone."""

    target_pose = DroneUniformPoseCommandCfg(
        asset_name="robot",
        body_name="base_link",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=DroneUniformPoseCommandCfg.Ranges(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Control commands for the four drone propellers."""

    thrust_command = mdp.ThrustActionCfg(
        asset_name="robot",
        scale=3.0,
        offset=3.5,
        preserve_order=False,
        use_default_offset=False,
        clip={
            "back_left_prop": (0.0, 6.0),
            "back_right_prop": (0.0, 6.0),
            "front_left_prop": (0.0, 6.0),
            "front_right_prop": (0.0, 6.0),
        },
    )


@configclass
class ObservationsCfg:
    """Information provided to the PPO policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Drone state observations."""

        base_link_position = ObsTerm(
            func=mdp.root_pos_w,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        base_orientation = ObsTerm(
            func=mdp.root_quat_w,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        last_action = ObsTerm(
            func=mdp.last_action,
            noise=Unoise(n_min=0.0, n_max=0.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


##
# Events
##


@configclass
class EventCfg:
    """Reset events and sensor recording."""

    # Reset drone position and velocity
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-1.0, 1.0),
                "yaw": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # --------------------------------------
    # Sensor Recording Event
    # --------------------------------------

    record_sensors = EventTerm(
        func=capture_sensors,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        is_global_time=True,
    )


##
# Rewards
##


@configclass
class RewardsCfg:
    """Rewards for reaching and maintaining target position."""

    distance_to_goal_exp = RewTerm(
        func=distance_to_goal_exp,
        weight=25.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "target_pose",
        },
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    yaw_aligned = RewTerm(
        func=yaw_aligned,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.0,
        },
    )

    lin_vel_xyz_exp = RewTerm(
        func=lin_vel_xyz_exp,
        weight=2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 2.0,
        },
    )

    ang_vel_xyz_exp = RewTerm(
        func=ang_vel_xyz_exp,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 10.0,
        },
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.05,
    )

    action_magnitude_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-0.05,
    )

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-5.0,
    )


##
# Termination Conditions
##


@configclass
class TerminationsCfg:
    """Conditions for ending a drone episode."""

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    crash = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": -3.0,
        },
    )


##
# Main Environment Configuration
##


@configclass
class TrackPositionNoObstaclesEnvCfg(ManagerBasedRLEnvCfg):
    """State-based drone position-control environment."""

    # Scene
    scene: ArlTrackPositionStateBasedSceneCfg = (
        ArlTrackPositionStateBasedSceneCfg(
            num_envs=4096,
            env_spacing=2.5,
        )
    )

    # Observations, actions, and commands
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    # Rewards and termination conditions
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Events, including sensor capture
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Initialize simulation settings."""

        # General settings
        self.decimation = 10
        self.episode_length_s = 5.0

        # Simulation timestep
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation

        # Physics material
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )

        # PhysX configuration
        self.sim.physics = PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15,
        )

        # Drone thrusters use Isaac Lab actuators
        self.sim.use_newton_actuators = False
