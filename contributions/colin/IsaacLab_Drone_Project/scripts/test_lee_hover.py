"""Shared Lee/PID runner; scripts/run_hover.py is the friendly entry point."""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import torch
from isaaclab_contrib.controllers import (
    LeePosController,
    LeePosControllerCfg,
)

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.cli import load_external_tasks

from isaaclab_tasks.utils import resolve_task_config

from IsaacLab_Drone_Project.controllers.cascaded_pid import CascadedPID, CascadedPIDCfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--validate", action="store_true", help="Run matched multi-episode Lee/PID validation")
    parser.add_argument("--duration", type=float, default=20.0, help="Seconds per validation episode")
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--validation-controllers", nargs="+", choices=("lee", "pid"), default=("lee", "pid"))
    parser.add_argument(
        "--reference-dir", type=Path, default=None, help="Prior Lee suite for matched PID-only validation"
    )
    parser.add_argument("--controller", choices=("lee", "pid"), default="lee")
    parser.add_argument("--yaw-target", choices=("zero", "initial"), default="zero")
    parser.add_argument("--initial-yaw", type=float, default=None, help="Optional initial yaw in radians")
    parser.add_argument("--target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--decimation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--motor-start", choices=("default", "hover"), default="default")
    add_launcher_args(parser)
    parser.set_defaults(device=None, visualizer=["kit"])
    args = parser.parse_args()
    if args.steps <= 0 or args.decimation <= 0:
        parser.error("steps and decimation must be positive")
    if not math.isfinite(args.duration) or args.duration < 10:
        parser.error("validation duration must be finite and at least 10 seconds")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("validation seeds must be unique")

    load_external_tasks()

    task = "MyDrone-TrackPosition-v0"
    # resolve_task_config invokes Hydra, which otherwise reparses our CLI flags.
    sys.argv = [sys.argv[0]]
    env_cfg, _ = resolve_task_config(task, "")
    env_cfg.scene.num_envs = 1

    # Update motor commands every 2 physics steps
    env_cfg.decimation = args.decimation
    env_cfg.seed = args.seed
    if args.validate:
        # Keep auto-reset outside the measured window; failures are checked explicitly.
        env_cfg.episode_length_s = args.duration + 1.0
        env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 1.5)
        # Ground-truth control does not consume camera images; skip costly RTX capture.
        env_cfg.scene.front_camera = None
    if args.initial_yaw is not None and not args.validate:
        env_cfg.scene.robot.init_state.rot = (0.0, 0.0, math.sin(args.initial_yaw / 2), math.cos(args.initial_yaw / 2))

    args.device = env_cfg.sim.device

    print("Physics timestep:", env_cfg.sim.dt)
    print("Controller timestep:", env_cfg.sim.dt * env_cfg.decimation)

    # For this test, actions directly represent
    # requested motor thrusts in Newtons.
    action_cfg = env_cfg.actions.thrust_command
    action_cfg.scale = 1.0
    action_cfg.offset = 0.0
    action_cfg.use_default_offset = False

    # Disable unrelated recording and command visuals.
    if hasattr(env_cfg.events, "record_sensors"):
        env_cfg.events.record_sensors = None

    if hasattr(env_cfg.commands, "target_pose"):
        env_cfg.commands.target_pose.debug_vis = False

    # Fixed gains for a reproducible first test.
    controller_cfg = LeePosControllerCfg(
        K_pos_range=(
            (3.0, 3.0, 2.0),
            (3.0, 3.0, 2.0),
        ),
        K_vel_range=(
            (2.5, 2.5, 1.5),
            (2.5, 2.5, 1.5),
        ),
        K_rot_range=(
            (1.85, 1.85, 0.4),
            (1.85, 1.85, 0.4),
        ),
        K_angvel_range=(
            (0.5, 0.5, 0.09),
            (0.5, 0.5, 0.09),
        ),
        max_inclination_angle_rad=math.pi / 6,
        max_yaw_rate=math.pi / 3,
    )

    with launch_simulation(env_cfg, args):
        env = gym.make(task, cfg=env_cfg)
        env.reset()

        robot = env.unwrapped.scene["robot"]
        device = env.unwrapped.device

        expected_order = [
            "back_left_prop",
            "back_right_prop",
            "front_left_prop",
            "front_right_prop",
        ]
        assert list(robot.thruster_names) == expected_order

        controller = LeePosController(
            cfg=controller_cfg,
            asset=robot,
            num_envs=1,
            device=device,
        )
        controller.reset()
        if args.validate:
            from validate_hover import run_suite

            try:
                with torch.no_grad():
                    run_suite(env, robot, controller, args)
            finally:
                env.close()
            return
        if args.controller == "pid":
            controller_cfg = CascadedPIDCfg()
            controller = CascadedPID(
                controller.mass, controller.gravity, env_cfg.sim.dt * env_cfg.decimation, controller_cfg
            )
        actuator = robot.actuators["thrusters"]
        startup_thrust = actuator.curr_thrust[0].clone()
        print("Configured default rps:", robot.data.default_thruster_rps.tolist())
        print("Actuator captured initial rps:", actuator._init_thruster_rps.tolist())
        print("Thrust coefficients:", actuator.thrust_const.tolist())
        if args.motor_start == "hover":
            # Diagnostic initial condition: motors are already spinning at hover.
            # Do this after reset, using each motor's sampled thrust coefficient.
            hover_thrust = controller.mass[:, None] * controller.gravity.norm(dim=1)[:, None] / 4
            actuator.curr_thrust.copy_(hover_thrust.expand_as(actuator.curr_thrust))
            robot.set_thrust_target(actuator.curr_thrust.clone())
            print("Equivalent hover rps:", torch.sqrt(actuator.curr_thrust / actuator.thrust_const).tolist())
        print("Mass (kg):", controller.mass.tolist())
        print("Actuator limits (N):", actuator.cfg.thrust_range)
        print("Actuator dt:", actuator.cfg.dt)
        print("Initial motor thrust (N):", actuator.curr_thrust[0].tolist())
        print("Rise/fall tau (s):", actuator.tau_inc_s.tolist(), actuator.tau_dec_s.tolist())
        assert math.isclose(actuator.cfg.dt, env_cfg.sim.dt)
        log_path = args.log or Path("outputs") / (
            args.controller + datetime.now().strftime("_hover_%Y%m%d_%H%M%S_%f") + ".jsonl"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print("Diagnostics:", log_path.resolve())

        # Hold the initial X, Y, and Z position,
        # with a desired yaw of zero.
        initial_pos = robot.data.root_pos_w.torch[0].clone()
        target = torch.cat([initial_pos, initial_pos.new_zeros(1)]).unsqueeze(0)
        target[:, :3] += target.new_tensor(args.target_offset)
        if args.yaw_target == "initial":
            qx, qy, qz, qw = robot.data.root_quat_w.torch[0]
            target[:, 3] = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

        # Matrix rows: [Fz, Tx, Ty, Tz]
        # Columns: [BL, BR, FL, FR]
        mixer = robot.allocation_matrix[2:6, :]

        print(f"\n=== {args.controller.upper()} HOVER TEST ===")
        print("Target:", target[0].tolist())
        print("Motor order:", robot.thruster_names)
        print("======================\n")

        with torch.no_grad(), log_path.open("w", encoding="utf-8") as log:
            log.write(
                json.dumps(
                    {
                        "metadata": {
                            "physics_dt": env_cfg.sim.dt,
                            "decimation": args.decimation,
                            "seed": args.seed,
                            "controller": args.controller,
                            "quaternion_log_order": "wxyz",
                            "simulator_quaternion_order": "xyzw",
                            "yaw_target": args.yaw_target,
                            "initial_yaw_override": args.initial_yaw,
                            "target": target[0].tolist(),
                            "allocation_matrix": robot.allocation_matrix.tolist(),
                            "mass": controller.mass.tolist(),
                            "gains": controller_cfg.to_dict(),
                            "actuator_limits": actuator.cfg.thrust_range,
                            "initial_thrust": actuator.curr_thrust[0].tolist(),
                            "motor_start": args.motor_start,
                            "unmodified_startup_thrust": startup_thrust.tolist(),
                            "default_rps": robot.data.default_thruster_rps.tolist(),
                            "actuator_initial_rps": actuator._init_thruster_rps.tolist(),
                            "thrust_const": actuator.thrust_const.tolist(),
                            "tau_inc": actuator.tau_inc_s.tolist(),
                            "tau_dec": actuator.tau_dec_s.tolist(),
                        }
                    },
                    default=str,
                )
                + "\n"
            )
            for step in range(args.steps):
                # Compute required forces and torques.
                if args.controller == "pid":
                    wrench = controller.compute(
                        target,
                        robot.data.root_pos_w.torch,
                        robot.data.root_lin_vel_w.torch,
                        robot.data.root_quat_w.torch[:, [3, 0, 1, 2]],
                        robot.data.root_ang_vel_b.torch,
                    )
                else:
                    wrench = controller.compute(target)

                # Convert Fz, Tx, Ty, Tz into 4 motor thrusts.
                motor_thrusts = torch.linalg.solve(mixer, wrench[0, 2:6])

                # Stay within the configured motor command limits.
                raw_thrusts = motor_thrusts.clone()
                motor_thrusts = motor_thrusts.clamp(0.0, 6.0)
                if args.controller == "pid":
                    controller.allocation_feedback(raw_thrusts.unsqueeze(0), motor_thrusts.unsqueeze(0))
                # Capture pre-action state: env.step auto-resets on termination.
                record = {
                    "step": step,
                    "time_s": step * env_cfg.sim.dt * env_cfg.decimation,
                    "position_w": robot.data.root_pos_w.torch[0].tolist(),
                    "velocity_w": robot.data.root_lin_vel_w.torch[0].tolist(),
                    "quaternion_wxyz": robot.data.root_quat_w.torch[0, [3, 0, 1, 2]].tolist(),
                    "angular_velocity_b": robot.data.root_ang_vel_b.torch[0].tolist(),
                    "desired_wrench_b": wrench[0].tolist(),
                    "motor_raw": raw_thrusts.tolist(),
                    "motor_command": motor_thrusts.tolist(),
                    "command_wrench_b": (robot.allocation_matrix @ motor_thrusts).tolist(),
                    "motor_applied_before": robot.data.applied_thrust[0].tolist(),
                    "saturated": ((raw_thrusts < 0) | (raw_thrusts > 6)).tolist(),
                }

                # Since scale=1 and offset=0,
                # raw actions are thrust commands.
                actions = motor_thrusts.unsqueeze(0)

                _, _, terminated, truncated, _ = env.step(actions)
                record["terminated"] = terminated.tolist()
                record["truncated"] = truncated.tolist()
                record["motor_applied_last_substep"] = robot.data.applied_thrust[0].tolist()
                record["post_state_is_reset"] = bool((terminated | truncated).any().item())
                log.write(json.dumps(record) + "\n")
                log.flush()

                if step % 25 == 0:
                    pos = robot.data.root_pos_w.torch[0]
                    quat = robot.data.root_quat_w.torch[0]

                    x, y, z, w = quat.tolist()
                    cos_tilt = 1.0 - 2.0 * (x * x + y * y)
                    cos_tilt = max(-1.0, min(1.0, cos_tilt))
                    tilt = math.degrees(math.acos(cos_tilt))

                    print(
                        f"Step {step:03d} | "
                        f"X: {pos[0].item():.3f} | "
                        f"Y: {pos[1].item():.3f} | "
                        f"Z: {pos[2].item():.3f} | "
                        f"Tilt: {tilt:.1f} deg"
                    )

                    print(
                        "Motor targets:",
                        [round(v, 3) for v in motor_thrusts.tolist()],
                    )

                if bool((terminated | truncated).any().item()):
                    print(
                        "Episode ended:",
                        "terminated =",
                        terminated.tolist(),
                        "truncated =",
                        truncated.tolist(),
                    )
                    break

        env.close()


if __name__ == "__main__":
    main()
