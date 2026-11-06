"""Multi-episode validation helpers, invoked through test_lee_hover.py --validate."""

import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from IsaacLab_Drone_Project.controllers.cascaded_pid import CascadedPID

THRESHOLDS = {
    "min_altitude_m": 0.5,
    "max_tilt_deg": 45.0,
    "max_error_m": 1.0,
    "tail_rmse_m": 0.05,
    "tail_max_speed_m_s": 0.1,
    "tail_max_tilt_deg": 10.0,
    "tail_max_yaw_error_deg": 5.0,
    "max_saturation_fraction": 0.01,
}


def summarize(rows, dt, expected_steps, failure):
    """Evaluate the actual sampled trajectory, including the last unreset state."""
    tail = [r for r in rows if r["time_s"] >= expected_steps * dt - 5.0]
    commands = [r for r in rows if "saturated" in r]
    result = {
        "failure": failure,
        "duration_observed_s": rows[-1]["time_s"],
        "position_rmse_m": math.sqrt(np.mean([r["error_m"] ** 2 for r in rows])),
        "tail_rmse_m": math.sqrt(np.mean([r["error_m"] ** 2 for r in tail])) if tail else None,
        "tail_max_speed_m_s": max((r["speed_m_s"] for r in tail), default=None),
        "tail_max_tilt_deg": max((r["tilt_deg"] for r in tail), default=None),
        "tail_max_yaw_error_deg": max((r["yaw_error_deg"] for r in tail), default=None),
        "min_altitude_m": min(r["position_w"][2] for r in rows),
        "max_tilt_deg": max(r["tilt_deg"] for r in rows),
        "max_error_m": max(r["error_m"] for r in rows),
        "saturation_fraction": sum(r["saturated"] for r in commands) / max(1, len(commands)),
        "last_position_w": rows[-1]["position_w"],
    }
    result["passed"] = (
        failure is None
        and len(commands) == expected_steps
        and result["min_altitude_m"] >= THRESHOLDS["min_altitude_m"]
        and result["max_tilt_deg"] <= THRESHOLDS["max_tilt_deg"]
        and result["max_error_m"] <= THRESHOLDS["max_error_m"]
        and result["tail_rmse_m"] <= THRESHOLDS["tail_rmse_m"]
        and result["tail_max_speed_m_s"] <= THRESHOLDS["tail_max_speed_m_s"]
        and result["tail_max_tilt_deg"] <= THRESHOLDS["tail_max_tilt_deg"]
        and result["tail_max_yaw_error_deg"] <= THRESHOLDS["tail_max_yaw_error_deg"]
        and result["saturation_fraction"] <= THRESHOLDS["max_saturation_fraction"]
    )
    return result


def snapshot(robot, target, time_s):
    pos = robot.data.root_pos_w.torch[0].tolist()
    vel = robot.data.root_lin_vel_w.torch[0].tolist()
    x, y, z, w = robot.data.root_quat_w.torch[0].tolist()
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return {
        "time_s": time_s,
        "position_w": pos,
        "velocity_w": vel,
        "quaternion_xyzw": [x, y, z, w],
        "angular_velocity_b": robot.data.root_ang_vel_b.torch[0].tolist(),
        "error_m": math.dist(pos, target[0, :3].tolist()),
        "speed_m_s": math.sqrt(sum(v * v for v in vel)),
        "tilt_deg": math.degrees(math.acos(max(-1, min(1, 1 - 2 * (x * x + y * y))))),
        "yaw_error_deg": abs(math.degrees(math.atan2(math.sin(yaw), math.cos(yaw)))),
    }


def initial_conditions(row, actuator):
    """Detach matched-run metadata from the trajectory that receives command fields."""
    return {
        "state": deepcopy(row),
        "tau_inc": actuator.tau_inc_s.tolist(),
        "tau_dec": actuator.tau_dec_s.tolist(),
        "thrust_const": actuator.thrust_const.tolist(),
    }


def run_suite(env, robot, lee, args):
    dt = env.unwrapped.step_dt
    steps = round(args.duration / dt)
    if not math.isclose(steps * dt, args.duration, abs_tol=1e-8):
        raise ValueError("duration must be an integer number of control steps")
    output = args.output_dir or Path("outputs") / datetime.now().strftime("validation_%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed suite: {output}")
    target = torch.tensor([[0.0, 0.0, 1.5, 0.0]], device=env.unwrapped.device)
    pid = CascadedPID(lee.mass, lee.gravity, dt)
    actuator = robot.actuators["thrusters"]
    mixer = robot.allocation_matrix[2:6]
    results, matched_starts = [], {}
    for name, controller in (("lee", lee), ("pid", pid)):
        if name not in args.validation_controllers:
            continue
        for seed in args.seeds:
            for scenario in ("calm", "position", "perturbed"):
                rng = np.random.default_rng(seed)
                offset, angles, velocity = np.zeros(3), np.zeros(3), np.zeros(6)
                if scenario != "calm":
                    offset = rng.uniform([-0.25, -0.25, -0.15], [0.25, 0.25, 0.15])
                if scenario == "perturbed":
                    angles = np.deg2rad(rng.uniform([-5, -5, -15], [5, 5, 15]))
                    velocity = rng.uniform([-0.2] * 3 + [-0.1] * 3, [0.2] * 3 + [0.1] * 3)
                reset_cfg = env.unwrapped.event_manager.get_term_cfg("reset_base")
                reset_cfg.params["pose_range"] = {
                    k: (float(v), float(v))
                    for k, v in zip(("x", "y", "z", "roll", "pitch", "yaw"), np.r_[offset, angles], strict=True)
                }
                reset_cfg.params["velocity_range"] = {
                    k: (float(v), float(v))
                    for k, v in zip(("x", "y", "z", "roll", "pitch", "yaw"), velocity, strict=True)
                }
                env.unwrapped.event_manager.set_term_cfg("reset_base", reset_cfg)
                env.reset(seed=seed)
                controller.reset()
                # Test-only already-spinning initialization is reapplied after EVERY reset.
                hover = controller.mass[:, None] * controller.gravity.norm(dim=1)[:, None] / 4
                actuator.curr_thrust.copy_(hover.expand_as(actuator.curr_thrust))
                robot.set_thrust_target(actuator.curr_thrust.clone())
                assert torch.allclose(actuator.curr_thrust.sum(-1), controller.mass * controller.gravity.norm(dim=1))
                if name == "pid":
                    assert not pid.velocity_integral.any() and not pid.rate_integral.any()
                    assert pid.previous_velocity is None and not pid.saturated.any()
                rows = [snapshot(robot, target, 0.0)]
                start = initial_conditions(rows[0], actuator)
                key = (seed, scenario)
                if name == "pid" and key not in matched_starts and args.reference_dir:
                    reference_path = args.reference_dir / f"lee_{scenario}_seed{seed}.jsonl"
                    with reference_path.open() as reference:
                        reference_metadata = json.loads(next(reference))["metadata"]
                    if (
                        reference_metadata["dt"] != dt
                        or reference_metadata["duration_s"] != args.duration
                        or reference_metadata["target"] != target.tolist()
                    ):
                        raise ValueError("Reference controller timing, duration or target differs from this run")
                    matched_starts[key] = reference_metadata["start"]
                if name == "lee":
                    matched_starts[key] = start
                elif key not in matched_starts:
                    raise ValueError("PID-only validation requires --reference-dir with matched Lee logs")
                elif start != matched_starts[key]:
                    (output / "start_mismatch.json").write_text(
                        json.dumps({"expected": matched_starts[key], "actual": start}, indent=2)
                    )
                    raise AssertionError(f"Unmatched initial state or motor parameters: {key}")
                metadata = {
                    "controller": name,
                    "seed": seed,
                    "scenario": scenario,
                    "target": target.tolist(),
                    "dt": dt,
                    "duration_s": args.duration,
                    "thresholds": THRESHOLDS,
                    "start": start,
                    "gains": controller.cfg.to_dict(),
                    "motor_start": "hover",
                    "initial_motor_thrust": actuator.curr_thrust.tolist(),
                }
                failure = None
                path = output / f"{name}_{scenario}_seed{seed}.jsonl"
                with path.open("w", encoding="utf-8") as stream:
                    stream.write(json.dumps({"metadata": metadata}, default=str) + "\n")
                    for step in range(steps):
                        if name == "pid":
                            wrench = pid.compute(
                                target,
                                robot.data.root_pos_w.torch,
                                robot.data.root_lin_vel_w.torch,
                                robot.data.root_quat_w.torch[:, [3, 0, 1, 2]],
                                robot.data.root_ang_vel_b.torch,
                            )
                        else:
                            wrench = lee.compute(target)
                        raw = torch.linalg.solve(mixer, wrench[0, 2:6])
                        if not torch.isfinite(raw).all():
                            failure = "nonfinite_command"
                            break
                        command = raw.clamp(0.0, 6.0)
                        if name == "pid":
                            pid.allocation_feedback(raw[None], command[None])
                        rows[-1].update(
                            {
                                "motor_raw": raw.tolist(),
                                "motor_command": command.tolist(),
                                "desired_wrench_b": wrench[0].tolist(),
                                "saturated": bool(((raw < 0) | (raw > 6)).any()),
                            }
                        )
                        _, _, terminated, truncated, _ = env.step(command[None])
                        rows[-1]["motor_applied_last_substep"] = robot.data.applied_thrust[0].tolist()
                        stream.write(json.dumps(rows[-1]) + "\n")
                        if bool((terminated | truncated).any()):
                            failure = "unexpected_env_termination_or_reset"
                            break  # Never score auto-reset state as the terminal state.
                        row = snapshot(robot, target, (step + 1) * dt)
                        rows.append(row)
                        if not all(
                            math.isfinite(v) for v in row["position_w"] + row["velocity_w"] + row["quaternion_xyzw"]
                        ):
                            failure = "nonfinite_state"
                        elif row["position_w"][2] < THRESHOLDS["min_altitude_m"]:
                            failure = "low_altitude"
                        elif row["tilt_deg"] > THRESHOLDS["max_tilt_deg"]:
                            failure = "excessive_tilt"
                        elif row["error_m"] > THRESHOLDS["max_error_m"]:
                            failure = "excessive_position_error"
                        if failure:
                            break
                    if "saturated" not in rows[-1]:
                        stream.write(json.dumps(rows[-1]) + "\n")
                result = {"controller": name, "seed": seed, "scenario": scenario, **summarize(rows, dt, steps, failure)}
                results.append(result)
                (output / "summary.json").write_text(
                    json.dumps({"thresholds": THRESHOLDS, "results": results}, indent=2)
                )
                print(
                    f"VALIDATION {name} {scenario} seed={seed}: {'PASS' if result['passed'] else 'FAIL'} "
                    f"RMSE={result['position_rmse_m']:.4f}m tail={result['tail_rmse_m']} "
                    f"max_tilt={result['max_tilt_deg']:.2f} failure={failure}",
                    flush=True,
                )
    print(
        f"Validation: {sum(r['passed'] for r in results)}/{len(results)} passed. Results: {output.resolve()}",
        flush=True,
    )
