
"""Record onboard drone camera and IMU data."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from datetime import datetime

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUTPUT = Path.cwd() / "outputs" / RUN_NAME
MAX_FRAMES = 100


def as_numpy(value):
    """Accept either an Isaac Lab array wrapper or a Torch tensor."""
    tensor = getattr(value, "torch", value)
    return tensor.detach().cpu().numpy()


def record_drone_state(env, frame):
    """Record environment 0 at the sensor capture interval."""
    robot = env.scene["robot"]
    thrusters = robot.actuators["thrusters"]

    row = {
        "frame": frame,
        "env_id": 0,
        "time_s": env.common_step_counter * env.step_dt,
    }

    def add_values(prefix, values, labels):
        values = as_numpy(values)[0].reshape(-1)
        for label, value in zip(labels, values, strict=True):
            row[f"{prefix}_{label}"] = float(value)

        # Ground-truth state. "_w" means world coordinates.
        add_values("position_m", robot.data.root_pos_w, ("x", "y", "z"))
        add_values("velocity_m_s", robot.data.root_lin_vel_w, ("x", "y", "z"))
        add_values(
            "angular_velocity_rad_s",
            robot.data.root_ang_vel_w,
            ("x", "y", "z"),
        )

        # Preserve the installed API's quaternion component order.
        add_values(
            "orientation_quat",
            robot.data.root_quat_w,
            ("component_0", "component_1", "component_2", "component_3"),
        )

        # Raw policy actions: these are not necessarily thrust in newtons.
        actions = as_numpy(env.action_manager.action)[0].reshape(-1)
        for index, value in enumerate(actions):
            row[f"policy_action_{index}"] = float(value)

        # Actual motor-model output, in the actuator's reported motor order.
        thrust = as_numpy(thrusters.applied_thrust)[0].reshape(-1)
        for motor, value in zip(thrusters.thruster_names, thrust, strict=True):
            row[f"applied_thrust_N_{motor}"] = float(value)

        row["sum_motor_thrust_N"] = float(thrust.sum())

        csv_path = OUTPUT / "drone_state.csv"
        write_header = not csv_path.exists()

        with csv_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

def capture_sensors(env, env_ids):
    """Called periodically by the Isaac Lab event manager."""
    frame = getattr(capture_sensors, "frame", 0)

    if frame >= MAX_FRAMES:
        return

    camera = env.scene["front_camera"]
    imu = env.scene["imu"]

    if not camera.data.output:
        return

    rgb = camera.data.output["rgb"].torch[0].cpu().numpy()
    depth = (
        camera.data.output["distance_to_image_plane"]
        .torch[0].cpu().numpy().squeeze()
    )

    acceleration = imu.data.lin_acc_b.torch[0].cpu().numpy()
    angular_velocity = imu.data.ang_vel_b.torch[0].cpu().numpy()

    for folder in ("rgb", "depth"):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)

    name = f"frame_{frame:04d}"

    record_drone_state(env, frame)

    # Save color image
    Image.fromarray(rgb[..., :3].astype(np.uint8)).save(
        OUTPUT / "rgb" / f"{name}.png"
    )

    # Save the original numerical depth measurements
    np.save(OUTPUT / "depth" / f"{name}.npy", depth)

    # Save IMU measurements
    csv_path = OUTPUT / "imu.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "time_s",
                "acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z",
            ])

        writer.writerow([
            env.common_step_counter * env.step_dt,
            *acceleration.tolist(),
            *angular_velocity.tolist(),
        ])

    capture_sensors.frame = frame + 1

    if frame == 0:
        print(f"[SENSOR] Saving this run to: {OUTPUT}")

    if frame % 10 == 0:
        print(f"[SENSOR] Captured frame {frame}")
