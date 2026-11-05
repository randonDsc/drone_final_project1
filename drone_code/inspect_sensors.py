"""Look at what the indoor sensors really report, in Isaac Sim.

    C:\\isaacsim\\python.bat drone_code\\inspect_sensors.py

Takes no arguments: settings are in run_config.INSPECT_SENSORS.

Loads the indoor scene with the warehouse active and lets the unpowered drone sit
on its pad. Prints IMU, rangefinder, and LiDAR readings, including the returns on
each LiDAR channel, and saves one scan under runs/sensor_inspection/.

This shows raw measurements only. No estimator or controller runs, and no drone
pose is read: nothing here produces a position.
"""
from pathlib import Path

from isaacsim import SimulationApp

import run_config


run = run_config.INSPECT_SENSORS

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": not run["gui"]})


def main():
    import numpy as np

    import hover_ppo_config
    import sensor_config

    from drone_sim import DroneSim
    from indoor_sensors import (
        LidarScanReader,
        RangefinderReader,
        channel_counts,
    )

    project_dir = Path(__file__).resolve().parents[1]

    sim = DroneSim(
        simulation_app,
        project_dir / hover_ppo_config.SCENES["indoor"],
    )
    sim.connect_imu(sensor_config.IMU_PATH)

    ranger = RangefinderReader(
        sim.stage,
        sensor_config.RANGEFINDER,
        np.random.default_rng(0),
    )
    lidar = LidarScanReader(sim.stage, sensor_config.LIDAR)

    sim.world.play()

    # RTX sensors only advance on rendered frames.
    render_every = 4
    steps = int(run["seconds"] / sim.dt)

    imu_samples = 0
    ranges = []
    misses = 0
    scans = []

    try:
        for step in range(steps):
            if not simulation_app.is_running():
                break

            sim.world.step(render=step % render_every == 0)

            reading = sim.read_imu()

            if reading is not None:
                imu_samples += 1
                gyro, accel = reading

            if ranger.model.next_time <= sim.world.current_time + 1e-9:
                measured = ranger.read(sim.world.current_time)

                if measured is None:
                    misses += 1
                else:
                    ranges.append(measured)

            scan = lidar.read()

            if scan is not None:
                scans.append(scan)

        print("\n========== Sensor readings ==========")
        print(f"Simulated {steps * sim.dt:.2f} s in {steps} physics steps")

        print(f"\nIMU: {imu_samples} valid samples")

        if imu_samples:
            print(f"  gyro  [rad/s]  {np.round(gyro, 4)}")
            print(f"  accel [m/s^2]  {np.round(accel, 3)} (+g on z at rest)")

        print(
            f"\nRangefinder: {len(ranges)} returns, {misses} with no return, "
            f"target {sensor_config.RANGEFINDER['update_rate_hz']} Hz"
        )

        if ranges:
            print(
                f"  range [m]  min {min(ranges):.3f}, max {max(ranges):.3f}, "
                f"mean {np.mean(ranges):.3f}"
            )

        print(f"\nLiDAR: {len(scans)} new sweeps (10 Hz expected)")

        if scans:
            scan = scans[-1]
            distance = np.linalg.norm(scan.points, axis=1)

            print(f"  last sweep: {len(scan)} points")
            print(f"  duration [s]  {scan.point_times.max():.3f}")
            print(
                f"  range [m]     min {distance.min():.2f}, "
                f"max {distance.max():.2f}"
            )
            print("  returns per channel, elevation degrees: count")

            for channel, count in channel_counts(scan).items():
                print(f"    {channel:4d}: {count}")

            output = project_dir / "runs" / "sensor_inspection"
            output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output / "last_scan.npz",
                points=scan.points,
                point_times=scan.point_times,
                intensity=scan.intensity,
                channel_elevation=scan.channel_elevation,
            )
            print(f"  saved {output / 'last_scan.npz'}")
        else:
            print(
                "  No sweeps. RTX sensors need rendering, and the XT32 is "
                "downloaded from an NVIDIA URL, so check the network too."
            )

        print("=====================================", flush=True)

    finally:
        lidar.close()
        sim.close()


try:
    main()
finally:
    simulation_app.close()
