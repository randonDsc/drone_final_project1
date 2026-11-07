# ARL Robot 1 — Isaac Lab Drone Experiments

**DSC 180A — Group D17, Group A**

## Overview

My main goal is to test basic drone hovering and navigation, collect sensor data, and study how external disturbances affect flight performance.

The experiments include sensor integration, constant-force disturbance testing, and point-to-point navigation using the Lee Position Controller.

## Completed Work

### 1. Basic Hovering
- Ran ARL Robot 1 in Isaac Lab.
- Used the Lee Position Controller.
- Set the target position to (0, 0, 1) meters.

### 2. Sensor Integration
- Added RGB and Depth cameras.
- Added a 16-channel LiDAR sensor.
- Added an IMU to collect linear acceleration and angular velocity.
- Saved sensor data at multiple frames, using a 100-frame interval.

### 3. Wind Disturbance Experiment
- Applied a simplified constant horizontal external force in the world X direction.
- Tested two conditions: 0 N and 0.1 N.
- Recorded drone position at Frames 100, 200, 300, and 400.
- Saved the results as CSV files.

### 4. Normal Navigation
- Implemented point-to-point navigation using the Lee Position Controller.
- Tested navigation distances of 2 m, 10 m, and 50 m.
- Used smooth interpolation to gradually change the target position.
- Maintained a target altitude of 1 m during navigation.
- Verified the final positions and velocities through simulation output.

### 5. Navigation CSV Data Logging
- Added CSV logging to record simulation time, target position, actual position, and velocity.
- Recorded data every 100 simulation steps.
- Added an optional `--save` argument to enable data logging only when needed.
- Opened and checked the saved CSV file in Excel.
- Compared the target and actual positions to observe tracking error.

## Code Files

| File | Description |
|---|---|
| `arl_robot_1.py` | Basic hovering using Lee Position Controller |
| `arl_robot_1_camera.py` | RGB, Depth, and LiDAR integration |
| `arl_robot_1_imu.py` | IMU integration and multi-frame sensor data collection |
| `arl_robot_1_wind.py` | Constant-force disturbance experiment with CSV logging |
| `arl_robot_1_navigation.py` | Point-to-point navigation with smooth trajectory and optional CSV logging |

## Experiment Results

The 0.1 N horizontal force caused a larger displacement in the X direction compared with the 0 N condition.

At Frame 400, the recorded X position was approximately 0.00012 m without the force and 0.02716 m with the force, giving a difference of about 2.7 cm.

The drone's height stayed close to the 1 m target in both conditions.

These are preliminary results from two runs, not a statistical evaluation across repeated trials.

### Navigation Results

The drone successfully reached target positions of 2 m, 10 m, and 50 m along the X direction while maintaining an altitude close to 1 m.

In the 50 m experiment, the final X position was approximately 50.00002 m, showing a very small final position error.

During navigation, the actual position sometimes lagged behind the moving target. This tracking error can be analyzed using the saved CSV data.

The uploaded `navigation_50m.csv` contains a partial flight record from a test run rather than a complete 50 m trajectory.

These experiments were performed in a simple environment without obstacles or wind disturbances during navigation.

## Project Structure

- `code/` — Python experiment files
- `result/sensor_samples/` — Sample RGB, Depth, LiDAR, and IMU data
- `result/wind_experiment/` — CSV results for 0 N and 0.1 N
- `result/navigation_50m.csv` — Partial 50 m navigation data
- `docs/` — Additional documentation

## Environment and Notes

The experiments were developed using Isaac Lab with the PhysX backend.

The scripts depend on the Isaac Lab installation, its ARL Robot 1 asset, and the Lee Position Controller. They are not standalone Python programs.

For sensor data saving, use the `--save` option with the relevant scripts.

The disturbance is modeled as a constant horizontal force, not a full aerodynamic wind model.

## Next Steps

1. Test longer navigation distances, including 100 m.
2. Combine normal navigation with external wind disturbances.
3. Collect position, velocity, and sensor data during navigation.
4. Compare navigation performance with and without disturbances.
5. Evaluate tracking error, trajectory deviation, and flight stability.
6. Study SAC and reward functions for future reinforcement learning experiments.
