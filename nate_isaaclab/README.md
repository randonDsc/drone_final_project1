# ARL Robot 1 — Isaac Lab Drone Experiments

**DSC 180A — Group D17, Group A**

## Overview

This project contains my drone simulation experiments using Isaac Lab, ARL Robot 1, and the Lee Position Controller.

My main goal was to test basic hovering, collect sensor data, and study how a constant horizontal external force affects the drone's position.

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

## Code Files

| File | Description |
|---|---|
| `arl_robot_1.py` | Basic hovering using Lee Position Controller |
| `arl_robot_1_camera.py` | RGB, Depth, and LiDAR integration |
| `arl_robot_1_imu.py` | IMU integration and multi-frame sensor data collection |
| `arl_robot_1_wind.py` | Constant-force disturbance experiment with CSV logging |

## Experiment Results

The 0.1 N horizontal force caused a larger displacement in the X direction compared with the 0 N condition.

At Frame 400, the recorded X position was approximately 0.00012 m without the force and 0.02716 m with the force, giving a difference of about 2.7 cm.

The drone's height stayed close to the 1 m target in both conditions.

These are preliminary results from two runs, not a statistical evaluation across repeated trials.

## Project Structure

- `code/` — Python experiment files
- `result/sensor_samples/` — Sample RGB, Depth, LiDAR, and IMU data
- `result/wind_experiment/` — CSV results for 0 N and 0.1 N
- `docs/` — Additional documentation

## Environment and Notes

The experiments were developed using Isaac Lab with the PhysX backend.

The scripts depend on the Isaac Lab installation, its ARL Robot 1 asset, and the Lee Position Controller. They are not standalone Python programs.

For sensor data saving, use the `--save` option with the relevant scripts.

The disturbance is modeled as a constant horizontal force, not a full aerodynamic wind model.

## Next Steps

1. Implement normal navigation between two positions.
2. Test navigation with and without external disturbances.
3. Collect sensor and position data during navigation.
4. Evaluate position error, trajectory deviation, and flight stability.
5. Study SAC and reward functions for future reinforcement learning experiments.