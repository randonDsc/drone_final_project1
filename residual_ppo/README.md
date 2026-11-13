# Residual PPO for Quadrotor Path Following

## Objective
Train a PPO-based residual controller for wind-disturbed
quadrotor trajectory tracking.

## Baseline
Existing PID + SAC implementation on the main branch.

## Architecture
- Baseline controller: PID
- Learning algorithm: PPO
- Observation/action space: Match Path SAC
- Simulation: NVIDIA Isaac Sim
- Evaluation: PID vs PID+SAC vs PID+PPO

## Status
Project structure created. Training not yet implemented.
