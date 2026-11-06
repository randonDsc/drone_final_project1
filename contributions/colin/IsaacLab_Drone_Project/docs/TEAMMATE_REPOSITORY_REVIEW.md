# Relationship to drone_final_project1

Reviewed the user-provided local checkout originally at `C:/Users/colin/Desktop/DSC106/drone_final_project1`. This is a source inspection, not a reproduction of its training results or hardware validation. No teammate code was copied into the PID implementation or executed.

The user requested moving only that checkout. Its new location is `C:/Users/colin/Desktop/DSC180/drone_final_project1`; the .git directory and origin URL were preserved. This project's directory and Isaac Lab dependency paths are unchanged. Open the new location in your editor; if a Git desktop client remembers the old location, point it to this existing repository.

```powershell
cd C:\Users\colin\Desktop\DSC180\drone_final_project1
git status
git remote -v
Get-Content .\drone_code\CONTROL.md
```

## Architecture and useful connections

| Area | Teammate source | Relationship to this project |
|---|---|---|
| Classical control | `drone_code/controller.py` | NumPy near-hover PID outputs collective thrust and body torques. Uses position/height integrals and saturation-aware integration. A potential third controller behind a state/wrench adapter. |
| Physics | `drone_code/drone_sim.py`, `motor.py` | Direct Isaac Sim World/backend with Iris views, motor model, mixer and COM compensation. Our ARL multirotor uses Isaac Lab managers and a different actuator model. |
| Residual learning | `hover_env.py`, `sac_hover_env.py`, `train_ppo_hover.py`, `train_sac_hover.py` | PID plus learned bounded correction. Corrections are collective thrust/torques before the mixer, not four individual motor-thrust residuals. Uses Stable-Baselines3 tooling, unlike our RSL-RL PPO. |
| Wind | `wind.py` | Mean wind plus temporally smooth shared/local stochastic gusts; quadratic drag at distributed body points yields net force and torque. Strong candidate for a future shared disturbance module after an adapter and tests. |
| Estimation | `state_estimation.py` | IMU noise, complementary attitude estimation and strapdown integration. IMU-only position and yaw drift are explicitly documented. It is not a complete navigation estimator. |
| Additional sensors | `indoor_sensors.py` | LiDAR and downward range measurements. Source explicitly says they are not connected to estimator/controller/policy; LiDAR-inertial odometry is not implemented there. |
| Navigation | `mapping.py`, `navigation.py`, `flight_navigation.py` | Scene-geometry occupancy mapping, 3D Dijkstra and waypoint/trajectory following. Scene-derived maps are not equivalent to mapping from real sensor observations. |
| Model compatibility | `model_contract.py` | Saved observation/controller contracts prevent incompatible PPO/SAC models from silently loading. Useful pattern for our future residual experiments. |

## Important interface differences

- Teammate PID consumes WXYZ orientation and world angular velocity, converting rates to body coordinates internally. Our installed Isaac Lab exposes XYZW orientation and we use body angular velocity. An adapter must explicitly convert both.
- Their SAC configuration runs physics/PID at 120 Hz and the learned policy at 10 Hz. Our current diagnostic runs physics at 100 Hz and both classical controllers at 50 Hz. A comparison must document or match these rates.
- Their SAC residual scales are 1.5 N collective thrust and (0.04, 0.04, 0.02) N m roll/pitch/yaw torque. These cannot be interpreted as motor commands or used directly with our older PPO checkpoint.
- Wind's point offsets are body coordinates; velocities, forces and angular velocity inputs use world coordinates. Preserve force application frames and moment arms, and avoid applying the same aerodynamic torque twice.
- Mass, inertia, rotor geometry, motor dynamics, sensor noise and failure definitions differ. Comparing raw rewards across repositories would not establish controller quality.

## Suggested integration order

1. Keep the repositories runnable independently and define a shared state/target/wrench schema with units, frames, quaternion order and timestamps.
2. Compare Lee and our PID on the same ARL simulation. Optionally adapt their PID to that simulation as a separate named baseline.
3. Share scenario definitions and metric output (position RMSE, recovery time, clipping and physical failure), keeping true state for evaluation separate from controller observations.
4. Port and verify the wind model once calm baselines pass. Its OU gust/drag approximation is useful but not CFD or validated hardware aerodynamics.
5. Add residual RL with explicit observation/action contracts, then integrate sensor estimates and navigation.

For self-review, start with `drone_code/CONTROL.md`, `SAC_RESIDUAL.md`, `PPO_RESIDUAL.md`, `INDOOR_SENSORS.md` and `FLIGHT_SAC.md`; the root README contains only a title. Inspect controller.py and hover_env.py together to follow how PID and residual commands reach the mixer. Training scripts and model files alone do not prove measured performance.
