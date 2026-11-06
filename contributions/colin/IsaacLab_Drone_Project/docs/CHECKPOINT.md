# Colin's independent Isaac Lab checkpoint

## Contribution

Developed and evaluated conventional stabilization baselines for an ARL quadrotor in Isaac Lab: cascaded PID, a Lee-controller runner, motor-allocation diagnostics, quaternion-order correction, repeated-reset validation, and reproducible measurement summaries. This is a separate runnable contribution, not an integration of the group's drones or simulators.

## Evidence

Two controllers x three seeds x three starting conditions x 20 seconds = **18 episodes, 360 simulated seconds**. All passed the provisional calm-flight criteria with zero motor-command clipping.

| Worst value over nine episodes per controller | Lee | PID |
|---|---:|---:|
| Final-five-second position RMSE | 0.0000836 m | 0.0017665 m |
| Maximum full-episode tilt | 5.759 degrees | 6.734 degrees |
| Final-five-second maximum speed | 0.0001567 m/s | 0.0002579 m/s |
| Final-five-second maximum yaw error | 0.0000966 degrees | 0.01421 degrees |

These are ideal simulation measurements, not hardware precision claims. Scenarios use random position error, up to 5-degree roll/pitch and 15-degree yaw error, and small linear/angular velocities. Control is 50 Hz; physics is 100 Hz. See [results JSON](results/calm_hover_validation.json) and [methodology](BASELINE_VALIDATION.md).

Eight targeted unit tests passed. A harness-only reference aliasing bug was corrected: command logging mutated the in-memory initial-state reference. PID was then tested against the unmodified saved Lee metadata, with exact initial-state and motor-parameter matching. The nine Lee and nine PID results are combined in the results JSON.

## Limits

- No RL training occurred during validation. Earlier direct PPO is preliminary; residual PPO is planned.
- Wind, sensor-noise robustness, obstacle avoidance and hardware performance are not established.
- Motors start at hover thrust after every explicit diagnostic reset. Automatic training reset behavior still needs work.
- Ground contact is not directly measured; altitude/attitude/position guards reject major excursions.
- Three seeds and modest disturbances are not broad robustness certification.

## Group connection and next steps

The teammate's simulator contains another PID, residual PPO/SAC, wind, sensor readers and navigation. Integration needs explicit frame, quaternion, control-rate, motor-model and action adapters; see [review](TEAMMATE_REPOSITORY_REVIEW.md).

Next: training-safe automatic resets, bounded residual PPO with zero-residual equivalence checks, separate training/evaluation trials, then shared wind scenarios and estimated state.
