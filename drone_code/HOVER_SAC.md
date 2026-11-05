# SAC wind correction

SAC learns a bounded correction on top of PID, to hold a hover point in wind:

    final command = PID command + SAC correction

**None of this has been run.** Rewards, limits, stages, and gains are starting values.
It uses simulator state only, with no IMU or LiDAR. See `CONTROL.md` for the controller.

## Commands

The scripts take no arguments. Edit `run_config.py`, then run from
`/c/drone_final_project1`:

    # Train or resume, from run_config.TRAIN_HOVER_SAC.
    /c/isaacsim/python.bat drone_code/train_hover_sac.py

    # Compare PID-only with PID + SAC on matching seeds, from run_config.EVAL_HOVER_SAC.
    /c/isaacsim/python.bat drone_code/eval_hover_sac.py

    # Navigation, with the mode from sim_config.CONTROL_MODE.
    /c/isaacsim/python.bat drone_code/motor_physics.py

| Goal | Setting in `run_config.py` |
| --- | --- |
| Start a new run | `TRAIN_HOVER_SAC`: `resume = None`, a new `run_name` |
| Resume | `TRAIN_HOVER_SAC`: `resume` = newest model; `total_timesteps` is the total to reach |
| Quick pipeline check | `TRAIN_HOVER_SAC`: `resume = None`, `total_timesteps = 5000`, `run_name = "check"` |
| Baseline with no model | `EVAL_HOVER_SAC`: `pid_only = True` |
| Choose the model or wind | `EVAL_HOVER_SAC`: `model`, `wind_stage` |
| Show the window | `gui = True` |

## Action

Four normalized values in [-1, 1], multiplied by `RESIDUAL_LIMITS` and added to the PID
command once, before the mixer. The mixer's rotor limits still apply.

| Action | At +-1 |
| --- | --- |
| Thrust correction | +-1.5 N |
| Roll torque correction | +-0.04 N m |
| Pitch torque correction | +-0.04 N m |
| Yaw torque correction | +-0.02 N m |

PID runs at 120 Hz. SAC decides at `RL_RATE_HZ` (10 Hz) and the correction is held for
12 physics steps. The rate must divide 120 exactly.

## Observation

One frame is 20 values. The observation is the last 5 frames, 100 values, oldest first,
one per decision.

| Values | Meaning | Frame | Scale |
| --- | --- | --- | --- |
| 3 | Target minus position | World | 1 m |
| 3 | Linear velocity | World | 1 m/s |
| 3 | Attitude error from level at target yaw, as a rotation vector | Body | 0.5 rad |
| 3 | Angular velocity | Body | 2 rad/s |
| 1 | PID thrust minus hover thrust | Body z | 5 N |
| 3 | PID torques | Body | 0.1 N m |
| 4 | Previous normalized action | n/a | 1 |

Values are divided by their scales and clipped to +-10. There is no learned normalization:
SAC replays old data, and fixed scales keep it valid. The rotation vector has no wrap at
+-180 degrees. The first frame of an episode fills the history, and nothing comes from
after the decision. Wind is never observed.

## Reward

`reward = -0.1 * (sum of weighted costs + failure penalty)`, from `hover_sac_config.REWARD`:

| Term | Weight | Cost |
| --- | --- | --- |
| `position` | 1.0 | Squared position error |
| `position_linear` | 0.5 | Position error, which still pulls when small |
| `speed` | 0.1 | Squared speed |
| `attitude` | 0.2 | Squared roll and pitch over (20 degrees)^2 |
| `angular_rate` | 0.02 | Squared angular speed |
| `action` | 0.01 | Squared normalized action |
| `action_rate` | 0.05 | Squared change in action |
| `saturation` | 0.5 | Share of the decision's physics steps saturated |
| `persistent_saturation` | 2.0 | Applied after 5 fully saturated decisions in a row |

Failure charges the bound on all future cost, as in the PPO task, so crashing never
beats continuing. Crossing a `FAILURE` limit is termination. The 20 s time limit is
truncation with no penalty, and SB3's replay buffer keeps them apart. There is no contact
sensor: in the open hover scene, `min_height` stands in for hitting the ground.

## Wind

Drag comes from the existing model in `wind.py`, applied at every physics step through
`DroneSim.step_command`. Each episode draws from the current stage, chosen by total
training steps, and the mean direction may turn slowly:

| Stage | Until step | No wind | Mean speed | Gust scale | Turn rate |
| --- | --- | --- | --- | --- | --- |
| `light` | 100,000 | 50% | 0 to 2 m/s | 0 to 0.25 | 0 |
| `steady` | 300,000 | 20% | 0 to 4 m/s | 0 to 0.5 | Up to 0.05 rad/s |
| `gusty` | 600,000 | 10% | 1 to 6 m/s | 0.25 to 0.75 | Up to 0.1 rad/s |
| `strong` | Then on | 10% | 2 to 8 m/s | 0.5 to 1 | Up to 0.2 rad/s |

Gust scale multiplies `BASE_WIND`: 1 m/s per horizontal axis. A given seed gives the same
wind and start state for both modes, and evaluation checks that.

## Checkpoints

Saved under `runs/sac_hover/<run_name>/`: every 100,000 steps in `checkpoints/`, at the
end as `final_*`, and on Ctrl+C as `interrupted_*`.

| Restored on resume | Source |
| --- | --- |
| Actor, critics, target critics | Model `.zip` |
| Actor, critic, and entropy optimizers; entropy coefficient | Model `.zip` |
| Step count, update count, wind stage | Model `.zip` |
| Replay buffer | Separate `.pkl` beside it |
| **Not restored** | Random generator state, the unfinished episode |

Each buffer can reach about 160 MB, so 1 million steps can use about 1.6 GB. A missing
or incompatible checkpoint, or a missing buffer, stops with a clear error before Isaac
starts. `allow_empty_replay_buffer` continues deliberately with an empty one. `resume`
continues an existing run; a run folder that already has a final model refuses a fresh
start.

`observation_contract.json` records the rates, scales, limits, history, and PID gains. If
any of these change, the model is refused. PPO models cannot load into SAC, and the two
corrections are never combined.

## Evaluation and logs

`eval_hover_sac.py` flies PID-only and PID + SAC on `EVAL_SEEDS`. It reports position RMSE,
RMSE over the second half of the episode, which shows steady error once the integrators
have settled, maximum error, failure rate, residual size, and saturation. Results are in
`eval_<stage>.json`, `_episodes.csv`, and `_summary.csv`.

Training writes `logs/progress_<time>.csv`, `logs/log_<time>.txt`, and TensorBoard files.

## Limits

- Hover only. In flight, position error is limited to 1 m for the policy, but moving
  waypoints are a separate evaluation that has not been designed or run.
- The first 2,000 steps use random actions, as SAC does, across the full correction range.
- There may be little left to correct: the integrator already cancels steady wind, so
  the gain may show mostly in gusts. Only evaluation can say.
