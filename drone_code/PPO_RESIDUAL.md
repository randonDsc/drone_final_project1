# PPO residual hover control

The controller is now PID; see `CONTROL.md`. SAC is the main residual method; see
`SAC_RESIDUAL.md`. Measurements below, and the design notes on limits, predate PID.

    IMU -> estimator -> PID command + PPO correction -> motor mixer -> Isaac Sim

PPO adds bounded corrections to the PID thrust and torques so the drone holds a
fixed hover point in variable wind. One drone, one environment, standalone
Isaac Sim, Gymnasium, and Stable-Baselines3. No Isaac Lab.

## Status of the IMU conversion

**Attitude rate and short-term attitude work from the IMU. Position hold does
not, and cannot, from an IMU alone.** The controller and policy no longer see
simulator state, but nothing tells them where the drone really is.

Estimated and measured in a NumPy rigid-body model with the real `controller.py`
and wind.py's drag, 12 seeds, 10 s, closed loop on the estimate. Not Isaac:

| Case | True error at end (median) | Estimate reports | Past 2 m limit |
| --- | --- | --- | --- |
| Noise-free IMU, calm | 0.00 m | 0.00 m | 0/2 |
| Configured noise, calm | 1.39 m | 0.05 m | 3/12 |
| Configured noise, 2 m/s wind | 2.98 m | 0.06 m | 9/12 |
| Configured noise, 10 m/s wind | 58.9 m | 0.05 m | 12/12 |
| Noise, 2 m/s wind, 30 s | 67.6 m | 0.48 m | 6/6 |

Neither filter gain nor the velocity leak changes this much. Removing either bias
alone does not fix it either: steady wind is the largest term. Isaac's real IMU is
likely worse, since logged flight shows about +-0.95 m/s^2 of horizontal scatter;
0.4 m/s^2 of added vibration raised the 2 m/s case to 12.2 m.

Why (full version in `state_estimation.py`):

- Wind carries the drone. Once it moves with the air there is no force left to
  measure, so an IMU cannot tell drifting from hovering.
- Thrust acts along body z, so a tilted, accelerating drone still reads "up".
  Only drag reveals tilt, and it is tiny at low airspeed.
- An accelerometer bias looks exactly like a small tilt.
- Yaw has no reference at all, and the start offset is never corrected.

Needed for real position hold, in order of value:

1. Horizontal position or velocity: optical flow, GNSS, motion capture, or LiDAR
   odometry from the HESAI already on the airframe.
2. Height: rangefinder or barometer.
3. Heading: magnetometer, or yaw from the sensor in item 1.

`INDOOR_SENSORS.md` has the chosen setup and its remaining work. The estimator is
the place to add them: each is a correction to `position`,
`velocity`, or `orientation` after the integration step.

Consequences today: most `weak` wind episodes will likely hit the 2 m failure
limit through drift nothing can observe. The reward still measures true position,
so it will punish what the policy cannot see. Treat IMU-only PPO results as a test
of the pipeline and of short-term disturbance rejection, not as position control.

## Where each input comes from

| Controller / policy input | Source with `state_source = "imu"` |
| --- | --- |
| Body rate | Gyro, minus estimated bias |
| Roll, pitch | Estimator: gyro integration, corrected by accelerometer |
| Yaw | Estimator: gyro integration only, starts at 0 |
| Linear velocity | Estimator: integrated acceleration, gravity removed |
| Position | Estimator: integrated velocity, starts at the commanded target |
| Previous action | The policy's own output |
| PID command in the observation | PID run on the same estimate |
| Target | `TARGET_POSITION`, compared against the estimate |

Isaac's IMU `orientation` output is simulator ground truth and is never read.

Ground truth stays in these places only, all outside the control path:

| Use | Why |
| --- | --- |
| Wind drag and all physics | The simulated world itself |
| Reward | Training signal; not available in flight, not an input |
| Failure limits | Ends the simulated episode |
| RMSE, max error, estimate error | Scoring |
| `teleport` at reset | Places the drone; the sampled start is hidden from the estimator |
| Smoke test comparisons | Diagnostics |
| Mass, motor constants, IMU mounting, COM | Fixed calibration, read once at startup |

`state_source = "ground_truth"` (in `watch_config.py` or `run_config.py`) is
the explicit old baseline, and the viewer prints it in capitals.
`tests/test_hover_task.py` feeds corrupted truth and requires identical commands
and observations. `motor_physics.py`, the navigation demo, still flies on ground
truth: it steers through obstacles, where this drift would mean collisions.

## Saved models

The observation is still 23 values in the same order, so SB3 would load `hover1`
without complaint, and it would be wrong: the position, velocity, attitude, rate,
and PID slots now hold estimates, and its normalization statistics describe
different signals. **Retraining is required.** `train_ppo_hover.py` writes
`observation_contract.json` in the run folder, and every script refuses a model
whose contract does not match. Models without one, such as `hover1`, are refused too:
they learned against the old PD controller, so PID needs new models.

## Install

Nothing to install. Isaac Sim 6.1's Python (3.12) already has everything:
gymnasium 1.4.0, stable-baselines3 2.9.0, torch 2.11.0, tensorboard 2.21.0.
To confirm:

    C:\isaacsim\python.bat -c "import gymnasium, stable_baselines3, torch, tensorboard; print(gymnasium.__version__, stable_baselines3.__version__, torch.__version__)"

If a package is ever missing, install it into Isaac's Python, not the system one:

    C:\isaacsim\python.bat -m pip install "stable-baselines3==2.9.0" "gymnasium==1.4.0" tensorboard

## Run

From `C:\drone_final_project1`. Run scripts one at a time.

    :: Unit tests, no SimulationApp (seconds). Run from drone_code.
    cd drone_code
    C:\isaacsim\python.bat -m unittest discover -s tests -p "test_*.py"
    cd ..

    :: Isaac smoke test (about 2 minutes). Do this before any long run.
    C:\isaacsim\python.bat drone_code\smoke_ppo_hover.py

    :: Train or resume, from run_config.TRAIN_PPO. 30 steps = 1 simulated second.
    C:\isaacsim\python.bat drone_code\train_ppo_hover.py

    :: Compare PID-only with PID+PPO on the fixed seeds, from run_config.EVAL_PPO.
    :: pid_only = True flies the baseline alone, with no model.
    C:\isaacsim\python.bat drone_code\eval_ppo_hover.py

    :: Watch the drone fly in the GUI in real time. No training, no arguments:
    :: edit drone_code\watch_config.py, then run this. Close the window to quit.
    C:\isaacsim\python.bat drone_code\watch_ppo_hover.py

    :: Learning curves, in a second terminal.
    C:\isaacsim\python.bat -m tensorboard.main --logdir runs\ppo_hover

    :: Navigation demo, unchanged.
    C:\isaacsim\python.bat drone_code\motor_physics.py

These scripts take no arguments: edit `drone_code\run_config.py` first. Set
`gui = True` there to show the window; it is much slower.
When resuming, `total_timesteps` is the total to reach, not extra steps.
Ctrl+C during training saves `interrupted_model.zip`, which `resume` accepts.
Each training launch writes `logs/progress_<time>.csv` and `logs/log_<time>.txt` in
the run folder, with losses and metrics per update. Evaluation writes
`_episodes.csv` and `_summary.csv` beside its JSON.
Expect about 0.2 million steps per hour.

## Files

| File | Purpose |
| --- | --- |
| `drone_sim.py` | Backend moved out of `main()`: scene, World, views, motor mixer, COM compensation, wind wrench, forces, midair reset |
| `state_estimation.py` | IMU noise model and estimator. NumPy only |
| `sensor_config.py` | IMU noise and estimator settings, shared by all scripts |
| `run_config.py` | What the train, eval, smoke, and sensor scripts do when run |
| `model_contract.py` | Records and checks what a model's observations mean |
| `indoor_sensors.py` | Reads LiDAR and rangefinder; see `INDOOR_SENSORS.md` |
| `check_sensor_assets.py` | Static USD checks, no Isaac Sim |
| `inspect_sensors.py` | Prints real sensor output in Isaac Sim |
| `hover_task.py` | Observations, reward, failure, episode sampling. NumPy only |
| `hover_env.py` | Gymnasium environment |
| `ppo_config.py` | Reward weights, action limits, timing, failure limits, wind, PPO |
| `train_ppo_hover.py` | Training with checkpoints and resume |
| `eval_ppo_hover.py` | PID-only against PID+PPO on matching seeds |
| `watch_ppo_hover.py` | Real-time GUI viewer, with or without PPO |
| `watch_config.py` | Viewer settings: PPO on/off, model, target, wind |
| `smoke_ppo_hover.py` | Isaac Sim validation |
| `tests/test_hover_task.py` | Task logic against a stand-in simulator |

## What changed in existing files

- `motor_physics.py` now uses `DroneSim`; the navigation loop is the same.
- Motor parameters and rotor positions moved to `sim_config.py`, values
  unchanged. Physics parameters and controller gains are unchanged.
- `wind.py` gained `reset_episode()`. `reset()` behaves as before.
- `drone_diagnostics.py`: COM now comes from one view of all five bodies, and
  `cross()` replaces `np.cross`. This matches the per-body formula to 1e-6 m
  at the state tested and about doubles the step rate.

## Design choices

**Scene.** The demo scene loads once with the warehouse and launch pad disabled
in memory only, leaving the same drone and physics scene over open ground. Sensors
are off. Each episode starts airborne near (0, 0, 2) m.

**Reset.** `DroneSim.teleport` overwrites root pose and velocity, rotor angles,
rotor velocities, and spin targets without stopping physics. The whole
articulation goes through the `Articulation` view because the rotors are links
of it. Forces last one physics step in this PhysX API (measured), so they cannot
carry across an episode. `smoke_ppo_hover.py` confirms a violent episode leaves
the next one identical.

**Timing.** PID, mixer, motors, and wind run at 120 Hz. PPO picks an action every
4 steps (30 Hz); the residual is held between decisions.

**Residual limits.** ±1.5 N thrust, ±0.04 N m roll and pitch, ±0.02 N m yaw. These
come from what the PD loops need rather than motor authority, which is far larger:
PD has no integral term, so wind leaves a steady offset, and holding a lean against
attitude gain kp = 0.2 takes kp × angle. ±0.04 N m buys 11 degrees. All four at full
scale use about 0.6 N of a rotor's 3.1 N headroom, so at hover the residual alone
cannot saturate a rotor. The mixer's clipping and saturation flag are unchanged;
the rate is logged per step.

**Failure penalty.** With only costs, ending the episode looks attractive because
it stops future costs. Inside the failure limits every cost term is bounded, so a
flight costs at most c per decision and its discounted future cost is below
c / (1 - gamma). Failure charges exactly that amount, about 1980 with the current
settings, so crashing cannot beat continuing from any surviving state. It uses the
infinite sum rather than the steps left because PPO bootstraps the value at the
time limit. Rewards are not normalized by VecNormalize, which would clip this
penalty; a fixed `scale` of 0.1 keeps value targets moderate instead. Time limits
are truncation, with no penalty.

**Wind.** Each episode draws mean speed, direction, gust strength, and gust seed
from the env's seeded generator, in the `weak` profile by default. The policy never
sees wind. Gust noise does not depend on the drone's motion, so a seed gives the
same wind realization to both controllers, and `eval_ppo_hover.py` verifies the
match on every seed. Evaluation seeds are fixed in `ppo_config.EVAL_SEEDS`.

**Normalization.** Observations are scaled by fixed physical constants, then by
VecNormalize. Its statistics are saved with every checkpoint; evaluation loads them
frozen, and `resume` continues updating them.

## Known limits

- `hover1` was trained against the old PD controller and can no longer be loaded. In
  `weak` wind it gave a mean RMSE of 0.060 m
  against PD's 0.131 m, better on 20 of 20 seeds, with no failures or saturation.
  `moderate` wind has not been evaluated. None of this carries over to IMU control.
- The IMU path has not been run in Isaac Sim. Mount handling, reset spikes, and
  IMU behavior after GUI Stop are checked first by `smoke_ppo_hover.py`.
- The refactored demo differed from the original by up to 4.6 cm over a 15 s
  windy headless run. The original's run-to-run noise has not been measured, so
  whether this is noise or a real change is unknown.

- `isaacsim.core.api` and `isaacsim.core.prims` are deprecated since Isaac Sim 6.0.
  The demo already used them, so the backend stays on them.
- Rotors never reach `PROPELLER_SPIN_SCALE × motor speed`. `iris.usd` caps link
  angular speed at 100 rad/s, so they settle near 111 rad/s and their drives sit at
  the torque limit. This adds a 0.03-0.05 rad/s body-rate oscillation at hover, in
  the demo too. Training keeps it, so the policy trains on the demo's dynamics.
- The residual is trained for hover. It has not been tried inside the navigation
  demo.
