# SAC flight residual

A new SAC model learns a bounded correction that helps PID follow trajectories in wind:

    u_final = u_PID + delta_u_SAC

**None of this has been run.** Every weight, limit, and speed here is a starting value.
It is separate from the hover and PPO workflows and changes none of their files.

## Commands

The scripts take no arguments. Edit `flight_config.py`, then run from
`/c/drone_final_project1`:

    # Train, fresh or resuming, from FLIGHT_TRAIN.
    /c/isaacsim/python.bat drone_code/train_flight_sac.py

    # Compare PID-only with PID + SAC, from FLIGHT_EVAL.
    /c/isaacsim/python.bat drone_code/eval_flight_sac.py

    # Fly the Dijkstra route, from FLIGHT_NAV.
    /c/isaacsim/python.bat drone_code/navigate_flight_sac.py

    # Train and evaluate each reward-weight variant, from FLIGHT_SWEEP.
    /c/isaacsim/python.bat drone_code/sweep_flight_sac.py

    # Unit tests, no Isaac Sim. Run from drone_code.
    /c/isaacsim/python.bat -m unittest tests.test_flight_sac

| Goal | Setting |
| --- | --- |
| Start a new model | `FLIGHT_TRAIN`: `mode = "fresh"`, a new `run_name` |
| Resume | `FLIGHT_TRAIN`: `mode = "resume"`, `resume_model` = newest model |
| Baseline with no model | `FLIGHT_EVAL`: `pid_only = True` |
| PID-only or SAC navigation | `FLIGHT_NAV`: `mode = "pid"` or `"pid_sac"` |

No new packages are needed: stable-baselines3, gymnasium, and NumPy are already used.

## Layers

    TimedReference            p_ref, v_ref, a_ref                 120 Hz
    ReferencePID (adapter)    lead point for the position-only PID 120 Hz
    PIDController (unchanged) [thrust, roll, pitch, yaw torque]    120 Hz
    + held SAC correction     added once, before the mixer         10 Hz, held
    DroneSim.step_command     mixer, motor limits, wind, physics   120 Hz

`flight_policy.FlightCore` is this loop. Training, evaluation, and navigation all use it,
so they share references, observations, scaling, and timing. They also share
`flight_reference.TimedReference`.

## How a position-only PID flies a reference

`controller.py` accepts position targets only. Its velocity request is
`kp/kd * (target - p)`. `ReferencePID` commands a lead point:

    target_xy = p_ref + (kd/kp) v_ref + (1/kp) a_ref
    target_z  = p_ref + (kd/kp) v_ref + (m/kp) a_ref

Substituting that into the controller's own equations gives

    horizontal acceleration = 0.6 e_p + 1.2 e_v + a_ref
    vertical force          = m g + 10 e_z + 5 e_vz + m a_z,ref

So the PID really is commanded the `v_ref` the reward measures, with acceleration
feedforward, without changing it. Gains stay fixed. The caps in `controller.py` still
apply: 10 degrees of tilt and 1.5 m/s^2 of horizontal demand.

Two details keep this honest:

- **Integrators.** The lead would wind them up in the direction of travel. So each step
  calls `update(dt=0)` with the lead, which reads the command and changes nothing, then
  `update(dt)` with `p_ref` itself, whose output is discarded. The integrators see only
  true tracking error. The lead's acceleration cap is passed in as saturation, which also
  holds the height and yaw integrators for that step: a small, conservative side effect.
- **Desired attitude.** The PID does not expose it. Its torque is exactly
  `kp * error - kd * rate + integral`, so `desired_attitude` inverts that from the real
  command rather than copying the controller.

## Action and residual bounds

Four values in [-1, 1], multiplied by `RESIDUAL_LIMITS`:

| Action | At +-1 |
| --- | --- |
| Thrust correction | +-0.8 N |
| Roll torque correction | +-0.05 N m |
| Pitch torque correction | +-0.03 N m |
| Yaw torque correction | +-0.02 N m |

The reference method bounds each rotor by `0.02 * T_lim`, with `T_lim = 1.5 m g`: 0.48 N
here. That is a per-rotor limit, so it does not map directly onto these four axes. A
correction of `delta` gives rotors `inverse(A) * S * delta`, where `A` is the mixer's
allocation matrix and `S` adds the torque the mixer shifts for the COM offset. The worst
case over all sign combinations is the sum of absolute values, per rotor:

    0.8/4 + 0.05/(4 * 0.21) + 0.03/(4 * 0.13) + 0.02/(4 * 0.06)  =  about 0.40 N

The arms and yaw ratio there are rounded. `FlightCore` computes this exactly from the real
matrix, measured mass, and COM at startup, prints it, and refuses to run if a rotor could
exceed 0.48 N. Rotors have about 3.1 N of headroom at hover.

Two different limits apply:

- **Residual bound.** SAC's output range. It cannot be exceeded, and hitting it is not
  penalized beyond the action cost.
- **Motor saturation.** The mixer solves for the combined command and clips each rotor
  to 0..7.07 N, unchanged. A clipped wrench is not the requested one, the flag reaches PID
  anti-windup, and the reward charges it. The residual is not added at the motors, and
  motor limits are not raised.

## Observation

One frame is 32 values; the observation is the last 4 frames, 128 values, oldest first,
one per decision. The first frame of an episode fills the history. Scale is what the raw
value is divided by; then values are clipped to +-10. There are no learned statistics:
SAC replays old data, and fixed scales keep it valid.

| Index | Meaning | Frame | Scale |
| --- | --- | --- | --- |
| 0:3 | `p_ref - p` | World | 1 m |
| 3:6 | `v_ref - v` | World | 1 m/s |
| 6:9 | `v_ref` | World | 2 m/s |
| 9:12 | `a_ref` | World | 1 m/s^2 |
| 12:18 | First two columns of R, body to world | World | 1 |
| 18:21 | Attitude error from PID's desired attitude, rotation vector | Body | 0.35 rad |
| 21:24 | Angular velocity | Body | 2 rad/s |
| 24:25 | PID thrust minus hover thrust | Body z | 5 N |
| 25:28 | PID torques | Body | 0.1 N m |
| 28:32 | Previous normalized action | n/a | 1 |

Frames are sampled at 10 Hz. Neither representation of attitude wraps: R columns have no
jump, and the rotation vector goes the short way around. Ground truth only; wind is never
observed. The hover model's 1 m cap on position error is not used: the clip passes +-10 m.

## Reward

From the reference method:

    H_d(e) = 0.5 e^2 when |e| <= d, else d (|e| - 0.5 d)
    L_vec  = sum_i H(e_p_i) + H(e_speed) + H(e_dir)
    L_att  = sum_i H(e_att_i) + sum_i H(e_rate_i)
    r_tracking = -(lambda_v L_vec + lambda_a L_att),  lambda_v = lambda_a = 1

with `e_p = p_ref - p`, `e_speed = |norm(v_ref) - norm(v)|`, and
`e_dir = w_dir (1 - dot(unit(v_ref), unit(v)))`.

Choices made for this project, not from the method:

| Choice | Value |
| --- | --- |
| Each error is divided by a scale, then Huber with delta = 1 | Position 0.25 m, speed 0.5 m/s, direction 1, attitude 0.2 rad, rate 1 rad/s |
| `w_dir` | `1 * g(|v_ref|) * g(|v|)`, `g` = smoothstep of speed / 0.3 m/s |
| Unit vectors | `v / max(|v|, 1e-6)` |
| `e_att` | Rotation vector from PID's desired attitude to the attitude, body frame |
| `e_rate` | `omega_ref - omega`, body frame |
| `omega_ref` | Rate of change of PID's desired attitude between decisions, limited to +-1 rad/s per axis |

Scaling makes the terms dimensionless, so radians and meters are comparable. With
delta = 1, a term turns linear when its error reaches its scale.

**Flight behavior.** Attitude is compared with what PID wants, so intended tilt costs
nothing. `omega_ref` allows intended rotation; it is a small-angle approximation of Euler
rates. Nothing pulls speed toward zero: `e_speed` compares with `v_ref`. In hovering and
braking, `w_dir` fades out and `e_speed` becomes plain speed. No separate vector velocity
penalty is added, since `e_speed` and `e_dir` already cover it and it would double-count.

**Added costs**, each logged under `cost/`:

| Term | Weight | Cost |
| --- | --- | --- |
| `action` | 0.005 | Squared normalized action |
| `action_rate` | 0.02 | Squared change in action |
| `saturation` | 0.5 | Share of the decision's physics steps saturated |
| `persistent_saturation` | 2.0 | After 5 fully saturated decisions |
| `unsafe_tilt` | 1.0 | Squared excess over 20 degrees, relative to that limit |
| `unsafe_rate` | 0.5 | Squared excess over 3 rad/s, relative to that limit |
| `route_deviation` | 1.0 | Squared excess over 1 m, relative to that limit |

Tracking terms are computed at every physics step and averaged over the decision, so one
decision's size does not depend on the physics rate. Then:

    reward = -0.1 * min(sum of costs, 10)                 surviving decision
    failure adds -0.1 * 10 / (1 - 0.99)  =  -100

**Avoiding cost by ending early.** Reward is only ever negative, so ending must not pay.
The cap bounds any surviving flight's discounted future cost by `10 / (1 - gamma)`, and
failure charges exactly that. Arriving never ends an episode: it runs for the route's
fixed duration, so finishing early gains nothing. Standing still lets the reference move
away, which costs more each step until the 3 m limit fails it.

## Curriculum

Chosen by total training steps, so a resume continues it. Wind comes from `wind.py`,
including its Ornstein-Uhlenbeck gusts, which are wind velocity in m/s and act through
drag. No separate disturbance force is added.

| Stage | Until step | Speed | Accel | Length | No wind | Mean wind | Gust scale | Turn rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `short_slow` | 150k | 0.3-0.7 | 0.3-0.4 | 1-3 m | 50% | 0-2 m/s | 0-0.25 | 0 |
| `turns` | 400k | 0.5-1.2 | 0.4-0.5 | 2-6 m | 20% | 0-4 m/s | 0-0.5 | to 0.05 rad/s |
| `climbs_loops` | 800k | 0.8-1.6 | 0.5-0.6 | 3-10 m | 10% | 1-6 m/s | 0.25-0.75 | to 0.1 rad/s |
| `full` | Then on | 1-2 | 0.5-0.6 | 4-16 m | 10% | 2-7 m/s | 0.5-1 | to 0.2 rad/s |

Route types are hover, straight, altitude, waypoints, circle, and figure-eight. Hover falls
to 5% of the last stage. Every route has acceleration, braking, and a settling hold; loops
and corners add turns. Corners may be sharp, which nearly stops the drone there, or rounded.

Yaw stays at 0, as in the existing navigation. "Turns" change the direction of travel, not
the heading.

## SAC

stable-baselines3 SAC. Separate optimizations:

- **Critics.** Twin Q networks regress to the Bellman target
  `r + gamma (1 - terminated) (min target Q - alpha log pi)`, using MSE. The Huber terms
  above build the reward; they are not this loss.
- **Actor.** Minimizes `alpha log pi - min Q`.
- **Temperature.** `alpha` is tuned to hold entropy near `target_entropy`.
- **Targets.** Polyak averaging with `tau`.

| Setting | Value | Why |
| --- | --- | --- |
| Networks | Actor and critics 256, 256 | Small input, as in hover |
| Learning rate | 3e-4 | SB3 default |
| Discount | 0.99 | About 10 s at 10 Hz, enough to see braking |
| Tau | 0.005 | SB3 default |
| Batch | 256 | SB3 default |
| Buffer | 500,000 | About 520 MB; holds several curriculum stages |
| Learning starts | 5,000 | Random actions until then |
| Updates | 1 per step | As in hover |
| Entropy | `auto`, target -4 | Minus the action size |
| Policy rate | 10 Hz | |
| Total steps | 1,500,000 | About 24 hours at 17 steps per second |
| Checkpoints | Every 50,000 | About 50 minutes |

These are project defaults, not published values. On this PC, 32 GB of RAM fits the buffer.
Disk is the tighter limit at 51 GB free, so only the 2 newest buffers are kept.

## Checkpoints

Saved under `runs/flight_sac/<run_name>/`:

- `checkpoints/flight_sac_N_steps.zip`, `flight_replay_buffer_N_steps.pkl`, and
  `flight_state_N_steps.json`
- `final_*`, `interrupted_*` on Ctrl+C, `crashed_*` on an error
- `flight_contract.json`, `config_snapshot_<time>.json`, `logs/`, `tensorboard/`

Files are written under a temporary name and moved into place, so a crash during a save
cannot damage a checkpoint.

| Restored on resume | From |
| --- | --- |
| Actor, critics, targets, their optimizers, temperature | Model `.zip` |
| Step and update counts, so the curriculum stage | Model `.zip` |
| Replay buffer | `.pkl` beside it |
| **Not restored** | Random generator state, the unfinished episode |

A resume starts a new episode and a new random stream, seeded from the step count. There
are no normalization statistics to restore. Fresh mode never loads anything. A resume
refuses a missing model, a missing buffer unless `allow_empty_replay_buffer` is set, hover
or PPO models, and models whose contract differs. Evaluation and navigation check only
what flying needs: rates, history, scales, limits, and PID gains.

## Evaluation

Scenarios: hover, straight, turns, altitude, circle, figure-eight, and the Dijkstra route.
Each flies in three wind profiles on `EVAL_SEEDS`, separate from training:

| Profile | Meaning |
| --- | --- |
| `training_range` | New seeds, the final stage's wind |
| `unseen_gusts` | Faster and stronger gusts than training |
| `outside_range` | Mean wind of 8-10 m/s, above training |

A seed fixes route, start, and wind, and the script stops if the two modes differ. The
policy is frozen and deterministic. Results go to `evaluation/<model>/`.

| Metric | Definition |
| --- | --- |
| `position_rmse` | Time-indexed: `sqrt(mean |p_ref(t) - p(t)|^2)`, sampled at 10 Hz |
| `cross_track_rmse`, `_max` | Geometric: distance to the nearest point of the path. Falling behind along the line adds nothing |
| `max_deviation` | Largest time-indexed error |
| `completion_rate`, `_time` | Within 0.15 m and 0.2 m/s of the end for 0.5 s; time counts only arrived flights |
| `failure_rate`, `collision_rate` | Crossed a limit. Collision: `too_low` in the open scene, obstacle in Dijkstra |
| `saturation_rate` | Share of physics steps with a clipped rotor |
| `residual_norm`, `residual_change` | Mean size of the action, and of its change per decision |

Failed flights stay in every average, measured over what they flew, so read RMSE together
with failure rate. `clearance_violation_rate`, Dijkstra only, is time inside the planner's
inflated geometry, which is not itself a crash.

**Sensitivity sweep.** `sweep_flight_sac.py` trains separate models at 0.5x and 2x of each
`lambda`, plus nominal, in their own folders. Rescoring one model would not test this.

## Navigation

`navigate_flight_sac.py` reuses the occupancy grid, `plan_flight`, PID, motors, and wind.
It replaces `WaypointFollower`, which stops at every waypoint, with a `TimedReference`.
Corners are rounded only where every chord passes `navigation.line_of_sight`; otherwise
they stay sharp. The reference keeps the polyline's points exactly, so nothing cuts
inside the checked route. A missing or incompatible model in `pid_sac` mode stops with an
error. Eval's Dijkstra part uses the same `fly_route`.

## Limits and tuning

- **Untested.** Nothing has been run, so bugs are likely on first launch. Start with a
  PID-only `eval_flight_sac.py` and `navigate_flight_sac.py` before training.
- **Reference speeds** assume PID can track them. If PID-only fails the `full` stage badly,
  lower its speed and wind before training.
- **Reward balance is a guess.** If `residual_norm` stays near zero or costs are dominated
  by `action_rate`, adjust the scales and action weights in `REWARD`.
- **Limits are tight.** Bounds are smaller than the hover model's, on purpose. Check
  `saturation_rate` and whether the policy pins at +-1.
- **PID caps remain.** 1.5 m/s^2 of horizontal demand and 10 degrees of tilt limit
  everything, and `outside_range` can exceed them.
- **Time-indexed tracking.** The reference does not wait for a drone that falls behind. In
  Dijkstra flight, more than 1 m of error stops the flight as a failure.
- **Clearance.** `SAFETY_MARGIN` is 0.01 m, smaller than expected tracking error. Line of
  sight proves the reference safe, not the flown path.
- **Collision detection** is position-based, with no contact sensor.
- **Takeoff.** Training never includes ground contact, so the correction stays zero during
  the launch departure.
- **Fixed yaw**, and ground-truth state only, with no IMU or LiDAR.
- **Duplicated route.** `motor_physics.py` keeps its destination inside `main()`, so
  `NAVIGATION` repeats it. Update both together. Its sensor logging is not included.
- **Sweep length.** Each variant trains for 500,000 steps, about 8 hours, so comparisons
  stop in the `climbs_loops` stage. Raise it to compare the final stage.
