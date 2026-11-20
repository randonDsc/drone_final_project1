# Path SAC training runs

What each path-following SAC model was trained with, how it went, and how to
reproduce it. The values for flight1 and flight2 are copied from what the runs saved
(`runs/flight_sac/<run>/flight_contract.json` and `config_snapshot_*.json`), not
from the current config. `tests/test_continual_backprop.py` checks that the
`TRAIN_RUNS` entries still produce exactly those contracts.

Drone physics (mass, motors, drag) is in `sim_config.py` and is not repeated here.

## Reproducing a run

Every run is an entry in `path_sac_config.TRAIN_RUNS`. Nothing else needs editing.

1. Set `TRAIN_RUN` to the run's name and `RESUME = None`.
2. Run, from the project folder: `C:\isaacsim\python.bat drone_code/train_path_sac.py`
3. To stop, press **Ctrl+C** in the terminal. That saves `interrupted_model.zip`.
   Closing the window instead loses everything since the last checkpoint, about 50
   minutes.
4. To continue, set `RESUME = "latest"` and run again. It picks the newest model in
   the run's folder that still has its replay buffer. Set it back to `None` after
   the run finishes.
5. To evaluate, set `PATH_EVAL["run_name"]` to the run and run `eval_path_sac.py`.

| Run | Needs first | Approx. time on an RTX 4070 SUPER |
| --- | --- | --- |
| `flight1` | nothing | 30 h |
| `flight2` | `runs/flight_sac/flight1/final_model.zip` and `final_replay_buffer.pkl`, from training flight1 or a copy of ours | 19 h |
| `flight6_cbp` | nothing | about 30 h, not yet run |
| `flight6_nocbp` | nothing | about 30 h, not yet run |

`runs/` is not in git, so a fresh clone has no models. A run folder that already
holds a model refuses a fresh start. To train a second copy beside it, copy the
entry under a new name.

Results will not match ours to the digit. Isaac Sim's GPU physics is not
deterministic across machines, and a resume does not restore the random generator
state. Expect the same trends.

## Settings shared by flight1 and flight2

The two runs' saved contracts differ only in the curriculum.

### Policy (what the model needs to fly)

| Setting | Value |
| --- | --- |
| Contract version | 1 (`workflow` = `flight_sac`) |
| Physics and PID rate | 120 Hz |
| Policy decision rate | 10 Hz, correction held for 12 physics steps |
| History | 4 frames of 32 values, so 128 inputs |
| Observation scales | position error 1 m, velocity error 1 m/s, reference velocity 2 m/s, reference acceleration 1 m/s², attitude error 0.35 rad, angular velocity 2 rad/s, PID thrust 5 N, PID torque 0.1 N·m |
| Observation clip | ±10 after scaling |
| Residual limits at action ±1 | thrust 0.8 N, roll 0.05 N·m, pitch 0.03 N·m, yaw 0.02 N·m |
| PID gains | `sim_config.PID`, unchanged: height kp 10, ki 3, kd 5; xy kp 0.6, ki 0.15, kd 1.2; attitude kp (0.2, 0.2, 0.1), ki (0, 0, 0.02), kd 0.05 |

### Reward

Each decision:

    reward = -0.1 * min(sum of costs, 10)
    failure adds -0.1 * 10 / (1 - 0.99) = -100

Tracking errors are divided by a scale, then go through a Huber loss with δ = 1.

| Term | Scale or weight |
| --- | --- |
| Position | 0.25 m |
| Speed | 0.5 m/s |
| Direction | 1 (dimensionless), weight 1, fades below 0.3 m/s |
| Attitude | 0.2 rad |
| Body rate | 1 rad/s, reference rate limited to 1 rad/s |
| Outer weights | λ_v = 1, λ_a = 1 |
| Action | 0.005 × \|action\|² |
| Action change | 0.02 × \|Δaction\|² |
| Saturation | 0.5 × share of saturated physics steps |
| Persistent saturation | 2.0 per decision after 5 fully saturated decisions |
| Unsafe tilt | 1.0 × ((tilt − 20°) / 20°)² above 20° |
| Unsafe rate | 0.5 × ((rate − 3) / 3)² above 3 rad/s |
| Route deviation | 1.0 × ((error − 1) / 1)² above 1 m |

### Episodes

| Setting | Value |
| --- | --- |
| Failure | position error 3 m, height below 0.5 m, tilt 45°, speed 7 m/s, angular speed 6 rad/s |
| Completion | within 0.15 m and 0.2 m/s of the end for 0.5 s |
| Time limit | 60 s; every route ends with a 3 s settling hold |
| Route start | ±3 m in x and y; routes stay between 1.2 and 4.8 m height |
| Start state | ±0.1 m position, ±0.1 m/s velocity, ±3° roll and pitch, ±5° yaw, ±0.1 rad/s |
| Reference spacing | 0.05 m |

### Wind model

Wind = mean + shared gust + local gust, applied through drag at the hull and four
rotors (`wind.py`).

| Part | Spread at gust scale 1 (x, y, z) | Correlation time |
| --- | --- | --- |
| Shared gust, whole drone | 1.0, 1.0, 0.35 m/s | 2.0 s |
| Local gust, per point | 0.3, 0.3, 0.15 m/s | 0.3 s, correlated over 0.4 m |

For each flight, the curriculum stage gives the following:

| Setting | How it is drawn |
| --- | --- |
| Calm flight | Chance of no wind at all |
| Mean speed | Drawn evenly from the range, horizontal, in a random direction |
| Gust scale | Drawn evenly from the range |
| Turn rate | Drawn evenly from the range, random sign; the mean wind turns at that rate for the whole flight |

### SAC (stable-baselines3)

| Setting | Value |
| --- | --- |
| Networks | actor 128 → 256 → 256 → 4 means + 4 log-std; two critics 132 → 256 → 256 → 1, plus target copies |
| Parameters | actor 100,872; each critic 100,097; 501,260 in the model file |
| Learning rate | 3e-4 |
| Discount γ | 0.99 |
| τ (target update) | 0.005 |
| Batch | 256 |
| Replay buffer | 500,000 transitions |
| Learning starts | 5,000 |
| Updates | 1 gradient step per decision |
| Entropy | `auto`, target −4 |
| Seed | 0 |
| Checkpoints | every 50,000 decisions, keeping the 2 newest replay buffers |

### Curriculum stages

A stage applies until the total step count reaches its end.

| Stage | Steps | Calm | Mean wind | Gust scale | Turn rate |
| --- | --- | --- | --- | --- | --- |
| `short_slow` | 0 – 150k | 50% | 0 – 2 m/s | 0 – 0.25 | 0 |
| `turns` | 150k – 400k | 20% | 0 – 4 m/s | 0 – 0.5 | 0 – 0.05 rad/s |
| `climbs_loops` | 400k – 800k | 10% | 1 – 6 m/s | 0.25 – 0.75 | 0 – 0.1 rad/s |
| `full` | 800k – (see runs) | 10% | 2 – 7 m/s | 0.5 – 1.0 | 0 – 0.2 rad/s |
| `strong_ramp` | 1.5M – 1.8M | 5% | 2 – 10 m/s | 0.5 – 1.0 | 0 – 0.2 rad/s |
| `strong` | 1.8M on | 5% | 2 – 14 m/s | 0.5 – 1.0 | 0 – 0.2 rad/s |

Routes per stage. `strong_ramp` and `strong` use `full`'s routes.

| Route setting | `short_slow` | `turns` | `climbs_loops` | `full` |
| --- | --- | --- | --- | --- |
| Mix: hover / straight / altitude / waypoints / circle / figure-eight | 0.2 / 0.45 / 0.25 / 0.1 / 0 / 0 | 0.1 / 0.25 / 0.15 / 0.35 / 0.15 / 0 | 0.05 / 0.15 / 0.15 / 0.3 / 0.2 / 0.15 | 0.05 / 0.15 / 0.15 / 0.3 / 0.175 / 0.175 |
| Speed | 0.3 – 0.7 m/s | 0.5 – 1.2 | 0.8 – 1.6 | 1.0 – 2.0 |
| Acceleration | 0.3 – 0.4 m/s² | 0.4 – 0.5 | 0.5 – 0.6 | 0.5 – 0.6 |
| Lateral acceleration | 0.3 m/s² | 0.4 | 0.5 | 0.6 |
| Vertical speed | 0.4 m/s | 0.6 | 0.8 | 1.0 |
| Length | 1 – 3 m | 2 – 6 | 3 – 10 | 4 – 16 |
| Height change | 0 – 0.5 m | 0 – 1.0 | 0 – 1.5 | 0 – 2.5 |
| Waypoints | 3 | 3 – 4 | 3 – 5 | 3 – 6 |
| Turn angle | 20 – 60° | 30 – 90 | 30 – 120 | 30 – 150 |
| Corner radius | 0.2 – 0.5 m | 0 – 0.6 | 0 – 0.8 | 0 – 1.0 |
| Loop radius | 1 – 1.5 m | 1 – 2.5 | 1.5 – 3.5 | 2 – 4 |
| Hover time | 4 – 8 s | 3 – 6 | 2 – 5 | 2 – 5 |

## flight1

`TRAIN_RUNS["flight1"]`: fresh, curriculum `base`, 1,500,000 steps.

**Curriculum `base`.** The first four stages. `full` never ends (`until_step` is
`None`), so it ran from 800k until training stopped at 1.5M.

**History**

| Launch | Mode | Steps | Wall time |
| --- | --- | --- | --- |
| 2026-11-01 22:11 | fresh | 0 → 79,168 | 1.4 h |
| 2026-11-02 00:16 | resume from the 50,000 checkpoint | 50,000 → 1,500,000 | 28.8 h |

The 29,168 steps after the 50,000 checkpoint in the first launch were discarded by the
resume. Wall time totals 30.2 h, at 13–15 decisions per second.

**End of training** (last log row, step 1,499,364; rates are rolling means over
recent episodes):

| Metric | Value |
| --- | --- |
| Episodes | 10,400 |
| Episode reward | −2.50 |
| Failure rate | 0% |
| Completion rate | 100% |
| Position error | 0.070 m |
| Entropy coefficient | 0.00039 |

**Evaluation** (`runs/flight_sac/flight1/evaluation/final_model/`, 2026-11-04). It
used 10 seeds per scenario and the three wind profiles that existed then. The table
averages the six open-field scenarios: hover, straight, turns, altitude, circle, and
figure-eight.

| Wind profile | Controller | Position RMSE | Worst deviation | Completion | Failure |
| --- | --- | --- | --- | --- | --- |
| `training_range` (2–7 m/s) | PID only | 0.502 m | 2.44 m | 5% | 0% |
| | PID + SAC | **0.082 m** | 0.36 m | 100% | 0% |
| `unseen_gusts` (2–6 m/s, faster gusts) | PID only | 0.497 m | 2.10 m | 5% | 0% |
| | PID + SAC | **0.082 m** | 0.29 m | 100% | 0% |
| `outside_range` (8–10 m/s) | PID only | 1.653 m | 3.15 m | 0% | 62% |
| | PID + SAC | **0.181 m** | 1.35 m | 90% | 0% |

On the Dijkstra warehouse route, PID + SAC completed 90% of flights in
`training_range` against 20% for PID alone. Neither completed any in `outside_range`.

## flight2

`TRAIN_RUNS["flight2"]`: fine-tune of flight1's final model, curriculum
`strong_wind`, to 2,500,000 total steps. The fine-tune loads flight1's weights,
replay buffer, and step count. Only the policy contract has to match.

**Curriculum `strong_wind`.** All six stages. `full` ends at 1,500,000. flight2 starts
at 1,500,000, so it trained only in `strong_ramp` (1.5M–1.8M) and `strong` (1.8M–2.5M).
The other training settings are identical to flight1's.

**History**

| Launch | Mode | Steps | Wall time |
| --- | --- | --- | --- |
| 2026-11-08 23:15 | finetune from flight1 `final_model.zip` | 1,500,000 → about 2,253,600 | 14.1 h |
| 2026-11-09 17:05 | resume from the 2,250,000 checkpoint | 2,250,000 → about 2,460,800 | 4.1 h |
| 2026-11-10 01:43 | resume from the 2,450,000 checkpoint | 2,450,000 → 2,500,000 | 1.0 h |

The first two launches ended with the window closed, so the steps after their last
checkpoint, about 3,600 and 10,800, were redone. Wall time totals 19.2 h.

**End of training** (last log row, step 2,499,518):

| Metric | Value |
| --- | --- |
| Episodes, counted from flight1's start | 16,132 |
| Episode reward | −17.58 |
| Failure rate | 0% |
| Completion rate | 75% |
| Position error | 0.26 m |
| Entropy coefficient | 0.0143, up from 0.0004 at the start of flight2 |

**Evaluation:** not run yet. `EVAL_WIND_PROFILES` now has five profiles. The first
three are unchanged, so they compare directly with flight1. `strong_range` (10–14
m/s) and `beyond_strong` (14–16 m/s) are new.

## flight6_cbp and flight6_nocbp (planned)

Both runs use flight1's wind and schedule (curriculum `base`, 1,500,000 steps,
fresh, seed 0) with different layer sizes. Only `flight6_cbp` adds Continual
Backprop. Comparing the two separates the effect of CBP from the effect of the
layer sizes. Comparing `flight6_nocbp` with flight1 shows the effect of the layer
sizes alone.

| Network | Hidden layers | Parameters |
| --- | --- | --- |
| Actor | 64 → 128 → 256 → 64 | 66,568 |
| Each critic | 128 → 256 → 256 → 128 | 148,865 |
| Model file | actor, 2 critics, 2 target critics | 662,028 |

**Continual Backprop** (`continual_backprop.py`, settings in
`path_sac_config.CONTINUAL_BACKPROP`), after every gradient update, on all 12 hidden
layers:

| Setting | Value |
| --- | --- |
| Utility | running average (decay 0.99) of mean\|activation\| × sum of \|outgoing weights\| |
| Maturity | 50 updates before a neuron can be replaced |
| Replacement rate | 1e-4 of a layer's eligible neurons per update, lowest utility first |
| Reset | incoming weights and bias redrawn as at initialization; outgoing weights zeroed; Adam moments zeroed; same reset in the target critics |
| Bias correction | on: the removed neuron's mean contribution moves into the next layer's bias |

Training logs `cbp/actor_replaced`, `cbp/critic_replaced`, and the share of neurons
silent on the last batch (`cbp/*_silent_share`). Ages and utilities are not saved with
the model, so after a resume every neuron is protected for 50 updates again. A CBP
model loads with plain `SAC.load`, so evaluation and navigation need no changes.
