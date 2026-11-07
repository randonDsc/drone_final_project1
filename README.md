# Improving Flying Stability of Traditional PID Controller Under Adverse Weather Condition using SAC Reinforcement Learning.

Po-Chen Lin, Cheng-I Lin, Colin Tran, Haoshuo Bi, and Claire(Please Replace your name)· UCSD DSC 180A Capstone · Advised by Tauhidur Rahman

## Notes for reading the github repo:

Since our project involves sensors and Isaac Sim and drones that majority of us never encounter before, in the first couple weeks, we each work individually and parallelly so all of us can stand on the same page regarding how everything actually works and learn the workflow of Isaac Sim and . Not until recently we integrate the whole project into a full pipeline of training and evaluation. As a result, the other 4 branches in the repository are the codebase for team members' approach in the last couple weeks. Thus, the contribution according to Github is not accurate.

## 1. Overview

A quadrotor flown by a classic cascaded **PID controller** drifts and overshoots in gusting wind. We keep the PID
controller unchanged and train a **Soft Actor-Critic (SAC)** policy to add a small, bounded correction on top of it:

```
u_final = u_PID + δu_SAC
```

`u` is the 4-value command `[total thrust, roll torque, pitch torque, yaw torque]`. PID runs at 120 Hz. SAC decides
at 10 Hz, and its correction is held between decisions and clipped to a few percent of rotor thrust. If the policy
behaves badly, the drone still flies on PID.

The whole stack runs in **NVIDIA Isaac Sim 6.1 (PhysX)** with a custom flight model built on top of it:

```
USD scene → occupancy grid → 3D Dijkstra path → waypoints / timed reference
         → PID controller (+ SAC residual) → motor mixer → rotor forces
         → gusting-wind drag model → PhysX rigid-body physics
```

There are two SAC tasks:

| Task | Goal | Main scripts |
| --- | --- | --- |
| **Hover SAC** | Hold a fixed point in wind | `train_hover_sac.py`, `eval_hover_sac.py` |
| **Path-following SAC** (main result) | Follow moving references (lines, turns, climbs, circles, figure-eights, and a Dijkstra route through a warehouse) in wind | `train_path_sac.py`, `eval_path_sac.py`, `navigate_path_sac.py` |


---

## 2. Results

Path-following SAC (`runs/flight_sac/flight1/final_model.zip`, 1.5M training steps), evaluated by
`eval_path_sac.py` on 10 seeds per scenario. PID-only and PID + SAC fly **the same** routes, start states, and wind
on each seed. Open-scene numbers are averaged over 6 route types: hover, straight, turns, altitude, circle, and
figure-eight.

**Wind profiles used in evaluation** (`path_sac_config.EVAL_WIND_PROFILES`). On each flight the wind starts from a
random horizontal direction, and its speed and gust strength are drawn uniformly from these ranges:

| Wind profile | Mean wind speed | Gust strength (std, horizontal / vertical) | Gust duration (shared / local) | Mean-wind turning rate | Compared with training |
| --- | --- | --- | --- | --- | --- |
| `training_range` | 2–7 m/s | 0.5–1.0 / 0.18–0.35 m/s | 2.0 s / 0.3 s | 0–0.2 rad/s | Same as the final training stage, but new seeds |
| `unseen_gusts` | 2–6 m/s | **1.2–1.5** / 0.42–0.53 m/s | **0.7 s / 0.15 s** | **0.1–0.3 rad/s** | Stronger, faster, more quickly turning gusts than training ever used |
| `outside_range` | **8–10 m/s** | 0.5–1.0 / 0.18–0.35 m/s | 2.0 s / 0.3 s | 0–0.2 rad/s | Mean wind above anything in training (max 7 m/s) |

Gust strength is the base gust in `BASE_WIND["gust_std"]` (1.0, 1.0, 0.35) m/s multiplied by a random gust scale:
0.5–1.0, or 1.2–1.5 for unseen gusts. Smaller local gusts across the airframe (`local_gust_std` = 0.3, 0.3, 0.15 m/s
× the same scale) add rotational disturbance. Drag is about 0.018·v² N on the 1.62 kg drone (≈ 0.9 N at 7 m/s,
≈ 1.8 N at 10 m/s).

| Scene | Wind profile | Controller | Position RMSE [m] | Max deviation [m] | Completed | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| Open | training range (2–7 m/s) | PID only | 0.502 | 1.77 | 5% | 0% |
| Open | training range (2–7 m/s) | **PID + SAC** | **0.082** | **0.27** | **100%** | 0% |
| Open | unseen gusts | PID only | 0.497 | 1.51 | 5% | 0% |
| Open | unseen gusts | **PID + SAC** | **0.082** | **0.25** | **100%** | 0% |
| Open | outside range (8–10 m/s) | PID only | 1.653 | 3.11 | 0% | 62% |
| Open | outside range (8–10 m/s) | **PID + SAC** | **0.181** | **0.83** | **90%** | 0% |
| Warehouse (Dijkstra) | training range | PID only | 0.391 | 0.98 | 20% | 50% |
| Warehouse (Dijkstra) | training range | **PID + SAC** | **0.186** | 0.95 | **90%** | 10% |
| Warehouse (Dijkstra) | unseen gusts | PID only | 0.399 | 0.99 | 0% | 50% |
| Warehouse (Dijkstra) | unseen gusts | **PID + SAC** | **0.221** | 1.00 | **70%** | 30% |
| Warehouse (Dijkstra) | outside range | PID only | 0.431 | 0.98 | 0% | 100% |
| Warehouse (Dijkstra) | outside range | PID + SAC | 0.444 | 1.00 | 0% | 100% |

On the open-scene routes, SAC cuts tracking error by about 6× and turns near-zero completion into 90–100%. In the
narrow warehouse it helps inside the training wind range, but neither controller survives wind beyond the range it
was trained on. Full per-scenario numbers are in `runs/flight_sac/flight1/evaluation/final_model/summary.csv`. A
written report is in `evaluation_report.pdf` in the same folder.

---

## 3. Setup

| Item | Version used |
| --- | --- |
| OS | Windows 11 Pro (10.0.26200) |
| GPU | NVIDIA GeForce RTX 4070 SUPER, driver 617.14 (any RTX GPU supported by Isaac Sim should work) |
| Simulator | **NVIDIA Isaac Sim 6.1.0** (standalone install at `C:\isaacsim`) |
| Python | **3.12.13**, bundled with Isaac Sim (`C:\isaacsim\python.bat`). Do **not** use a separate system Python. |
| stable-baselines3 | 2.9.0 |
| gymnasium | 1.4.0 |
| torch | 2.11.0 + CUDA 12.8 |
| numpy | 2.3.1 |
| tensorboard | 2.21.0 |


## 4. Directory Structure

```
drone_final_project1/
├── README.md                    ← this file
├── AI_USE.md                    ← log of AI-assisted work (see Section 10)
├── LICENSE
├── RL_nav_fc.png, control_rl_redem_sim.png   ← pipeline figures
│
├── drone_code/                  ← all source code (scripts are run from the project root)
│   │  ── Configuration (edit these, the scripts take no command-line arguments) ──
│   ├── sim_config.py            PID gains, motors, wind, grid map, navigation, flight mode for motor_physics.py
│   ├── run_config.py            What to run for hover SAC / hover PPO / sensor inspection
│   ├── hover_sac_config.py      Hover SAC: task, reward, residual limits, wind curriculum, SAC hyperparameters
│   ├── path_sac_config.py       Path SAC: train/eval/navigate settings, reward, curriculum, eval wind profiles
│   ├── hover_ppo_config.py      Hover PPO (earlier baseline) settings
│   ├── sensor_config.py         IMU noise, estimator, rangefinder / LiDAR settings
│   ├── watch_config.py          Settings for the PPO viewer
│   │
│   │  ── Simulation and physics ──
│   ├── drone_sim.py             Loads the USD scene, sets up the PhysX World, applies rotor forces each step
│   ├── motor.py                 Motor model: rotor speed → thrust and reaction torque, mixer, limits
│   ├── wind.py                  Gusting wind (Ornstein-Uhlenbeck gusts) + quadratic drag at sample points
│   ├── controller.py            Cascaded PID: position → acceleration → tilt → body torques
│   ├── drone_diagnostics.py     Centre-of-mass and thrust-offset checks
│   │
│   │  ── Mapping and navigation ──
│   ├── mapping.py, grid_map.py  USD geometry → 3D occupancy grid
│   ├── navigation.py            26-connected 3D Dijkstra, path shortening, waypoint follower
│   ├── path_reference.py        Timed references (p, v, a) along routes; corner rounding
│   ├── path_navigation.py       Flies a Dijkstra route as a timed reference (used by path SAC)
│   │
│   │  ── RL ──
│   ├── hover_task.py            Shared state, failure, and reset helpers
│   ├── hover_sac_env.py / hover_sac_policy.py   Hover SAC Gym environment and observation builder
│   ├── path_sac_env.py  / path_sac_policy.py    Path SAC Gym environment, PathCore control loop
│   ├── hover_env.py             Hover PPO environment + shared hover-scene loader
│   ├── model_contract.py        Refuses models trained with different observation/PID settings
│   │
│   │  ── Entry points (run these) ──
│   ├── motor_physics.py         Demo: Dijkstra navigation in the warehouse with the Isaac Sim GUI
│   ├── train_hover_sac.py, eval_hover_sac.py
│   ├── train_path_sac.py, eval_path_sac.py, navigate_path_sac.py, sweep_path_sac.py
│   ├── train_hover_ppo.py, eval_hover_ppo.py, smoke_hover_ppo.py, watch_hover_ppo.py
│   ├── inspect_sensors.py, check_sensor_assets.py
│   │
│   ├── tests/                   Unit tests (no Isaac Sim window needed)
│   └── CONTROL.md, HOVER_SAC.md, PATH_SAC.md, HOVER_PPO.md, INDOOR_SENSORS.md   ← design notes
│
├── sim_asset/                   Drone models: iris.usd, iris_indoor.usda (with sensors)
├── sim_scene/                   Scenes: dij3d_boxes.usd (warehouse with racks), dij3d.usd, build_box_scene.py
├── navigation_output/           Written by motor_physics.py: occupancy grid, path cells, waypoints
├── papers/                      Reference papers
└── runs/                        (git-ignored) Models, checkpoints, logs, evaluation results
    ├── sac_hover/<run_name>/
    ├── flight_sac/<run_name>/   ← path SAC (it keeps its earlier name "flight" so old runs still load)
    └── ppo_hover/<run_name>/
```

---

## 5. Quick Start

```bash
cd /c/drone_final_project1

# 1. Watch the drone plan a path and fly through the warehouse on PID (opens the Isaac Sim window)
/c/isaacsim/python.bat drone_code/motor_physics.py

# 2. Fly the same warehouse route with PID + trained path SAC, in wind
#    (settings: path_sac_config.PATH_NAV; needs runs/flight_sac/flight1/final_model.zip)
/c/isaacsim/python.bat drone_code/navigate_path_sac.py

# 3. Compare PID-only vs PID + SAC on all evaluation scenarios (headless, about 45 min on an RTX 4070 SUPER)
/c/isaacsim/python.bat drone_code/eval_path_sac.py
```
### Notes: The model is not uploaded to the GitHub, please download it from this link: 
#### https://huggingface.co/pcldsc/Iris_drone_residual_correction_SAC/tree/main 

After downloading, put the files in a `runs/` folder in the project root, exactly like this:

```
drone_final_project1/
└── runs/
    ├── flight_sac/flight1/
    │   ├── final_model.zip              ← path-following SAC (main model)
    │   └── flight_contract.json         ← keep next to final_model.zip
    └── sac_hover/wind1/
        ├── observation_contract.json    ← keep in wind1/
        └── checkpoints/
            └── sac_200000_steps.zip     ← hover SAC
```

The files on Hugging Face are already in this folder layout, so you can copy the downloaded `runs/` folder straight
into `drone_final_project1/`. Keep each `.json` file where it is shown: the scripts refuse to load a model without it.

---

## 6. How to Use the Codebase

### 6.1 How configuration works

**None of the scripts take command-line arguments.** Edit the matching config file, save it, and rerun the
script. Changes take effect the next time the script starts.

| Script | Edit this |
| --- | --- |
| `motor_physics.py` | `sim_config.py` (`CONTROL_MODE`, `DEMO_WIND`, `HOVER_SAC_MODEL_PATH`); destination inside `motor_physics.py` |
| `train_hover_sac.py`, `eval_hover_sac.py` | `run_config.py` (`TRAIN_HOVER_SAC`, `EVAL_HOVER_SAC`); learning details in `hover_sac_config.py` |
| `train_path_sac.py`, `eval_path_sac.py`, `navigate_path_sac.py`, `sweep_path_sac.py` | `path_sac_config.py` (`PATH_TRAIN`, `PATH_EVAL`, `PATH_NAV`, `PATH_SWEEP`) |
| `train_hover_ppo.py`, `eval_hover_ppo.py`, `smoke_hover_ppo.py` | `run_config.py`; details in `hover_ppo_config.py` |
| `watch_hover_ppo.py` | `watch_config.py` |
| `inspect_sensors.py` | `run_config.INSPECT_SENSORS`, `sensor_config.py` |

Every config value has a comment next to it with its meaning and unit. Units are meters, seconds, Newtons, and
radians unless the name says degrees.

`"gui": True` opens the Isaac Sim window so you can watch, but it is much slower. Use `"gui": False` for
training and evaluation.

---

### 6.2 Watch the drone fly: `motor_physics.py`

This is the end-to-end flight demo. It loads the warehouse scene `sim_scene/dij3d_boxes.usd` and builds an occupancy
grid from the scene's collision geometry. It then plans a 3D Dijkstra path from the drone's start pose to the
destination, shortens it with line-of-sight checks, and flies the waypoints with the PID controller, motor mixer, and
(optionally) wind.

```bash
/c/isaacsim/python.bat drone_code/motor_physics.py
```

- The Isaac Sim window opens and the flight starts automatically.
- **Pause → Play** resumes. **Stop → Play** resets the drone and replans.
- Outputs: `flight_log.txt` (the full flight log), plus the occupancy grid, path cells, and
  `waypoints_world.csv` in `navigation_output/`.

**What you can change**

| Want to… | Edit |
| --- | --- |
| Fly with wind | `sim_config.DEMO_WIND = True`; the strength is in `sim_config.WIND` (`mean_wind`, `gust_std`, …) |
| Add the hover SAC correction | `sim_config.CONTROL_MODE = "pid_sac"` and point `sim_config.HOVER_SAC_MODEL_PATH` at a hover model, e.g. `runs/sac_hover/wind1/checkpoints/sac_200000_steps.zip`. If the model is missing or incompatible, the flight says so and falls back to PID. |
| Change the destination | In `motor_physics.py` `main()`: `target["height"]` (world Z) and the XY offset `np.array([-2.0, 20.0])` from the start pose. To keep the path SAC route identical, change `path_sac_config.NAVIGATION` (`goal_offset_xy`, `goal_height`) to match. |
| Tune PID | `sim_config.PID` (height, xy, attitude loops) |
| Waypoint arrival tolerance | `sim_config.WAYPOINT_RADIUS`, `WAYPOINT_SPEED`, `WAYPOINT_DWELL` |
| Grid resolution / clearance | `sim_config.CELL_SIZE`, `SAFETY_MARGIN` |

> For the **path-following** SAC on this same route, use `navigate_path_sac.py` (Section 6.4). `motor_physics.py`
> flies waypoint-to-waypoint and stops briefly at each corner. `navigate_path_sac.py` flies a smooth timed reference
> along the same path.

---

### 6.3 Hover SAC: train and evaluate

SAC learns a correction that holds the drone at `(0, 0, 2)` m in an open scene while the wind gets harder in stages
(`light → steady → gusty → strong`). Settings are in `run_config.py`, and learning details are in
`hover_sac_config.py`.

**Train**

```bash
/c/isaacsim/python.bat drone_code/train_hover_sac.py
```

| `run_config.TRAIN_HOVER_SAC` key | Meaning |
| --- | --- |
| `run_name` | Output folder `runs/sac_hover/<run_name>/` |
| `total_timesteps` | Total SAC decisions to reach (10 per simulated second), **including** steps already in a resumed checkpoint |
| `resume` | `None` = start fresh; otherwise a model `.zip` to continue, e.g. `runs/sac_hover/wind1/checkpoints/sac_200000_steps.zip` |
| `allow_empty_replay_buffer` | `True` = resume even if the replay-buffer `.pkl` is missing |
| `gui` | Show the Isaac Sim window |

- Checkpoints are saved every 100,000 steps (`hover_sac_config.TRAINING`) to `checkpoints/sac_N_steps.zip` with a
  matching replay buffer.
- **Ctrl+C** saves `interrupted_model.zip`. To continue, set `resume` to that file or to the newest checkpoint.
- Quick pipeline check: `resume = None`, `total_timesteps = 5000`, `run_name = "check"`.
- Monitor training with `/c/isaacsim/python.bat -m tensorboard.main --logdir runs/sac_hover`.

**Evaluate** (PID-only vs PID + SAC on the same 20 seeds)

```bash
/c/isaacsim/python.bat drone_code/eval_hover_sac.py
```

| `run_config.EVAL_HOVER_SAC` key | Meaning |
| --- | --- |
| `run_name`, `model` | Which model; `model = None` uses `final_model.zip` of the run |
| `wind_stage` | `light`, `steady`, `gusty`, or `strong` (defined in `hover_sac_config.WIND_STAGES`) |
| `episodes` | First N of the 20 evaluation seeds, or `None` for all |
| `pid_only` | `True` = baseline only, needs no model |
| `gui` | Show the window (try `episodes = 3, gui = True` to watch) |

Outputs: `runs/sac_hover/<run_name>/eval_<stage>.json`, `eval_<stage>_summary.csv` (RMSE, late error, max error,
failure rate, mean |action|, saturation, cost), and `eval_<stage>_episodes.csv`.

---

### 6.4 Path-following SAC: train, evaluate, navigate

This is the main result. SAC learns a correction that helps PID follow **moving** references in wind. Every setting
is in `path_sac_config.py`.

**Train**

```bash
/c/isaacsim/python.bat drone_code/train_path_sac.py
```

| `path_sac_config.PATH_TRAIN` key | Meaning |
| --- | --- |
| `mode` | `"fresh"` = new model (refuses a folder that already has one); `"resume"` = continue `resume_model` |
| `resume_model` | e.g. `runs/flight_sac/flight1/final_model.zip` or `checkpoints/flight_sac_N_steps.zip` |
| `run_name` | Output folder `runs/flight_sac/<run_name>/` |
| `total_timesteps` | Total decisions to reach, including resumed ones (our model: 1.5M) |
| `seed`, `checkpoint_interval` | Training seed; save every N decisions (50,000 ≈ 50 min) |
| `keep_replay_buffers` | Older replay buffers (~520 MB each) beyond this many are deleted |
| `gui` | Show the window |

- The curriculum (`CURRICULUM`) moves through 4 stages by training step: `short_slow` (<150k) → `turns` (<400k)
  → `climbs_loops` (<800k) → `full`. Each stage makes the routes longer and faster and the wind stronger.
- A resumed run checks `flight_contract.json` and refuses to continue if the observation, PID, reward, or
  curriculum settings changed since the model was trained.
- Logs: `runs/flight_sac/<run>/logs/`, TensorBoard at `runs/flight_sac/<run>/tensorboard/`, plus a
  `config_snapshot_<time>.json` for each launch.

**Evaluate** (PID-only vs PID + SAC, the numbers in Section 2)

```bash
/c/isaacsim/python.bat drone_code/eval_path_sac.py
```

| `path_sac_config.PATH_EVAL` key | Meaning |
| --- | --- |
| `run_name`, `model` | Which model; `None` = `final_model.zip` of the run |
| `pid_only` | `True` = fly the PID baseline only, no model needed |
| `episodes` | First N of the 10 `EVAL_SEEDS`, or `None` for all |
| `include_dijkstra` | `False` skips the warehouse route, which loads a second scene |
| `gui` | Show the window |

The script flies every scenario (`EVAL_SCENARIOS`: hover, straight, turns, altitude, circle, figure_eight, plus the
Dijkstra route) under each wind profile in `EVAL_WIND_PROFILES`:

| Wind profile | Meaning |
| --- | --- |
| `training_range` | New seeds from the final training distribution (mean 2–7 m/s) |
| `unseen_gusts` | Faster, stronger gusts than training ever used |
| `outside_range` | Mean wind 8–10 m/s, above anything in training |

Outputs go to `runs/flight_sac/<run>/evaluation/<model>/`: `summary.csv` (per scenario × wind × controller),
`flights.csv` (one row per flight), and `results.json`.

**Navigate: fly the warehouse Dijkstra route with path SAC**

```bash
/c/isaacsim/python.bat drone_code/navigate_path_sac.py
```

| `path_sac_config.PATH_NAV` key | Meaning |
| --- | --- |
| `mode` | `"pid"` = PID only; `"pid_sac"` = PID + SAC. It stops with an error if the model is missing; it never silently falls back to PID |
| `run_name`, `model` | Which model |
| `wind_profile` | A name from `EVAL_WIND_PROFILES` |
| `seed` | Fixes the wind realization, so `pid` and `pid_sac` fly the same wind |
| `gui` | `True` to watch |

The route, speed, and destination are in `path_sac_config.NAVIGATION` (`goal_offset_xy`, `goal_height`, `speed`,
`corner_radius`, `max_position_error`, …). The result is saved to
`runs/flight_sac/<run>/navigation/<mode>_<wind_profile>.json`.

To compare the two controllers side by side, run once with `mode = "pid"` and once with `mode = "pid_sac"`, using
the same `seed`.

**Reward-weight sweep (optional)**

```bash
/c/isaacsim/python.bat drone_code/sweep_path_sac.py
```

This trains and evaluates one fresh model for each variant in `PATH_SWEEP["variants"]`, scaling the velocity
(`lambda_v`) and acceleration (`lambda_a`) reward weights. Each variant gets its own `runs/flight_sac/sweep_<variant>/`.

---

### 6.5 Hover PPO (earlier baseline)

Our first experiment used PPO instead of SAC, with IMU-based state estimation. It is kept for comparison.

```bash
/c/isaacsim/python.bat drone_code/smoke_hover_ppo.py   # short pipeline check
/c/isaacsim/python.bat drone_code/train_hover_ppo.py   # run_config.TRAIN_HOVER_PPO
/c/isaacsim/python.bat drone_code/eval_hover_ppo.py    # run_config.EVAL_HOVER_PPO
/c/isaacsim/python.bat drone_code/watch_hover_ppo.py   # watch_config.py: live viewer, wind, PPO on/off
```

`state_source` is `"imu"` (noisy IMU + estimator, see `sensor_config.py`) or `"ground_truth"`. `wind_profile` is
`none`, `weak`, or `moderate`. See `drone_code/HOVER_PPO.md`.

---

### 6.6 Sensors and unit tests

```bash
# Spawn the sensor-equipped drone (IMU, rangefinder, LiDAR) and print readings (run_config.INSPECT_SENSORS)
/c/isaacsim/python.bat drone_code/inspect_sensors.py

# Unit tests: controller, wind, navigation, state estimation, hover/path SAC. No GUI. Run from drone_code/.
cd drone_code
/c/isaacsim/python.bat -m unittest discover -s tests
```

> The current suite runs 162 tests, and 3 fail because their expected values predate the latest config tuning
> (`sim_config.WIND["mean_wind"]` and the loosened waypoint and tracking tolerances). The code itself is not broken.

---

## 7. Reproducing Our Results

All results use the configs as committed. Evaluation seeds are fixed (`EVAL_SEEDS`), so PID-only and PID + SAC face
identical routes and wind.

**A. Path-following SAC (Section 2 table)**

1. In `path_sac_config.py`, set `PATH_TRAIN`: `mode = "fresh"`, `run_name = "flight1"`,
   `total_timesteps = 1_500_000`, `seed = 0`, `gui = False`. Leave every other section unchanged.
2. Train: `/c/isaacsim/python.bat drone_code/train_path_sac.py`. Our run took about 30 hours on an RTX 4070 SUPER.
   You can stop with Ctrl+C at any time. To resume, set `mode = "resume"` and `resume_model` to the newest file in
   `runs/flight_sac/flight1/checkpoints/` (or `interrupted_model.zip`), then rerun.
3. Evaluate: in `PATH_EVAL`, set `run_name = "flight1"`, `model = None`, `pid_only = False`, `episodes = None`,
   `include_dijkstra = True`, then run `/c/isaacsim/python.bat drone_code/eval_path_sac.py`.
4. Compare your `runs/flight_sac/flight1/evaluation/final_model/summary.csv` with ours.
5. Watch one flight: in `PATH_NAV`, set `mode = "pid"` and then `"pid_sac"`, both with `wind_profile = "training_range"`,
   `seed = 18`, `gui = True`, then run `navigate_path_sac.py`.

To skip training, put our `final_model.zip` and `flight_contract.json` in `runs/flight_sac/flight1/` (Section 3.2)
and start at step 3.

**B. Hover SAC**

1. In `run_config.TRAIN_HOVER_SAC`, set `run_name = "wind1"`, `resume = None`, `total_timesteps = 200_000`, and
   train with `train_hover_sac.py`.
2. In `run_config.EVAL_HOVER_SAC`, set `model = "runs/sac_hover/wind1/checkpoints/sac_200000_steps.zip"` and
   `wind_stage = "gusty"`, then run `eval_hover_sac.py`.
3. Our gusty-wind result (20 seeds): PID only had an RMSE of 0.162 m and a max error of 0.864 m. PID + SAC had an
   RMSE of 0.061 m and a max error of 0.304 m. Neither failed any episode.

> Isaac Sim physics on the GPU is not bit-for-bit deterministic across machines, so expect small numeric
> differences. The trends should match.

---

## 8. Configuration Reference

These are the most important parameters. Every file comments every value.

### `sim_config.py` (physics, controller, wind, navigation)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `PID["height"]` | kp 10, ki 3, kd 5 | Height error [m] → vertical force [N] |
| `PID["xy"]` | kp 0.6, ki 0.15, kd 1.2 | Horizontal error [m] → acceleration [m/s²] → tilt |
| `PID["attitude"]` | kp (0.2, 0.2, 0.1), kd 0.05 | Roll/pitch/yaw error [rad] → torque [N·m] |
| `MAX_TILT_DEGREES` | 10 | Tilt limit of the controller |
| `MOTOR` | kT = 5.84e-6, max 1100 rad/s | Thrust = kT·ω², reaction torque = 0.06·thrust |
| `WIND["mean_wind"]` | (1.5, 0, 0) m/s | Mean wind for `motor_physics.py` |
| `WIND["gust_std"]`, `gust_timescale` | (0.5, 0.5, 0.2) m/s, 2 s | Shared gust strength and duration |
| `WIND["local_gust_std"]` | (0.15, 0.15, 0.08) m/s | Uneven wind across the airframe, which causes rotation |
| `WIND["effective_area"]` | 0.03 m² | Drag = ½ρ·Cd·A·\|v\|² (≈1.8 N at 10 m/s) |
| `CELL_SIZE`, `SAFETY_MARGIN` | 0.25 m, 0.01 m | Occupancy-grid resolution; extra clearance around the drone |
| `WAYPOINT_RADIUS / SPEED / DWELL` | 0.3 m, 0.4 m/s, 0.25 s | When a waypoint counts as reached |
| `CONTROL_MODE` | `"pid"` | `"pid"` or `"pid_sac"` for `motor_physics.py` |
| `DEMO_WIND` | `False` | Wind on or off in `motor_physics.py` |

### `path_sac_config.py` (path-following SAC)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `PHYSICS_RATE_HZ`, `RL_RATE_HZ` | 120, 10 | PID rate; SAC decision rate (correction held 12 physics steps) |
| `RESIDUAL_LIMITS` | thrust 0.8 N, roll 0.05, pitch 0.03, yaw 0.02 N·m | Correction at action = ±1; at most ~0.40 N per rotor |
| `HISTORY_LENGTH` | 4 | Observation frames stacked (0.3 s of history) |
| `REWARD` | `lambda_v` 1, `lambda_a` 1, `scale` 0.1, `cost_cap` 10 | Huber tracking costs (position, speed, direction, attitude, rate) + action, smoothness, and saturation costs |
| `FAILURE` | 3 m error, 0.5 m min height, 45° tilt, 7 m/s, 6 rad/s | Any of these ends the episode as a crash |
| `COMPLETION` | 0.15 m, 0.2 m/s, 0.5 s | Arrival at the final point |
| `CURRICULUM` | 4 stages | Route mix, speeds, lengths, and wind ranges per training stage |
| `SAC` | lr 3e-4, γ 0.99, τ 0.005, batch 256, buffer 500k, net [256, 256] | stable-baselines3 SAC hyperparameters |
| `EVAL_SEEDS` | 300000–300009 | Fixed evaluation seeds |
| `NAVIGATION` | goal (−2, 20) m offset at 1.2 m, 0.8 m/s | Warehouse Dijkstra route for eval / navigate |

### `hover_sac_config.py` (hover SAC)

| Parameter | Default | Meaning |
| --- | --- | --- |
| `TASK["target_position"]` | (0, 0, 2) | Hover point |
| `TASK["episode_seconds"]` | 20 | Episode length |
| `RESIDUAL_LIMITS` | thrust 1.5 N, roll/pitch 0.04, yaw 0.02 N·m | Correction at action = ±1 |
| `HISTORY_LENGTH` | 5 | 0.5 s of stacked observations |
| `WIND_STAGES` | light (<100k), steady (<300k), gusty (<600k), strong | Wind curriculum by training step |
| `SAC` | lr 3e-4, γ 0.99, buffer 200k, net [256, 256] | SAC hyperparameters |
| `EVAL_SEEDS` | 200000–200019 | Fixed evaluation seeds |

> **Changing any observation, residual, or PID setting makes existing models incompatible.** The scripts check
> each model's contract file and refuse to load a mismatched model instead of flying it wrong. Restore the original
> settings or train a new model.

---

## 9. Further Documentation

| File | Contents |
| --- | --- |
| `drone_code/CONTROL.md` | Controller design, PID loops, how the residual is added |
| `drone_code/HOVER_SAC.md` | Hover SAC: action, observation, reward, wind curriculum |
| `drone_code/PATH_SAC.md` | Path SAC: reference adapter, observation, reward, metrics |
| `drone_code/HOVER_PPO.md` | PPO baseline and IMU state estimation |
| `drone_code/INDOOR_SENSORS.md` | Rangefinder and LiDAR sensor models |
| `papers/` | Reference papers: SAC-assisted hybrid control, PID + RL |

---

## 10. AI Usage

### Haoshuo Bi
AI contribution: I used ChatGPT to help understand Isaac Lab, troubleshoot simulation errors, and modify Python scripts for drone sensors, wind disturbances, navigation, and CSV data logging. ChatGPT also helped me understand the simulation results and organize the experimental data.

Human Contribution:
I installed and configured Isaac Lab on Windows, modified and tested the drone simulation scripts, collected RGB, Depth, LiDAR, and IMU data, and compared drone behavior under 0 N and 0.1 N external forces. I also tested 2 m, 10 m, and 100 m navigation and verified the results using terminal outputs and CSV files.
### Colin Tran

### Cheng-I Lin
AI contribution: I used ChatGPT to first understand how to install Isaac Sim on my MacBook, which is not possible since I need Linux to run the simulation platform. Therefore, I switched to the MacBook friendly platform MuJoCo, which is another physics based simulator. Using AI, I learned how to navigate through the codes of certain example GitHub repositories to create testing experiments of drone flights, such as generating wind forces, incorporating camera, and capturing sensory data. 

Human Contribution: Given the outline of the coding structure, I experimented with the simulation and looked at the resulting outputs, debugging sensor readings (camera orientations), testing motor failures during certain simulated runs, and implementing different values to test wind conditions to see how results vary.

### Claire Tu

### Po-Chen Lin
AI contribution: I use ChatGPT to learn to install Isaac Sim and Isaac lab, learn about reinforcemnet learning, learn the physics, explain the approach of the paper we want to copy, and explain the raw data of the sensors. I also used it to discuss the possible approach to actually implement things like wind and navigation algorithm. Then, from the method I discussed with ChatGPT, I used Claude to do the actual implementation. 

Human Contribution: Giving the idea such as implementation of wind, alternate reward function, reduce the size of lidar data and other sensor data, construct the tested scene for Isaac sim, tested the result, debugged the code, and verify whether the approach is applicable with ChatGPT.


The full log of AI-assisted prompts of everyone is in [`AI_USE.md`](AI_USE.md).
---

## 11. License

See [`LICENSE`](LICENSE).
