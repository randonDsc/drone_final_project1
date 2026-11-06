# Drone flight under environmental disturbances

UC San Diego Data Science capstone: a reproducible MuJoCo framework for
measuring how wind affects flight stability and trajectory tracking. The
original Isaac Sim implementation remains in `drone_code/`. The local backend
is `mujoco_drone/`, developed on the `alan/mujoco` branch.

The MuJoCo PID baseline comes from **`mujoco-project/drone-example/main.py`**, as
requested. `point_to_point_wind.py` is a wind experiment;
`point_to_point_wind_corrected.py` is excluded as an implementation source.
The original project is not modified or needed at runtime.

## Install on Apple Silicon macOS

Use Python 3.11 or newer. From a fresh checkout:

```sh
cd drone_final_project1
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[rl,plot,test]'
python -m mujoco_drone inspect
```

No NVIDIA software, Isaac Sim, CUDA, or external Menagerie checkout is needed.
The licensed Skydio X2 MJCF, meshes, and texture are included. SAC uses CPU
PyTorch. See `requirements-macos-tested.txt` for the direct dependency versions
actually used in validation. The development `.venv` uses the existing
`mujoco-env` interpreter with system packages; a fresh install can use an
ordinary isolated venv as above.

## Run flight experiments

```sh
python -m mujoco_drone run --config configs/hover_calm.json --output mujoco_output/hover
python -m mujoco_drone run --config configs/line_gust.json --seed 42 --output mujoco_output/line_gust
python -m mujoco_drone run --task circle --wind turbulence --seed 42 --output mujoco_output/circle
python -m mujoco_drone plot mujoco_output/line_gust
```

Output directories must be new: runs refuse to overwrite existing results.
`--duration` sets a maximum horizon and also sets the duration of hover tasks.
Moving tasks follow their complete reference and end hold unless that maximum
cuts them short. Successful arrival does not stop collection early. Failures do.

Supported tasks: `hover`, `straight`, `altitude`, `waypoints`, `circle`,
`figure_eight`, and obstacle navigation with `dijkstra`.
The new controlled references have analytic, consistent position/velocity/
acceleration. Waypoints stop smoothly at each corner. All default paths start
at 1.5 m; waypoints include an altitude change. These are airborne-start
experiments. Warehouse navigation additionally validates an upward departure
from the source launch area; landing is not implemented.

## Dijkstra and the original Isaac experiments

The warehouse collision scene and original 3D Dijkstra algorithm now run in
MuJoCo. The runtime includes all 3,491 converted boxes and the ground plane;
it does not require USD, Isaac Sim, or an external scene download.

```sh
python -m mujoco_drone plan --scene warehouse --output mujoco_output/warehouse_plan
python -m mujoco_drone plot mujoco_output/warehouse_plan
python -m mujoco_drone navigate --scene warehouse --output mujoco_output/warehouse
python -m mujoco_drone plot mujoco_output/warehouse
```

`plan` saves a checked route without flying. `navigate` flies it with the
original adapted PID. The default `quintic_stops` reference stops smoothly at
each shortened waypoint. Two explicit configs reproduce the source navigation
reference/follower logic:

```sh
python -m mujoco_drone navigate --config configs/warehouse_dijkstra_isaac_timed.json \
  --output mujoco_output/warehouse_timed
python -m mujoco_drone navigate --config configs/warehouse_dijkstra_isaac_waypoints.json \
  --wind unseen_gusts --seed 300000 --output mujoco_output/warehouse_feedback
```

Use `--scene blocks` for a fast, small diagnostic course. Each navigation run
saves its occupancy grid, raw colliders, Dijkstra cells, shortened waypoints,
launch audit, native contacts and clearance/voxel metrics, alongside flight
logs. Plotting adds `navigation.png` and `clearance.png`.

The original open-scene route sampler and evaluation wind profiles are also
available. This runs hover, straight, altitude, waypoints, circle and figure-eight
with `training_range`, `unseen_gusts` and `outside_range` winds:

```sh
python -m mujoco_drone suite --config configs/isaac_path_evaluation.json \
  --seeds 300000 300001 300002 --output mujoco_output/isaac_protocol
python -m mujoco_drone run --config configs/altitude.json --output mujoco_output/altitude
bash scripts/reproduce_mujoco_navigation.sh mujoco_output/navigation_reproduction
```

The X2 envelope and a 15 cm tracking reserve replace the source Iris envelope
and 1 cm margin. The source launch stand is explicitly frozen. These are
documented experiment changes, so results are not identical Isaac reproductions.
The original wind distributions can exceed the controller's ability; inspect
failures and completion as well as RMSE. See
[navigation usage and fidelity](docs/MUJOCO_NAVIGATION.md) and
[measured navigation results](docs/MUJOCO_NAVIGATION_RESULTS.md).

Supported weather: `calm`, `constant`, `gust`, `periodic`, `turbulence`,
`randomized`. JSON inputs configure velocity vectors, onset/duration, frequency,
OU standard deviations and correlation times, direction rotation, random speed
range, density, coefficient, and area. The experiment seed controls both initial
state sampling and an independent wind stream. Speeds are **m/s**; resulting
drag forces are **N**. Calm means still air with aerodynamic drag on a moving
vehicle. Setting `wind.enabled=false` disables aerodynamic forces altogether.
`turbulence` uses `mean_velocity` plus OU fluctuations; `randomized` samples a
mean horizontal speed/direction once per seeded episode. `gust` is a finite
velocity pulse and `periodic` a sinusoid around zero, both using `gust_velocity`.

```sh
python -m mujoco_drone suite --config configs/evaluation.json \
  --seeds 300000 300001 300002 --output mujoco_output/pid_suite
python -m mujoco_drone plot mujoco_output/pid_suite
```

This runs all five tasks against all six weather types, with identical initial
conditions for each seed. Use `--tasks` and `--winds` to select a smaller matrix.
The measured validation report is [docs/MUJOCO_RESULTS.md](docs/MUJOCO_RESULTS.md).
`bash scripts/reproduce_mujoco_validation.sh mujoco_output/reproduction` runs
the complete headless validation sequence. Set `DRONE_RENDER_TEST=1` to also
exercise camera rendering with macOS graphics access.

## Controllers and the original PID

`original_adapted` is the default. It retains `main.py`'s cascaded position,
velocity and attitude loops and gains, while correcting coordinate and timing
issues, converting motor corrections to SI wrenches, adding bounded integrals
and reference feedforward, and evaluating at 120 Hz. See the exact changes in
[docs/MUJOCO_DIFFERENCES.md](docs/MUJOCO_DIFFERENCES.md).

`original_literal` is a diagnostic transcription of the original feedback and
mixer, including quaternion-component feedback and integer planner velocity
assignment. It replaces wall-clock scheduling with explicit simulation time.
Use the 100 Hz replay configuration to inspect its behavior:

```sh
python -m mujoco_drone run --config configs/original_pid_replay.json \
  --output mujoco_output/original_pid
```

This diagnostic can fail even in calm conditions. It is retained to make the
changes reviewable, and is not accepted as a SAC training baseline. The original
source is preserved verbatim in `docs/reference/original_mujoco_main.py.txt`.

`isaac_pid` uses the teammate's reference/PID algorithm with inertia-based X2
attitude gains. Select it with `--pid-family isaac_pid`. It is a separate policy
family; checkpoints trained against one family are rejected by another.

## Sensors and graphics

Every flight CSV includes ground truth and the ideal simulated gyroscope and
accelerometer. The accelerometer reports **specific force at the IMU site**,
in sensor axes: approximately +9.81 m/s² when upright and supported, and zero
in free fall. It is distinct from world acceleration. Controller and policy
inputs use ground truth; no camera navigation or IMU-only localization is claimed.

```sh
python -m mujoco_drone run --config configs/camera_hover.json --output mujoco_output/camera
python -m mujoco_drone plot mujoco_output/camera
```

Camera capture needs access to macOS graphics services, even without a visible
window. RGB/depth are stored per frame in `camera/frame_*.npz`, indexed by
`camera/frames.csv`. `camera/camera.json` records intrinsics, mount, clipping
planes, resolution and units. `depth_m` is MuJoCo's metric optical-axis depth,
not radial range. Background depth approaches the far clipping plane. The
platform depth warning about `ARB_clip_control` is a precision limitation;
validation checks a known 1.4 m plane, not arbitrary long-range accuracy.

For an interactive viewer on macOS, launch with MuJoCo's `mjpython` executable:

```sh
mjpython -m mujoco_drone run --task straight --wind gust --viewer --output mujoco_output/viewer
```

The viewer follows the drone and paces simulation to wall time when computation
permits. Interactive warehouse viewing and offscreen RGB/depth capture have been
tested. A fresh MuJoCo installation supplies `mjpython`; for this development
venv's inherited installation, the verified launcher is:

```sh
.venv/bin/python /opt/anaconda3/envs/mujoco-env/bin/mjpython -m mujoco_drone \
  navigate --scene warehouse --viewer --output mujoco_output/warehouse_viewer
```

Physics and file capture can run slower than real time on a large scene.

## Train and evaluate residual SAC

```sh
python -m mujoco_drone train --config configs/sac_train.json \
  --steps 20000 --seed 0 --checkpoint-interval 10000 --output mujoco_output/sac_short
python -m mujoco_drone suite --tasks hover straight waypoints circle figure_eight \
  --winds calm gust turbulence --seeds 400000 400001 400002 \
  --model mujoco_output/sac_short/final_model.zip --output mujoco_output/paired
python -m mujoco_drone plot mujoco_output/paired
```

A model causes each suite trial to run **both PID and PID+SAC**. Every pair
shares its reference, initial state and entire prescribed wind realization.
Aerodynamic forces can differ because drag depends on the flown state.
`paired/paired.csv` and `paired/comparison.json` report measured differences,
including failures. A single `run --model ...` flies PID+SAC only.

SAC retains the source's four 32-value historical frames, four residual wrench
actions, Huber-based costs, failure penalty, and four-stage curriculum. Physics
and PID run at 120 Hz; SAC acts at 10 Hz. The X2's lower yaw authority requires a
documented uniform residual reduction to stay inside the original per-rotor
budget. Training routes use the source's legacy `TimedReference` interpolation
for experimental traceability; controlled evaluation uses analytic references.

The default full run is 2,000,000 decisions. The 20,000-decision run is a pipeline
validation, entirely inside the first curriculum stage. It does not establish
convergence, improvement over PID, or competence in later stages.

```sh
python -m mujoco_drone train --config configs/sac_train.json \
  --resume mujoco_output/sac_short/final_model.zip --steps 200000 \
  --output mujoco_output/sac_continue
```

Resume requires the matching `.replay.pkl` and `.contract.json`, identical
settings, and verified file hashes. `--steps` means additional decisions. RNG
streams restart from a new deterministic seed; resume is not bitwise identical
to uninterrupted training. Fresh training with fixed seeds is reproducible on
the tested stack; exact replay across OS/library versions is not guaranteed.

Models are saved with the simulator, physical model hashes, source hashes,
controller, observations, normalization scales, residual bounds, aerodynamic
parameters, reward/curriculum/settings, and training state. Unknown checkpoints,
including Isaac weights, are rejected. Changing even a guarded source file
requires a new compatible checkpoint. No silent fallback to PID occurs.
The navigation extension uses contract version 2: checkpoints from the earlier
open-scene implementation must be retrained. The policy still learns tracking
in open space; Dijkstra supplies the obstacle route. It does not learn a map or
perform obstacle avoidance from camera images. Navigation keeps the residual
zero during its validated launch departure by default.

The original 500,000-transition buffer occupies about 0.52 GB, and each replay
snapshot has similar disk cost. Snapshots are retained for review. Long runs
should choose an appropriate checkpoint interval and archive outputs; a smaller
`--buffer-size` is available and is recorded as a training-setting change.

## Outputs and metric definitions

Each flight saves:

* `metadata.json`: complete inputs, initial state, units/frames, model audit,
  software versions, source hashes and protocol hash.
* `flight.csv`: post-step timestamps; body-origin and COM truth; quaternion;
  velocities and accelerations; reference; IMU; requested/clipped/actual rotor
  forces; per-rotor saturation; PID and residual; each sample-point wind and drag;
  and the interval during which those controls/forces were applied.
* `reference.npz`, `metrics.json`, and optional camera frames.
* For Dijkstra, `navigation/` with the occupancy map, geometry, waypoints and
  initial/final planner audit. The CSV also records clearance, launch, occupancy
  status, envelope limits and native contact counts.
* `error.json` if execution fails. Such a run is not silently counted as a
  completed trial. A suite stops so the execution error can be investigated.

Metrics use every physics sample, including failed flights. Position RMSE is
`sqrt(mean(||p(t)-p_ref(t)||²))`. Cross-track RMSE is distance to the geometric
polyline, independent of timing; loop geometry is sampled at 501 points.
Maximum deviation is the largest time-indexed position error. Altitude RMSE and
maximum altitude error use world Z. Saturation is reported both as the fraction
of steps with any clipped rotor and the fraction of individual rotor samples
clipped. `motor_*_at_limit` is separately logged.

Completion requires reference motion to finish, survival through the scheduled
horizon and a final dwell of 0.5 s within 0.15 m at speed <=0.2 m/s. A failure is
a collision, invalid simulation, height <0.5 m, tracking error >3 m, tilt >45°,
speed >7 m/s, or angular speed >6 rad/s. A threshold failure is not automatically
called a physical crash; `crashed` specifically records detected contact.
Navigation uses a 1 m error limit and separately validates its launch exception.
For `isaac_waypoints`, error to the active feedback target is not timed tracking
lag; compare geometric cross-track and clearance between follower types.

Gust recovery is the delay after gust end until the start of a confirmed 1 s
period with position error <=0.15 m and velocity error <=0.2 m/s. `null` means no
confirmed recovery in the available horizon. Zero can mean the disturbance
never pushed the drone outside those bands; it does not mean zero tracking error.

Suites save per-flight CSVs, grouped means/sample standard deviations, completion,
failure/crash and recovery rates. Three seeds provide limited evidence, not
reliable estimates of rare failure probability.

## Compare with Isaac measurements

No matching Isaac evaluation output was available, so no cross-simulator
agreement is claimed. `compare` accepts per-flight CSVs only when protocol hashes,
seeds, controller family, and metric definitions match:

```sh
python -m mujoco_drone compare left.csv right.csv --output mujoco_output/comparison
```

An Isaac export must be instrumented to use the same analytic reference,
wind-speed definitions, initial states, controller settings and metric schema.
Export the verified protocol key from actual metadata; do not merely rename
incompatible legacy metrics or copy a hash. One mode per CSV and one row per
protocol/seed are required. Even then, the differing Iris/X2 physical parameters
must be reported, and measured differences cannot be attributed only to the
physics engine. See [docs/MUJOCO_DIFFERENCES.md](docs/MUJOCO_DIFFERENCES.md).

## Validation and code layout

```sh
python -m pytest tests_mujoco -q
MUJOCO_TEST_RENDER=1 python -m pytest tests_mujoco/test_recording.py::test_camera_rgb_depth_and_known_plane_distance -q
PYTHONPATH=drone_code python -m pytest drone_code/tests -q
```

The graphics test is opt-in because macOS must permit CoreGraphics access. The
original Isaac suite has pre-existing failures; the report identifies them and
checks the untouched `HEAD` version separately. Isaac itself was not executed.

| Module | Responsibility |
| --- | --- |
| `config.py` | Validated JSON experiment inputs |
| `physics.py` | Compiled model audit, allocation, COM wrench, stepping |
| `weather.py` | Named velocity profiles around the shared OU/drag model |
| `controllers.py`, `core.py` | Original PID adaptation, diagnostic replay, Isaac PID option, residual loop |
| `references.py` | Analytic hover, lines, smooth waypoints and curves |
| `scenes.py`, `planning.py` | Packaged static scenes, compiled geometry, shared Dijkstra and reference adapters |
| `sensors.py`, `recording.py` | Simulated IMU, RGB/depth and synchronized CSV |
| `env.py`, `training.py` | Gymnasium, SAC training, save/resume/contracts |
| `experiments.py`, `metrics.py` | Repeated experiments and evaluation |
| `comparison.py`, `plotting.py`, `cli.py` | Paired comparisons, figures and commands |

[Design/audit](docs/MUJOCO_DESIGN.md),
[implementation differences](docs/MUJOCO_DIFFERENCES.md), and
[measured results and limitations](docs/MUJOCO_RESULTS.md) describe the research
scope. Sensor-based control, dynamic obstacles, rotor dynamics,
ground effect, battery effects, identified aerodynamics, and real-flight
validation remain future work. No such results are implied by stable simulation.

The Skydio assets retain their Apache-2.0 license in
`mujoco_drone/models/skydio_x2/LICENSE` and their source provenance manifest.
The repository's existing license is retained.
