# MuJoCo navigation and source experiment port

This extension runs the teammate's planner and experiment definitions with the
Skydio X2 backend. `original_adapted`, based on the user's original MuJoCo
`drone-example/main.py`, remains the default PID. No files in the original
`mujoco-project` are edited. The source Iris physics and NVIDIA runtime are not
required for MuJoCo execution.

## Implemented source workflows

| Source | MuJoCo equivalent | Fidelity |
| --- | --- | --- |
| `grid_map.py`, `mapping.py`, `navigation.py` | `planning.py` with the original shared search/map code | Original 26-neighbor Dijkstra, closed-cell corner checks, continuous collision checks, farthest-visible shortening and upward launch validation |
| `sim_scene/dij3d_boxes.usd` | Packaged `models/scenes/warehouse.json` and static MuJoCo geoms | All 3,491 authored collision boxes and one plane; frozen launch stand; visual-only assets excluded |
| `path_navigation.py` | `isaac_timed` | Original corner rounding, `TimedReference`, speed/acceleration/vertical limits and launch residual masking |
| `motor_physics.py` / `WaypointFollower` | `isaac_waypoints` | Original guided waypoint target generation, dwell and timeout; same feedback-dependent semantics |
| Source route evaluation / `path_sac_config.py` | `--protocol isaac_path` | Original full-stage sampler, six route types, wind-before-route-before-initial RNG order and evaluation wind profiles |
| Residual path SAC | Existing training and `--model` on navigation | Same observation/action interface and open-space curriculum; fresh MuJoCo policy required |
| Existing MuJoCo diagnostic experiments | `quintic_stops`, controlled routes and weather | Smooth analytic references for a stable, separately labeled baseline |

The shared Isaac modules only gain package imports and optional explicit map,
margin and search-budget parameters. Their defaults retain the original Isaac
behavior. Dijkstra itself is unchanged. The pre-existing original follower test
whose maximum sloped-route error exceeds its 1 cm reserve remains a documented
source issue; it has not been disguised by changing its assertion.

## Run, inspect and visualize

From the repository with `.[rl,plot,test]` installed:

```sh
python -m mujoco_drone inspect --scene warehouse
python -m mujoco_drone plan --scene warehouse --output mujoco_output/plan
python -m mujoco_drone plot mujoco_output/plan
python -m mujoco_drone navigate --scene warehouse --output mujoco_output/flight
python -m mujoco_drone plot mujoco_output/flight
```

Use new output paths on every invocation. On macOS use `mjpython` for `--viewer`;
the exact development-environment launcher is in the README. The viewer tracks
the drone and waits to maintain real time when computation permits. Headless
flights do not require graphics. Camera capture does require graphics services.

Choose a source reference mode:

```sh
python -m mujoco_drone navigate --config configs/warehouse_dijkstra_isaac_timed.json \
  --wind training_range --seed 300000 --output mujoco_output/timed
python -m mujoco_drone navigate --config configs/warehouse_dijkstra_isaac_waypoints.json \
  --wind unseen_gusts --seed 300000 --output mujoco_output/feedback
```

`--scene blocks` provides a small diagnostic scene; corresponding configs exist
for all three reference modes. `quintic_stops` is the default: it stops at each
shortened waypoint using analytic position/velocity/acceleration and respects
both total and vertical speed limits. It changes reference timing relative to
the source, which is why modes are recorded in metadata and protocol hashes.
`isaac_timed` preserves the original piecewise timed reference, including its
interpolation behavior. `isaac_waypoints` preserves the original feedback
follower and runs to the configured horizon, including final hover. Its command
can jump to a distant waypoint, so large active-target errors are normal. The
1 m navigation failure limit uses nearest-route deviation for that mode; timed
modes use error to their time-indexed reference. Native contact and other
physical failure limits apply to every mode.

The source open-route protocol intentionally draws its own geometry and initial
state from the original full-stage distributions. `task.start`, `goal`, speed
and initial-range overrides apply to controlled routes/navigation, not these
sampled open routes. `max_seconds` still limits execution. To change source route
distributions, edit the source experiment definitions and retrain policies whose
contracts guard them.

Dijkstra plans once from the known static map before flight. Wind changes the
executed trajectory, not the map. There is no online replanning, local obstacle
avoidance, camera mapping or IMU-only localization in this experiment.

```sh
python -m mujoco_drone suite --config configs/isaac_path_evaluation.json \
  --seeds 300000 300001 300002 --output mujoco_output/source_protocol
```

Profiles retain source definitions: `training_range` draws 2–7 m/s mean wind,
`unseen_gusts` draws 2–6 m/s with stronger/faster gusts, and `outside_range` draws
8–10 m/s mean wind. Fluctuations can exceed those mean speeds. These names refer
to the original full training distribution; a 20,000-decision smoke policy only
experiences the first, milder curriculum stage. Runtime metadata records the
actual seeded mean, gust standard deviations, time constants and rotation rate.

## Geometry, launch and safety interpretation

The manifest is generated from the flattened warehouse USD with OpenUSD, without
starting Isaac. The exporter validates metre units, Z-up, static axis-aligned
colliders and box-shaped support meshes. It rejects dynamic rigid obstacles
unless an exact support path is explicitly frozen. `/World/Cube` is authored as
dynamic, so its conversion is an explicit, recorded approximation. Regeneration
is optional; ordinary users do not need OpenUSD:

```sh
python -m pip install -e '.[usd]'
python scripts/export_isaac_warehouse.py --freeze-support /World/Cube \
  --output /tmp/warehouse_review.json
```

The source USD and saved initial-pose SHA-256 values are retained in the manifest.
Exported bounds were checked against all 3,491 source occupancy reports: the
maximum difference is 2.54 micrometres at the dynamic stand and under 0.05
micrometres elsewhere. These small differences reflect authored versus saved
runtime transforms. Runtime mapping reads compiled MuJoCo geoms, so planning
and physics see the same converted scene.

The drone envelope is derived from the X2's actual collision geoms about its
body origin. It includes arbitrary yaw and the configured 10-degree tilt
allowance: approximately `[0.42672, 0.42672, 0.16139]` m half extents. Obstacles
are expanded by that envelope plus an explicit 0.15 m tracking reserve. The
source 0.01 m reserve remains configurable, but is not the baseline here. The
grid remains 0.25 m; any voxel intersecting an expanded obstacle is blocked.
Consequently a blocked voxel can be entered while continuous geometric
clearance still exceeds the reserve. These are separate measurements.

The saved source body-origin start `[-2.30, 0, 0.12845]` and goal
`[-4.30, 20, 1.2]` m are retained. With the different X2 body geometry, its
lowest point starts about 4 cm above the small support; this is an upward
departure from the saved launch pose, not an Iris ground-contact reproduction.
The planner independently validates the vertical exit. Only verified named
support geoms and the supporting floor are exempt from contact failure during
that departure. Other contacts still fail the flight. The normal minimum-height
rule resumes after launch. Landing and moving obstacles are unsupported.

Native MuJoCo contacts detect physical collision. Signed AABB clearance measures
distance to the conservative vehicle envelope, excluding the added reserve;
negative values mean envelope overlap, which need not equal native contact.
Clearance and occupancy violation rates exclude the validated launch interval;
launch samples remain logged. Tilt above the envelope's allowance is reported
separately and invalidates reliance on that nominal envelope. Map bounds, grid
cell budget, search expansion budget and blocked destinations fail explicitly;
the implementation does not silently clear cells or move the requested goal.

## Results, SAC and reproducibility

See [measured results](MUJOCO_NAVIGATION_RESULTS.md). Planning-only output never
claims flight success. Flight output adds `navigation/occupancy_grid.npz`,
`path_cells.csv`, `waypoints_world.csv`, `collision_geometry.json`,
`occupancy_obstacles.txt`, `plan.json` and `final_navigation.json`. The normal
flight log includes occupancy/clearance, launch, tilt-envelope and native-contact
fields. Scientific plots show the route, reference, executed trajectory and
clearance. Tiny boxes and the high roof may be hidden in the plot for legibility;
all colliders remain active in planning and physics.

The navigation update increments policy contracts to version 2. Existing
checkpoints are rejected if their guarded code/model differs. Training remains
open-space residual tracking, with no camera or obstacle observations. Dijkstra
provides the route and the SAC residual is zero during launch by default. An
execution check with short retraining is not evidence of improved navigation or
SAC convergence. No teammate Isaac-trained checkpoint or matching measurement
archive was available; no cross-simulator performance equivalence is claimed.

```sh
bash scripts/reproduce_mujoco_navigation.sh mujoco_output/reproduced_navigation
```

This runs tests, three reference modes on the warehouse and compact scenes,
the six original open-route tasks under three original wind profiles, a fresh
20,000-decision SAC execution check and paired navigation evaluation. Raw outputs
are ignored by Git; small measured summaries and input hashes are saved for
review in `results/2026-11-06/`. The earlier November 5 report remains historical.
