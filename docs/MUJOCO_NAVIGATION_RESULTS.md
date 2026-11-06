# Measured navigation extension — 2026-11-06

The original 3D Dijkstra pipeline and warehouse collision environment now run in
MuJoCo using the original adapted `main.py` PID. The validation below contains
117 scheduled flights, plus separate development, unit and graphics checks.
All scheduled outcomes, including failures and incomplete flights, are retained.
This is a port of source algorithms/experiment definitions onto a different
vehicle and physics engine, not measured agreement with Isaac Sim.

Small review artifacts and raw-input hashes are in
[results/2026-11-06](../results/2026-11-06/). Full flight CSVs, occupancy grids,
references, camera arrays and checkpoints remain in ignored local
`mujoco_output/`. The implementation is on `alan/mujoco`; it has not been
committed or pushed as part of this extension. The earlier November 5 report
remains a historical open-scene baseline.

## What was reproduced

The packaged warehouse contains all 3,491 collision boxes and one infinite
ground plane from `sim_scene/dij3d_boxes.usd`. Its authored dynamic launch stand
is explicitly frozen and reported in provenance. The original search,
no-corner-cutting checks, continuous visibility checks, shortening and launch
entry logic are shared, rather than reimplemented. Source `TimedReference`,
corner rounding and adaptive `WaypointFollower` are available alongside a
separate analytic stop-at-waypoint baseline.

The original saved start/goal are retained. X2 collision geoms produce a
conservative yaw/tilt envelope of `[0.42672443, 0.42672443, 0.16138609]` m half
extents. The grid is 0.25 m and the added tracking reserve is 0.15 m. These differ
from the source Iris envelope and 0.01 m reserve. The stand/vehicle difference
also leaves the X2 about 4 cm above the authored support at initialization.
See [fidelity and launch details](MUJOCO_NAVIGATION.md).

The resulting warehouse plan has a `105 × 164 × 55` grid, 314,590 blocked cells,
521,941 expanded Dijkstra nodes, a 45.862 m grid route and a 42.856 m shortened
route with ten points including the vertical departure. The path climbs to
6.875 m before descending to the 1.2 m destination. Planning took about 3.3 s
in representative runs on this machine; this is not a controlled speed benchmark.

## Warehouse PID results: 36 flights

Every cell below uses seeds 300000–300002. Navigation initial perturbations are
zero, so the three calm seeds repeat the same deterministic flight. Windy seeds
draw independent weather. Reference horizons are about 110.72 s for smooth
stops, 62.92 s for the original timed reference, and a fixed 240 s for the
feedback follower, including final hover. These timing differences prevent
treating aggregate tracking RMSE as a controlled comparison between modes.

| Reference mode | Calm completed | Training range completed | Unseen gusts completed | Outside range completed | Physical collisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `quintic_stops` | 3/3 | 3/3 | 3/3 | 1/3 | 0/12 |
| `isaac_timed` | 3/3 | 3/3 | 3/3 | 0/3 | 0/12 |
| `isaac_waypoints` | 3/3 | 3/3 | 3/3 | 0/3 | 0/12 |

The 27 calm/training-range/unseen-gust flights all completed and maintained the
continuous 15 cm reserve. Some entered conservatively blocked voxels: voxel
occupancy includes cell-volume inflation, so this can happen while continuous
clearance still exceeds the reserve. No tested flight exceeded the envelope's
10-degree tilt allowance or saturated a rotor.

| Calm reference | Active-target position RMSE | Geometric cross-track RMSE | Minimum envelope clearance | Simulated horizon |
| --- | ---: | ---: | ---: | ---: |
| Smooth stops | 0.01433 m | 0.00602 m | 0.30588 m | 110.725 s |
| Original timed | 0.02221 m | 0.01608 m | 0.30767 m | 62.925 s |
| Original feedback follower | 0.40915 m | 0.14090 m | 0.33073 m | 240 s |

For timed modes, position RMSE is error to a scheduled reference. The feedback
follower commands distant active waypoints or sliding targets, so its position
RMSE measures error to those commands, not tracking lag. Its larger geometric
cross-track error is a real difference; it briefly enters blocked voxels even
in calm air. Keep these metrics distinct when discussing the original code.

Outside-range winds draw 8–10 m/s mean speed plus fluctuations. Across the nine
warehouse cases, one completed, two terminated for the 1 m navigation error
limit, and six survived without meeting final completion. Minimum conservative
clearance fell to 0.10282 m, 0.11378 m and 0.11005 m for smooth, timed and feedback
modes respectively. This consumes the 15 cm reserve despite no physical contact.
These results do not establish robust navigation in that wind range.

## Compact course: 9 flights

Each reference mode ran calm, finite gust and turbulence at seed 300000. All
nine flights completed, with no native contact or continuous reserve violation.
The route is 7.276 m around three local diagnostic obstacles. Mean position
RMSE was 0.01667 m for smooth stops and 0.02870 m for the original timed mode.
The feedback follower had 0.07409 m mean geometric cross-track RMSE and about
1.06% blocked-voxel occupancy despite at least 0.22576 m continuous clearance.
These are diagnostic checks, not a multi-seed reliability estimate.

## Original open-scene protocol: 54 flights

The six source route types ran at seeds 300000–300002 under each original
evaluation profile. Wind sampling precedes route sampling and initial-state
sampling, exactly as in the source evaluation code. Tests independently recreate
that RNG sequence and match paths, initial states and wind parameters.

| Wind profile | Flights | Completed | Tracking-limit failures | Survived without final completion | Mean position RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `training_range` | 18 | 18 | 0 | 0 | 0.08117 m |
| `unseen_gusts` | 18 | 18 | 0 | 0 | 0.08665 m |
| `outside_range` | 18 | 2 | 5 | 11 | 0.73563 m |

No native collisions or rotor saturation occurred. The source failure label
`out_of_bounds` means tracking error beyond the configured limit, not a physical
crash or navigation map exit. The open-route limit is 3 m; navigation uses 1 m.
Open-route geometry, random initial states and timing differ from the controlled
November 5 experiments, so their RMSE values are not interchangeable.

## Fresh short SAC check: 18 paired flights

Contract version 2 rejects the older backend's checkpoints. A fresh CPU SAC run
used `configs/sac_train.json`, seed 0, 20,000 decisions, 15,000 gradient updates,
the original 500,000-transition buffer and a 20,000-decision checkpoint interval.
The monitor saved 240 episode records (19,964 decisions plus the unfinished
last episode). Every decision lies in the first curriculum stage; this is a
training/save/load/navigation integration check, not convergence evidence.

Nine PID/PID+SAC pairs used the original timed warehouse route, seeds
400000–400002 and calm/unseen-gust/outside-range profiles. Each pair has identical
initial state, planned reference, prescribed weather and protocol hash. Drag
forces can differ because each flown state differs. The residual stays zero
during launch. No camera or obstacle observations are supplied to SAC.

| Wind | PID mean position RMSE | PID+SAC mean position RMSE | PID completed | PID+SAC completed |
| --- | ---: | ---: | ---: | ---: |
| Calm | 0.02221 m | 0.02052 m | 3/3 | 3/3 |
| Unseen gusts | 0.07226 m | 0.07310 m | 3/3 | 3/3 |
| Outside range | 0.31995 m | 0.31976 m | 1/3 | 1/3 |

Across all nine pairs, mean position RMSE changed by **−0.000347 m**, while mean
geometric cross-track RMSE changed by **+0.004593 m** and altitude RMSE by
**+0.001287 m**. Both modes completed 7/9; both had two tracking-limit failures
and no native collisions. Outside-range minimum clearance was 0.03954 m for PID
and 0.02656 m for PID+SAC, below the 0.15 m reserve in both cases. A small
time-indexed tracking improvement in one condition is not evidence of better
overall navigation. Failed flights also have shorter horizons, which complicates
cross-condition RMSE averages. The calm seeds repeat an identical realization.

## Verification

* MuJoCo suite: **43 passed, 1 graphics test skipped** in headless mode.
* Explicit graphics calibration: **1 passed**, verifying a known 1.4 m plane.
* Three-second warehouse viewer/camera run completed its requested horizon
  without failure and saved 16 finite RGB/depth frames. It intentionally did
  not run long enough to complete the mission.
* The built wheel ran a warehouse plan and short physics flight outside the
  checkout, with all 3,491 boxes and **no `pxr` import**.
* A saved 20-decision pre-extension open-flight trace (random residual actions,
  seeded turbulence, 2 s horizon) matches all 2,840 values **exactly**, maximum
  absolute difference 0.0 on this stack.
* All six original MuJoCo input-file hashes remain unchanged.
* Original navigation tests: **34 passed, 1 failed**. The identical failure was
  reproduced from untouched `HEAD`: source guided-follower sloped deviation
  1.16485 m versus its 0.01 m reserve. This is retained as a source limitation.

Run `bash scripts/reproduce_mujoco_navigation.sh <new-output-directory>` for the
headless sequence. Graphics tests require macOS graphics access. Software
versions and every raw-flight input hash are retained with the review artifacts.
Three windy seeds do not estimate rare failure probability, and the model still
uses uncalibrated drag assumptions, ideal sensors and simplified motor physics.
No original Isaac SAC measurement archive was available for numerical comparison.
