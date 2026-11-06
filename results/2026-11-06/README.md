# Navigation extension review artifacts

Measured local outputs for 2026-11-06, using the Skydio X2 and `original_adapted`
PID. See [the results report](../../docs/MUJOCO_NAVIGATION_RESULTS.md) for
interpretation, failures and limitations. These measurements do not establish
equivalence to Isaac Sim or a converged SAC policy.

| Directory | Recorded flights |
| --- | ---: |
| `warehouse_quintic_stops` | 12 |
| `warehouse_isaac_timed` | 12 |
| `warehouse_isaac_waypoints` | 12 |
| Three `blocks_*` directories | 9 total |
| `isaac_protocol` | 54 |
| `navigation_paired` | 18: nine PID/PID+SAC pairs |
| **Total scheduled** | **117** |

Each suite includes per-flight metrics, grouped summaries, configurations and
a summary figure. Navigation suites also include a calm example with metadata,
planner audit, waypoints, flight/route figures and clearance curves.
`navigation_paired/paired/` contains the strict paired tracking comparison.
`sac_short/` contains actual training settings, updates and episode records.
`graphics/` records the separate short viewer/camera check; it is not a completed
warehouse mission. `tests/` includes the headless/render checks and the identical
pre-existing original-navigation test failure from the working tree and untouched
HEAD. `verification.json` records numerical regression, packaging and input-file
checks. The saved regression trace is a small measured baseline on this exact
software stack, rather than a portable bitwise assertion across library versions.

`ARTIFACTS.json` retains SHA-256 values and local source paths for every copied
artifact, all 117 complete raw-flight input sets, and the actual final checkpoint,
contract and replay buffer. Large raw flight CSVs, grids, model weights and replay
snapshots are deliberately retained under ignored `mujoco_output/` locally.

Reproduce the headless sequence with a new output directory:

```sh
bash scripts/reproduce_mujoco_navigation.sh mujoco_output/navigation_reproduced
```

Collect small outputs after the full sequence finishes:

```sh
python scripts/collect_navigation_results.py \
  --validation mujoco_output/navigation_reproduced \
  --isaac mujoco_output/navigation_reproduced/isaac_protocol_pid \
  --paired mujoco_output/navigation_reproduced/navigation_paired \
  --training mujoco_output/navigation_reproduced/sac_short \
  --output results/navigation_reproduced
```

The extra graphics, baseline-regression and packaging verification records were
collected separately. Their exact local inputs are identified in the manifest.
