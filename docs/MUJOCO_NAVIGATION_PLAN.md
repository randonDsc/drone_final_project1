# Navigation extension — implementation and validation plan

Starting point: clean `alan/mujoco` at `d067015`. The source experiment is the
teammate's 26-connected 3D Dijkstra pipeline in `navigation.py`, `mapping.py`,
`path_navigation.py`, `motor_physics.py` and `path_sac_config.NAVIGATION`.

## Architecture

1. Reuse the NumPy grid, no-corner-cutting Dijkstra, continuous line checks,
   path shortening and launch-entry validation. Add package imports and explicit
   configuration arguments with the original defaults retained for Isaac.
2. Convert the existing flattened `dij3d_boxes.usd` collision geometry to a
   checked-in scene manifest using OpenUSD, without starting Isaac. Runtime
   MuJoCo scene construction uses that manifest and the existing X2 assets.
   Include a small obstacle course for fast diagnosis.
3. Read the compiled MuJoCo collision geoms to build the map. Derive the X2
   envelope from its actual collision bounds, with yaw/tilt allowance and a
   configurable tracking reserve. Preserve original obstacle/plane identities.
4. Run the planned route through smooth stop-at-waypoint references, the source
   timed reference with collision-checked rounding, or the original adaptive
   WaypointFollower. Record reference semantics; the adaptive target is feedback
   dependent and its target-error RMSE is not a time-scheduled tracking metric.
5. Share physics, wind, PID/SAC, sensors and logging. Add native contact checks,
   continuous clearance, voxel occupancy violations, launch status, full map and
   route artifacts, and 3D obstacle/trajectory figures. A validated launch support
   may contact the drone during departure; unrelated contacts are never exempt.
6. Add the original open-scene evaluation protocol: original route sampler,
   initial-state draw order and `training_range`, `unseen_gusts`, `outside_range`
   wind definitions. This is an experiment-definition port, not measured
   equivalence between different vehicles or physics engines.
7. Validate planning independently, then fly calm cases, seeded wind cases,
   contact/failure cases and PID/SAC comparisons. Recheck the existing calm/wind
   backend against the saved pre-extension state trace.

## Assumptions and limits

Only static collision environments are supported. Rotated finite primitives are
mapped conservatively by AABBs; tilted infinite planes and dynamic obstacles are
rejected. The warehouse source already represents colliders as axis-aligned
boxes, so its conversion does not require visual meshes or NVIDIA assets.
The Skydio replaces the Iris: its mass, envelope, gains and actuation differ.
The source saved start/goal are retained where configured, and the resulting
launch is audited rather than assumed to be identical ground contact.

The source 1 cm planner margin is smaller than its documented flight tracking
errors. New baseline configs use an explicit 15 cm tracking reserve; the original
value remains selectable and differences are reported. A safe nominal path is
not a guarantee that a disturbed flight remains safe.

Physics/environment integration changes guarded source files. Old SAC contracts
will continue to reject incompatible source; no existing policy is silently
blessed. A fresh short training run can validate PID+SAC execution on the new
backend, but cannot establish convergence or obstacle-aware learning. No original
MuJoCo files or `main` branch are modified, and no new changes are pushed without
review.
