# Iris Flight Rebuild v4 — physics-first isolation

Rebuilt from the user's **previously successful** `iris_flight_v2.py`, with optional sensors and carefully isolated scene settings. This is a diagnostic/recovery tool, **not a claimed successful 100 m flight**. It must be run in Isaac Sim 6.2 on the user's Windows machine.

## Why this version is different

The v3 log reported a nearly five-meter drop in just 0.092 seconds, followed by an upward velocity near the ground. This is inconsistent with simple gravity and indicates possible USD/PhysX pose discontinuity, collision, or invalid initialization. It is not enough to conclude the motor controller is underpowered. The code is deliberately reset to the previous working sequence:

1. Open personal USD; move root to requested height **before** physics world creation.
2. Create optional sensor prims, but do not warm up the camera with simulation steps.
3. Create world, call `world.reset()` **exactly once**, set body default state, call `post_reset()`.
4. Create rigid-body mass views and motor mixer exactly as before.
5. Apply motor forces on **every first and subsequent physics step**; render at 120 Hz during isolation.
6. Print the state before and after each of the first 15 steps, and flag an implausible pose jump immediately.

## Files to install

Extract ZIP to `C:\Users\tucla\Desktop\DSC180A\my_drone_sim`, preserving the `scripts/` and `configs/` directories. Do NOT overwrite the existing `controller.py`, `motor.py`, `drone_diagnostics.py`, or USD scene.

- `scripts/iris_rebuild_v4.py` (new)
- `configs/flight_rebuild_v4.json` (new)
- `scripts/scene_presets.py` (same existing preset helper, provided in case missing)

## PowerShell commands

Run from `C:\Users\tucla\Desktop\DSC180A`:

```powershell
$isaac = "C:\isaacsim\isaac-sim-standalone-6.2.0-windows-x86_64\python.bat"

# Test A: known baseline altitude (no sensors, no new geometry)
& $isaac ".\my_drone_sim\scripts\iris_rebuild_v4.py" --mode base --env empty --height 1.5 --distance 0 --duration 3

# Test B: new 5-m altitude (no sensors)
& $isaac ".\my_drone_sim\scripts\iris_rebuild_v4.py" --mode base --env empty --height 5 --distance 0 --duration 3

# Only if A and B hover successfully:
& $isaac ".\my_drone_sim\scripts\iris_rebuild_v4.py" --mode imu --env empty --height 5 --distance 0 --duration 3
& $isaac ".\my_drone_sim\scripts\iris_rebuild_v4.py" --mode camera --env empty --height 5 --distance 0 --duration 3
& $isaac ".\my_drone_sim\scripts\iris_rebuild_v4.py" --mode both --env empty --height 5 --distance 0 --duration 5

# Only after all the tests are successful:
& $isaac ".\my_drone_sim\scripts\iris_rebuild_v4.py" --mode both --env empty --height 5 --distance 10 --duration 45
& $isaac ".\my_drone_sim\scripts\iris_rebuild_v4.py" --mode both --env empty --height 5 --distance 100 --duration 180

# Then try selectable environments (no new learning/obstacle avoidance)
& $isaac ".\my_drone_sim\scripts\iris_rebuild_v4.py" --mode both --env warehouse --height 5 --distance 10 --duration 45
& $isaac ".\my_drone_sim\scripts\iris_rebuild_v4.py" --mode both --env outdoor --height 5 --distance 10 --duration 45
```

For faster rendering after physics is proven stable, add `--fast-render` (10 Hz vs 120 Hz). Do not use `--fast-render` for initial diagnosis; keeping rendering the same as the old working script removes one variable.

## Results

Each run creates exactly two new user data files:

- `results/rebuild_MODE_ENV_TIMESTAMP/ground_truth.csv`: physics body position, quaternion, velocity, angular rate, thrust, waypoint, etc.
- `results/rebuild_MODE_ENV_TIMESTAMP/sensors.sqlite`: IMU rows, RGB frames compressed with zlib, metadata. With `--mode base`, sensor tables are present but empty; with `--mode imu`, only IMU has rows; with `--mode both`, both have rows.

Sampling: physics 120 Hz, truth 60 Hz, IMU 60 Hz, RGB 10 FPS at 320x240. Ground truth and sensor time aligned by `physics_step` and `sim_time_s`.

## Interpretation

- `[END] status=hover_completed`: hover flight remained near requested height/low speed for full test duration.
- `[END] status=arrived_and_held`: target reached with low speed for 2 s.
- `[ROOT-CAUSE] PHYSICS_POSE_JUMP`: body moved >0.18 m in one physics tick during first second. Likely a stage/tensor/collision discontinuity, not normal thrust or gravity. Inspect the **first** `[BEFORE]` and `[AFTER]` difference.
- `fell_below_safe_altitude`: gravitational/attitude/control failure; use short-step `[BEFORE]` / `[AFTER]` data.
- `RGB=0` after a flight shorter than 0.1 s is NOT camera failure, and is not raised as an exception.

### If baseline already fails

Do NOT proceed to sensors. Share the first 15 `[BEFORE]` and `[AFTER]` lines from the no-sensor test and the `[ROOT-CAUSE]` line if present. That tells us whether physics state is stable before adding more systems. If the 1.5 m baseline passes but 5 m fails, likely an initial pose/scene issue. If the 5 m baseline passes but IMU or camera breaks, sensor attachment/initialization is implicated.

The file has been syntax-checked locally but **cannot be executed in this environment's Isaac Sim**. Success is determined only by the Windows test results.
