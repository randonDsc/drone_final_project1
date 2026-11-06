# Longer calm-flight validation

This suite tests Lee and PID across repeated resets and controlled starting errors. It uses the existing gains, the ARL drone, 100 Hz physics, 50 Hz control, ground-truth state, and 0–6 N motor command limits. No RL training or wind is introduced.

## Reproduce in PowerShell

```powershell
cd C:\Users\colin\Desktop\IsaacLab_Drone_Project
uv run --extra isaacsim python -u scripts/test_lee_hover.py --validate --duration 20 --seeds 0 1 2 --viz none
```

This creates a unique `outputs/validation_DATE_TIME` directory. Read the path printed at completion. To use a chosen directory, add `--output-dir outputs/my_validation`; choose a new directory for each run. A suite with an existing summary is not overwritten. Use `--viz kit` instead of `--viz none` to watch the suite. The camera is disabled for validation because neither controller consumes images; this avoids unnecessary RTX sensor work. The ordinary single-episode command still retains its camera.

For a faster six-episode check use `--seeds 0 --duration 10`. For longer validation use `--duration 30`; unlike `--steps` in the original diagnostic, this also sets the validation environment's episode duration. Suite mode always tests both controllers, initializes motors at hover thrust, holds target (0,0,1.5) and yaw zero, and ignores single-run controller/target/startup selections.

```powershell
# Substitute the directory printed by your run.
$validation = Get-Content outputs/my_validation/summary.json -Raw | ConvertFrom-Json
$validation.results | Format-Table controller,seed,scenario,passed,tail_rmse_m,max_tilt_deg,saturation_fraction,failure

.venv/Scripts/python.exe -m pytest tests/test_cascaded_pid.py tests/test_hover_validation.py -q
```

## Method

There are 18 episodes: 2 controllers x 3 seeds x 3 scenarios, each lasting 20 simulated seconds.

- `calm`: level at the target, zero velocity.
- `position`: random initial X/Y offsets within +/-0.25 m and Z offset within +/-0.15 m.
- `perturbed`: the same position ranges plus roll/pitch within +/-5 degrees, yaw within +/-15 degrees, linear velocities within +/-0.2 m/s per world axis, and angular velocities within +/-0.1 rad/s per world axis.

Each controller/scenario/seed pair resets the simulator with the same seed. The suite asserts exact matching of observed initial position, orientation and velocities, as well as sampled motor coefficients and rise/fall time constants. The nominal target remains fixed; it is not moved to each randomized start.

After EVERY explicit episode reset, PID integrators, previous-velocity history, and saturation state are cleared. The suite asserts these are empty before use. Hover motor thrust is reapplied after reset and total thrust is checked against weight. This fixes initialization across episodes in this diagnostic harness; it does not repair Isaac Lab's actuator initialization or the RL environment's automatic reset path.

The environment timeout is placed one second beyond the measurement window. The runner stops after exactly the requested number of actions, captures the final state, and never treats an auto-reset state as a successful ending. Unexpected environment termination is a failed episode. All trajectories and motor commands are recorded in per-episode JSONL files; summary.json is updated after each completed episode. If interrupted, check the number of entries: the default complete suite has 18.

You can also run PID against saved Lee reference logs without repeating completed Lee trials:

```powershell
uv run --extra isaacsim python -u scripts/test_lee_hover.py --validate --validation-controllers pid --reference-dir outputs/validation_calm_20s --duration 20 --seeds 0 1 2 --viz none --output-dir outputs/new_pid_validation
```

This checks reference timing, duration, target, and exact initial-state/motor-parameter equality. The output has nine PID episodes; interpret it alongside the nine Lee episodes in the reference directory.

## Provisional acceptance criteria (chosen before the runs)

All conditions must pass:

| Condition | Threshold |
|---|---:|
| Complete measured duration without unexpected reset | 20 s |
| Minimum sampled root altitude | >= 0.5 m |
| Maximum sampled tilt | <= 45 degrees |
| Maximum sampled 3D position error | <= 1 m |
| Final-five-second position RMSE | <= 0.05 m |
| Final-five-second maximum speed | <= 0.1 m/s |
| Final-five-second maximum tilt | <= 10 degrees |
| Final-five-second maximum absolute yaw error | <= 5 degrees |
| Actions with any clipped motor | <= 1% |

State is sampled at the control rate, so extrema between actions can be missed. The altitude guard detects major loss of clearance; it is not a contact-sensor collision test. These are modest calm-flight tests with pre-spun motors and perfect state, not certification of wind performance, real-world stability, sensor estimation, takeoff, or inverted recovery. A failed accuracy criterion is kept distinct from a failure that stops an episode early.

## Files changed

- `scripts/test_lee_hover.py`: validation CLI, longer test-specific episode duration, camera disable in validation mode, and suite dispatch.
- `scripts/validate_hover.py`: matched scenarios, per-episode resets, initialization checks, trajectory logging and pass/fail scoring.
- `tests/test_hover_validation.py`: ensures incomplete/reset, inaccurate and saturated trajectories cannot pass scoring.

Suggested commit message: `Add repeatable multi-seed validation for PID and Lee baselines`
