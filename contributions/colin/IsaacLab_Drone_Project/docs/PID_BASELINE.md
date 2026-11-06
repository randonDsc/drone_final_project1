# PID baseline and reproduction

The controller in `src/IsaacLab_Drone_Project/controllers/cascaded_pid.py` implements position P, velocity PID, attitude P, and body-rate PI. It outputs a body-frame wrench to the same allocation matrix and 0–6 N command limits as Lee. Velocity differentiation uses measured velocity and a low-pass filter. Integrators are bounded and freeze on the step following motor clipping. This conservative anti-windup does not handle every intermediate limit; aggressive commands need further validation.

This is an independent near-hover controller, not a PX4 port. It uses Euler attitude errors and is not intended for inverted recovery. No RL training is required. Parameters are in `CascadedPIDCfg` and saved with every run.

## Run in PowerShell

```powershell
cd C:\Users\colin\Desktop\IsaacLab_Drone_Project

# Exact hover, motors already spinning
uv run --extra isaacsim python scripts/test_lee_hover.py --controller pid --motor-start hover --yaw-target initial

# Nontrivial tracking: move 20 cm in X and 10 cm up
uv run --extra isaacsim python scripts/test_lee_hover.py --controller pid --motor-start hover --yaw-target initial --target-offset 0.2 0 0.1 --log outputs/pid_comparison.jsonl

# Matched Lee run
uv run --extra isaacsim python scripts/test_lee_hover.py --controller lee --motor-start hover --yaw-target initial --target-offset 0.2 0 0.1 --log outputs/lee_comparison.jsonl

.venv/Scripts/python.exe scripts/summarize_lee_hover.py outputs/pid_comparison.jsonl outputs/lee_comparison.jsonl
.venv/Scripts/python.exe -m pytest tests/test_cascaded_pid.py -q
```

The simulator opens a window. Expect movement toward (0.2, 0, 1.6) from (0, 0, 1.5), with small tilt and a normal timeout at five seconds. The console reports state every 25 actions. JSONL logs contain per-action state, requested wrench, raw and clipped thrusts, actuator thrust, and termination flags. Explicit log paths are overwritten on rerun; omit `--log` for unique timestamped filenames.

Defaults are seed 0, 100 Hz physics, 50 Hz control, and 250 actions. Both controllers use the same sensor-free simulator state. `--motor-start default` retains the original startup deficit; `hover` is a first-episode diagnostic initialization, not a fix to library reset behavior. The environment still times out after five seconds even if `--steps` is increased. Timeout alone is not a successful-hover criterion because the crash height is -3 m.

## Verified result

Matched corrected five-second offset tests (seed 0, starting yaw 0):

| Metric | PID | Lee |
|---|---:|---:|
| 3D position RMSE, including initial error | 0.08454 m | 0.08691 m |
| Maximum tilt | 3.308 degrees | 2.208 degrees |
| Actions requiring clipping | 0% | 0% |
| Last sampled position | (0.199356, -0.000071, 1.600164) m | (0.201372, -0.000001, 1.597860) m |

Both ended by timeout. Logs are outputs/pid_corrected_offset.jsonl and outputs/lee_corrected_offset.jsonl. Do not rank controller robustness from this single seed and maneuver.

After correcting quaternion conversion, the PID offset run completed 250 actions: 3D position RMSE 0.08454 m including the initial error, altitude 1.500–1.60412 m, max tilt 3.30845 degrees, zero clipped actions. Last pre-action position was (0.199356, -0.0000711, 1.600164) m. Three simulator-free tests passed (force direction/equilibrium, saturation/reset, yaw wrapping). This is one short seeded trial, not a robustness claim.

## Quaternion correction

The locally installed Isaac Lab math uses XYZW quaternions. The previous handoff and script assumed WXYZ. PID takes WXYZ explicitly, so the runner now reorders state at that boundary. Initial-yaw configuration and console tilt use XYZW correctly; new logs store true WXYZ with explicit metadata. The summary script recognizes legacy logs lacking that metadata and interprets their mislabeled arrays as XYZW.

Consequently, earlier statements that the drone started at 180-degree yaw were incorrect: [0,0,0,1] is identity here. Previously reported tilt values must be recomputed. Position and thrust metrics are unaffected. Initial failed PID trials are retained in outputs for traceability; they used the wrong orientation interpretation and are not valid PID performance comparisons.
