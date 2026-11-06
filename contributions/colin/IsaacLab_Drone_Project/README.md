# Colin's Isaac Lab Drone Simulator

Independent DSC180 checkpoint: conventional Lee and cascaded PID stabilization for an ARL quadrotor. This contribution is not yet integrated with the group's other simulator. No RL training occurred in these validation runs; residual PPO and wind testing are next steps.

## Main files

| File | Purpose |
|---|---|
| scripts/run_hover.py | Start a visual experiment or validation suite |
| scripts/test_lee_hover.py | Shared runner, motor allocation and logging |
| src/IsaacLab_Drone_Project/controllers/cascaded_pid.py | Position/velocity/attitude/rate control with bounded integrators |
| scripts/validate_hover.py | Matched multi-seed scenarios, reset checks and scoring |
| scripts/summarize_lee_hover.py | Summarize single-run logs for either controller |
| tests/test_cascaded_pid.py, tests/test_hover_validation.py | Controller and scoring tests |

Lee comes from isaaclab_contrib. The tasks directory contains environment and RL configurations. The earlier altitude-only controller, PPO playback helper, navigation and sensor recorder are experimental and were not validated by this checkpoint.

## Setup

Tested with Windows, RTX 4060 Laptop GPU, Python 3.12, uv and Isaac Lab revision `0c114a24a3b9e854d299d3527ada507139fc4e40`. pyproject.toml expects IsaacLab beside this project. From contributions/colin, if that dependency is not already present:

```powershell
git clone https://github.com/isaac-sim/IsaacLab.git IsaacLab
git -C IsaacLab checkout 0c114a24a3b9e854d299d3527ada507139fc4e40
cd IsaacLab_Drone_Project
uv sync --extra isaacsim --locked
```

NVIDIA runtime/driver prerequisites and remote USD assets are required. The revision and local environment were verified; a clean-machine installation has not been tested. Models, virtual environments and simulator installations are not bundled.

## Run from this project folder

```powershell
# Watch PID move 20 cm sideways and 10 cm upward.
uv run --extra isaacsim python scripts/run_hover.py --controller pid --motor-start hover --yaw-target initial --target-offset 0.2 0 0.1

# Use --controller lee for the other baseline.

# Repeat the full suite; prints PASS/FAIL and an output folder path.
uv run --extra isaacsim python -u scripts/run_hover.py --validate --duration 20 --seeds 0 1 2 --viz none

# Simulator-independent checks.
uv run --extra isaacsim python -m pytest tests/test_cascaded_pid.py tests/test_hover_validation.py -q
```

Use `--viz kit` to watch validation. Logs and summary.json are saved in a timestamped outputs/validation_* folder. The single visual run lasts five seconds; validation runs 20 seconds per episode.

## Checkpoint results

**18/18 episodes passed**: two controllers, three seeds and three scenarios (level hover, position error, combined position/attitude/velocity error). Eight targeted unit tests passed. Physics runs at 100 Hz and control at 50 Hz.

| Worst over nine episodes each | Lee | PID |
|---|---:|---:|
| Final-five-second position RMSE | 0.0000836 m | 0.0017665 m |
| Maximum tilt | 5.759 degrees | 6.734 degrees |
| Actions requiring motor clipping | 0% | 0% |

[Machine-readable results](docs/results/calm_hover_validation.json) include every episode and threshold. Passing requires full duration; altitude >=0.5 m; tilt <=45 degrees; position error <=1 m; final-five-second RMSE <=5 cm, speed <=0.1 m/s, tilt <=10 degrees and yaw error <=5 degrees; and <=1% clipped actions.

These use perfect simulator state, already-spinning motors and no wind. They do not establish hardware precision, takeoff, contact-free flight, realistic sensing or broad robustness. Hover thrust and controller state are reset for every diagnostic episode; the automatic training-reset path still needs work. The installed Isaac Lab uses XYZW quaternions, explicitly converted to WXYZ for PID.

Next: training-safe resets, bounded residual PPO, then matched wind and sensor-estimation experiments.
