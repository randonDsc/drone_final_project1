# Colin's Drone Simulator — Isaac Lab

An independent DSC180 contribution for testing drone stabilization. This checkpoint is not yet integrated with the group's other simulator.

## Current status

- Classical **Lee** and **cascaded PID** controllers; neither requires RL training.
- **18/18 calm-flight validation episodes passed**: 20 seconds each, three seeds, three starting conditions, both controllers.
- Ground-truth simulator state and motors already spinning at hover thrust. Wind, estimated-state control, residual RL and hardware validation remain future work.

## Start here

Run from this project folder after following [setup](docs/SETUP.md):

```powershell
# Watch PID recover to a nearby target.
uv run --extra isaacsim python scripts/run_hover.py --controller pid --motor-start hover --yaw-target initial --target-offset 0.2 0 0.1

# Use --controller lee for the other baseline.

# Full numerical validation, without a viewport.
uv run --extra isaacsim python -u scripts/run_hover.py --validate --duration 20 --seeds 0 1 2 --viz none
```

The visual test runs five seconds. The suite runs eighteen 20-second episodes, prints PASS/FAIL and saves a timestamped folder under outputs/.

## Understand the project

| Question | Guide |
|---|---|
| What did Colin complete? | [Checkpoint summary](docs/CHECKPOINT.md) |
| What does each file do? | [Code map](docs/CODE_MAP.md) |
| How do I install it? | [Setup and dependency revision](docs/SETUP.md) |
| How does PID work? | [PID baseline](docs/PID_BASELINE.md) |
| How was it validated? | [Methodology](docs/BASELINE_VALIDATION.md) and [measured results](docs/results/calm_hover_validation.json) |
| How does it relate to the group? | [Integration review](docs/TEAMMATE_REPOSITORY_REVIEW.md) |

## Tests and next steps

```powershell
uv run --extra isaacsim python -m pytest tests/test_cascaded_pid.py tests/test_hover_validation.py -q
```

Eight targeted tests passed. This does not claim coverage of older sensor, navigation or RL code. Next: training-safe automatic resets, bounded residual PPO, then wind and realistic sensing.

Original development-template instructions are preserved in [development notes](docs/ISAACLAB_DEVELOPMENT.md).
