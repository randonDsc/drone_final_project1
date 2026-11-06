# Code map

## Read these first

1. `scripts/run_hover.py`: the command you run; delegates to the existing runner.
2. `scripts/test_lee_hover.py`: loads the ARL task, selects Lee/PID, allocates motor commands, and logs a run. The historical filename remains compatible.
3. `src/IsaacLab_Drone_Project/controllers/cascaded_pid.py`: position P -> velocity PID -> attitude P -> body-rate PI. Lee is supplied by isaaclab_contrib in the sibling Isaac Lab checkout.
4. `scripts/validate_hover.py`: matched multi-seed scenarios, reset checks, logging and acceptance scoring; invoked through `run_hover.py --validate`.
5. `tests/test_cascaded_pid.py` and `tests/test_hover_validation.py`: focused simulator-free tests.

```text
Target + state -> Lee or PID -> body force/torque
    -> allocation matrix -> four motor requests -> limits
    -> motor dynamics -> drone motion -> next state
```

Installed Isaac Lab uses XYZW quaternions; the runner converts to WXYZ for PID. Position and linear velocity are world frame; controller angular velocity and torque are body frame. Thrust is in newtons.

## Supporting and experimental files

| Location | Purpose/status |
|---|---|
| scripts/summarize_lee_hover.py | Single-run summaries for either controller, including legacy quaternion correction. Suite results already have summary.json. |
| scripts/inspect_hover.py | Physical-property inspection. |
| scripts/test_hover.py and controllers/hover_controller.py | Earlier altitude-only PD experiment; not full stabilization. |
| scripts/drone_control.py | Older direct-PPO playback; requires an external checkpoint. |
| tasks/drone_flight/position_tracking/config/arl_robot_1/ | Scene, action/observation/reward/reset settings and task registration. |
| tasks/drone_flight/mdp/ | Task commands and reward/observation functions. |
| tasks/drone_flight/**/agents/ | RL configurations; not evidence of successful training. |
| tasks/drone_flight/obstacle_navigation/ | Experimental navigation task, outside current validation. |
| navigation.py | Fixed target configuration helper, not a full planner. |
| sensor_recorder.py | Camera/IMU recording experiment; disabled during baseline diagnostics and not validated here. |
| tests/test_registration.py | Older template test, excluded from targeted test claims. |
| docs/results/ | Small tracked measurement summaries. |
| outputs/, logs/, backups/, checkpoints/, .venv/ | Local outputs, models and dependencies excluded from submission. |

Files were preserved to keep existing entry points and experiment history. Start with the five files above to follow the tested system.
