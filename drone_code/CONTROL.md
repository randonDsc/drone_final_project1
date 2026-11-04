# Flight control

`controller.PIDController` is the only flight controller. It replaced the PD
controller in place. **The gains, and all of this code, have not been run.**

    Dijkstra planner -> waypoints -> WaypointFollower -> active target
    Drone state + active target -> PID -> thrust and torques
    Drone state and history -> SAC -> held correction (mode "pid_sac" only)
    PID command + correction -> motor mixer -> rotor commands -> forces

| Part | Input | Output | Rate |
| --- | --- | --- | --- |
| Planner | Occupancy grid, start, goal | World waypoints [m] | Once per flight |
| Follower | Position, velocity | Active target [m] | 120 Hz |
| PID | State, target | Thrust [N], body torques [N m] | 120 Hz |
| SAC | Observation history | Normalized action, held | 10 Hz |
| Mixer | Thrust, torques | Four rotor speeds [rad/s] | 120 Hz |

Planning stays separate from stabilization. Neither the planner nor the follower
knows which control mode is flying.

## PID layers

Inputs are world position [m], quaternion [w, x, y, z], world velocity [m/s], and
world angular velocity [rad/s]. Gains are in `sim_config.PID`. The proportional and
derivative gains are the old PD values, unchanged.

| Loop | Frame | Error | Output | Integral adds to |
| --- | --- | --- | --- | --- |
| Height | World z | Height [m] | Vertical force [N], tilt-compensated into thrust | Force [N] |
| Horizontal | World x, y | Position [m] | Acceleration [m/s^2], then roll and pitch targets | Acceleration [m/s^2] |
| Attitude | Body | Roll, pitch, yaw [rad] | Body torques [N m] | Torque [N m] |

Gravity compensation, tilt compensation, speed and tilt caps, mixing, and motor limits
are unchanged.

The roll and pitch integral gains are 0 by default. The horizontal loop already
trims their steady error, so an inner integrator would fight it. The yaw loop has no
outer loop, so its integrator is on.

## Integrators

- **Timestep:** each step adds `ki * error * dt`, with `dt` passed in by the caller.
- **Reading without integrating:** `dt = 0` changes nothing. Observations use this, so
  the integrators advance once per physics step.
- **Limits:** each integrator is clamped to `integral_limit`.
- **Zone:** an axis holds its value while its error exceeds `integral_zone`, so a far
  waypoint on a long leg does not wind it up.
- **Anti-windup:** when its output is saturated, an integrator may shrink but not grow.
  Saturation means the thrust clip, the acceleration cap, or the mixer clipping a rotor
  on the previous step, which also covers the SAC correction.
- **Resets:** `reset()` runs on every env reset and on each flight start or Stop ->
  Play. It does not run between waypoints, because the wind has not changed.

Gains stay fixed during SAC training, and a changed gain makes old SAC models
incompatible.

## Modes

Set `CONTROL_MODE` in `sim_config.py`.

| Mode | Meaning |
| --- | --- |
| `"pid"` | PID only. The default |
| `"pid_sac"` | PID plus the correction from `SAC_MODEL_PATH` |

In `"pid_sac"`, a missing or incompatible model prints `FALLBACK to PID only` and the
flight continues on PID. The log's `Control mode:` line shows what is flying.
