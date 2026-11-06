# Implementation differences and physical conventions

## Original MuJoCo PID is the source baseline

The user selected `drone-example/main.py`. A verbatim copy is retained in
`reference/original_mujoco_main.py.txt`; `ORIGINAL_SOURCES.json` records input
hashes. No files in the original project are edited. The mistaken corrected
wind script is not an implementation source.

| Original `main.py` behavior | Validated adaptation |
| --- | --- |
| PID updates depend on elapsed wall time | Explicit simulation dt; all adapted loops at 120 Hz |
| Outer update every 20 steps at original 0.01 s dt | Diagnostic replay retains 5 Hz outer / 100 Hz inner; adapted controller evaluates every physics step |
| Quaternion x/y treated as roll/pitch and w as yaw | Signed Euler errors and body angular rates; roll/pitch divided by two to preserve the original near-hover quaternion gain |
| Yaw setpoint w=1 | Signed yaw target 0 rad; quaternion w cannot encode signed heading |
| Planner assigns float velocities into an integer array | Floating-point velocity requests |
| Fixed axis mapping assumes zero heading | Heading stabilized at zero for these tasks; this remains a near-hover controller |
| Hardcoded 3.2495 N per rotor | Weight feedforward from compiled mass: 3.2495625 N per rotor |
| Per-rotor altitude correction | Converted to total thrust with factor four and tilt compensation |
| Roll/pitch/yaw motor increments | Converted using the original rotor arms and yaw coefficients to physical wrench units |
| Point-to-point targets | Reference velocity/acceleration feedforward, plus original cascade and gains |
| Integrals/output clipping through `simple_pid` | Explicit bounded integrals, integrator zones and saturation feedback |
| Raw truth called a sensor | Explicit ground truth and separate ideal simulated IMU measurements |

The original gains retained in the adapted family are: position (2, .15, 1.5),
velocity (.1, .003, .02), per-rotor altitude (5.50844, .57871, 1.2), roll/pitch
(2.6785, .56871, 1.2508), and yaw (.54, 0, 5.358333). Their *units and effective
mapping* are made explicit in `controllers.py`. These changes are material:
results from `original_adapted` must not be described as an unchanged original
controller. `original_literal` retains the old feedback/mixer defects for
diagnosis under the common physics backend; deterministic timing and the common
drag/integrator still differ from the wall-clock GUI script.

## Model and actuator audit

The model is the Skydio X2 loaded by the original `main.py`, from its nested
Menagerie copy. Its yaw signs match that copy's committed model. The separate
top-level Menagerie checkout has opposite yaw signs; it is not substituted.
The onboard camera comes from the existing sensor model. Provenance and original
asset hashes are included with the copied files.

* Mass: 1.325 kg. COM: (0, 0, 0.0539622641509434) m in body axes.
* Body-frame inertia, kg m²:

```text
[[ 0.0366516981, 0, -0.0021000000],
 [ 0,            0.0254116981, 0],
 [-0.0021000000, 0,  0.0605280000]]
```

* Principal inertia is positive. The off-diagonal body inertia is preserved.
* Rotor positions: (-.14,-.18,.05), (-.14,.18,.05), (.14,.18,.08),
  (.14,-.18,.08) m.
* Direct site motors: controls 0..13 N, unity force gain, no motor activation
  dynamics; reaction coefficient (+.0201,-.0201,+.0201,-.0201) m.
* Allocation is derived about COM from site positions and actuator gear. For
  this model, rows [thrust, roll torque, pitch torque, yaw torque] are:

```text
[[ 1,      1,      1,      1     ],
 [-.18,   +.18,   +.18,   -.18   ],
 [+.14,   +.14,   -.14,   -.14   ],
 [+.0201, -.0201, +.0201, -.0201]]
```

MuJoCo computes inertia from the authored mass distribution. These numbers
verify what the *model* simulates, not the mass distribution of a real X2.
The source model's 13 N rotor limits are retained, not experimentally identified.

## Frames, dynamics and weather

World +Z is up, body +Z is thrust. Quaternions are Hamilton [w,x,y,z]. Freejoint
translation velocity is world-frame velocity of the body origin; freejoint
angular velocity is body-frame. `mj_objectVelocity` provides world angular and
COM linear velocity. The controller/trajectory use body-origin position and its
derivative; wind uses COM velocity plus angular point motion. Both origin and
COM states/accelerations are logged. Torque sign and frame tests exercise the
compiled model, including non-diagonal inertia.

The original MJCF timestep is .01 s. Adapted experiments use 1/120 s and
`implicitfast`, matching the teammate's physics/control rate. Built-in MuJoCo
density/viscosity/wind are zeroed to prevent double-counting aerodynamic forces.
The distributed source wind algorithm is the only drag model:

```text
v_point = v_COM + omega_world cross r_COM
v_relative = v_point - wind_at_point
F_point = -0.5 * rho * Cd * area_point * |v_relative| * v_relative
F_total = sum(F_point)
tau_COM = sum(r_COM cross F_point)
```

MuJoCo's applied wrench is world-frame at COM. The Isaac helper's original
`body_wrench()` returns body-frame torque about the body origin, so that wrench
cannot be copied directly. The backend uses the shared point forces and reduces
their torque about COM instead. Areas sum to the configured total. Hull and rotor
sample points follow X2 geometry rather than Iris offsets. Exact OU updates and
spatial covariance are reused; named temporal mean-wind programs are added.

Density 1.225 kg/m³, Cd 1 and total area .03 m² are inherited starting assumptions,
not measured aerodynamic identification. Distributed fluctuations remain frozen
to the airframe, not spatially advected. No blade/rotor flow, lift changes,
ground effect, battery, motor lag or rotor gyroscopic dynamics are modeled.

After `mj_step`, `mj_forward` refreshes derived pose, acceleration and sensor
arrays so post-step state and IMU share a timestamp. The held wind wrench is
logged with its preceding control interval; it is not relabeled as an
instantaneous recomputation at the new state.

## Teammate algorithms and SAC

Shared NumPy modules now support package imports and their original script
imports. The original Isaac scripts, USD files, gains and default experiment
settings remain available. `ReferencePID` accepts optional gains without changing
its default behavior. The MuJoCo backend reuses `PathCore`'s observation, held
residual and command sequencing, and calls the same reward functions.

The optional `isaac_pid` family retains the teammate's outer feedback/lead-point
algorithm. Its X2 attitude gains are `kp=I_diagonal*omega_n²` and
`kd=2*.9*I_diagonal*omega_n`, with omega_n=(8,8,5) rad/s. This diagonal gain design
is a tuning choice; the actual engine retains full coupled inertia. This
family caps requested horizontal speed at 2 m/s in norm and vertical speed at
1 m/s, rather than Isaac's current 5 m/s settings. The original-adapted family
retains per-axis horizontal velocity clipping at +/-2 m/s and its original
altitude-force loop; `max_z_speed` applies only to the Isaac PID family. Its
vertical reference is speed limited by the task generator. Horizontal
acceleration remains capped at 1.5 m/s² in both adapted families.

The source Iris motor model uses kf=5.84e-6 N/(rad/s)², yaw reaction coefficient
.06 m, and max speed 1100 rad/s. No fictitious kf is assigned to the X2. Its motor
commands remain newtons throughout, with no RPM conversion.

One SAC observation frame has exactly the source layout:

| Slots | Value |
| --- | --- |
| 0:3 | p_ref - p, world |
| 3:6 | v_ref - v, world |
| 6:9 | Reference velocity |
| 9:12 | Reference acceleration |
| 12:18 | First two columns of body-to-world rotation |
| 18:21 | Source quaternion rotation error, q_desired^-1 * q |
| 21:24 | Body angular velocity |
| 24:28 | PID thrust minus hover thrust, then three PID torques |
| 28:32 | Previous normalized residual action |

Four frames, oldest first, make 128 values. Initial history repeats the first
frame. Normalization/clipping are unchanged. Wind is not observed. The same
Huber, direction fade, action/change, attitude/rate, saturation and failure costs
are used, with gamma=.99 and scale=.1. MuJoCo additionally detects real contacts
and checks failures at each physics substep, rather than waiting for a decision.

The original wrench residual limits (.8 N, .05/.03/.02 N m) exceed the X2's
per-rotor budget because its yaw moment arm is smaller. Exact inversion of the
COM allocation gives a uniform reduction factor **0.6819981344**, yielding
(.5455985075 N, .0340999067/.0204599440/.0136399627 N m). The worst rotor
correction is .3899475 N, equal to .02*1.5*m*g. This preserves the normalized
four-action architecture and budget, not identical physical action magnitudes.

SAC retains the source MLP [256,256] actors/critics, lr=3e-4, batch=256,
buffer=500000, learning_starts=5000, tau=.005, entropy auto, target entropy=-4,
one gradient per decision, and original four curriculum stages. Device is CPU.
The short validation run cannot reach later stages. Those stages are exercised
for reset/stepping validity in tests, not claimed as learned capabilities.

The source `TimedReference` independently interpolates positions/velocities/
accelerations; they are approximate feedforward signals. Training retains that
code for traceability. Controlled evaluation uses analytic quintic references;
curves use an analytic phase law and waypoints stop at corners. This difference
is recorded in training/run metadata. Obstacle clearance guarantees from the
Isaac Dijkstra pipeline are not transferred to these open-space experiments.

## Sources and comparison limits

The copied MJCF and actual compiled state are authoritative for these tests.
Relevant API semantics were checked against primary documentation:

* [MuJoCo Python and macOS viewer](https://mujoco.readthedocs.io/en/stable/python.html)
* [MuJoCo data and applied wrench fields](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html)
* [MJCF motors, sites and sensors](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
* [Stable-Baselines3 SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)

The models are different aircraft with different mass, inertia, rotor geometry,
actuation and aerodynamic approximations. No actual Isaac evaluation table with
matching tasks/conditions was found. A future comparison must standardize those
parameters and the reference schedule, seed/initial conditions, wind model,
logging and metric definitions before interpreting simulator differences. Same
observation dimensions do not establish checkpoint compatibility or comparable
physical experiments.
