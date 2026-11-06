# MuJoCo implementation plan and repository audit

This branch (`alan/mujoco`, starting at `d40cac3`) adds a local experimental
backend. The original `mujoco-project` is an input only. No Isaac checkpoint is
assumed transferable, and no changes are pushed.

## Audit

* `drone_code/drone_sim.py` owns Isaac/PhysX, USD loading, geometry, and rotor
  application. `mapping.py`, cameras, and GUI entry points require Isaac.
* `controller.py`, `wind.py`, `path_reference.py`, most of `path_sac_policy.py`,
  and task/configuration helpers are NumPy code. Reuse these algorithms through
  package imports; retain direct-script imports for the Isaac workflows.
* Path SAC actually stacks four 32-value frames (128 values), uses four held
  residual wrench actions, updates PID/physics at 120 Hz and SAC at 10 Hz,
  and has four curriculum stages. Reward includes tracking, direction,
  attitude/rate, action change, saturation, and terminal failure costs.
* Per the user's clarification, `drone-example/main.py` is the original PID
  baseline; `point_to_point_wind.py` is a wind experiment. Exclude the mistaken
  `point_to_point_wind_corrected.py` as an implementation source. The original
  controller uses `simple_pid`, wall-clock calls, a hand-written mixer, and
  quaternion components as attitude feedback. `dummySensor` exposes truth.
* The original project contains TWO Skydio copies. The one loaded by the
  original `main.py`, `drone-example/mujoco_menagerie-main/skydio_x2`, adds the
  onboard camera and reverses all four yaw gear signs relative to the separate
  `mujoco_menagerie/skydio_x2` checkout. Use the script's actual model and derive
  signs from MJCF rather than assuming them.
  Its yaw signs also match `git show HEAD:.../x2.xml` in `drone-example`;
  they are not a change introduced by the excluded wind script.
* Compiling that model yields mass 1.325 kg, body COM (0,0,0.053962264) m,
  positive principal inertias, four direct site motors with controls 0..13 N,
  and reaction coefficients +/-0.0201 m. The model is a simplified rigid drone,
  not an identified model of real motors or propeller flow.
* The Isaac default instead uses `kf=5.84e-6`, motor angular speeds in rad/s,
  reaction coefficient 0.06 m, a different rotor layout and vehicle inertia.
  Copying its motor mixer or gains without checking would change the physics.
* Existing `TimedReference.sample` linearly interpolates position, velocity and
  acceleration separately; the latter are not exactly derivatives of the
  interpolated position. Preserve that legacy option for reproduction and
  provide differentiable references for the new controlled experiments.
* No usable Isaac evaluation metric CSV/JSON was found. Navigation waypoints,
  occupancy maps, and text logs are not evidence of equivalent performance.

## Architecture and sequence

1. Package the existing simulator-independent algorithms with backward
   compatible imports. Copy the licensed Skydio model/assets into this repo.
2. `mujoco_drone.physics`: compile MJCF, audit mass/inertia/actuators, derive
   allocation about COM, transform angular velocities, apply world-frame drag
   forces/torques at COM, step and refresh sensors at a coherent timestamp.
3. `weather`: reuse distributed quadratic drag and exact OU updates. Add calm,
   directional, gust, periodic, turbulent, and seeded randomized profiles.
   Disable MuJoCo's built-in fluid forces to avoid counting drag twice.
4. `sensors`, `recording`, `metrics`: distinguish ideal simulated IMU from truth,
   stream structured flight logs, optionally capture RGB/depth, retain complete
   configurations, model/source hashes, units, versions, and wind realizations.
5. `references` and `core`: preserve the original MuJoCo PID as a diagnostic
   baseline, adapt its cascaded controller using explicit simulation time and
   proper attitude coordinates, and support the teammate's reference PID as a
   separate controller family. Use the same residual loop and observation/reward
   definitions for hover, straight, waypoints, circle and figure eight.
6. `env`, `training`: Gymnasium/SB3 SAC, same observation/reward definitions and
   curriculum, explicit MuJoCo checkpoint contracts and resumable replay data.
   Bound residuals using the X2 allocation, not the Iris motor constants.
7. `experiments` and CLI: paired seeds/configurations, fixed horizons, aggregate
   completion/failure/saturation and tracking metrics, gust recovery, plotting,
   and comparison imports that reject mismatched or missing experiment data.
8. Tests and real simulations gate the stages. Document measured results and
   distinguish training-pipeline validation from evidence of a robust policy.

## Assumptions and validation gates

SI units, world +Z up, body +Z thrust, Hamilton quaternions [w,x,y,z], and body
angular velocities at the free joint must be checked with impulse/rotation and
hover tests. Wrenches are reduced about COM before writing `xfrc_applied`.
Air density 1.225 kg/m^3, drag coefficient 1, and effective area 0.03 m^2 are
inherited research assumptions, not measured X2 properties. Sample points follow
X2 rotor/body geometry; area weights are normalized. OU fluctuations are frozen
to the airframe as in the source model, not an advected meteorological field.

Baseline control uses ground truth. IMU and rendered cameras are simulated ideal
measurements; they do not establish sensor-based autonomy. Rotor lag, ground
effect, battery, rotor gyros, blade aerodynamics and real-world validation remain
outside this model. Obstacle navigation is deferred until the earlier phases are
validated. macOS camera capture and GUI must be tested separately from headless
physics. A short SAC run proves learning/save/load plumbing only; convergence
requires longer training and independent evaluation.
