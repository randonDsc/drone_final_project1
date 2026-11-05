# Indoor sensing

**Chosen approach: LiDAR-inertial odometry (LIO) from the XT32 and IMU, with a
downward rangefinder for height.** The hardware and measurement readers exist. The
odometry does not, so the drone still has no usable position estimate.

## Three separate layers

| Sensor | 1. Hardware in USD | 2. Code reading it | 3. Estimation |
| --- | --- | --- | --- |
| IMU | Done; settings unchanged | Done: `DroneSim.read_imu` | Attitude filter done; position drifts |
| Hesai XT32 | Done; mount raised | Written, never run: `LidarScanReader` | **Missing: LIO** |
| Rangefinder | Added | Written, never run: `RangefinderReader` | **Missing: height update** |

Having a sensor in the USD gives no localization. Nothing here feeds the controller
or policy yet; `hover_env.py` still uses only the IMU estimator.

## Sensor settings

| Setting | IMU | Hesai XT32 | Rangefinder |
| --- | --- | --- | --- |
| Prim type | `IsaacImuSensor` | `OmniLidar` in the vendor asset | `IsaacRaycastSensor` |
| Mount, body frame | Origin | (-0.003, 0.001, 0.14) m | (0, 0, -0.072) m |
| Orientation | Aligned with body | Aligned with body | Ray along -z |
| Rate | Every physics step, 120 Hz | 10 Hz sweep | 50 Hz, sampled in code |
| Range | n/a | 0.05 to 120 m | 0.05 to 12 m |
| Channels | n/a | 32, +15 to -16 degrees | 1 ray |
| Noise | `IMU_NOISE`, in code | Vendor profile: 2 cm range, 0.015 degree azimuth | `RANGEFINDER`, in code |
| Needs rendering | No | **Yes** | No |

Noise values are typical figures, not measurements of a part. Settings live in
`sensor_config.py`. The XT32's own profile is the vendor's and is unchanged.

## Findings

- **IMU:** configured correctly, and logged data shows +9.81 m/s^2 on z. Its
  `orientation` output is simulator truth and must never be read.
- **XT32 profile:** correct. New sweeps arrive at 10 Hz with per-point timestamps,
  which LIO needs to undo motion distortion. Reads faster than 10 Hz repeat a sweep,
  and some points have zero distance. `LidarScanReader` handles both. There is no
  ring id by default, so the reader rounds elevation instead.
- **XT32 mount:** at 0.0958 m the airframe rises to -12.0 degrees. Raised to 0.14 m,
  it peaks at -18.9 degrees, below the lowest channel, by vertices in the USD.
- **Unexplained:** recordings at the old mount lost all of -12 to -16 degrees and
  part of -9 to -11, more than the airframe geometry predicts. Something else may
  also block them, such as the sensor's own housing. `inspect_sensors.py` prints
  per-channel counts and will show whether the raised mount fixes it.

Static ray-and-box check in the warehouse, using the box colliders, ideal beams, and
point-to-plane information. Not an Isaac LiDAR run:

| Pose | Clearance | Result |
| --- | --- | --- |
| (5, 5, 2) | 3.10 m | All axes and yaw constrained |
| Demo goal (3.6, 1.1, 1.1) | 3.41 m | All constrained |
| (0, -10, 2) | 2.10 m | All constrained; z weakest |
| (-8, -14, 2) | 1.79 m | **x has no information** |
| (0, 0, 2) | 0.24 m | Constrained, but inside the 0.34 m rotor reach |
| Open ground (hover scene) | n/a | **Only z. x, y, and yaw have none** |

So the warehouse supports LIO at most poses, not all. Long, uniform stretches can
lose a direction, and the odometry must detect that.

## Why no camera or GPS

- **Optical flow camera: not added.** The warehouse boxes have flat colors and no
  texture, so flow has little to track. It would need another RTX render stream.
  Isaac's `motion_vectors` output is simulator truth, not measured flow, so real flow
  also needs an image pipeline. Add one if LIO proves unreliable and the scene gains
  texture.
- **GPS: not added.** There is no outdoor sky view in a warehouse.
- **Magnetometer: not added.** Isaac has no sensor for it here, and LIO provides yaw.

## What remains

1. Run `inspect_sensors.py` and confirm real data from every sensor.
2. Build the LIO. Isaac includes none, so this needs a new component: point
   deskewing with IMU data, scan-to-map matching, and a fused state output. Options
   are FAST-LIO style code in Python or C++, or an external ROS 2 stack through
   `isaacsim.ros2.bridge`. Either way it needs a map frame and a start pose.
3. Fuse range into height, correcting with estimated tilt. A range is height over
   whatever is below, so pallets and shelves appear as steps to reject.
4. Replace `ImuEstimator` translation in `hover_env.py` with the fused state, and
   add sensor delay: 10 Hz scans arrive after the sweep.
5. Move hover into the warehouse: keep `/World/BoxWarehouse` active and pick a target
   with clearance, such as (5, 5, 2).
6. Retrain. Observations change meaning again, so bump `model_contract.VERSION`.

Cost to plan for: the XT32 needs rendering. Training now renders nothing, so LIO in
the loop will run much slower than the current 62 decisions per second.

Other known gaps: the XT32 is fetched from an NVIDIA URL at run time, so it needs
network or a local cache. Its weight is not modeled. The rangefinder is a perfect
thin ray before added noise, with no cone or surface effects. Box colliders are what
the rangefinder hits, and rendered boxes are what the LiDAR hits; they match here.

## Selecting the assets

| Item | Original, unchanged | With indoor sensors |
| --- | --- | --- |
| Drone | `sim_asset/iris.usd` | `sim_asset/iris_indoor.usda` |
| Scene | `sim_scene/dij3d_boxes.usd` | `sim_scene/dij3d_boxes_indoor.usda` |

`ACTIVE_SCENE` in `hover_ppo_config.py` selects `"original"` (the default) or `"indoor"`.
The hover env deactivates the warehouse in either case. The navigation demo always
loads the original.

## Commands

From `C:\drone_final_project1`.

    :: Static: no Isaac Sim, seconds. Checks the USD and that physics is unchanged.
    C:\isaacsim\python.bat drone_code\check_sensor_assets.py

    :: Live: opens Isaac Sim and prints each sensor's real output.
    C:\isaacsim\python.bat drone_code\inspect_sensors.py

To see the prims, open `sim_scene\dij3d_boxes_indoor.usda` in Isaac Sim and expand
`/World/iris/body`.
