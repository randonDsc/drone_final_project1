"""IMU noise and state estimator settings, shared by training, evaluation, and
the viewer, so a policy sees the same kind of signal it was trained on.

Units: meters, seconds, radians. Theory and limits: state_estimation.py.

These values are starting assumptions in the range of a good MEMS IMU after a
factory calibration. They are not measurements of any particular part.
"""

# Prim of the IMU already in the scene. Connection fails loudly if it is missing;
# there is no fallback to simulator state.
IMU_PATH = "/World/iris/body/imu"

# Added on top of what Isaac's IMU reports, which has no noise of its own.
# Set every value to zero for a noise-free sensor.
IMU_NOISE = {
    # White noise density. Per-sample std = density * sqrt(sample rate).
    "gyro_noise_density": 5.0e-5,      # [rad/s/sqrt(Hz)]
    "accel_noise_density": 7.0e-4,     # [m/s^2/sqrt(Hz)]

    # Bias drawn once per episode (switch-on error), per axis.
    "gyro_bias_std": 1.0e-3,           # [rad/s]
    "accel_bias_std": 0.02,            # [m/s^2]

    # Bias random walk during the episode.
    "gyro_bias_walk": 1.0e-5,          # [rad/s^2/sqrt(Hz)]
    "accel_bias_walk": 1.0e-4,         # [m/s^3/sqrt(Hz)]
}

# Indoor sensor hardware; see INDOOR_SENSORS.md. These exist only in the "indoor"
# scene (hover_ppo_config.ACTIVE_SCENE). Nothing feeds them to the estimator yet.
LIDAR = {
    "path": "/World/iris/body/HESAI_XT32_SD10/PandarXT_32_10hz",

    # Shorter returns are discarded. Recorded scans contain zero-distance points.
    "min_valid_range": 0.1,            # [m]
}

RANGEFINDER = {
    "path": "/World/iris/body/rangefinder",

    # Must match minRange and maxRange in sim_asset/iris_indoor.usda.
    "min_range": 0.05,                 # [m]
    "max_range": 12.0,                 # [m]

    "update_rate_hz": 50.0,

    # Range noise std = noise_std + noise_fraction * range. The ray itself is
    # perfect, so all noise comes from here. Typical small-laser values, not
    # measurements of a particular part.
    "noise_std": 0.02,                 # [m]
    "noise_fraction": 0.005,
    "resolution": 0.01,                # [m]
    "dropout_probability": 0.002,
}

ESTIMATOR = {
    # Attitude filter. kp pulls tilt toward the direction of the measured
    # specific force, with a time constant of about 1/kp seconds. That direction
    # is only gravity when not accelerating, so higher is not simply better.
    "attitude_kp": 0.5,                # [1/s]

    # Learns roll and pitch gyro bias. Yaw bias is not observable.
    "attitude_ki": 0.02,               # [1/s^2]

    # Skip tilt correction when |specific force| differs from g by more than
    # this, since the reading is then mostly not gravity.
    "accel_gate": 2.0,                 # [m/s^2]

    # IMU samples discarded after each reset. Isaac may compute acceleration
    # from velocity changes, so a teleport can look like a huge spike, and the
    # sensor's own averaging window must clear.
    "reset_ignore_samples": 6,

    # Samples averaged after that to set the initial roll and pitch.
    "alignment_samples": 6,

    # 0 is pure integration. Above 0, velocity decays with this time constant
    # [s]: a "probably hovering" assumption that bounds velocity error but also
    # hides real motion and steady wind drift. It is not a measurement.
    "velocity_leak_time": 0.0,
}
