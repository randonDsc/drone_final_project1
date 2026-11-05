"""Settings for watch_hover_ppo.py. Edit this file, then run:

    C:\\isaacsim\\python.bat drone_code\\watch_hover_ppo.py

Changes apply the next time the script starts. Only the viewer reads this file:
training, evaluation, and the navigation demo are not affected.

Units: meters, seconds, meters per second.
"""

# ---------- Control state ----------

# "imu": the controller and policy use the IMU and estimator. Noise and estimator
# settings are in sensor_config.py, shared with training.
# "ground_truth": they use perfect simulator state, the old baseline.
STATE_SOURCE = "imu"

# ---------- Controller ----------

# True: PID + trained PPO residual. False: PID only.
#
# False until a model is trained on the PID baseline. hover1 learned against
# the old PD controller, so it can no longer be loaded.
USE_PPO = False

# Model .zip, relative to the project root or absolute. Used when USE_PPO is
# True. The matching normalization .pkl next to it is found automatically, for
# final_model.zip, interrupted_model.zip, and checkpoints/hover_N_steps.zip.
MODEL_PATH = "runs/ppo_hover/imu1/final_model.zip"

# ---------- Hover target ----------

# World-frame [x, y, z]. Sets the controller's target, the green marker, the
# camera, and where each episode starts. z must be at least 0.7: episodes end
# below 0.5 and start up to 0.2 from the target.
TARGET_POSITION = (0.0, 0.0, 2.0)

# ---------- Wind ----------

# Used exactly as written on every episode, with no random mean, direction, or
# gust scale. Only the gust noise changes between episodes, fluctuating around
# mean_wind.
#
# The default PPO profile trains on mean speeds up to 3 m/s.
WIND = {
    # False: no wind force at all.
    "enabled": True,

    # Mean wind velocity, world frame [m/s]; the direction the air moves toward.
    "mean_wind": (10.0, 0.0, 0.0),

    # Gust strength: standard deviation per axis [m/s]. Zero disables gusts.
    "gust_std": (0.5, 0.5, 0.2),

    # Gust timescale [s]. Larger = slower, longer gusts.
    "gust_timescale": 2.0,

    # Small, fast variation across the airframe, per axis [m/s]. Zero disables
    # it. Set this and gust_std to zero for perfectly steady wind.
    "local_gust_std": (0.15, 0.15, 0.08),
}

# ---------- Playback ----------

# Seed of the first episode; later episodes count up from it. Episodes with the
# same seed get the same gusts and start state, with PPO on or off.
FIRST_SEED = 100_000

# Stop after this many 10 s episodes. 0 runs until the window is closed.
EPISODES = 0

# 1.0 is real time, 0.5 is half speed.
PLAYBACK_SPEED = 1.0
