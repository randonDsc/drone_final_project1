"""What each script does when you run it. Edit this file, then run the script with
no arguments, for example:

    /c/isaacsim/python.bat drone_code/train_hover_sac.py

Changes apply the next time a script starts. Paths are relative to the project
root, or absolute. A path of None uses the run's final files.

Other settings stay where they were: hover_sac_config.py and hover_ppo_config.py for rewards,
wind stages, seeds, and checkpoint interval, sim_config.py for PID and flight
mode, watch_config.py for the viewer.
"""

# ---------- SAC ----------

# train_hover_sac.py
TRAIN_HOVER_SAC = {
    # SAC decisions to reach, 10 per simulated second. When resuming, this is the
    # total, counting the steps already in the checkpoint.
    "total_timesteps": 1_000_000,

    # Saved under runs/sac_hover/<run_name>/. Evaluate with the same name.
    "run_name": "wind1",

    # Model .zip to continue from, or None to start fresh. After each stop, point
    # this at the newest save: checkpoints/sac_N_steps.zip or interrupted_model.zip.
    "resume": "runs/sac_hover/wind1/checkpoints/sac_200000_steps.zip",

    # True: resume with an empty replay buffer if its .pkl is missing.
    # False: stop with an error instead.
    "allow_empty_replay_buffer": False,

    # Show the Isaac Sim window (slower).
    "gui": False,
}

# eval_hover_sac.py
EVAL_HOVER_SAC = {
    "run_name": "wind1",

    # Model .zip, or None for the run's final_model.zip.
    "model": "runs/sac_hover/wind1/checkpoints/sac_200000_steps.zip",

    # light, steady, gusty, or strong; see hover_sac_config.WIND_STAGES.
    "wind_stage": "gusty",

    # Use only the first N of hover_sac_config.EVAL_SEEDS, or None for all 20.
    "episodes": None,

    # True: fly the baseline alone, which needs no model.
    "pid_only": False,

    "gui": False,
}

# ---------- PPO ----------

# train_hover_ppo.py
TRAIN_HOVER_PPO = {
    # PPO decisions to reach, 30 per simulated second. When resuming, this is the
    # total, counting the steps already in the checkpoint.
    "total_timesteps": 500_000,

    # Saved under runs/ppo_hover/<run_name>/.
    "run_name": "run1",

    # Checkpoint .zip to continue from, or None to start fresh.
    "resume": None,

    # none, weak, or moderate; see hover_ppo_config.WIND_PROFILES.
    "wind_profile": "weak",

    # "imu" or "ground_truth"; see hover_ppo_config.STATE_SOURCES.
    "state_source": "imu",

    "gui": False,
}

# eval_hover_ppo.py
EVAL_HOVER_PPO = {
    "run_name": "run1",

    # Model .zip and statistics .pkl, or None for the run's final files.
    "model": None,
    "vecnormalize": None,

    "wind_profile": "weak",
    "state_source": "imu",

    # Use only the first N of hover_ppo_config.EVAL_SEEDS, or None for all 20.
    "episodes": None,

    # True: fly the baseline alone, which needs no model.
    "pid_only": False,

    "gui": False,
}

# smoke_hover_ppo.py
SMOKE_HOVER_PPO = {
    "state_source": "imu",
    "gui": False,
}

# ---------- Sensors ----------

# inspect_sensors.py
INSPECT_SENSORS = {
    "seconds": 3.0,    # simulated time to run
    "gui": True,
}
