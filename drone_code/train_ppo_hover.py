"""Train PPO residual hover. Run with Isaac Sim's python.bat.

    C:\\isaacsim\\python.bat drone_code\\train_ppo_hover.py

Takes no arguments: set run_config.TRAIN_PPO first. When resuming, total_timesteps
is the total to reach, counting the steps already in the checkpoint.

Saves under runs/ppo_hover/<run_name>/: checkpoints, TensorBoard logs, and
final_model.zip with final_vecnormalize.pkl. Each launch also writes
logs/progress_<time>.csv, with losses and metrics per update, and
logs/log_<time>.txt, the same tables as text.
"""
import time
from pathlib import Path

from isaacsim import SimulationApp

import model_contract
import ppo_config
import run_config
import sensor_config


run = run_config.TRAIN_PPO

project_dir = Path(__file__).resolve().parents[1]
run_dir = project_dir / ppo_config.RUN_DIR / run["run_name"]

if run["state_source"] not in ppo_config.STATE_SOURCES:
    raise SystemExit(f"state_source must be one of {ppo_config.STATE_SOURCES}.")

# Refuse now, not after Isaac Sim has started: never continue a policy, or
# overwrite a run, whose observations mean something else.
checkpoint = project_dir / run["resume"] if run["resume"] else None
existing_model = checkpoint or run_dir / "final_model.zip"

if existing_model.is_file():
    problem = model_contract.incompatibility(existing_model, run["state_source"])

    if problem:
        raise SystemExit(
            problem + "\nUse a new run_name in run_config.TRAIN_PPO."
        )

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": not run["gui"]})


def main():
    # Deferred, like the demo's imports, until SimulationApp is running.
    import numpy as np

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
    )
    from stable_baselines3.common.logger import Logger, make_output_format
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    import hover_task

    from hover_env import SimulationClosed, make_hover_env

    class HoverMetricsCallback(BaseCallback):
        """Send reward components and failure stats to TensorBoard."""

        def _on_step(self):
            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                for name, value in info["costs"].items():
                    self.logger.record_mean(f"cost/{name}", value)

                self.logger.record_mean(
                    "hover/position_error", info["position_error"]
                )
                self.logger.record_mean(
                    "hover/estimate_error", info["estimate_error"]
                )
                self.logger.record_mean(
                    "hover/residual_norm", info["residual_norm"]
                )
                self.logger.record_mean(
                    "hover/saturation_rate",
                    info["saturated_steps"]
                    / max(info["physics_steps"], 1),
                )

                if done:
                    self.logger.record_mean(
                        "hover/failure_rate",
                        float(info["failure_reason"] is not None),
                    )

            return True

    training = ppo_config.TRAINING

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_hover_env(
        simulation_app,
        wind_profile=run["wind_profile"],
        render_sim=run["gui"],
        state_source=run["state_source"],
    )

    model_contract.write_contract(run_dir, model_contract.make_contract(
        run["state_source"],
        sensor_config.IMU_NOISE,
        sensor_config.ESTIMATOR,
        hover_task.OBSERVATION_SIZE,
    ))

    # One env, so one DummyVecEnv slot. SB3's Monitor adds ep_rew_mean.
    venv = DummyVecEnv([lambda: Monitor(env)])

    try:
        if checkpoint:
            venv = VecNormalize.load(
                str(ppo_config.vecnormalize_path_for(checkpoint)), venv
            )
            venv.training = True

            model = PPO.load(
                str(checkpoint),
                env=venv,
                device=ppo_config.PPO["device"],
            )
            print(
                f"Resumed {checkpoint} at {model.num_timesteps} steps.",
                flush=True,
            )
        else:
            # Rewards are not normalized: VecNormalize would clip the failure
            # penalty, and hover_task.failure_penalty relies on its full size.
            venv = VecNormalize(
                venv,
                norm_obs=True,
                norm_reward=False,
                clip_obs=training["observation_clip"],
            )

            model = PPO(
                "MlpPolicy",
                venv,
                seed=training["seed"],
                verbose=1,
                tensorboard_log=str(run_dir / "tensorboard"),
                **ppo_config.PPO,
            )

        remaining = run["total_timesteps"] - model.num_timesteps

        if remaining <= 0:
            print(
                f"Already at {model.num_timesteps} steps; nothing to do.",
                flush=True,
            )
            return

        # SB3 opens these files for writing, so a fixed name would erase the
        # earlier history on a resume. Each launch gets its own.
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = run_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        model.set_logger(Logger(
            folder=str(log_dir),
            output_formats=[
                make_output_format("stdout", str(log_dir)),
                make_output_format("csv", str(log_dir), f"_{stamp}"),
                make_output_format("log", str(log_dir), f"_{stamp}"),
                make_output_format(
                    "tensorboard",
                    str(run_dir / "tensorboard" / f"ppo_{stamp}"),
                ),
            ],
        ))

        callbacks = [
            HoverMetricsCallback(),
            CheckpointCallback(
                save_freq=training["checkpoint_interval"],
                save_path=str(checkpoint_dir),
                name_prefix="hover",
                save_vecnormalize=True,
            ),
        ]

        try:
            model.learn(
                total_timesteps=remaining,
                callback=callbacks,
                reset_num_timesteps=False,
                tb_log_name="ppo",
            )

            model.save(str(run_dir / "final_model"))
            venv.save(str(run_dir / "final_vecnormalize.pkl"))
            print(f"Saved final model in {run_dir}", flush=True)

        except (KeyboardInterrupt, SimulationClosed) as stop:
            # Keep what was learned; set resume to interrupted_model.zip.
            model.save(str(run_dir / "interrupted_model"))
            venv.save(str(run_dir / "interrupted_vecnormalize.pkl"))
            print(
                f"Stopped ({type(stop).__name__}); saved in {run_dir}",
                flush=True,
            )

    finally:
        venv.close()


try:
    main()
finally:
    simulation_app.close()
