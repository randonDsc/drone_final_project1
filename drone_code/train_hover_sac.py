"""Train the SAC wind correction on top of PID. Run with Isaac Sim's python.bat.

    C:\\isaacsim\\python.bat drone_code\\train_hover_sac.py

Takes no arguments: set run_config.TRAIN_HOVER_SAC first. When resuming, total_timesteps
is the total to reach, counting the steps already in the checkpoint.

Saves under runs/sac_hover/<run_name>/: checkpoints/, final_model.zip with
final_replay_buffer.pkl, TensorBoard logs, and for each launch
logs/progress_<time>.csv and logs/log_<time>.txt. Ctrl+C saves interrupted_*.
"""
import time
from pathlib import Path

from isaacsim import SimulationApp

import hover_task
import model_contract
import run_config
import hover_sac_config


run = run_config.TRAIN_HOVER_SAC

project_dir = Path(__file__).resolve().parents[1]
run_dir = project_dir / hover_sac_config.RUN_DIR / run["run_name"]

# Report these now, not after Isaac Sim has started.
if run["resume"]:
    checkpoint = project_dir / run["resume"]
    buffer_path = hover_sac_config.replay_buffer_path_for(checkpoint)

    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    problem = model_contract.sac_incompatibility(
        checkpoint, hover_sac_config.contract()
    )

    if problem:
        raise SystemExit(problem)

    if not buffer_path.is_file() and not run["allow_empty_replay_buffer"]:
        raise SystemExit(
            f"Replay buffer not found: {buffer_path}\n"
            "Set allow_empty_replay_buffer in run_config.TRAIN_HOVER_SAC to continue "
            "with an empty one."
        )

elif (run_dir / "final_model.zip").is_file():
    raise SystemExit(
        f"{run_dir} already holds a model. Set resume or a new run_name in "
        "run_config.TRAIN_HOVER_SAC."
    )

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": not run["gui"]})


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
    )
    from stable_baselines3.common.logger import Logger, make_output_format
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from hover_env import SimulationClosed
    from hover_sac_env import make_hover_sac_env

    class HoverMetricsCallback(BaseCallback):
        """Advance the wind stages; log reward terms and failure stats."""

        def _on_step(self):
            env.set_training_step(self.num_timesteps)

            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                for name, value in info["costs"].items():
                    self.logger.record_mean(f"cost/{name}", value)

                self.logger.record_mean(
                    "hover/position_error", info["position_error"]
                )
                self.logger.record_mean(
                    "hover/residual_norm", info["residual_norm"]
                )
                self.logger.record_mean(
                    "hover/saturation_rate",
                    info["saturated_steps"]
                    / max(info["physics_steps"], 1),
                )
                self.logger.record("hover/wind_stage", info["wind_stage"])

                if done:
                    self.logger.record_mean(
                        "hover/failure_rate",
                        float(info["failure_reason"] is not None),
                    )

            return True

    training = hover_sac_config.TRAINING

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_hover_sac_env(simulation_app, render_sim=run["gui"])

    model_contract.write_contract(run_dir, {
        **hover_sac_config.contract(),
        "observation_size": env.observation_space.shape[0],
        "state_source": "ground_truth",
    })

    venv = DummyVecEnv([lambda: Monitor(env)])

    try:
        if run["resume"]:
            model = SAC.load(
                str(checkpoint),
                env=venv,
                device=hover_sac_config.SAC["device"],
            )

            if buffer_path.is_file():
                model.load_replay_buffer(str(buffer_path))
                print(
                    f"Replay buffer restored: {model.replay_buffer.size()} "
                    "transitions.",
                    flush=True,
                )
            else:
                print("WARNING: resuming with an empty replay buffer.", flush=True)

            print(
                f"Resumed {checkpoint} at {model.num_timesteps} steps.",
                flush=True,
            )
        else:
            model = SAC(
                "MlpPolicy",
                venv,
                seed=training["seed"],
                verbose=1,
                **hover_sac_config.SAC,
            )

        env.set_training_step(model.num_timesteps)

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
                    str(run_dir / "tensorboard" / f"sac_{stamp}"),
                ),
            ],
        ))

        callbacks = [
            HoverMetricsCallback(),
            CheckpointCallback(
                save_freq=training["checkpoint_interval"],
                save_path=str(checkpoint_dir),
                name_prefix="sac",
                save_replay_buffer=True,
            ),
        ]

        try:
            model.learn(
                total_timesteps=remaining,
                callback=callbacks,
                reset_num_timesteps=False,
            )

            model.save(str(run_dir / "final_model"))
            model.save_replay_buffer(str(run_dir / "final_replay_buffer.pkl"))
            print(f"Saved final model in {run_dir}", flush=True)

        except (KeyboardInterrupt, SimulationClosed) as stop:
            # Keep what was learned; resume from interrupted_model.zip.
            model.save(str(run_dir / "interrupted_model"))
            model.save_replay_buffer(
                str(run_dir / "interrupted_replay_buffer.pkl")
            )
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
