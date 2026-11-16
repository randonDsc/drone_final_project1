"""Train the SAC flight residual on top of PID. Run with Isaac Sim's python.bat.

    /c/isaacsim/python.bat drone_code/train_path_sac.py

By default reads path_sac_config.PATH_TRAIN; --wind-adaptation enables explicit continuation options. mode "fresh" starts a new
randomly initialized model and refuses a folder that already has one. mode "resume"
continues resume_model, and refuses a missing or incompatible one. When resuming,
total_timesteps is the total to reach.

Saves under runs/flight_sac/<run_name>/: checkpoints/, final_* or interrupted_* or
crashed_*, TensorBoard logs, logs/progress_<time>.csv, logs/log_<time>.txt,
flight_contract.json, and config_snapshot_<time>.json for each launch.
"""
import argparse
import json
import zipfile
import time
from pathlib import Path

from isaacsim import SimulationApp

import path_sac_config


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--wind-adaptation", action="store_true")
parser.add_argument("--additional-steps", type=int, default=1000)
parser.add_argument("--run-name", default="flight3")
parser.add_argument("--resume-model", type=Path)
args = parser.parse_args()
run = path_sac_config.train_settings()
if args.wind_adaptation:
    if args.additional_steps <= 0 or Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("Use positive additional steps and a simple run name")
    if args.resume_model is None:
        parser.error("--wind-adaptation requires --resume-model")
    source = args.resume_model.resolve()
    if not source.is_file():
        parser.error(f"Missing checkpoint: {source}")
    with zipfile.ZipFile(source) as archive:
        parent_steps = int(json.loads(archive.read("data"))["num_timesteps"])
    run = {**run, "mode": "resume", "resume_model": str(source),
           "run_name": args.run_name, "total_timesteps": parent_steps + args.additional_steps,
           "seed": 730000, "checkpoint_interval": 500, "keep_replay_buffers": 2, "gui": False}


project_dir = Path(__file__).resolve().parents[1]
run_dir = project_dir / path_sac_config.RUN_DIR / run["run_name"]
checkpoint_dir = run_dir / "checkpoints"

if run["mode"] not in ("fresh", "resume"):
    raise SystemExit('PATH_TRAIN mode must be "fresh" or "resume".')

checkpoint = None
buffer_path = None

# Report these now, not after Isaac Sim has started.
if run["mode"] == "resume":
    if not run["resume_model"]:
        raise SystemExit('Mode "resume" needs resume_model in PATH_TRAIN.')

    checkpoint = project_dir / run["resume_model"]
    buffer_path = path_sac_config.replay_buffer_path_for(checkpoint)

    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    problem = path_sac_config.incompatibility(checkpoint, check_training=True)

    if problem:
        raise SystemExit(problem)

    if not buffer_path.is_file() and not run["allow_empty_replay_buffer"]:
        raise SystemExit(
            f"Replay buffer not found: {buffer_path}\n"
            "Set allow_empty_replay_buffer in PATH_TRAIN to continue with an "
            "empty one."
        )

else:
    existing = [
        *run_dir.glob("*_model.zip"),
        *checkpoint_dir.glob("flight_sac_*.zip"),
    ]

    if existing:
        raise SystemExit(
            f"{run_dir} already holds a model. Use a new run_name, or set mode "
            '"resume" with resume_model in PATH_TRAIN.'
        )

if args.wind_adaptation:
    from wind_events import ADAPTATION_SETTINGS
    import model_contract
    saved_contract = json.loads((model_contract.run_folder(checkpoint) / path_sac_config.CONTRACT_FILE).read_text())
    previous = saved_contract.get("training", {}).get("wind_training")
    if previous is not None and previous != ADAPTATION_SETTINGS:
        raise SystemExit("The source model used a different wind-adaptation configuration")
    existing_models = [*run_dir.glob("*_model.zip"), *checkpoint_dir.glob("flight_sac_*_steps.zip")]
    same_run = model_contract.run_folder(checkpoint).resolve() == run_dir.resolve()
    if same_run:
        if previous != ADAPTATION_SETTINGS:
            raise SystemExit("Cannot adapt in place over an original baseline run")
        for existing_model in existing_models:
            with zipfile.ZipFile(existing_model) as archive:
                steps = int(json.loads(archive.read("data"))["num_timesteps"])
            if steps > parent_steps:
                raise SystemExit(f"A newer checkpoint exists: {existing_model}; resume from that instead")
    elif existing_models:
        raise SystemExit("Destination already contains trained checkpoints; choose a new run name")
    if (run_dir / "STOP_TRAINING").exists():
        raise SystemExit("STOP_TRAINING exists; remove it only when ready to resume")
    path_sac_config.WIND_TRAINING = ADAPTATION_SETTINGS
elif checkpoint is not None:
    import model_contract
    source_contract = json.loads((model_contract.run_folder(checkpoint) / path_sac_config.CONTRACT_FILE).read_text())
    if source_contract.get("training", {}).get("wind_training") is not None:
        raise SystemExit("This checkpoint requires --wind-adaptation to preserve its training curriculum")

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": not run["gui"], "create_new_stage": False})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger, make_output_format
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    from path_sac_env import make_path_sac_env
    from hover_env import SimulationClosed

    def save_all(model, stem, buffer_stem):
        """Write complete files, then move them into place, so a crash during a
        save cannot leave a damaged checkpoint under the real name."""
        model_path = stem.with_name(f"{stem.name}.zip")
        partial_model = stem.with_name(f"{stem.name}_partial.zip")
        final_buffer = buffer_stem.with_name(f"{buffer_stem.name}.pkl")
        partial_buffer = buffer_stem.with_name(f"{buffer_stem.name}_partial.pkl")

        model.save(str(partial_model))
        model.save_replay_buffer(str(partial_buffer))

        partial_model.replace(model_path)
        partial_buffer.replace(final_buffer)

    class PathCallback(BaseCallback):
        """Advance the curriculum; log costs; save checkpoints and their state."""

        def _on_step(self):
            env.set_training_step(self.num_timesteps)

            for info, done in zip(self.locals["infos"], self.locals["dones"]):
                for name, value in info["costs"].items():
                    self.logger.record_mean(f"cost/{name}", value)

                self.logger.record_mean(
                    "flight/position_error", info["position_error"]
                )
                self.logger.record_mean(
                    "flight/residual_norm", info["residual_norm"]
                )
                self.logger.record_mean(
                    "flight/saturation_rate",
                    info["saturated_steps"] / max(info["physics_steps"], 1),
                )
                self.logger.record("flight/stage", info["stage"])
                self.logger.record_mean("wind/event_episode", float(info.get("wind_event_episode", False)))
                self.logger.record_mean("wind/event_active", float(info.get("wind_event_active", False)))

                if done:
                    self.logger.record_mean(
                        "flight/failure_rate",
                        float(info["failure_reason"] is not None),
                    )
                    self.logger.record_mean(
                        "flight/completion_rate", float(info["completed"])
                    )
                    self.logger.record_mean(
                        f"route_failure/{info['route_type']}",
                        float(info["failure_reason"] is not None),
                    )

            if self.num_timesteps % run["checkpoint_interval"] == 0:
                self.save_checkpoint()

            if args.wind_adaptation and (run_dir / "STOP_TRAINING").exists():
                print("Stop requested; ending learning and saving model/buffer.", flush=True)
                return False
            return True

        def save_checkpoint(self):
            steps = self.num_timesteps

            save_all(
                self.model,
                checkpoint_dir / f"flight_sac_{steps}_steps",
                checkpoint_dir / f"flight_replay_buffer_{steps}_steps",
            )

            write_json(checkpoint_dir / f"flight_state_{steps}_steps.json", {
                "timesteps": steps,
                "stage": path_sac_config.stage_for_step(steps)["name"],
                "run": run,
            })

            # Buffers are large; keep the newest. Models stay.
            buffers = sorted(
                checkpoint_dir.glob("flight_replay_buffer_*_steps.pkl"),
                key=lambda path: int(path.stem.split("_")[-2]),
            )

            for old in buffers[:-max(run["keep_replay_buffers"], 1)]:
                old.unlink()

            print(f"Checkpoint saved at {steps} steps.", flush=True)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = make_path_sac_env(simulation_app, render_sim=run["gui"])

    path_sac_config.write_contract(run_dir)

    stamp = time.strftime("%Y%m%d_%H%M%S")

    write_json(run_dir / f"config_snapshot_{stamp}.json", {
        "run": run,
        "contract": {
            "policy": path_sac_config.policy_contract(),
            "training": path_sac_config.training_contract(),
        },
    })

    venv = DummyVecEnv([lambda: Monitor(env)])

    try:
        if checkpoint is not None:
            model = SAC.load(
                str(checkpoint),
                env=venv,
                device=path_sac_config.SAC["device"],
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

            # Generators are not saved, so draw from a new stream, not a repeat.
            model.set_random_seed(run["seed"] + model.num_timesteps + 1)

            print(
                f"Resumed {checkpoint} at {model.num_timesteps} steps.",
                flush=True,
            )
        else:
            model = SAC(
                "MlpPolicy",
                venv,
                seed=run["seed"],
                verbose=1,
                **path_sac_config.SAC,
            )
            print("Started a new randomly initialized model.", flush=True)

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
                    str(run_dir / "tensorboard" / f"flight_{stamp}"),
                ),
            ],
        ))

        try:
            model.learn(
                total_timesteps=remaining,
                callback=PathCallback(),
                reset_num_timesteps=False,
            )

            save_all(
                model, run_dir / "final_model", run_dir / "final_replay_buffer"
            )
            print(f"Saved final model in {run_dir}", flush=True)

        except (KeyboardInterrupt, SimulationClosed) as stop:
            # Keep what was learned; resume from interrupted_model.zip.
            save_all(
                model,
                run_dir / "interrupted_model",
                run_dir / "interrupted_replay_buffer",
            )
            print(
                f"Stopped ({type(stop).__name__}); saved in {run_dir}",
                flush=True,
            )

        except Exception:
            # Keep what was learned, then show the real error.
            save_all(
                model,
                run_dir / "crashed_model",
                run_dir / "crashed_replay_buffer",
            )
            print(f"Training failed; saved crashed_* in {run_dir}", flush=True)
            raise

    finally:
        venv.close()


try:
    main()
finally:
    simulation_app.close()
