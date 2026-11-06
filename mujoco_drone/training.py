"""SAC training, replay-buffer resume, and strict MuJoCo checkpoint contracts."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from drone_code import path_sac_config
from .config import ExperimentConfig, WindConfig
from .env import DroneEnv
from .experiments import json_write, source_state, versions


def sha256(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def policy_contract(env):
    # Code affecting controller/observation/dynamics must match, not only shape.
    source = source_state()["sha256"]
    required = ["mujoco_drone/physics.py", "mujoco_drone/weather.py", "mujoco_drone/controllers.py",
                "mujoco_drone/core.py", "mujoco_drone/env.py", "mujoco_drone/references.py",
                "drone_code/controller.py", "drone_code/hover_task.py", "drone_code/wind.py",
                "drone_code/path_sac_policy.py", "drone_code/path_sac_config.py",
                "drone_code/state_estimation.py", "drone_code/path_reference.py"]
    return dict(simulator="mujoco", algorithm="sac_residual", version=1,
                state_source="ground_truth_body_origin", observation_size=128,
                action_size=4, history_length=4, rl_rate_hz=10,
                physics=env.config.to_dict()["physics"],
                model=env.sim.audit(), mujoco_version=versions()["mujoco"],
                aerodynamics=dict(air_density=env.sim.wind.air_density,
                                  drag_coefficient=env.sim.wind.drag_coefficient,
                                  effective_area=env.sim.wind.effective_area,
                                  offsets=env.sim.wind.offsets.tolist(),
                                  area_weights=env.sim.wind.area_weights.tolist()),
                observation_scales=path_sac_config.OBSERVATION_SCALES,
                observation_clip=path_sac_config.OBSERVATION_CLIP,
                residual_limits=env.core.limits.tolist(),
                source_sha256={key: source[key] for key in required})


def validate_checkpoint(path, env, training=False):
    path = Path(path)
    sidecar = path.with_suffix(".contract.json")
    if not path.is_file() or not sidecar.is_file():
        raise ValueError("A MuJoCo SAC .zip and matching .contract.json are required. Isaac checkpoints cannot be assumed compatible; retrain locally.")
    saved = json.loads(sidecar.read_text())
    expected = policy_contract(env)
    different = [key for key, value in expected.items() if saved.get("policy", {}).get(key) != value]
    if different:
        raise ValueError("Checkpoint incompatible with this MuJoCo backend: "+", ".join(different))
    if saved.get("model_sha256") != sha256(path):
        raise ValueError("Checkpoint hash does not match its contract")
    if training and saved.get("training", {}).get("environment") != env.config.to_dict():
        # JSON converts tuples to lists; compare canonical encodings instead.
        if json.dumps(saved.get("training", {}).get("environment"), sort_keys=True) != json.dumps(env.config.to_dict(), sort_keys=True):
            raise ValueError("Resume environment differs from the training configuration")
    return saved


def load_policy(path, env):
    validate_checkpoint(path, env)
    from stable_baselines3 import SAC
    model = SAC.load(str(path), device="cpu")
    if model.observation_space != env.observation_space or model.action_space != env.action_space:
        raise ValueError("Saved policy spaces do not match the flight environment")
    return model


def save_checkpoint(model, path, env, training):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_model = path.with_suffix(".partial.zip")
    partial_buffer = path.with_suffix(".partial.pkl")
    model.save(str(partial_model))
    model.save_replay_buffer(str(partial_buffer))
    partial_model.replace(path)
    buffer = path.with_suffix(".replay.pkl")
    partial_buffer.replace(buffer)
    state = dict(policy=policy_contract(env), training=training,
                 total_timesteps=model.num_timesteps, model_sha256=sha256(path),
                 replay_sha256=sha256(buffer))
    partial_contract = path.with_suffix(".partial.json")
    json_write(partial_contract, state)
    partial_contract.replace(path.with_suffix(".contract.json"))


def train(output, steps=2_000_000, seed=0, resume=None, checkpoint_interval=5000,
          learning_starts=None, buffer_size=None, config=None):
    if steps < 1 or checkpoint_interval < 1:
        raise ValueError("steps and checkpoint_interval must be positive")
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.logger import configure
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = config or ExperimentConfig(seed=seed, max_seconds=60,
                  wind=WindConfig(scenario="curriculum", gust_std=(1, 1, .35), local_gust_std=(.3, .3, .15)))
    config = replace(config, seed=seed)
    if config.physics.controller == "original_literal":
        raise ValueError("The literal historical controller is diagnostic only; validate an adapted baseline before SAC")
    env = DroneEnv(config, training=True)
    monitored = Monitor(env, str(output/"episodes.csv"))
    settings = deepcopy(path_sac_config.SAC)
    settings["device"] = "cpu"
    # SB3 inserts this into policy_kwargs in place. Record it explicitly so a
    # saved training contract compares identically when resumed.
    settings["policy_kwargs"]["use_sde"] = False
    if learning_starts is not None:
        if learning_starts < 0:
            raise ValueError("learning_starts cannot be negative")
        settings["learning_starts"] = learning_starts
    if buffer_size is not None:
        if buffer_size < 256:
            raise ValueError("buffer_size must be >= 256")
        settings["buffer_size"] = buffer_size
    training = dict(environment=config.to_dict(), sac=settings,
                    reward=env.reward_config, curriculum=path_sac_config.CURRICULUM,
                    episode=path_sac_config.EPISODE, failure=path_sac_config.FAILURE,
                    reference="legacy Isaac TimedReference, retained for curriculum reproduction")

    class TrainingCallback(BaseCallback):
        def _on_step(self):
            env.training_step = self.num_timesteps
            for info in self.locals["infos"]:
                self.logger.record_mean("flight/position_error", info["position_error"])
                self.logger.record_mean("flight/saturation_rate", info["saturated_steps"]/max(info["physics_steps"], 1))
                self.logger.record_mean("flight/residual_norm", info["residual_norm"])
                for key, value in info["costs"].items():
                    self.logger.record_mean("cost/"+key, value)
                self.logger.record("curriculum/stage", info["stage"])
            if self.num_timesteps % checkpoint_interval == 0:
                save_checkpoint(self.model, output/"checkpoints"/f"sac_{self.num_timesteps}.zip", env, training)
                print(f"Saved checkpoint at {self.num_timesteps} decisions", flush=True)
            return True

    try:
        if resume:
            saved = validate_checkpoint(resume, env, training=True)
            canonical = lambda value: json.dumps(value, sort_keys=True)
            if canonical(saved["training"]) != canonical(training):
                raise ValueError("Resume training/reward/curriculum settings differ")
            buffer = Path(resume).with_suffix(".replay.pkl")
            if not buffer.is_file() or sha256(buffer) != saved["replay_sha256"]:
                raise ValueError("Matching replay buffer is required to resume")
            model = SAC.load(str(resume), env=monitored, device="cpu")
            model.load_replay_buffer(str(buffer))
            env.training_step = model.num_timesteps
            # SB3 does not checkpoint every RNG; resume is not bitwise replay.
            model.set_random_seed(seed+model.num_timesteps+1)
        else:
            model = SAC("MlpPolicy", monitored, seed=seed, verbose=1, **deepcopy(settings))
        model.set_logger(configure(str(output), ["stdout", "csv"]))
        json_write(output/"run.json", dict(training=training, additional_steps=steps, seed=seed,
                   resume=str(resume) if resume else None, versions=versions(), source=source_state()))
        model.learn(total_timesteps=steps, callback=TrainingCallback(), reset_num_timesteps=not bool(resume), log_interval=10)
        final = output/"final_model.zip"
        save_checkpoint(model, final, env, training)
        json_write(output/"training_result.json", dict(timesteps=model.num_timesteps,
                   gradient_updates=model._n_updates, final_model=str(final),
                   note="Training execution does not establish convergence or robustness; use independent paired evaluation."))
        return final
    finally:
        monitored.close()
