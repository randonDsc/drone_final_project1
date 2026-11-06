import json
from dataclasses import replace

import pytest

from mujoco_drone.config import ExperimentConfig, WindConfig
from mujoco_drone.env import DroneEnv
from mujoco_drone.training import load_policy, train, validate_checkpoint


def test_real_gradient_updates_checkpoint_prediction_and_resume(tmp_path):
    config = ExperimentConfig(max_seconds=.5, wind=WindConfig(scenario="curriculum"))
    checkpoint = train(tmp_path/"first", steps=16, learning_starts=4, buffer_size=512, config=config)
    result = json.loads((tmp_path/"first"/"training_result.json").read_text())
    assert result["gradient_updates"] == 12
    env = DroneEnv(config)
    observation, _ = env.reset(seed=4)
    model = load_policy(checkpoint, env)
    assert env.action_space.contains(model.predict(observation, deterministic=True)[0])
    resumed = train(tmp_path/"resume", steps=8, resume=checkpoint, learning_starts=4, buffer_size=512, config=config)
    resumed_result = json.loads((tmp_path/"resume"/"training_result.json").read_text())
    assert resumed_result["timesteps"] == 24
    assert resumed_result["gradient_updates"] == 20
    changed = DroneEnv(replace(config, physics=replace(config.physics, max_tilt_degrees=12)))
    with pytest.raises(ValueError, match="incompatible"):
        load_policy(resumed, changed)
    # A matching shape must never be sufficient to load another simulator.
    sidecar = resumed.with_suffix(".contract.json")
    contract = json.loads(sidecar.read_text())
    contract["policy"]["simulator"] = "isaac_sim"
    sidecar.write_text(json.dumps(contract))
    with pytest.raises(ValueError, match="simulator"):
        validate_checkpoint(resumed, env)
    env.close()
    changed.close()


def test_unknown_checkpoint_is_not_silently_used(tmp_path):
    path = tmp_path/"isaac.zip"
    path.write_bytes(b"unverified weights")
    env = DroneEnv()
    with pytest.raises(ValueError, match="Isaac"):
        load_policy(path, env)
    env.close()
