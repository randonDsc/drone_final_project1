"""Continual Backprop and the training-run presets. No Isaac Sim, but needs torch
and stable-baselines3, so run with Isaac Sim's python, from drone_code:

    C:\\isaacsim\\python.bat -m unittest tests.test_continual_backprop -v
"""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import path_sac_config

from continual_backprop import ContinualBackpropSAC, hidden_layers
from stable_baselines3 import SAC


PROJECT = Path(__file__).resolve().parents[2]


class TinyEnv(gym.Env):
    """Reward for keeping the action near a fixed target. Enough to train on."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (6,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (4,), np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self.steps += 1
        reward = -float(np.sum((action - 0.3) ** 2))
        return self.observation_space.sample(), reward, False, self.steps >= 20, {}


SETTINGS = {
    "replacement_rate": 0.05,
    "maturity": 5,
    "decay": 0.9,
    "bias_correction": True,
}


def tiny_model(**changes):
    model = ContinualBackpropSAC(
        "MlpPolicy",
        TinyEnv(),
        learning_starts=10,
        batch_size=16,
        seed=0,
        device="cpu",
        policy_kwargs={"net_arch": {"pi": [8, 16, 8], "qf": [16, 8]}},
    )
    model.attach_continual_backprop(dict(SETTINGS, **changes))
    return model


class Layers(unittest.TestCase):
    def test_every_hidden_layer_is_found(self):
        model = tiny_model()
        names = [layer.name for layer in model.continual_backprop.layers]

        self.assertEqual(names, [
            "actor_0", "actor_1", "actor_2",
            "critic0_0", "critic0_1", "critic1_0", "critic1_1",
        ])

    def test_last_actor_layer_feeds_both_heads(self):
        model = tiny_model()
        last = model.continual_backprop.layers[2]

        self.assertEqual(last.outgoing, [model.actor.mu, model.actor.log_std])

    def test_last_critic_layer_feeds_the_q_output(self):
        network = tiny_model().critic.q_networks[0]
        incoming, _, outgoing = hidden_layers(network)[-1]

        self.assertEqual(incoming.out_features, 8)
        self.assertEqual(outgoing[0].out_features, 1)

    def test_flight6_layer_sizes_are_found(self):
        model = ContinualBackpropSAC(
            "MlpPolicy", TinyEnv(), device="cpu",
            policy_kwargs={"net_arch": path_sac_config.TRAIN_RUNS[
                "flight6_cbp"]["net_arch"]},
        )
        model.attach_continual_backprop(SETTINGS)
        sizes = [
            layer.incoming.out_features
            for layer in model.continual_backprop.layers
        ]

        self.assertEqual(sizes, [64, 128, 256, 64] + [128, 256, 256, 128] * 2)


class Reset(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        # Gives Adam moments to clear.
        self.model.learn(60)
        self.layer = self.model.continual_backprop.layers[3]   # critic0_0

    def test_reset_redraws_incoming_and_zeroes_outgoing(self):
        layer = self.layer
        before = layer.incoming.weight.detach().clone()
        index = torch.tensor([2, 5])

        layer.reset(index, SETTINGS)

        self.assertFalse(torch.equal(layer.incoming.weight[index], before[index]))
        self.assertTrue(torch.equal(layer.incoming.weight[0], before[0]))
        outgoing = layer.outgoing[0].weight.detach()
        self.assertEqual(float(outgoing[:, index].abs().sum()), 0.0)
        self.assertEqual(int(layer.age[index].sum()), 0)

    def test_reset_clears_adam_moments(self):
        layer = self.layer
        index = torch.tensor([1])
        state = layer.optimizer.state[layer.incoming.weight]

        self.assertGreater(float(state["exp_avg"][1].abs().sum()), 0.0)

        layer.reset(index, SETTINGS)

        self.assertEqual(float(state["exp_avg"][1].abs().sum()), 0.0)
        self.assertEqual(float(state["exp_avg_sq"][1].abs().sum()), 0.0)

        outgoing_state = layer.optimizer.state[layer.outgoing[0].weight]
        self.assertEqual(float(outgoing_state["exp_avg"][:, 1].abs().sum()), 0.0)

    def test_target_critic_gets_the_same_reset(self):
        layer = self.layer
        index = torch.tensor([0, 3])

        layer.reset(index, SETTINGS)

        target_in, target_out = layer.mirror
        self.assertTrue(torch.equal(
            target_in.weight[index], layer.incoming.weight[index]
        ))
        self.assertEqual(float(target_out[0].weight[:, index].abs().sum()), 0.0)

    def test_bias_correction_keeps_the_mean_output(self):
        layer = self.layer
        index = torch.tensor([4])
        layer.mean_activation = torch.full_like(layer.mean_activation, 0.5)
        out = layer.outgoing[0]
        expected = out.bias + out.weight[:, index] @ layer.mean_activation[index]
        expected = expected.detach().clone()

        layer.reset(index, SETTINGS)

        torch.testing.assert_close(out.bias.detach(), expected)


class Training(unittest.TestCase):
    def test_young_neurons_are_never_replaced(self):
        model = tiny_model(maturity=10_000)
        model.learn(200)

        self.assertEqual(
            sum(layer.replaced for layer in model.continual_backprop.layers), 0
        )

    def test_neurons_are_replaced_and_training_stays_finite(self):
        model = tiny_model()
        model.learn(400)

        replaced = sum(layer.replaced for layer in model.continual_backprop.layers)
        self.assertGreater(replaced, 0)

        for parameter in model.policy.parameters():
            self.assertTrue(torch.isfinite(parameter).all())

    def test_flying_does_not_count_as_activity(self):
        model = tiny_model()
        model.predict(TinyEnv().observation_space.sample())

        self.assertTrue(all(
            layer.batches == 0 for layer in model.continual_backprop.layers
        ))

    def test_saved_model_loads_with_plain_sac(self):
        model = tiny_model()
        model.learn(50)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.zip")
            model.save(path)
            loaded = SAC.load(path, device="cpu")

        observation = TinyEnv().observation_space.sample()
        np.testing.assert_allclose(
            loaded.predict(observation, deterministic=True)[0],
            model.predict(observation, deterministic=True)[0],
            atol=1e-6,
        )


def using_run(name):
    return mock.patch.object(path_sac_config, "TRAIN_RUN", name)


class Presets(unittest.TestCase):
    def test_every_entry_is_complete(self):
        for name, entry in path_sac_config.TRAIN_RUNS.items():
            self.assertIn(entry["mode"], ("fresh", "finetune"), name)
            self.assertIn(entry["curriculum"], path_sac_config.CURRICULA, name)

            if entry["mode"] == "finetune":
                self.assertTrue(entry["resume_model"], name)

    def test_flight1_and_flight2_keep_the_original_network(self):
        for name in ("flight1", "flight2"):
            with using_run(name):
                self.assertEqual(
                    path_sac_config.sac_settings(), path_sac_config.SAC
                )
                self.assertNotIn(
                    "continual_backprop", path_sac_config.training_contract()
                )

    def test_flight6_runs_differ_only_in_continual_backprop(self):
        with using_run("flight6_cbp"):
            cbp = path_sac_config.training_contract()

        with using_run("flight6_nocbp"):
            plain = path_sac_config.training_contract()

        self.assertEqual(cbp.pop("continual_backprop"),
                         path_sac_config.CONTINUAL_BACKPROP)
        self.assertEqual(cbp, plain)
        self.assertEqual(
            plain["sac"]["policy_kwargs"]["net_arch"],
            {"pi": [64, 128, 256, 64], "qf": [128, 256, 256, 128]},
        )

    def test_flight1_curriculum_never_leaves_full(self):
        with using_run("flight1"):
            self.assertEqual(path_sac_config.stage_for_step(10**9)["name"], "full")

        with using_run("flight2"):
            self.assertEqual(
                path_sac_config.stage_for_step(1_500_000)["name"], "strong_ramp"
            )

    def test_saved_runs_match_their_presets(self):
        """The presets reproduce what flight1 and flight2 were trained with."""
        for name in ("flight1", "flight2"):
            model = PROJECT / path_sac_config.RUN_DIR / name / "final_model.zip"

            if not model.is_file():
                self.skipTest(f"{name} is not on this machine")

            with using_run(name):
                self.assertIsNone(
                    path_sac_config.incompatibility(model, check_training=True),
                    name,
                )

    def test_a_cbp_model_is_refused_without_cbp(self):
        with using_run("flight6_cbp"):
            contract = path_sac_config.training_contract()

        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / path_sac_config.CONTRACT_FILE).write_text(
                __import__("json").dumps({
                    "policy": path_sac_config.policy_contract(),
                    "training": contract,
                }),
                encoding="utf-8",
            )
            model = Path(folder) / "final_model.zip"

            with using_run("flight6_nocbp"):
                problem = path_sac_config.incompatibility(
                    model, check_training=True
                )

        self.assertIn("training.continual_backprop", problem)

    def test_resume_latest_picks_the_newest_model_with_a_buffer(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / "checkpoints").mkdir()

            def touch(path):
                path.write_bytes(b"")
                time.sleep(0.02)

            touch(run / "checkpoints" / "flight_sac_100_steps.zip")
            touch(run / "checkpoints" / "flight_replay_buffer_100_steps.pkl")
            touch(run / "interrupted_model.zip")
            touch(run / "interrupted_replay_buffer.pkl")
            # Newest, but its buffer is gone.
            touch(run / "checkpoints" / "flight_sac_200_steps.zip")

            latest = path_sac_config.latest_model(run)

        self.assertTrue(latest.endswith("interrupted_model.zip"), latest)

    def test_resume_settings(self):
        with using_run("flight1"), \
                mock.patch.object(path_sac_config, "RESUME", "some/model.zip"):
            settings = path_sac_config.train_settings()

        self.assertEqual(settings["mode"], "resume")
        self.assertEqual(settings["resume_model"], "some/model.zip")
        self.assertEqual(settings["run_name"], "flight1")
        self.assertEqual(settings["total_timesteps"], 1_500_000)


if __name__ == "__main__":
    unittest.main()
