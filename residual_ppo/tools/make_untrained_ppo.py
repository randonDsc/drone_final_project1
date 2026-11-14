"""M2.5 adapter smoke test helper: save an UNTRAINED SB3 PPO (MlpPolicy, actor and critic 256x256) with the
PathSacEnv spaces, plus a matching ppo_contract.json ({"policy": path_sac_config.policy_contract()}).

No Isaac, no SimulationApp, no training. Output goes to a gitignored folder (default residual_ppo/scratch/...).
Run with Isaac's python.bat so the model is saved by the same SB3 version that will load it:

    # Directory: <repo root>
    & "<Isaac Sim>\\python.bat" residual_ppo\\tools\\make_untrained_ppo.py
"""
import argparse
import json
import sys
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))

import torch  # noqa: E402,F401  (torch first: two torch installs in Isaac Python; harmless elsewhere)
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402

import path_sac_config  # noqa: E402

# PathSacEnv spaces (confirmed in M1.5): obs Box(-10, 10, (128,)) float32, action Box(-1, 1, (4,)) float32.
OBS_SPACE = gym.spaces.Box(-10.0, 10.0, (128,), dtype=np.float32)
ACT_SPACE = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)


class SpacesOnlyEnv(gym.Env):
    """Carries the spaces for PPO construction; never stepped (no training happens)."""

    observation_space = OBS_SPACE
    action_space = ACT_SPACE

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(128, np.float32), {}

    def step(self, action):
        raise RuntimeError("untrained smoke model: never stepped")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESIDUAL_DIR / "scratch" / "m2_5_untrained_ppo"))
    args = parser.parse_args()
    from stable_baselines3 import PPO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = PPO("MlpPolicy", SpacesOnlyEnv(), seed=0, device="cpu",
                policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}})
    model.save(out / "model")
    (out / "ppo_contract.json").write_text(json.dumps(
        {"policy": path_sac_config.policy_contract(),
         "note": "UNTRAINED smoke-test model (M2.5 adapter smoke). Not a result."}, indent=2), encoding="utf-8")
    print("saved", out / "model.zip", "| actor", model.policy.mlp_extractor.policy_net,
          "| critic", model.policy.mlp_extractor.value_net, "| num_timesteps", model.num_timesteps)


if __name__ == "__main__":
    main()
