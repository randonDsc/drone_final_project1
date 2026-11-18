"""Adapter for a PPO DIRECT-control model (M8 arm), written for the M9 demo and meant to be REUSED by the direct
evaluation runner (prespec section 8 item 2, option (b)). No Isaac imports.

Same interface and order of checks as the frozen evaluation/controllers.py adapters (imported, never edited):
  name, deterministic, act(observation) -> float32[4], record() -> dict, bind_env(env).
  1. Contract FIRST, before any SB3 import: ppo_contract.json NEXT TO the model must have a "policy" group equal to
     config/direct_ppo_config.policy_contract() (the residual contract + "arm": "direct" + the action mapping).
     A residual model (policy group == path_sac_config.policy_contract(), no arm) is therefore refused, and the
     frozen pid_ppo adapter refuses every direct model (its rule is equality with the residual contract).
  2. PPO.load(model, device="cpu"), called BEFORE SimulationApp by the runner (two torch installs; CLAUDE.md ENV-1).
  3. bind_env refuses unless the env is a direct_control.direct_env.DirectEnv (its core a DirectCore) AND the model's
     spaces equal the env's. (Spaces alone cannot tell the arms apart: both are Box(-1,1,(4,)) / Box(-10,10,(128,)).)
  4. act = PPO.predict(observation, deterministic=True)[0] as float32.
"""
import json
from pathlib import Path

import numpy as np

from evaluation import controllers as frozen

ControllerRefused = frozen.ControllerRefused
PPO_CONTRACT_FILE = frozen.PPO_CONTRACT_FILE


def check_direct_contract(model_path):
    """Refuse unless ppo_contract.json next to the model has a "policy" group equal to the direct policy contract."""
    from config import direct_ppo_config

    path = Path(model_path).parent / PPO_CONTRACT_FILE
    if not path.is_file():
        raise ControllerRefused(f"no {PPO_CONTRACT_FILE} next to {frozen._repo_rel(model_path)}")
    stored = json.loads(path.read_text(encoding="utf-8")).get("policy")
    expected = frozen._normalized(direct_ppo_config.policy_contract())
    if stored is None:
        raise ControllerRefused(f"{PPO_CONTRACT_FILE} has no 'policy' group")
    if stored != expected:
        keys = sorted(set(expected) | set(stored))
        different = [k for k in keys if stored.get(k) != expected.get(k)]
        raise ControllerRefused(f"{PPO_CONTRACT_FILE} policy group differs from direct_ppo_config.policy_contract(): "
                                f"{different}")
    return path


class DirectPpoController(frozen._ModelController):
    name = "ppo_direct"

    def __init__(self, model=None):
        self.model_path = frozen._require_model(model, self.name)
        self.contract_path = check_direct_contract(self.model_path)  # before any SB3 import
        from stable_baselines3 import PPO

        self.model = PPO.load(str(self.model_path), device="cpu")

    def bind_env(self, env):
        from direct_control.direct_env import DirectCore, DirectEnv

        if not isinstance(env, DirectEnv) or not isinstance(env.core, DirectCore):
            raise ControllerRefused(f"{self.name} needs a DirectEnv with a DirectCore, got "
                                    f"{type(env).__name__} / {type(getattr(env, 'core', None)).__name__}")
        super().bind_env(env)
        self.env_checked = True

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def record(self):
        return {**self._base_record(self.contract_path,
                                    f"{PPO_CONTRACT_FILE} 'policy' == direct_ppo_config.policy_contract()"),
                "env_type_checked": bool(getattr(self, "env_checked", False)),
                "action": "PPO.predict(obs, deterministic=True)[0] -> DirectCore (velocity + yaw command)"}
