"""PPO hyperparameters for PPO DIRECT control (M8; decided by Claire 2026-11-15/16, plan section 3 / 3b).

Everything is taken from the residual arm (config/ppo_config.py) EXCEPT, by Claire's decision:
  - net_arch pi [64, 128, 256, 64], vf [128, 256, 256, 128] (the paper's actor/critic; residual arm 256x256 each);
  - log_std_init = ln(0.4) = -0.9163 (residual arm -1.0).
The log_std is SB3's standard one: state-independent, learned, one value per action dimension (as in the residual
arm). target_kl stays 0.02 for parity (D2), with the pre-declared undertraining check (prespec section 4, A5).

Same API as ppo_config, so the training script and the audit take either module. Importable without Isaac/torch.
"""
import math

import path_sac_config
from config import ppo_config as _residual

ARM = "direct"
POLICY = _residual.POLICY
PPO = dict(_residual.PPO)                       # identical to the residual arm, including seed 0 and target_kl 0.02
PROVISIONAL = _residual.PROVISIONAL

NET_ARCH = {"pi": [64, 128, 256, 64], "vf": [128, 256, 256, 128]}
ACTIVATION_FN = _residual.ACTIVATION_FN         # ReLU
LOG_STD_INIT = math.log(0.4)                    # -0.916290731874155
ORTHO_INIT = _residual.ORTHO_INIT

TOTAL_DECISIONS = _residual.TOTAL_DECISIONS     # 1,500,000 (733 rollouts = 1,501,184)

# The direct action mapping (direct_control/direct_env.py). Stored in the policy contract so a direct model can
# never be flown by the pid_ppo adapter (its policy group must equal path_sac_config.policy_contract()).
ACTION_MAPPING = {
    "frame": "world",
    "velocity_limits_mps": [3.0, 3.0, 1.5],
    "yaw": "angle, pi * a3 rad",
    "inner_loop": "shared PIDController (controller.py, 1.5 m/s^2 cap) via target substitution p + (kd/kp) v_cmd",
    "integrators": "integrate at the current position: xy/height stay 0, yaw integrates",
    "residual_correction": 0.0,
}

assert PPO == _residual.PPO
assert PPO["gamma"] == path_sac_config.SAC["gamma"]


def policy_kwargs():
    """SB3 policy_kwargs (imports torch)."""
    import torch

    return {"net_arch": {k: list(v) for k, v in NET_ARCH.items()},
            "activation_fn": getattr(torch.nn, ACTIVATION_FN),
            "log_std_init": LOG_STD_INIT,
            "ortho_init": ORTHO_INIT}


def policy_kwargs_record():
    return {"net_arch": {k: list(v) for k, v in NET_ARCH.items()}, "activation_fn": ACTIVATION_FN,
            "log_std_init": LOG_STD_INIT, "ortho_init": ORTHO_INIT}


def model_kwargs():
    return {**PPO, "policy_kwargs": policy_kwargs()}


rollouts = _residual.rollouts
planned_decisions = _residual.planned_decisions
minibatches_per_rollout = _residual.minibatches_per_rollout
max_gradient_steps = _residual.max_gradient_steps


def initial_std():
    return math.exp(LOG_STD_INIT)


def expected_initial_clip(sigma=None, dims=4):
    return _residual.expected_initial_clip(initial_std() if sigma is None else sigma, dims)


def policy_contract():
    """The residual policy contract plus the arm and its action mapping (so the two arms can never be confused)."""
    return {**path_sac_config.policy_contract(), "arm": ARM, "action_mapping": ACTION_MAPPING}


def training_contract():
    return {
        "algorithm": "PPO",
        "arm": ARM,
        "ppo": PPO,
        "policy": POLICY,
        "policy_kwargs": policy_kwargs_record(),
        "total_decisions": TOTAL_DECISIONS,
        "reward": path_sac_config.reward_settings(),
        "failure": path_sac_config.FAILURE,
        "episode": path_sac_config.EPISODE,
        "curriculum": path_sac_config.CURRICULUM,
    }


def contract():
    """ppo_contract.json content for a direct model."""
    return {"policy": policy_contract(), "training": training_contract()}
