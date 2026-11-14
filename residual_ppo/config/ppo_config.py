"""PPO hyperparameters for the residual PPO (M3 design, confirmed by Claire 2026-11-14).

Only the RL algorithm differs from the team's Path SAC: same env, PID, observation/action/reward, curriculum.
ent_coef and target_kl are PROVISIONAL (may be revisited before the M4 run; any change = a new run).

Importable without Isaac and without torch (the activation is stored by name; policy_kwargs() maps it).
Callers put residual_ppo/ on sys.path first (path_sac_config is a shared module at its top level).

    # Directory: <repo root>
    & "<Isaac Sim>\\python.bat" -c "import sys; sys.path.insert(0, 'residual_ppo'); from config import ppo_config"
"""
import math

import path_sac_config

POLICY = "MlpPolicy"

# P1-P3: SB3 PPO keyword arguments. SB3 defaults are written out (P3) so a future default change cannot alter a run.
PPO = {
    "learning_rate": 3e-4,          # constant (matches the team SAC)
    "n_steps": 2048,                # decisions per rollout (1 env)
    "batch_size": 256,              # 8 minibatches per epoch
    "n_epochs": 10,
    "gamma": 0.99,                  # MUST equal path_sac_config.SAC["gamma"] (failure penalty = cost_cap / (1 - gamma))
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,          # P3 (SB3 default)
    "normalize_advantage": True,    # P3 (SB3 default)
    "ent_coef": 0.001,              # P1, PROVISIONAL
    "vf_coef": 0.5,                 # P3 (SB3 default)
    "max_grad_norm": 0.5,           # P3 (SB3 default)
    "use_sde": False,               # P3 (SB3 default): plain diagonal Gaussian
    "target_kl": 0.02,              # P2, PROVISIONAL: SB3 stops a rollout's epochs once approx_kl > 1.5 * 0.02
    "device": "cpu",                # Isaac torch is a CPU build; SB3 recommends CPU for small-MLP PPO
    "seed": 0,                      # same seed as the team SAC (single-seed limitation)
}
PROVISIONAL = ("ent_coef", "target_kl")

# P4-P6: policy network. Separate actor and critic, 256x256 each (SAC: pi/qf 256x256).
NET_ARCH = {"pi": [256, 256], "vf": [256, 256]}
ACTIVATION_FN = "ReLU"              # P5: matches the team SAC actor (SB3 PPO default is Tanh)
LOG_STD_INIT = -1.0                 # P6: initial std exp(-1) = 0.368 (SB3 default 0 -> std 1.0, ~32% components clipped)
ORTHO_INIT = True                   # P3 (SB3 default; action_net gain 0.01, value_net gain 1)

TOTAL_DECISIONS = 1_500_000         # budget (matches the team SAC); learn() runs whole rollouts

# Checks built into the config (fail at import).
assert PPO["gamma"] == path_sac_config.SAC["gamma"], "PPO gamma must equal the SAC gamma used by the failure penalty"
assert PPO["n_steps"] % PPO["batch_size"] == 0, "n_steps must be a multiple of batch_size"
assert not PPO["use_sde"]


def policy_kwargs():
    """SB3 policy_kwargs (imports torch)."""
    import torch

    return {"net_arch": {k: list(v) for k, v in NET_ARCH.items()},
            "activation_fn": getattr(torch.nn, ACTIVATION_FN),
            "log_std_init": LOG_STD_INIT,
            "ortho_init": ORTHO_INIT}


def policy_kwargs_record():
    """JSON form of policy_kwargs() (activation by name)."""
    return {"net_arch": {k: list(v) for k, v in NET_ARCH.items()}, "activation_fn": ACTIVATION_FN,
            "log_std_init": LOG_STD_INIT, "ortho_init": ORTHO_INIT}


def model_kwargs():
    """Everything passed to stable_baselines3.PPO(POLICY, env, **model_kwargs())."""
    return {**PPO, "policy_kwargs": policy_kwargs()}


def rollouts(total=TOTAL_DECISIONS):
    return math.ceil(total / PPO["n_steps"])


def planned_decisions(total=TOTAL_DECISIONS):
    """learn() stops after the rollout that reaches `total`: 733 x 2048 = 1,501,184 for 1.5M."""
    return rollouts(total) * PPO["n_steps"]


def minibatches_per_rollout():
    return PPO["n_epochs"] * (PPO["n_steps"] // PPO["batch_size"])


def max_gradient_steps(total=TOTAL_DECISIONS):
    """Upper bound on optimizer steps (target_kl early stopping can only lower it)."""
    return rollouts(total) * minibatches_per_rollout()


def initial_std():
    return math.exp(LOG_STD_INIT)


def expected_initial_clip(sigma=None, dims=4):
    """Clip fraction of N(0, sigma) samples outside [-1, 1]: per component, and per action (any of `dims`)."""
    sigma = initial_std() if sigma is None else sigma
    per_component = math.erfc(1.0 / (sigma * math.sqrt(2.0)))
    return per_component, 1.0 - (1.0 - per_component) ** dims


def training_contract():
    """What must stay the same for a PPO run to be continued or compared (stored in ppo_contract.json)."""
    return {
        "algorithm": "PPO",
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
    """ppo_contract.json content. The evaluation adapter (pid_ppo) requires "policy" == policy_contract()."""
    return {"policy": path_sac_config.policy_contract(), "training": training_contract()}
