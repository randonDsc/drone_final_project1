"""Shared PPO training setup (M3; reused by the M4 training script). No Isaac imports at module level.

Order inside an Isaac process (M2.1 torch issue, see CLAUDE.md ENV-1):
    import_torch_first()  ->  SimulationApp(...)  ->  build_training_env(app)  ->  build_model(venv)
torch and torch._dynamo must be imported BEFORE SimulationApp, otherwise building Adam after the app pulls in the
stale torch from extsDeprecated/omni.isaac.ml_archive (ImportError _EvalFrameOverride).
"""
import platform
import sys
from pathlib import Path

import gymnasium as gym

RESIDUAL_DIR = Path(__file__).resolve().parents[1]


def import_torch_first():
    """Import torch + torch._dynamo before SimulationApp. Returns the torch module."""
    import torch
    import torch._dynamo  # noqa: F401

    return torch


class SeedGuard(gym.Wrapper):
    """Refuses any training reset with a forbidden seed (eval / validation) or with evaluation options.

    Only the first reset gets a seed (0, from PPO's seed); later resets continue the env's RNG stream (seed None).
    """

    def __init__(self, env, forbidden):
        super().__init__(env)
        self.forbidden = frozenset(forbidden)
        self.seeded_resets = []      # non-None seeds, in order
        self.unseeded_resets = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None and int(seed) in self.forbidden:
            raise RuntimeError(f"SeedGuard: seed {seed} is an evaluation/validation seed; refusing a training reset")
        if options:
            raise RuntimeError(f"SeedGuard: training resets take no options (got {options}); "
                               "route_stage / wind_profile / route_type are evaluation-only")
        if seed is None:
            self.unseeded_resets += 1
        else:
            self.seeded_resets.append(int(seed))
        return self.env.reset(seed=seed, options=options)


def build_training_env(app, forbidden_seeds):
    """PathSacEnv -> SeedGuard -> Monitor -> DummyVecEnv (no VecNormalize). Returns (env, guard, venv)."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    import path_sac_env

    env = path_sac_env.make_path_sac_env(app)
    guard = SeedGuard(env, forbidden_seeds)
    venv = DummyVecEnv([lambda: Monitor(guard)])
    return env, guard, venv


def build_model(venv):
    from stable_baselines3 import PPO

    from config import ppo_config

    return PPO(ppo_config.POLICY, venv, verbose=0, **ppo_config.model_kwargs())


def make_curriculum_callback(env):
    """env.set_training_step(num_timesteps) after every step, like the team SAC's PathCallback."""
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        def _on_step(self):
            env.set_training_step(self.num_timesteps)
            return True

    return CurriculumCallback()


def count_gradient_steps(optimizer):
    """Counts optimizer.step() calls (SB3 PPO's _n_updates counts EPOCHS, not minibatch steps)."""
    counter = {"steps": 0}

    def hook(opt, args, kwargs):
        counter["steps"] += 1

    optimizer.register_step_post_hook(hook)
    return counter


def parameter_counts(model):
    """Actor = policy_net + action_net + log_std; critic = value_net + value_net head."""
    pol = model.policy

    def n(module):
        return sum(p.numel() for p in module.parameters())

    actor = n(pol.mlp_extractor.policy_net) + n(pol.action_net) + pol.log_std.numel()
    critic = n(pol.mlp_extractor.value_net) + n(pol.value_net)
    features = n(pol.features_extractor) + (0 if pol.share_features_extractor else
                                            n(pol.pi_features_extractor) + n(pol.vf_features_extractor))
    total = n(pol)
    return {"actor": actor, "critic": critic, "features_extractor": features, "total": total}


def state_dict_sha256(model):
    import hashlib

    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _redact(path):
    import run_common  # residual_ppo/evaluation is on sys.path (set by the calling script)

    return run_common.redact_isaac(path)


def versions():
    """Versions and origins from the IMPORTED modules (never importlib.metadata: stale after SimulationApp)."""
    import numpy as np
    import stable_baselines3
    import torch
    from stable_baselines3.common.utils import get_device

    return {
        "python": platform.python_version(),
        "numpy": np.__version__, "numpy_file": _redact(np.__file__),
        "gymnasium": gym.__version__, "gymnasium_file": _redact(gym.__file__),
        "stable_baselines3": stable_baselines3.__version__, "stable_baselines3_file": _redact(stable_baselines3.__file__),
        "torch": torch.__version__, "torch_file": _redact(torch.__file__),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "sb3_auto_device": str(get_device("auto")),
    }


def torch_origin_report():
    """Every loaded torch* module must come from the imported torch's package dir; none from ml_archive."""
    import torch

    root = Path(torch.__file__).resolve().parent
    loaded = outside = 0
    bad, synthetic, ml_archive_any, prebundle = [], [], [], []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not isinstance(f, str) or not f:
            continue
        low = f.lower()
        if "ml_archive" in low:
            ml_archive_any.append(name)
            if "pip_prebundle" in low:
                prebundle.append(name)   # the stale torch 2.11.0+cu128 lives here
        if name == "torch" or name.startswith("torch."):
            if not Path(f).is_absolute():
                # torch sets a placeholder __file__ on namespace objects (torch.ops: "_ops.py", torch.classes:
                # "_classes.py"); not a file on disk, so not an origin.
                synthetic.append((name, f))
                continue
            loaded += 1
            p = Path(f).resolve()
            if root not in p.parents:
                outside += 1
                bad.append((name, _redact(f)))
    return {"torch_package_dir": _redact(root), "torch_modules_with_file": loaded, "outside_package_dir": outside,
            "outside_examples": bad[:10], "synthetic_file_attributes": synthetic,
            "modules_from_ml_archive": len(ml_archive_any), "ml_archive_examples": sorted(ml_archive_any)[:10],
            "modules_from_ml_archive_pip_prebundle": len(prebundle), "prebundle_examples": sorted(prebundle)[:10]}
