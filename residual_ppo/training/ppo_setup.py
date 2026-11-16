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
    """Import torch + the lazily imported submodules SB3 needs before SimulationApp. Returns the torch module.

    torch._dynamo: imported by Adam (M2.1). torch.utils.serialization(.config): imported lazily by torch.load
    (torch/serialization.py), i.e. by PPO.load; after SimulationApp it resolved to the stale ml_archive copy
    (M5 S2 first attempt, caught by the origin check)."""
    import torch
    import torch._dynamo  # noqa: F401
    import torch.utils.serialization  # noqa: F401
    import torch.utils.serialization.config  # noqa: F401

    return torch


def pin_torch_imports():
    """Call right AFTER SimulationApp(...). Kit's OmniFinder (first on sys.meta_path) maps not-yet-imported module
    names to extension prebundles, so lazily imported torch.* submodules resolved to the stale
    extsDeprecated/omni.isaac.ml_archive torch (M5: torch.utils.serialization via PPO.load,
    torch._subclasses.schema_check_mode in an error path). This finder, inserted ahead of it, resolves torch /
    torch.* only through the parent package's own __path__ (the imported kit torch). Nothing on disk changes.
    Returns a check record."""
    import importlib.abc
    import importlib.machinery
    import importlib.util

    import torch

    class _PinTorchFinder(importlib.abc.MetaPathFinder):
        pinned = True

        def find_spec(self, name, path=None, target=None):
            if not name.startswith("torch."):
                return None
            if path is None:
                parent = sys.modules.get(name.rpartition(".")[0])
                path = getattr(parent, "__path__", None)
            if path is None:
                return None
            return importlib.machinery.PathFinder.find_spec(name, path, target)

    if not any(getattr(f, "pinned", False) for f in sys.meta_path):
        sys.meta_path.insert(0, _PinTorchFinder())
    probe = "torch._subclasses.schema_check_mode"
    origin = None if probe in sys.modules else importlib.util.find_spec(probe).origin
    root = Path(torch.__file__).resolve().parent
    ok = origin is None or root in Path(origin).resolve().parents
    return {"pinned": True, "meta_path_first": type(sys.meta_path[0]).__name__, "probe": probe,
            "probe_origin": None if origin is None else _redact(origin), "probe_inside_kit_torch": ok}


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


ARMS = ("residual", "direct")


def arm_configs(arm):
    """(experiment_config, ppo_config) modules of an arm. residual = PID + PPO residual; direct = PPO direct control."""
    if arm == "residual":
        from config import experiment_config, ppo_config
        return experiment_config, ppo_config
    if arm == "direct":
        from config import direct_experiment_config, direct_ppo_config
        return direct_experiment_config, direct_ppo_config
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")


def make_arm_env(app, arm="residual"):
    """The arm's env (needs SimulationApp running): PathSacEnv (residual) or direct_control DirectEnv (direct)."""
    if arm == "residual":
        import path_sac_env
        return path_sac_env.make_path_sac_env(app)
    if arm == "direct":
        from direct_control import direct_env
        return direct_env.make_direct_env(app)
    raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")


def build_training_env(app, forbidden_seeds, monitor_path=None, arm="residual"):
    """Arm env -> SeedGuard -> Monitor -> DummyVecEnv (no VecNormalize). Returns (env, guard, venv).

    monitor_path: optional per-launch Monitor CSV (SB3 appends ".monitor.csv")."""
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = make_arm_env(app, arm)
    guard = SeedGuard(env, forbidden_seeds)
    venv = DummyVecEnv([lambda: Monitor(guard, filename=None if monitor_path is None else str(monitor_path))])
    return env, guard, venv


def build_model(venv, config=None):
    """SB3 PPO from an arm's ppo_config module (default: the residual arm's)."""
    from stable_baselines3 import PPO

    if config is None:
        from config import ppo_config as config

    return PPO(config.POLICY, venv, verbose=0, **config.model_kwargs())


def make_curriculum_callback(env):
    """env.set_training_step(num_timesteps) after every step, like the team SAC's PathCallback."""
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        def _on_step(self):
            env.set_training_step(self.num_timesteps)
            return True

    return CurriculumCallback()


def all_parameters_finite(model):
    import torch

    with torch.no_grad():
        return all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())


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
