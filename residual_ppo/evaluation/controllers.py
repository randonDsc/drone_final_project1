"""Controllers for the evaluation protocol (docs/eval_protocol_v1.md, M2.3 runner design, M2.5 adapters). No Isaac.

Interface: name, act(observation) -> float32[4] (deterministic, normalized residual action), record() -> dict,
bind_env(env) (called once after the env is built and before any flight; refuses on a space mismatch).
- pid      zero residual (team PID-only).
- pid_ppo  SB3 PPO.load(model, device="cpu"); refused unless ppo_contract.json NEXT TO the model has a "policy" group
           equal to path_sac_config.policy_contract(), and unless the model's spaces equal the env's.
- pid_sac  refused unless path_sac_config.incompatibility(model, check_training=True) is None; act through the team's
           path_sac_policy.load_path_policy (actor mean, deterministic); spaces checked as for PPO.
Models are loaded in the constructor, which the runner calls BEFORE SimulationApp (two torch installs; CLAUDE.md).
Refusals raise ControllerRefused before anything is loaded with SB3 (contract checks come first).
"""
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

PPO_CONTRACT_FILE = "ppo_contract.json"
REPO_DIR = Path(__file__).resolve().parents[2]


class ControllerRefused(RuntimeError):
    """The model may not be evaluated under this protocol."""


def _sha256(path):
    """SHA-256 of the raw bytes (model zips and contract files are compared as stored)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repo_rel(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        return f"<outside repo>/{p.name}"


def _normalized(obj):
    return json.loads(json.dumps(obj))


def saved_model_info(model_path):
    """From the SB3 zip without unpickling: saved SB3 version (system_info.txt) and num_timesteps (data JSON)."""
    with zipfile.ZipFile(model_path) as z:
        info = z.read("system_info.txt").decode("utf-8", "replace") if "system_info.txt" in z.namelist() else ""
        data = json.loads(z.read("data").decode("utf-8")) if "data" in z.namelist() else {}
    version = None
    for line in info.splitlines():
        if "Stable-Baselines3" in line:
            version = line.split(":", 1)[1].strip()
    return {"saved_sb3_version": version, "num_timesteps": data.get("num_timesteps")}


def check_spaces(model_obs, model_act, env_obs, env_act):
    """Refuse unless the model's observation and action spaces equal the env's (gymnasium Box equality)."""
    problems = []
    if model_obs != env_obs:
        problems.append(f"observation space {model_obs} != env {env_obs}")
    if model_act != env_act:
        problems.append(f"action space {model_act} != env {env_act}")
    if problems:
        raise ControllerRefused("; ".join(problems))


def check_ppo_contract(model_path):
    """Refuse unless ppo_contract.json next to the model has a "policy" group equal to policy_contract()."""
    import path_sac_config

    path = Path(model_path).parent / PPO_CONTRACT_FILE
    if not path.is_file():
        raise ControllerRefused(f"no {PPO_CONTRACT_FILE} next to {_repo_rel(model_path)}")
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = _normalized(path_sac_config.policy_contract())
    stored = saved.get("policy")
    if stored is None:
        raise ControllerRefused(f"{PPO_CONTRACT_FILE} has no 'policy' group (missing policy group)")
    if stored != expected:
        keys = sorted(set(expected) | set(stored or {}))
        different = [k for k in keys if (stored or {}).get(k) != expected.get(k)]
        raise ControllerRefused(f"{PPO_CONTRACT_FILE} policy group differs from policy_contract(): "
                                f"{different or 'missing policy group'}")
    return path


def check_sac_contract(model_path):
    import path_sac_config

    reason = path_sac_config.incompatibility(model_path, check_training=True)
    if reason is not None:
        raise ControllerRefused(f"path_sac_config.incompatibility(check_training=True): {reason}")
    import model_contract

    return model_contract.run_folder(model_path) / path_sac_config.CONTRACT_FILE


def _require_model(model_path, name):
    if model_path is None:
        raise ControllerRefused(f"controller {name!r} needs --model")
    path = Path(model_path)
    if not path.is_file():
        raise ControllerRefused(f"model file not found: {_repo_rel(path)}")
    return path


class PidController:
    """PID only: zero residual through PathSacEnv, exactly as the team's eval_path_sac.py (policy None).

    M2.2 proved this bit-exact equal to a pure PID loop.
    """

    name = "pid"
    deterministic = True

    def __init__(self, model=None):
        if model is not None:
            raise ControllerRefused("controller 'pid' takes no model")

    def bind_env(self, env):
        pass

    def act(self, observation):
        return np.zeros(4, dtype=np.float32)

    def record(self):
        return {"name": self.name, "model": None, "contract": None, "device": None,
                "action": "np.zeros(4, float32) at every decision (team PID-only)",
                "equivalence": "M2.2: bit-exact equal to a pure PID loop (results/m2/zero_residual.json)"}


class _ModelController:
    name = None
    deterministic = True

    def bind_env(self, env):
        check_spaces(self.model.observation_space, self.model.action_space, env.observation_space, env.action_space)
        self.spaces_checked = True

    def _base_record(self, contract_path, contract_rule):
        import stable_baselines3

        return {"name": self.name, "model": _repo_rel(self.model_path), "model_sha256": _sha256(self.model_path),
                **saved_model_info(self.model_path), "runtime_sb3_version": stable_baselines3.__version__,
                "loaded_num_timesteps": int(self.model.num_timesteps),
                "contract": _repo_rel(contract_path), "contract_sha256": _sha256(contract_path),
                "contract_rule": contract_rule, "device": str(self.model.device), "deterministic": True,
                "spaces_checked_against_env": bool(getattr(self, "spaces_checked", False))}


class PidPpoController(_ModelController):
    name = "pid_ppo"

    def __init__(self, model=None):
        self.model_path = _require_model(model, self.name)
        self.contract_path = check_ppo_contract(self.model_path)  # before any SB3 import
        from stable_baselines3 import PPO

        self.model = PPO.load(str(self.model_path), device="cpu")

    def act(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float32)

    def record(self):
        return {**self._base_record(self.contract_path, f"{PPO_CONTRACT_FILE} 'policy' == policy_contract()"),
                "action": "PPO.predict(obs, deterministic=True)[0]"}


class PidSacController(_ModelController):
    name = "pid_sac"

    def __init__(self, model=None):
        self.model_path = _require_model(model, self.name)
        self.contract_path = check_sac_contract(self.model_path)  # before any SB3 import
        from stable_baselines3 import SAC

        import path_sac_policy

        self.model = SAC.load(str(self.model_path), device="cpu")  # metadata + space check only
        self.policy = path_sac_policy.load_path_policy(str(self.model_path))  # the team's loader acts

    def act(self, observation):
        return np.asarray(self.policy(observation), dtype=np.float32)

    def record(self):
        return {**self._base_record(self.contract_path, "path_sac_config.incompatibility(check_training=True) None"),
                "action": "path_sac_policy.load_path_policy(model) (SAC.predict deterministic = actor mean)"}


CONTROLLERS = {c.name: c for c in (PidController, PidPpoController, PidSacController)}


def make_controller(name, model=None):
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller {name!r}; known: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](model=model)
