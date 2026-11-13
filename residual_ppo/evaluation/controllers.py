"""Controllers for the evaluation protocol (docs/milestone2_plan.md, M2.3 runner design). No Isaac.

Interface: name, act(observation) -> float32[4] (deterministic, normalized residual action), record() -> dict.
Only "pid" is implemented. "pid_ppo" (M3+) and "pid_sac" (M7) plug in here later; the runner does not change.
"""
import numpy as np


class PidController:
    """PID only: zero residual through PathSacEnv, exactly as the team's eval_path_sac.py (policy None).

    M2.2 proved this bit-exact equal to a pure PID loop.
    """

    name = "pid"
    deterministic = True

    def act(self, observation):
        return np.zeros(4, dtype=np.float32)

    def record(self):
        return {"name": self.name, "model": None, "contract": None, "device": None,
                "action": "np.zeros(4, float32) at every decision (team PID-only)",
                "equivalence": "M2.2: bit-exact equal to a pure PID loop (results/m2/zero_residual.json)"}


class _NotImplementedController:
    name = None

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"controller {self.name!r} is not implemented yet. Planned: {self.planned}")


class PidPpoController(_NotImplementedController):
    name = "pid_ppo"
    planned = "M3+: SB3 PPO.load(model); act = model.predict(obs, deterministic=True)[0]; device cpu."


class PidSacController(_NotImplementedController):
    name = "pid_sac"
    planned = ("M7: refuse unless path_sac_config.incompatibility(model) is None; "
               "act = path_sac_policy.load_path_policy(model) (actor mean, deterministic).")


CONTROLLERS = {c.name: c for c in (PidController, PidPpoController, PidSacController)}


def make_controller(name, **kwargs):
    if name not in CONTROLLERS:
        raise ValueError(f"unknown controller {name!r}; known: {sorted(CONTROLLERS)}")
    return CONTROLLERS[name](**kwargs)
