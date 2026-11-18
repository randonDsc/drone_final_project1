"""M9 demo environments: the unchanged PathSacEnv / DirectEnv, with only reset() replaced. DESCRIPTIVE DEMO, NOT eval-v1.

What differs from training / eval-v1 (and nothing else):
- The scene: demo_config.SCENES[scene]; obstacles stay ACTIVE, only /World/Cube is deactivated. The sim is built
  exactly like hover_env.make_hover_sim (same wind config from path_sac_config.BASE_WIND), just with that scene path
  and deactivation list. No shared module is patched.
- reset(): instead of drawing a curriculum route and a wind, it takes a fixed TimedReference and a fixed wind dict
  (options["reference"], options["wind"]), then makes the same calls as PathSacEnv.reset in the same order:
  core.begin -> core.apply_wind -> sim.teleport, with the start state drawn by hover_task.sample_initial_state from
  the demo seed (>= 600000; eval / validation seeds refused).
- The core records the drone and the reference every 2nd physics step (60 Hz) after its normal step. Reading the
  state is side-effect free; check_scene.py proves the recorded env is bit-identical to a pure PID loop.
step(), observation, reward, failure rules, completion and the policy interface are inherited unchanged.

Import needs NumPy + gymnasium only (like path_sac_env); make_demo_sim needs a running SimulationApp.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np

import hover_task
import path_sac_config
import sim_config
from direct_control.direct_env import DirectCore, DirectEnv
from path_sac_env import PathSacEnv
from path_sac_policy import PathCore, PathTrace

from demo import demo_config

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent

RECORD_EVERY = 2  # physics steps (120 Hz) per recorded row -> 60 Hz
RECORD_COLUMNS = (
    ["t"]
    + [f"pos_{a}" for a in "xyz"]
    + [f"quat_{a}" for a in "wxyz"]
    + [f"vel_{a}" for a in "xyz"]
    + [f"angvel_{a}" for a in "xyz"]
    + [f"ref_pos_{a}" for a in "xyz"]
    + [f"ref_vel_{a}" for a in "xyz"]
    + ["rotor_saturated"]
)


def make_demo_sim(simulation_app, scene, log=print):
    """hover_env.make_hover_sim with the demo scene and deactivation list; then world.play() as make_path_sac_env."""
    from drone_sim import DroneSim

    spec = demo_config.SCENES[scene]
    profile = path_sac_config.BASE_WIND
    wind_config = dict(sim_config.WIND)
    wind_config.update(
        enabled=profile["enabled"],
        gust_std=profile["gust_std"],
        local_gust_std=profile["local_gust_std"],
    )
    for name in ("mean_wind", "gust_timescale"):
        if name in profile:
            wind_config[name] = profile[name]

    sim = DroneSim(
        simulation_app,
        REPO_DIR / spec["usd"],
        log=log,
        wind_config=wind_config,
        deactivate_prims=spec["deactivate"],
    )
    root = sim.stage.GetPrimAtPath(spec["obstacle_root"])
    if not (root.IsValid() and root.IsActive()):
        raise RuntimeError(f"{scene}: obstacle root {spec['obstacle_root']} is missing or inactive")
    sim.world.play()
    return sim


class _RecordingCore:
    """Mixin for PathCore / DirectCore: after each physics step, every RECORD_EVERY-th step, append one row."""

    def _init_recording(self):
        self.recording = None
        self._physics_count = 0

    def start_recording(self):
        self.recording = []
        self._physics_count = 0
        self._record(self.sim.read_drone_state(), self.saturated)

    def stop_recording(self):
        rows = np.array(self.recording, dtype=np.float64).reshape(-1, len(RECORD_COLUMNS))
        self.recording = None
        return rows

    def _record(self, truth, saturated):
        sample = self.sample()
        self.recording.append(np.concatenate([
            [self.time],
            np.asarray(truth["position"], dtype=np.float64).reshape(3),
            np.asarray(truth["orientation"], dtype=np.float64).reshape(4),
            np.asarray(truth["linear_velocity"], dtype=np.float64).reshape(3),
            np.asarray(truth["angular_velocity"], dtype=np.float64).reshape(3),
            np.asarray(sample.position, dtype=np.float64).reshape(3),
            np.asarray(sample.velocity, dtype=np.float64).reshape(3),
            [float(bool(saturated))],
        ]))

    def step(self, state, render=False):
        saturated = super().step(state, render=render)
        if self.recording is not None:
            self._physics_count += 1
            if self._physics_count % RECORD_EVERY == 0:
                self._record(self.sim.read_drone_state(), saturated)
        return saturated


class RecordingPathCore(_RecordingCore, PathCore):
    def __init__(self, sim, log=print):
        super().__init__(sim, log=log)
        self._init_recording()


class RecordingDirectCore(_RecordingCore, DirectCore):
    def __init__(self, sim, log=print):
        super().__init__(sim, log=log)
        self._init_recording()


class _DemoReset:
    """PathSacEnv.reset with a fixed reference and wind (see the module docstring)."""

    def reset(self, seed=None, options=None):
        options = options or {}
        demo_config.assert_demo_seed(seed)
        gym.Env.reset(self, seed=seed)

        reference = options["reference"]
        wind = options["wind"]

        task = dict(path_sac_config.EPISODE)
        task["target_position"] = reference.path[0]
        initial = hover_task.sample_initial_state(self.np_random, task)

        core = self.core
        core.begin(reference)
        core.apply_wind(wind, options.get("wind_profile", {}))

        self.sim.teleport(
            initial["position"],
            initial["orientation"],
            initial["linear_velocity"],
            initial["angular_velocity"],
            np.full(4, core.hover_motor_speed),
        )

        self.route_type = options.get("track_name", "demo")
        self.stage_name = "demo"
        self.decisions = 0
        self.saturated_streak = 0
        self.needs_reset = False
        self.trace = PathTrace()
        self.max_decisions = int(np.ceil(
            min(reference.duration, path_sac_config.EPISODE["max_seconds"])
            * path_sac_config.RL_RATE_HZ
        ))

        info = {
            "wind_mean": np.asarray(wind["mean_wind"], dtype=float).copy(),
            "wind_seed": int(wind["seed"]),
            "gust_scale": float(wind["gust_scale"]),
            "route_type": self.route_type,
            "stage": self.stage_name,
            "route_seconds": float(reference.duration),
            "initial_state": {k: np.asarray(v, dtype=float).tolist() for k, v in initial.items()},
        }
        return core.observe(self.sim.read_drone_state()), info


class DemoResidualEnv(_DemoReset, PathSacEnv):
    """PathSacEnv (PID + residual action; zero action = PID) with the demo reset and 60 Hz recording."""

    def __init__(self, sim, render_sim=False, log=print):
        super().__init__(sim, render_sim=render_sim, log=log)
        self.core = RecordingPathCore(sim, log=log)


class DemoDirectEnv(_DemoReset, DirectEnv):
    """DirectEnv (velocity + yaw command into the shared capped inner loop) with the demo reset and recording."""

    def __init__(self, sim, render_sim=False, log=print):
        super().__init__(sim, render_sim=render_sim, log=log)
        self.core = RecordingDirectCore(sim, log=log)


def make_demo_env(sim, arm, log=print):
    """arm: "residual" (PID and PID+PPO) or "direct". Several envs may share one sim; reset() re-initializes it."""
    if arm == "residual":
        return DemoResidualEnv(sim, log=log)
    if arm == "direct":
        return DemoDirectEnv(sim, log=log)
    raise ValueError(f"unknown arm {arm!r}")
