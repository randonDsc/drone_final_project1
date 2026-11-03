"""DSC 180A: PPO residual lateral wind rejection on a physical Iris drone.

Run with Isaac Sim 6.2's python.bat. Run from any directory.

This deliberately preserves the working team-derived PD altitude, attitude, and
motor controller. PPO learns ONE bounded *lateral acceleration correction* by
shifting the controller's desired Y-position. This is real policy learning, but
not end-to-end motor control, visual navigation, or sensor-based RL.

No USD scenes or team Git files are saved/edited by this program.
"""
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mode", choices=["torchcheck", "smoke", "train", "evaluate", "all"], default="all")
parser.add_argument("--steps", type=int, default=2048,
                    help="PPO environment interactions, not physics steps; use 8192+ for exploration")
parser.add_argument("--gui", action="store_true", help="Show simulator (much slower than headless)")
parser.add_argument("--model", type=str, default="", help="Trained PPO .zip; for --mode evaluate")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# IMPORTANT: Isaac Sim's deprecated ML archive may insert an older PyTorch
# directory into the module search paths when SimulationApp starts. Preload
# the entire *matching* Dynamo package + optimizer BEFORE starting Kit so
# later optimizer imports reuse the correct modules from sys.modules.
# This imports no omni/Isaac APIs and changes no installed files.
def preload_torch_before_kit():
    import importlib
    import torch
    from pathlib import Path

    torch_dir = Path(torch.__file__).resolve().parent
    expected_dynamo = torch_dir / "_dynamo" / "__init__.py"
    if not expected_dynamo.exists():
        raise RuntimeError(
            "The active torch package does not have its matching _dynamo package: "
            f"{expected_dynamo}. Do not use deprecated ML-archive modules as a substitute."
        )
    print(f"[PRE-KIT] torch={torch.__version__} at {torch_dir}", flush=True)
    dynamo = importlib.import_module("torch._dynamo")
    origin = Path(dynamo.__file__).resolve()
    print(f"[PRE-KIT] torch._dynamo at {origin}", flush=True)
    if origin != expected_dynamo:
        raise RuntimeError(
            "Before Kit startup, torch._dynamo is already being imported from "
            f"a different installation: {origin}. Expected {expected_dynamo}"
        )
    parameter = torch.nn.Parameter(torch.zeros(1))
    torch.optim.Adam([parameter], lr=3e-4)
    print("[PRE-KIT] Adam optimizer initialization PASSED", flush=True)


if args.mode in ("torchcheck", "train", "evaluate", "all"):
    preload_torch_before_kit()
if args.mode == "torchcheck":
    print("[PRE-KIT] PyTorch compatibility test complete; Isaac Sim not started.", flush=True)
    raise SystemExit(0)

# CRITICAL: Launch SimulationApp before importing Isaac Sim/Omniverse APIs.
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": not args.gui})

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import omni.usd
from pxr import Gf, UsdGeom, UsdPhysics
from isaacsim.core.api import World
from isaacsim.core.prims import RigidPrim

# Gymnasium is needed for smoke tests; Stable-Baselines3 is imported only
# when training or evaluation is requested. Some Windows environments block
# compiled dependencies (such as pandas) imported by Stable-Baselines3.
import gymnasium as gym
from gymnasium import spaces

from controller import controller
from motor import MotorModel
from drone_diagnostics import read_total_com_in_body_frame, rotate_vector

PROJECT = Path(__file__).resolve().parents[1]
SCENE = PROJECT / "scenes" / "my_world_with_iris_drone.usd"
RESULTS = PROJECT / "results"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
RUN_DIR = RESULTS / f"ppo_{RUN_ID}"

# Keep the same flight/dynamics/wind settings as your PD experiments.
START_HEIGHT = 1.5
TARGET_OFFSET_XY = np.array([8.0, 0.0], dtype=float)
TARGET_HEIGHT = 1.5
PHYSICS_HZ = 120
CONTROL_HZ = 10
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ   # PPO holds one correction for 12 physics steps
DT = 1.0 / PHYSICS_HZ
EPISODE_SECONDS = 20.0
MAX_ACTION_STEPS = int(EPISODE_SECONDS * CONTROL_HZ)
WIND_START_S = 5.0
WIND_SPEED_MPS = 3.0
DRAG_KG_PER_S = 0.25

# RL supplies a world-Y acceleration bias of +/-0.60 m/s^2.
# Baseline controller uses kp_xy=0.3, so shifting its Y setpoint by +/-2 m
# adds +/-0.6 m/s^2 BEFORE its existing acceleration/tilt safety clipping.
MAX_LATERAL_ACCEL_MPS2 = 0.60
BASELINE_KP_XY = 0.3
MAX_VIRTUAL_Y_OFFSET_M = MAX_LATERAL_ACCEL_MPS2 / BASELINE_KP_XY


def wind_world(t: float, mode: str) -> np.ndarray:
    if t < WIND_START_S or mode == "calm":
        return np.zeros(3, dtype=float)
    if mode == "steady":
        return np.array([0.0, WIND_SPEED_MPS, 0.0], dtype=float)
    if mode == "gust":
        return np.array([0.0, WIND_SPEED_MPS + 1.5 * math.sin(2 * math.pi * 0.25 * t), 0.0])
    raise ValueError(f"Unknown wind mode: {mode}")


class IrisResidualEnv(gym.Env):
    """Gymnasium environment driving the actual PhysX drone, not a fake 2D model.

    Action: one number [-1, 1], added as a lateral acceleration request via
    the existing PD waypoint interface. Observations: ground truth (NOT IMU).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, scene_path: Path, wind_mode="steady", gui=False):
        super().__init__()
        if not scene_path.is_file():
            raise FileNotFoundError(f"Personal drone USD scene does not exist: {scene_path}")
        self.wind_mode = wind_mode
        self.gui = bool(gui)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        # [position error xyz (3), velocity xyz (3), quaternion wxyz (4),
        #  angular velocity xyz (3), previous lateral action (1)] = 14
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32)
        self.previous_action = 0.0
        self.episode_i = 0
        self.action_steps = 0
        self.physics_steps = 0
        self.last_info = {}

        if not omni.usd.get_context().open_stage(str(scene_path)):
            raise RuntimeError(f"USD stage could not be opened: {scene_path}")
        simulation_app.update()
        stage = omni.usd.get_context().get_stage()
        iris = stage.GetPrimAtPath("/iris")
        body_prim = stage.GetPrimAtPath("/iris/body")
        if not iris.IsValid() or not body_prim.IsValid():
            raise RuntimeError("Expected /iris and /iris/body in the personal USD scene")
        if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError("/iris/body does not have RigidBodyAPI")
        UsdGeom.XformCommonAPI(iris).SetTranslate(Gf.Vec3d(0, 0, START_HEIGHT))

        physics_scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(physics_scenes) > 1:
            raise RuntimeError(f"Multiple USD physics scenes found: {physics_scenes}")
        physics_path = physics_scenes[0] if physics_scenes else "/World/PhysicsScene"
        self.world = World(physics_dt=DT, rendering_dt=DT,
                           stage_units_in_meters=1.0, physics_prim_path=physics_path)
        self.body = self.world.scene.add(
            RigidPrim(prim_paths_expr="/iris/body", name="ppo_iris_body")
        )
        self.world.reset()
        initial_p, initial_q = self.body.get_world_poses()
        self.initial_p = np.asarray(initial_p, dtype=np.float32).copy()
        self.initial_q = np.asarray(initial_q, dtype=np.float32).copy()
        self.target_xy = self.initial_p[0, :2].astype(float) + TARGET_OFFSET_XY
        self.true_target = np.array([*self.target_xy, TARGET_HEIGHT], dtype=float)

        mass_views = []
        mass = 0.0
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if not p.startswith("/iris/") or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            api = UsdPhysics.RigidBodyAPI(prim)
            if api.GetRigidBodyEnabledAttr().Get() is False:
                continue
            v = RigidPrim(prim_paths_expr=p, name=f"ppo_mass_{len(mass_views)}",
                          reset_xform_properties=False)
            v.initialize()
            mass_views.append(v)
            mass += float(np.asarray(v.get_masses()).reshape(-1)[0])
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(f"Bad drone mass: {mass}")
        self.mass = mass
        # Drone mass geometry is fixed for this experiment. Cache the COM offset
        # rather than querying each mass prim at every physics step.
        self.com_offset_body = np.asarray(
            read_total_com_in_body_frame(self.body, mass_views), dtype=float
        )
        self.motors = MotorModel(5.84e-6, 0.06 * 5.84e-6, 1100.0, [-1, -1, +1, +1])
        self.rotor_positions = np.array([
            [0.13798545, -0.20671639, 0.023],
            [-0.12511168, 0.21875414, 0.023],
            [0.138, 0.20257577, 0.023],
            [-0.12415360, -0.22234580, 0.023],
        ], dtype=float)
        self.allocation = np.vstack([
            np.ones(4), self.rotor_positions[:, 1], -self.rotor_positions[:, 0],
            self.motors.reaction_signs * self.motors.km / self.motors.kf,
        ])
        self.rotor_max_force = self.motors.kf * self.motors.max_speed ** 2
        self.base_target = dict(height=TARGET_HEIGHT, xy=self.target_xy,
                                roll=0., pitch=0., yaw=0., mass=self.mass)
        print(f"[IRIS] mass={self.mass:.4f}kg start={self.initial_p[0].tolist()} "
              f"target={self.true_target.tolist()} physics={PHYSICS_HZ}Hz, control={CONTROL_HZ}Hz",
              flush=True)

    def state(self):
        p, q = self.body.get_world_poses()
        v = self.body.get_linear_velocities()
        w = self.body.get_angular_velocities()
        return (np.asarray(p[0], dtype=float), np.asarray(q[0], dtype=float),
                np.asarray(v[0], dtype=float), np.asarray(w[0], dtype=float))

    def observation(self):
        p, q, v, w = self.state()
        err = self.true_target - p
        return np.array([err[0]/8, err[1]/3, err[2]/2,
                         v[0]/4, v[1]/4, v[2]/4,
                         q[0], q[1], q[2], q[3],
                         w[0]/4, w[1]/4, w[2]/4,
                         self.previous_action], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_i += 1
        self.action_steps = 0
        self.physics_steps = 0
        self.previous_action = 0.0
        self.world.reset()
        self.body.set_world_poses(self.initial_p.copy(), self.initial_q.copy())
        self.body.set_linear_velocities(np.zeros((1, 3), dtype=np.float32))
        self.body.set_angular_velocities(np.zeros((1, 3), dtype=np.float32))
        self.world.play()
        return self.observation(), {"episode": self.episode_i, "wind_mode": self.wind_mode}

    def step(self, action):
        action_y = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        # PPO controls a BOUNDED setpoint offset, not the drone's USD pose.
        target = dict(self.base_target)
        target["xy"] = self.target_xy + np.array([0.0, action_y * MAX_VIRTUAL_Y_OFFSET_M])
        self.previous_action = action_y
        saturated_any = False
        last_wind = np.zeros(3)
        for _ in range(SUBSTEPS):
            if not simulation_app.is_running():
                raise RuntimeError("Isaac Sim window closed during PPO training")
            p, q, v, w = self.state()
            thrust, tx, ty, tz = controller(p, q, v, w, target)
            wrench = np.array([thrust, tx, ty, tz], dtype=float)
            wrench[1:] += np.cross(self.com_offset_body, [0., 0., thrust])
            raw_rotor_thrust = np.linalg.solve(self.allocation, wrench)
            saturated_any |= bool(np.any((raw_rotor_thrust < 0) |
                                         (raw_rotor_thrust > self.rotor_max_force)))
            rotor_thrust = np.clip(raw_rotor_thrust, 0, self.rotor_max_force)
            speeds = np.sqrt(rotor_thrust / self.motors.kf)
            individual_thrust, reaction_torque = self.motors.calculate(speeds)
            rotor_force = np.zeros((4, 3), dtype=float)
            rotor_force[:, 2] = individual_thrust
            force_body = rotor_force.sum(axis=0)
            torque_body = np.cross(self.rotor_positions, rotor_force).sum(axis=0)
            torque_body[2] += reaction_torque.sum()

            t = self.physics_steps * DT
            last_wind = wind_world(t, self.wind_mode)
            air_world = DRAG_KG_PER_S * (last_wind - v)
            air_world[2] = 0  # horizontal drag only
            q_inverse = q.copy()
            q_inverse[1:] *= -1
            air_body = rotate_vector(q_inverse, air_world)
            combined_force = force_body + air_body
            combined_torque = torque_body + np.cross(self.com_offset_body, air_body)
            self.body.apply_forces_and_torques_at_pos(
                forces=combined_force.astype(np.float32).reshape(1, 3),
                torques=combined_torque.astype(np.float32).reshape(1, 3),
                is_global=False,
            )
            self.world.step(render=self.gui)
            self.physics_steps += 1

        self.action_steps += 1
        p, q, v, w = self.state()
        error = self.true_target - p
        err_3d = float(np.linalg.norm(error))
        # Reward emphasizes returning to the true Y=0 line despite steady wind.
        # Very small action penalty discourages chattering. The existing PD
        # controller still supplies every motor action and controls X/Z/attitude.
        reward = (-0.25 * abs(error[1]) - 0.08 * abs(v[1])
                  - 0.015 * abs(error[0]) / 8
                  - 0.005 * action_y**2)
        failed = (not np.all(np.isfinite(p)) or not np.all(np.isfinite(v))
                  or p[2] < 0.25 or p[2] > 6.0
                  or abs(p[0]) > 35.0 or abs(p[1]) > 15.0)
        if failed:
            reward -= 100.0
        truncated = self.action_steps >= MAX_ACTION_STEPS
        info = {
            "time_s": round(self.physics_steps * DT, 5),
            "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
            "vx": float(v[0]), "vy": float(v[1]), "vz": float(v[2]),
            "error_m": err_3d, "lateral_abs_error_m": float(abs(error[1])),
            "target_x": float(self.true_target[0]),
            "target_y": float(self.true_target[1]),
            "target_z": float(self.true_target[2]),
            "wind_vy_mps": float(last_wind[1]),
            "ppo_action_y": action_y,
            "saturated": int(saturated_any),
        }
        self.last_info = info
        obs = self.observation()
        # Never return NaN observations to PPO.
        if not np.all(np.isfinite(obs)):
            failed = True
            reward -= 100.0
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        return obs, float(reward), bool(failed), bool(truncated), info

    def close(self):
        try:
            self.world.pause()
        except Exception:
            pass
        super().close()


def verify_torch_consistency():
    """Verify preloaded Torch/Dynamo survived Kit startup without mixing installs."""
    import sys
    import torch

    torch_dir = Path(torch.__file__).resolve().parent
    expected_dynamo = torch_dir / "_dynamo" / "__init__.py"
    loaded = sys.modules.get("torch._dynamo")
    origin = getattr(loaded, "__file__", None)
    print(f"[POST-KIT] torch._dynamo loaded from {origin}", flush=True)
    if origin is None or Path(origin).resolve() != expected_dynamo:
        raise RuntimeError(
            "Kit replaced or removed the preloaded torch._dynamo. "
            f"Expected {expected_dynamo}, got {origin}."
        )
    parameter = torch.nn.Parameter(torch.zeros(1))
    torch.optim.Adam([parameter], lr=3e-4)
    print("[POST-KIT] Adam optimizer initialization PASSED", flush=True)


def load_ppo_backend():
    """Load SB3 only for real training/evaluation (not the physics smoke test)."""
    verify_torch_consistency()
    print("[PPO] Importing Stable-Baselines3...", flush=True)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "Stable-Baselines3 could not load. If the exception mentions "
            "'An Application Control policy has blocked this file', "
            "Windows has denied loading a compiled Python dependency "
            "(often pandas._libs.sparse). This cannot be corrected by "
            "reinstalling Stable-Baselines3 alone. Inspect Windows "
            "CodeIntegrity/Operational events for the blocked file and policy."
        ) from exc

    class EpisodeReporter(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ep = 0

        def _on_step(self) -> bool:
            dones = self.locals.get("dones")
            if dones is not None and any(dones):
                self.ep += 1
                info = self.locals["infos"][0]
                print(f"[TRAIN] episode={self.ep} t={info.get('time_s', '?')}s "
                      f"end_y={info.get('y', float('nan')):.2f}m "
                      f"error={info.get('error_m', float('nan')):.2f}m", flush=True)
            return True

    return PPO, EpisodeReporter


def evaluate(env, model, condition, label, directory):
    env.wind_mode = condition
    obs, _ = env.reset()
    path = directory / f"{label}_{condition}.csv"
    rows = []
    for _ in range(MAX_ACTION_STEPS):
        if model is None:
            act = np.zeros((1,), dtype=np.float32)
        else:
            act, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(act)
        rows.append(dict(controller=label, wind=condition, run_id=RUN_ID,
                         wall_time_utc=datetime.now(timezone.utc).isoformat(), **info))
        if terminated or truncated:
            break
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    tail = [r for r in rows if r["time_s"] >= max(0, rows[-1]["time_s"] - 3)]
    tail_lateral = float(np.mean([r["lateral_abs_error_m"] for r in tail]))
    rms_lateral = float(np.sqrt(np.mean([r["y"]**2 for r in rows])))
    last = rows[-1]
    result = dict(controller=label, condition=condition, run_id=RUN_ID,
                  duration_s=last["time_s"], final_error_m=last["error_m"],
                  final_y_m=last["y"], lateral_rmse_m=rms_lateral,
                  mean_abs_lateral_error_last3s_m=tail_lateral,
                  any_motor_saturation=int(any(r["saturated"] for r in rows)),
                  file=path.name)
    print(f"[EVAL] {label:4s} {condition:6s} final_error={last['error_m']:.3f}m "
          f"tail_mean_|y|={tail_lateral:.3f}m  -> {path.name}", flush=True)
    return result


def main():
    np.random.seed(args.seed)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "run_id": RUN_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene": str(SCENE),
        "training_controller": "PPO residual Y-setpoint + team-derived PD/motor baseline",
        "policy_state": "physics ground truth; not onboard sensors",
        "action": "bounded +/-0.60 m/s^2 lateral acceleration via PD setpoint bias",
        "train_mode": "steady 3m/s crosswind beginning t=5s",
        "steps_requested": args.steps,
        "physics_hz": PHYSICS_HZ, "rl_control_hz": CONTROL_HZ,
        "episode_s": EPISODE_SECONDS, "seed": args.seed,
        "wind_model": "F_xy = 0.25*(wind_xy - drone_velocity_xy); gust_y=3+1.5*sin(2*pi*.25*t)",
        "caveat": "not a full motor-control PPO policy; short training may not outperform PD",
    }
    (RUN_DIR / "configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Run folder: {RUN_DIR}", flush=True)
    print("[BOOT] Creating Iris environment...", flush=True)
    env = IrisResidualEnv(SCENE, wind_mode="steady", gui=args.gui)
    print("[BOOT] Iris environment created.", flush=True)
    try:
        if args.mode == "smoke":
            env.reset()
            for i in range(4):
                _, reward, term, trunc, info = env.step([0.0])
                print(f"Smoke {i+1}: time={info['time_s']}, position="
                      f"({info['x']:.3f},{info['y']:.3f},{info['z']:.3f}), reward={reward:.3f}")
                if term or trunc:
                    break
            print("Smoke test complete (physics stepping worked).")
            return
        # Defer SB3/pandas imports until the physical simulation smoke test has passed.
        PPO, EpisodeReporter = load_ppo_backend()
        model = None
        if args.mode in ("all", "train"):
            model = PPO(
                "MlpPolicy", env, device="cpu", verbose=1, seed=args.seed,
                n_steps=256, batch_size=64, n_epochs=5,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                policy_kwargs=dict(net_arch=[64, 64]),
            )
            print(f"[TRAIN] PPO starting: requested_timesteps={args.steps}", flush=True)
            model.learn(total_timesteps=args.steps, callback=EpisodeReporter(), progress_bar=False)
            model_name = RUN_DIR / "iris_ppo_residual.zip"
            model.save(str(model_name))
            print(f"[TRAIN] Saved trained policy: {model_name}", flush=True)
        elif args.mode == "evaluate":
            if not args.model:
                raise ValueError("--mode evaluate requires --model PATH_TO_ZIP")
            model = PPO.load(args.model, device="cpu")

        if args.mode in ("all", "evaluate"):
            summaries = []
            for mode in ("calm", "steady", "gust"):
                summaries.append(evaluate(env, None, mode, "pd", RUN_DIR))
                summaries.append(evaluate(env, model, mode, "ppo", RUN_DIR))
            summary_path = RUN_DIR / "comparison.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
                w.writeheader()
                w.writerows(summaries)
            print("\n=== Comparison (lower is better) ===")
            for r in summaries:
                print(f"  {r['controller']:3s} {r['condition']:6s} "
                      f"mean |y| last3s = {r['mean_abs_lateral_error_last3s_m']:.3f}m")
            print(f"Data written to: {RUN_DIR}")
    finally:
        env.close()


try:
    main()
except Exception:
    # Print before closing SimulationApp, so the real exception stays visible
    # even if Isaac Sim outputs extensive shutdown logs.
    import traceback
    print("\n[ERROR] Iris experiment failed; full Python traceback follows:", flush=True)
    traceback.print_exc()
    raise
finally:
    # Prevent lingering background Isaac Sim processes even after exceptions.
    simulation_app.close()
