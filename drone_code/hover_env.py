"""Gymnasium environment: PID hover plus a bounded learned residual.

    IMU -> estimator -> PID command + held residual -> mixer -> Isaac Sim

state_source "imu" is the default. Two kinds of state stay separate here:

    control state   What the PID controller and policy see. From the estimator,
                    fed only by sim.read_imu(). See _control_state.
    truth           sim.read_drone_state(). Physics (wind drag), reward, failure
                    limits, and logged metrics only. Never reaches the
                    controller or observation.

state_source "ground_truth" sets the control state to truth, explicitly: the
perfect-state baseline. hover_sac_env.py builds on this class.

One drone in one environment. The env never starts or closes SimulationApp;
the launching script owns it, because it must exist before Isaac imports.
Build the sim with make_hover_env(), or pass any object with the same methods
as drone_sim.DroneSim (the unit tests use a small stand-in).
"""
from pathlib import Path

import gymnasium as gym
import numpy as np

import hover_task
import model_contract
import hover_ppo_config
import sensor_config
import sim_config

from controller import PIDController
from state_estimation import ImuEstimator, ImuNoise


# Consecutive physics steps without an IMU sample before giving up [1/120 s].
MAX_MISSED_IMU_SAMPLES = 120


class SimulationClosed(RuntimeError):
    """The Isaac Sim window was closed during an episode."""


def wind_settings(wind_profile):
    """A hover_ppo_config profile by name, or a settings dict used as it is."""
    if isinstance(wind_profile, str):
        return hover_ppo_config.WIND_PROFILES[wind_profile]

    return wind_profile


class DroneHoverEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        sim,
        wind_profile=hover_ppo_config.DEFAULT_WIND_PROFILE,
        render_sim=False,
        task=hover_ppo_config.HOVER,
        residual_limits=hover_ppo_config.RESIDUAL_LIMITS,
        reward=hover_ppo_config.REWARD,
        failure=hover_ppo_config.FAILURE,
        scales=hover_ppo_config.OBSERVATION_SCALES,
        gamma=hover_ppo_config.PPO["gamma"],
        state_source=hover_ppo_config.DEFAULT_STATE_SOURCE,
        imu_noise=sensor_config.IMU_NOISE,
        estimator=sensor_config.ESTIMATOR,
    ):
        super().__init__()

        if state_source not in hover_ppo_config.STATE_SOURCES:
            raise ValueError(f"Unknown state source: {state_source}")

        self.sim = sim
        self.state_source = state_source
        self.estimator = None
        self.imu_noise = None
        self.missed_imu_samples = 0

        if state_source == "imu":
            self.estimator = ImuEstimator(
                estimator,
                sim.dt,
                hover_task.GRAVITY,
                sim.imu_mount_rotation,
                sim.imu_mount_offset,
            )
            self.imu_noise = ImuNoise(imu_noise, sim.dt)

        self.wind_profile = wind_settings(wind_profile)
        self.render_sim = bool(render_sim)
        self.task = task
        self.reward_weights = reward
        self.failure_limits = failure
        self.scales = scales

        self.target_position = np.asarray(
            task["target_position"], dtype=float
        )
        self.decimation = int(task["control_decimation"])
        self.max_decisions = int(task["max_decisions"])

        if self.target_position[2] <= failure["min_height"]:
            raise ValueError("Hover target is below the failure height.")

        self.residual_limits = hover_task.residual_limits_array(
            residual_limits
        )
        self.penalty = hover_task.failure_penalty(reward, failure, gamma)

        self.hover_thrust = sim.total_mass * hover_task.GRAVITY

        self.pid = PIDController()
        self.last_saturated = False

        self.pid_target = sim.base_target()
        self.pid_target["xy"] = self.target_position[:2].copy()
        self.pid_target["height"] = float(self.target_position[2])

        # Motors spin at this speed in level hover; used to restart the rotors.
        self.hover_motor_speed = np.sqrt(
            self.hover_thrust / (4 * sim.motors.kf)
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(hover_task.OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.previous_action = np.zeros(4, dtype=np.float32)
        self.decisions = 0
        self.wind_episode = None
        self.needs_reset = True

    def _pid_command(self, state, dt=0.0):
        """The PID command. dt = 0 only reads it, for observations."""
        return np.array(
            self.pid.update(
                positions=state["position"],
                orientations=state["orientation"],
                velocities=state["linear_velocity"],
                angular_velocities=state["angular_velocity"],
                target=self.pid_target,
                dt=dt,
                actuators_saturated=self.last_saturated,
            ),
            dtype=float,
        )

    def _sample_wind(self):
        """One episode's reset_episode arguments."""
        return hover_task.sample_wind(self.np_random, self.wind_profile)

    def _before_physics_step(self):
        """Called before every physics step, for time-varying wind."""

    def _step_costs(self, truth, action, saturated_steps, physics_steps):
        return hover_task.step_costs(
            truth,
            self.target_position,
            action,
            self.previous_action,
            self.reward_weights,
        )

    def _control_state(self, truth):
        """The only state the controller and policy may use."""
        if self.estimator is None:
            return truth

        return self.estimator.state()

    def _update_estimator(self):
        """Feed one IMU sample, after a physics step. No simulator state."""
        if self.estimator is None:
            return

        reading = self.sim.read_imu()

        if reading is None or not all(np.all(np.isfinite(r)) for r in reading):
            self.missed_imu_samples += 1

            if self.missed_imu_samples > MAX_MISSED_IMU_SAMPLES:
                raise RuntimeError(
                    "The IMU stopped giving valid readings; refusing to "
                    "continue on a stale estimate."
                )

            return

        gyro, accel = self.imu_noise.apply(*reading)

        self.estimator.update(
            gyro,
            accel,
            dt=(self.missed_imu_samples + 1) * self.sim.dt,
        )
        self.missed_imu_samples = 0

    def _observation(self, control_state):
        return hover_task.build_observation(
            control_state,
            self.target_position,
            self._pid_command(control_state),
            self.hover_thrust,
            self.previous_action,
            self.scales,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Order matters for matching realizations: wind first, then the start.
        # Nothing here depends on the policy, so a given seed reproduces both.
        self.wind_episode = self._sample_wind()
        initial = hover_task.sample_initial_state(self.np_random, self.task)

        self.previous_action = np.zeros(4, dtype=np.float32)
        self.decisions = 0
        self.needs_reset = False

        self.pid.reset()
        self.last_saturated = False

        self.sim.wind.reset_episode(**self.wind_episode)
        self.sim.teleport(
            initial["position"],
            initial["orientation"],
            initial["linear_velocity"],
            initial["angular_velocity"],
            np.full(4, self.hover_motor_speed),
        )

        truth = self.sim.read_drone_state()

        if self.estimator is not None:
            # The estimator starts from what an operator would assume: at the
            # commanded target, level, at rest. The sampled start stays hidden.
            self.imu_noise.reset(np.random.default_rng(
                int(self.np_random.integers(0, 2**63 - 1))
            ))
            self.estimator.reset(self.target_position)
            self.missed_imu_samples = 0

        info = {
            "wind_mean": self.wind_episode["mean_wind"].copy(),
            "wind_gust_scale": self.wind_episode["gust_scale"],
            "wind_seed": self.wind_episode["seed"],
        }

        return self._observation(self._control_state(truth)), info

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32).reshape(4)

        if not np.all(np.isfinite(action)):
            raise ValueError(f"Invalid action: {action}")

        action = np.clip(action, -1.0, 1.0)

        if not self.sim.is_running():
            raise SimulationClosed("Simulation closed during an episode.")

        # Held for every physics step of this decision. PID, motors, and wind
        # are all recomputed from the fresh state inside the loop.
        residual = action.astype(float) * self.residual_limits

        saturated_steps = 0
        simulated_steps = 0
        truth = self.sim.read_drone_state()

        for substep in range(self.decimation):
            if hover_task.failure_reason(
                truth, self.target_position, self.failure_limits
            ) == "invalid_state":
                break

            self._before_physics_step()

            command = self._pid_command(
                self._control_state(truth), dt=self.sim.dt
            ) + residual

            # truth is passed for the wind drag physics only.
            _, saturated, _, _ = self.sim.step_command(
                truth,
                *command,
                render=(
                    self.render_sim
                    and substep == self.decimation - 1
                ),
            )

            self.last_saturated = bool(saturated)
            saturated_steps += int(saturated)
            simulated_steps += 1
            truth = self.sim.read_drone_state()
            self._update_estimator()

        self.decisions += 1

        reason = hover_task.failure_reason(
            truth, self.target_position, self.failure_limits
        )

        terminated = reason is not None
        truncated = not terminated and self.decisions >= self.max_decisions

        if reason == "invalid_state":
            # Non-finite values must not reach the reward or the policy.
            costs = {name: 0.0 for name in (
                "position", "speed", "attitude",
                "angular_rate", "action", "action_rate",
            )}
            position_error = float("nan")
            observation = np.zeros(
                self.observation_space.shape, dtype=np.float32
            )
        else:
            costs = self._step_costs(
                truth, action, saturated_steps, simulated_steps
            )
            position_error = float(np.linalg.norm(
                self.target_position - truth["position"]
            ))

        total_cost = sum(costs.values())
        penalty = self.penalty if terminated else 0.0
        reward = -self.reward_weights["scale"] * (total_cost + penalty)

        self.previous_action = action.copy()

        estimate_error = float("nan")

        if reason != "invalid_state":
            control_state = self._control_state(truth)
            observation = self._observation(control_state)

            # A metric only: how far the estimate has drifted from truth.
            estimate_error = float(np.linalg.norm(
                control_state["position"] - truth["position"]
            ))

        self.needs_reset = terminated or truncated

        info = {
            "costs": costs,
            "total_cost": total_cost,
            "failure_penalty": penalty,
            "failure_reason": reason,
            "position_error": position_error,
            "estimate_error": estimate_error,
            "residual_norm": float(np.linalg.norm(action)),
            "residual": residual.copy(),
            "saturated_steps": saturated_steps,
            "physics_steps": simulated_steps,
        }

        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.sim.close()


def load_ppo_policy(env, model_path, stats_path):
    """Load a saved model; return a deterministic observation -> action."""
    problem = model_contract.incompatibility(model_path, env.state_source)

    if problem:
        raise RuntimeError(problem)

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    # The policy only understands observations scaled by the training
    # statistics. training=False freezes them.
    statistics = VecNormalize.load(
        str(stats_path),
        DummyVecEnv([lambda: env]),
    )
    statistics.training = False
    statistics.norm_reward = False

    model = PPO.load(str(model_path), device="cpu")

    def policy(observation):
        action, _ = model.predict(
            statistics.normalize_obs(observation[None]),
            deterministic=True,
        )
        return action[0]

    return policy


def make_hover_sim(simulation_app, profile, log=print):
    """Load the open hover scene once. Call after SimulationApp has started."""
    from drone_sim import DroneSim

    project_dir = Path(__file__).resolve().parents[1]

    # Episode wind comes from the profile; the rest is the demo's model.
    wind_config = dict(sim_config.WIND)
    wind_config.update(
        enabled=profile["enabled"],
        gust_std=profile["gust_std"],
        local_gust_std=profile["local_gust_std"],
    )

    for name in ("mean_wind", "gust_timescale"):
        if name in profile:
            wind_config[name] = profile[name]

    return DroneSim(
        simulation_app,
        project_dir / hover_ppo_config.SCENE,
        log=log,
        wind_config=wind_config,
        deactivate_prims=hover_ppo_config.DEACTIVATE_PRIMS,
    )


def make_hover_env(
    simulation_app,
    wind_profile=hover_ppo_config.DEFAULT_WIND_PROFILE,
    render_sim=False,
    log=print,
    task=hover_ppo_config.HOVER,
    state_source=hover_ppo_config.DEFAULT_STATE_SOURCE,
):
    """Load the open hover scene once and wrap it in a DroneHoverEnv.

    wind_profile: a hover_ppo_config.WIND_PROFILES name, or a settings dict with the
    same keys. A dict with "mean_wind" is fixed; see hover_task.sample_wind.

    Call only after SimulationApp has started.
    """
    profile = wind_settings(wind_profile)
    sim = make_hover_sim(simulation_app, profile, log=log)

    if state_source == "imu":
        sim.connect_imu(sensor_config.IMU_PATH)

    sim.world.play()

    return DroneHoverEnv(
        sim,
        wind_profile=profile,
        render_sim=render_sim,
        task=task,
        state_source=state_source,
    )
