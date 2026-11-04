"""Hover in randomized wind, for SAC: DroneHoverEnv with its own wind stages,
observation history, reward, and decision rate. Simulator state only.

    Drone state/history -> SAC -> held correction
    PID command + correction -> mixer -> rotor commands

Wind is the demo's drag model from wind.py, applied at every physics step through
DroneSim.step_command. Each episode draws a mean, gust strength, and direction
rate from the current stage in sac_config.WIND_STAGES. The policy never sees it.
"""
import gymnasium as gym
import numpy as np

import hover_task
import sac_config

from hover_env import DroneHoverEnv, make_hover_sim
from sac_policy import OBSERVATION_SIZE, ObservationHistory, build_frame


def sac_task():
    decimation = sac_config.physics_steps_per_decision()
    task = dict(sac_config.TASK)

    task["control_decimation"] = decimation
    task["max_decisions"] = int(round(
        task["episode_seconds"] * sac_config.RL_RATE_HZ
    ))

    return task


class SacHoverEnv(DroneHoverEnv):
    def __init__(self, sim, render_sim=False, wind_stage=None):
        """wind_stage: a stage name to always use, or None to follow training."""
        super().__init__(
            sim,
            wind_profile=sac_config.BASE_WIND,
            render_sim=render_sim,
            task=sac_task(),
            residual_limits=sac_config.RESIDUAL_LIMITS,
            reward=sac_config.REWARD,
            failure=sac_config.FAILURE,
            gamma=sac_config.SAC["gamma"],
            state_source="ground_truth",
        )

        if abs(self.decimation * sim.dt - 1.0 / sac_config.RL_RATE_HZ) > 1e-9:
            raise ValueError(
                "sac_config.PHYSICS_RATE_HZ does not match the simulation."
            )

        # The base penalty bounds the six shared costs. Add the other three.
        extra = (
            sac_config.REWARD["position_linear"]
            * sac_config.FAILURE["max_position_error"]
            + sac_config.REWARD["saturation"]
            + sac_config.REWARD["persistent_saturation"]
        )
        self.penalty += extra / (1.0 - sac_config.SAC["gamma"])

        limit = sac_config.OBSERVATION_CLIP

        self.observation_space = gym.spaces.Box(
            low=-limit,
            high=limit,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.history = ObservationHistory()
        self.fixed_stage = (
            None if wind_stage is None else sac_config.stage_by_name(wind_stage)
        )
        self.training_step = 0
        self.direction_rate = 0.0
        self.saturated_streak = 0

    def set_training_step(self, step):
        """Total training steps so far; chooses the wind stage."""
        self.training_step = int(step)

    @property
    def stage(self):
        if self.fixed_stage is not None:
            return sac_config.WIND_STAGES[self.fixed_stage]

        for stage in sac_config.WIND_STAGES:
            limit = stage["until_step"]

            if limit is None or self.training_step < limit:
                return stage

        return sac_config.WIND_STAGES[-1]

    def _sample_wind(self):
        wind = hover_task.sample_stage_wind(self.np_random, self.stage)
        self.direction_rate = wind.pop("direction_rate")

        return wind

    def _before_physics_step(self):
        """Turn the mean wind; gusts keep fluctuating around it."""
        if self.direction_rate != 0.0:
            angle = self.direction_rate * self.sim.dt
            c, s = np.cos(angle), np.sin(angle)
            x, y, z = self.sim.wind.mean_wind

            self.sim.wind.mean_wind = np.array([c*x - s*y, s*x + c*y, z])

    def _observation(self, control_state):
        return self.history.push(build_frame(
            control_state,
            self.target_position,
            float(self.pid_target.get("yaw", 0.0)),
            self._pid_command(control_state),
            self.hover_thrust,
            self.previous_action,
        ))

    def _step_costs(self, truth, action, saturated_steps, physics_steps):
        weights = self.reward_weights

        costs = super()._step_costs(
            truth, action, saturated_steps, physics_steps
        )

        share = saturated_steps / max(physics_steps, 1)
        self.saturated_streak = (
            self.saturated_streak + 1 if share >= 1.0 else 0
        )
        persistent = (
            self.saturated_streak
            >= sac_config.PERSISTENT_SATURATION_DECISIONS
        )

        costs["position_linear"] = float(
            weights["position_linear"]
            * np.linalg.norm(self.target_position - truth["position"])
        )
        costs["saturation"] = float(weights["saturation"] * share)
        costs["persistent_saturation"] = float(
            weights["persistent_saturation"] * persistent
        )

        return costs

    def reset(self, seed=None, options=None):
        self.history.clear()
        self.saturated_streak = 0

        observation, info = super().reset(seed=seed, options=options)
        info["wind_stage"] = self.stage["name"]

        return observation, info

    def step(self, action):
        result = super().step(action)
        result[4]["wind_stage"] = self.stage["name"]

        return result


def make_sac_env(simulation_app, render_sim=False, wind_stage=None, log=print):
    """Load the open hover scene and wrap it. Needs SimulationApp running."""
    sim = make_hover_sim(simulation_app, sac_config.BASE_WIND, log=log)
    sim.world.play()

    return SacHoverEnv(sim, render_sim=render_sim, wind_stage=wind_stage)
