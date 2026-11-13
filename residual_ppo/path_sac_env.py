"""Gymnasium environment: trajectory flight in wind, PID plus SAC correction.

One SAC transition is one decision: the correction is held for every physics step
while PID, the reference, wind, and motors keep updating underneath. Costs are
averaged over those steps. Simulator ground truth only.

Episodes end in one of two ways:

    terminated   A failure limit was crossed. Charges the failure penalty, and
                 SAC does not bootstrap past it.
    truncated    The reference and its settling time ran out, or the time limit.
                 No penalty; SB3 keeps it apart from termination and bootstraps.

A finished route is a truncation too, at a time fixed by the route: arriving never
ends the episode, so no behavior shortens it. Being late just keeps costing, and
giving up costs the failure penalty when the reference has moved 3 m away.
"""
import gymnasium as gym
import numpy as np

import path_sac_config
import hover_task

from path_sac_policy import (
    PathCore,
    PathTrace,
    OBSERVATION_SIZE,
    decision_costs,
    failure_penalty,
    physics_terms,
)
from path_reference import sample_reference
from hover_env import SimulationClosed, make_hover_sim


COST_NAMES = (
    "position", "speed", "direction", "attitude", "rate",
    "unsafe_tilt", "unsafe_rate", "route_deviation",
    "action", "action_rate", "saturation", "persistent_saturation",
)


class PathSacEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sim, render_sim=False, log=print):
        super().__init__()

        self.sim = sim
        self.render_sim = bool(render_sim)
        self.core = PathCore(sim, log=log)
        self.reward = path_sac_config.reward_settings()
        self.penalty = failure_penalty(
            self.reward, path_sac_config.SAC["gamma"]
        )

        limit = path_sac_config.OBSERVATION_CLIP

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-limit, high=limit, shape=(OBSERVATION_SIZE,), dtype=np.float32
        )

        self.training_step = 0
        self.max_decisions = 0
        self.decisions = 0
        self.saturated_streak = 0
        self.needs_reset = True
        self.route_type = None
        self.stage_name = None
        self.trace = PathTrace()

    def set_training_step(self, step):
        """Total training steps so far; chooses the curriculum stage."""
        self.training_step = int(step)

    def reset(self, seed=None, options=None):
        """options, for evaluation: route_type, route_stage, wind_profile (name)."""
        super().reset(seed=seed)

        options = options or {}
        stage = (
            path_sac_config.stage_by_name(options["route_stage"])
            if "route_stage" in options
            else path_sac_config.stage_for_step(self.training_step)
        )
        profile = (
            path_sac_config.EVAL_WIND_PROFILES[options["wind_profile"]]
            if "wind_profile" in options
            else stage["wind"]
        )

        # Order matters for matching realizations: wind, route, then the start.
        # Nothing here depends on the policy, so a seed reproduces all of them.
        wind = hover_task.sample_stage_wind(self.np_random, profile)

        self.route_type, reference = sample_reference(
            self.np_random,
            stage["route"],
            path_sac_config.EPISODE,
            options.get("route_type"),
        )

        task = dict(path_sac_config.EPISODE)
        task["target_position"] = reference.path[0]
        initial = hover_task.sample_initial_state(self.np_random, task)

        core = self.core
        core.begin(reference)
        core.apply_wind(wind, profile)

        self.sim.teleport(
            initial["position"],
            initial["orientation"],
            initial["linear_velocity"],
            initial["angular_velocity"],
            np.full(4, core.hover_motor_speed),
        )

        self.stage_name = stage["name"]
        self.decisions = 0
        self.saturated_streak = 0
        self.needs_reset = False
        self.trace = PathTrace()

        self.max_decisions = int(np.ceil(
            min(reference.duration, path_sac_config.EPISODE["max_seconds"])
            * path_sac_config.RL_RATE_HZ
        ))

        info = {
            "wind_mean": wind["mean_wind"].copy(),
            "wind_seed": wind["seed"],
            "route_type": self.route_type,
            "stage": self.stage_name,
            "route_seconds": reference.duration,
        }

        return core.observe(self.sim.read_drone_state()), info

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Call reset() before step().")

        action = np.asarray(action, dtype=np.float32).reshape(4)

        if not np.all(np.isfinite(action)):
            raise ValueError(f"Invalid action: {action}")

        if not self.sim.is_running():
            raise SimulationClosed("Simulation closed during an episode.")

        core = self.core
        core.apply_action(action)

        sums = None
        saturated_steps = 0
        simulated_steps = 0
        truth = self.sim.read_drone_state()

        for substep in range(core.decimation):
            if not hover_task.state_is_valid(truth):
                break

            saturated = core.step(
                truth,
                render=(
                    self.render_sim and substep == core.decimation - 1
                ),
            )

            saturated_steps += int(saturated)
            simulated_steps += 1
            truth = self.sim.read_drone_state()

            if hover_task.state_is_valid(truth):
                sample = core.sample()

                terms = physics_terms(
                    truth, sample, core.desired, core.omega_ref, self.reward
                )

                sums = terms if sums is None else {
                    name: sums[name] + value for name, value in terms.items()
                }

                core.completion.update(
                    truth,
                    sample,
                    core.reference.final_position,
                    self.sim.dt,
                    core.time,
                )

        self.decisions += 1

        sample = core.sample()

        reason = hover_task.failure_reason(
            truth, sample.position, path_sac_config.FAILURE
        )

        terminated = reason is not None
        truncated = not terminated and self.decisions >= self.max_decisions

        share = saturated_steps / max(simulated_steps, 1)
        self.saturated_streak = (
            self.saturated_streak + 1 if share >= 1.0 else 0
        )
        persistent = (
            self.saturated_streak
            >= path_sac_config.PERSISTENT_SATURATION_DECISIONS
        )

        if sums is None or reason == "invalid_state":
            # Non-finite values must not reach the reward or the policy.
            costs = {name: 0.0 for name in COST_NAMES}
            total_cost = self.reward["cost_cap"]
            position_error = float("nan")
            position = np.full(3, np.nan)
            observation = np.zeros(
                self.observation_space.shape, dtype=np.float32
            )
        else:
            costs = decision_costs(
                {
                    name: value / simulated_steps
                    for name, value in sums.items()
                },
                core.action,
                core.previous_action,
                share,
                persistent,
                self.reward,
            )
            total_cost = min(sum(costs.values()), self.reward["cost_cap"])
            position_error = float(np.linalg.norm(
                sample.position - truth["position"]
            ))
            position = np.array(truth["position"], dtype=float)
            observation = core.observe(truth)

        penalty = self.penalty if terminated else 0.0
        reward = -self.reward["scale"] * (total_cost + penalty)

        self.trace.record(
            position, position_error, core.action, core.previous_action
        )
        self.trace.saturated_steps += saturated_steps
        self.trace.physics_steps += simulated_steps

        self.needs_reset = terminated or truncated

        info = {
            "costs": costs,
            "total_cost": total_cost,
            "failure_penalty": penalty,
            "failure_reason": reason,
            "position_error": position_error,
            "residual_norm": float(np.linalg.norm(core.action)),
            "residual": core.correction.copy(),
            "saturated_steps": saturated_steps,
            "physics_steps": simulated_steps,
            "route_type": self.route_type,
            "stage": self.stage_name,
            "completed": core.completion.completed,
        }

        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.sim.close()


def make_path_sac_env(simulation_app, render_sim=False, log=print):
    """Load the open scene once and wrap it. Needs SimulationApp running.

    The real Isaac Sim drone, mixer, and wind drag; only the hover scene is shared.
    """
    sim = make_hover_sim(simulation_app, path_sac_config.BASE_WIND, log=log)
    sim.world.play()

    return PathSacEnv(sim, render_sim=render_sim, log=log)
