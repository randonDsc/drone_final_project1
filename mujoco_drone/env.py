"""Gymnasium interface shared by training and measured flight experiments."""
from dataclasses import asdict, replace

import gymnasium as gym
import numpy as np

from drone_code import hover_task, path_sac_config
from drone_code.path_reference import sample_reference
from drone_code.path_sac_policy import (OBSERVATION_SIZE, decision_costs,
                                       failure_penalty, physics_terms)
from .config import ExperimentConfig
from .core import FlightCore
from .physics import DroneSimulation
from .references import make_reference


class DroneEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config=None, training=False, on_step=None):
        super().__init__()
        self.config = config or ExperimentConfig()
        self.training = training
        if training and (self.config.scene.name != "open" or self.config.task.kind == "dijkstra"):
            raise ValueError("SAC curriculum trains the tracking controller in open space; obstacle planning is evaluated separately")
        self.sim = DroneSimulation(self.config.physics, self.config.wind, self.config.seed, scene=self.config.scene)
        self.core = FlightCore(self.sim)
        self.reward_config = path_sac_config.reward_settings()
        self.penalty = failure_penalty(self.reward_config, path_sac_config.SAC["gamma"])
        self.action_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        clip = path_sac_config.OBSERVATION_CLIP
        self.observation_space = gym.spaces.Box(-clip, clip, (OBSERVATION_SIZE,), dtype=np.float32)
        self.on_step = on_step
        self.training_step = 0
        self.needs_reset = True
        self.saturated_streak = 0
        self.stage_name = "controlled_experiment"
        self.failure = None
        self.initial = None
        self.navigation = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.np_random
        self.navigation = None
        if self.training:
            stage = (path_sac_config.stage_by_name(options["stage"])
                     if "stage" in options else path_sac_config.stage_for_step(self.training_step))
            self.stage_name = stage["name"]
            wind = hover_task.sample_stage_wind(rng, stage["wind"])
            self.route_type, reference = sample_reference(rng, stage["route"], path_sac_config.EPISODE, options.get("route_type"))
            task = dict(path_sac_config.EPISODE, target_position=reference.path[0])
            initial = hover_task.sample_initial_state(rng, task)
            self.core.apply_wind(wind, stage["wind"])
            # apply_wind clears direction rate in begin; restore it below.
            direction_rate = wind["direction_rate"]
        else:
            from .weather import ScenarioWind
            wind_config = self.config.wind
            profile = path_sac_config.EVAL_WIND_PROFILES.get(wind_config.scenario)
            if self.config.protocol == "isaac_path" and profile is None:
                if wind_config.scenario != "calm":
                    raise ValueError("Isaac path protocol uses calm or an original Isaac evaluation wind profile")
                profile = dict(no_wind_probability=1.0, mean_speed_range=(0, 0), gust_scale_range=(0, 0), direction_rate_range=(0, 0))
            if profile is not None:
                # Match Isaac evaluation's order: wind, route, initial state.
                wind = hover_task.sample_stage_wind(rng, profile)
                wind_config = replace(wind_config, scenario="curriculum",
                     gust_std=path_sac_config.BASE_WIND["gust_std"],
                     local_gust_std=path_sac_config.BASE_WIND["local_gust_std"])
            self.sim.wind = ScenarioWind(wind_config, self.sim.rotor_positions, self.config.seed if seed is None else seed)
            self.route_type = self.config.task.kind
            if self.config.protocol == "isaac_path" and self.route_type != "dijkstra":
                stage = path_sac_config.stage_by_name(path_sac_config.EVAL_ROUTE_STAGE)
                self.stage_name = stage["name"]
                self.route_type, reference = sample_reference(rng, stage["route"], path_sac_config.EPISODE, self.route_type)
                initial = hover_task.sample_initial_state(rng, dict(path_sac_config.EPISODE, target_position=reference.path[0]))
            else:
                reference = None if self.route_type == "dijkstra" else make_reference(self.config.task)
                start = self.config.task.start if reference is None else reference.path[0]
                initial = hover_task.sample_initial_state(rng, dict(asdict(self.config.task), target_position=start))
            direction_rate = 0.0
            if profile is not None:
                self.core.apply_wind(wind, profile)
                direction_rate = wind["direction_rate"]
        self.sim.reset(initial["position"], initial["orientation"], initial["linear_velocity"], initial["angular_velocity"])
        if self.route_type == "dijkstra":
            from .planning import NavigationMission
            self.navigation = NavigationMission(self.sim, self.config, initial["position"])
            reference = self.navigation.reference
        self.initial = {name: value.copy() for name, value in initial.items()}
        self.core.begin(reference)
        self.core.direction_rate = direction_rate
        seconds = min(reference.duration, self.config.max_seconds,
                      path_sac_config.EPISODE["max_seconds"] if self.training or (self.config.protocol == "isaac_path" and self.route_type != "dijkstra") else float("inf"))
        self.max_physics_steps = max(1, int(np.ceil(seconds/self.sim.dt-1e-10)))
        self.physics_steps = 0
        self.saturated_streak = 0
        self.needs_reset = False
        self.failure = None
        self.total_reward = 0.0
        info = dict(route_type=self.route_type, stage=self.stage_name,
                    wind_seed=self.sim.wind.seed, initial_state={k: v.tolist() for k, v in self.initial.items()},
                    route_seconds=reference.duration)
        field = self.sim.wind
        info["wind_realization"] = dict(mean_velocity_m_s=field.mean_wind.tolist(),
                    gust_std_m_s=field.gust_std.tolist(), local_gust_std_m_s=field.local_gust_std.tolist(),
                    gust_timescale_s=field.gust_timescale, local_gust_timescale_s=field.local_gust_timescale,
                    direction_rate_rad_s=direction_rate, enabled=field.enabled)
        if self.navigation:
            info["navigation"] = self.navigation.info()
        return self.core.observe(self.sim.read_drone_state()), info

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Call reset() before step() or after episode end")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain four finite normalized residuals")
        core = self.core
        if self.navigation and self.config.navigation.residual_after_launch and self.navigation.launch_active(core.time):
            action = np.zeros(4, dtype=np.float32)
        core.apply_action(action)
        sums = None
        saturated_steps = 0
        substeps = 0
        self.failure = None
        for _ in range(min(core.decimation, self.max_physics_steps-self.physics_steps)):
            before = self.sim.read_drone_state()
            if self.navigation:
                try:
                    self.navigation.before_step(before, core.time, self.sim.dt)
                except RuntimeError:
                    self.failure = "waypoint_timeout"
                    break
                if self.config.navigation.residual_after_launch and self.navigation.launch_active(core.time):
                    core.correction[:] = 0
            try:
                saturated = core.step(before)
            except FloatingPointError:
                self.failure = "invalid_state"
                break
            state = self.sim.read_drone_state()
            sample = core.sample()
            terms = physics_terms(state, sample, core.desired, core.omega_ref, self.reward_config)
            sums = terms if sums is None else {k: sums[k]+v for k, v in terms.items()}
            substeps += 1
            self.physics_steps += 1
            saturated_steps += int(saturated)
            core.completion.update(state, sample, core.reference.final_position, self.sim.dt, core.time)
            limits = path_sac_config.FAILURE
            allowed_supports = ()
            failure_reference = sample.position
            if self.navigation:
                snapshot = self.navigation.observe(state, core.time)
                limits = dict(limits, max_position_error=self.config.navigation.max_position_error)
                failure_reference = self.navigation.failure_reference(state, sample)
                if snapshot["launch_active"]:
                    limits["min_height"] = -float("inf")
                    allowed_supports = self.navigation.support_ids
                if snapshot["out_of_bounds"]:
                    self.failure = "out_of_map"
            self.failure = self.failure or hover_task.failure_reason(state, failure_reference, limits)
            if self.sim.collided(allowed_supports):
                self.failure = "collision"
            if self.on_step is not None:
                self.on_step(self, state, sample)
            if self.failure:
                break
        terminated = self.failure is not None
        truncated = not terminated and self.physics_steps >= self.max_physics_steps
        share = saturated_steps/max(substeps, 1)
        self.saturated_streak = self.saturated_streak+1 if share >= 1 else 0
        persistent = self.saturated_streak >= path_sac_config.PERSISTENT_SATURATION_DECISIONS
        if sums is None or self.failure == "invalid_state":
            costs = {}
            total_cost = self.reward_config["cost_cap"]
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            costs = decision_costs({k: v/substeps for k, v in sums.items()}, core.action,
                                   core.previous_action, share, persistent, self.reward_config)
            total_cost = min(sum(costs.values()), self.reward_config["cost_cap"])
            observation = core.observe(self.sim.read_drone_state())
        reward = -self.reward_config["scale"]*(total_cost+(self.penalty if terminated else 0))
        self.total_reward += reward
        self.training_step += 1
        self.needs_reset = terminated or truncated
        state = self.sim.read_drone_state()
        info = dict(costs=costs, total_cost=total_cost, failure_reason=self.failure,
                    position_error=float(np.linalg.norm(core.sample().position-state["position"])),
                    residual_norm=float(np.linalg.norm(core.action)),
                    saturated_steps=saturated_steps, physics_steps=substeps,
                    completed=core.completion.completed and not terminated,
                    route_type=self.route_type, stage=self.stage_name)
        return observation, float(reward), terminated, truncated, info

    def close(self):
        self.sim.close()
