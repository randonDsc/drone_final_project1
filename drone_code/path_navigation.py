"""Fly the existing Dijkstra route through the shared flight loop.

Reuses the existing planner unchanged: mapping.py builds the occupancy grid and
navigation.plan_flight gives collision-checked waypoints. Those waypoints then
become a TimedReference and fly through path_sac_policy.PathCore, the same loop as
training. WaypointFollower is not used: it stops at each waypoint and has no
reference velocity for the policy to see.

Corners are rounded only where every chord passes navigation.line_of_sight, the
planner's own check against the inflated grid and expanded obstacles. The launch
corner always stays sharp, because the planner's launch departure deliberately
starts inside the launch pad's clearance.

Import only after SimulationApp has started.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import path_sac_config
import hover_task

from path_sac_policy import PathCore, PathTrace
from path_reference import TimedReference, round_corners


@dataclass
class Route:
    reference: TimedReference
    geometry: object
    occupancy: object
    waypoints: np.ndarray
    start: np.ndarray
    launch_end: float


def load_navigation_sim(simulation_app, log=print):
    """The obstacle scene, paused at its USD pose, wind model from BASE_WIND."""
    import sim_config

    from drone_sim import DroneSim

    project_dir = Path(__file__).resolve().parents[1]

    wind_config = dict(sim_config.WIND)
    wind_config.update(path_sac_config.BASE_WIND)

    return DroneSim(
        simulation_app,
        project_dir / path_sac_config.NAVIGATION["scene"],
        log=log,
        wind_config=wind_config,
    )


def plan_route(sim, simulation_app, log=print):
    """Plan with the existing Dijkstra pipeline; return the timed Route."""
    from mapping import build_occupancy_map, read_scene_geometry
    from navigation import line_of_sight, plan_flight

    settings = path_sac_config.NAVIGATION

    sim.world.pause()

    positions, _ = sim.body.get_world_poses()
    start = np.asarray(positions[0], dtype=float).copy()

    goal = np.array([
        start[0] + settings["goal_offset_xy"][0],
        start[1] + settings["goal_offset_xy"][1],
        settings["goal_height"],
    ])

    log(f"Planning from {start} to {goal}")

    geometry = read_scene_geometry(sim.stage, start)
    occupancy = build_occupancy_map(geometry, start, goal)

    def progress(expanded):
        sim.world.pause()
        simulation_app.update()

        if not simulation_app.is_running():
            raise RuntimeError("Simulation closed during planning.")

    plan = plan_flight(occupancy, geometry, start, goal, progress=progress)

    log(
        f"Dijkstra route: {len(plan.waypoints)} waypoints, "
        f"{plan.distance:.2f} m"
    )

    # A start inside the pad's clearance means plan_flight added a launch
    # departure, and corner 1 is its end.
    has_launch = bool(occupancy.grid[occupancy.index(start)])

    points = round_corners(
        plan.waypoints,
        settings["corner_radius"],
        spacing=path_sac_config.EPISODE["spacing"],
        is_safe=lambda a, b: line_of_sight(occupancy, a, b),
        locked=(1,) if has_launch else (),
    )

    reference = TimedReference(
        points,
        settings["speed"],
        settings["acceleration"],
        settings["lateral_acceleration"],
        settings["vertical_speed"],
        start_hold=settings["start_hold"],
        end_hold=settings["settle_seconds"],
        spacing=path_sac_config.EPISODE["spacing"],
    )

    launch_end = (
        reference.time_at_point(plan.waypoints[1])
        if has_launch
        else settings["start_hold"]
    )

    log(f"Reference: {reference.duration:.1f} s, launch ends {launch_end:.1f} s")

    return Route(
        reference, geometry, occupancy, plan.waypoints, start, launch_end
    )


class ObstacleMonitor:
    """Where the drone is relative to the map, checked from true position.

    No contact sensor exists, so a collision means the body origin came within
    collision_half_size of an obstacle's real bounds. A clearance violation means
    it entered the planner's inflated geometry, which is more conservative and
    not itself a crash. The launch departure is exempt from both.
    """

    def __init__(self, geometry, occupancy):
        half_size = np.asarray(
            path_sac_config.NAVIGATION["collision_half_size"], dtype=float
        )

        self.lower = np.array([box.lower - half_size for box in geometry.boxes])
        self.upper = np.array([box.upper + half_size for box in geometry.boxes])
        self.planes = [
            (plane.normal, plane.offset + float(np.abs(plane.normal) @ half_size))
            for plane in geometry.planes
        ]
        self.occupancy = occupancy
        self.half_size = half_size

    def collided(self, position):
        if len(self.lower) and np.any(np.all(
            (position >= self.lower) & (position <= self.upper), axis=1
        )):
            return True

        return any(position @ normal <= offset for normal, offset in self.planes)

    def clearance_violated(self, position):
        try:
            return bool(self.occupancy.grid[self.occupancy.index(position)])
        except ValueError:
            return True


def fly_route(core, route, policy, wind, profile, render, log=print):
    """One flight; policy None flies PID only. Returns the metrics dictionary."""
    sim = core.sim
    settings = path_sac_config.NAVIGATION

    limits = dict(path_sac_config.FAILURE)
    limits["max_position_error"] = settings["max_position_error"]

    # The route starts on the pad, under the training height limit.
    limits["min_height"] = -np.inf

    monitor = ObstacleMonitor(route.geometry, route.occupancy)

    sim.teleport(
        route.start,
        sim.initial_orientations[0],
        np.zeros(3),
        np.zeros(3),
        np.zeros(4),
    )

    core.begin(route.reference)
    core.apply_wind(wind, profile)

    trace = PathTrace()

    reason = None
    collided = False
    clearance_steps = 0
    step = 0
    final_step = int(np.ceil(route.reference.duration / sim.dt))

    state = sim.read_drone_state()

    while step < final_step and sim.is_running():
        if step % core.decimation == 0:
            sample = core.sample()

            observation = core.observe(state)

            trace.record(
                state["position"],
                np.linalg.norm(sample.position - state["position"]),
                core.action,
                core.previous_action,
            )

            flying = (
                core.time >= route.launch_end
                or not settings["residual_after_launch"]
            )

            core.apply_action(
                policy(observation)
                if policy is not None and flying
                else np.zeros(4, dtype=np.float32)
            )

        saturated = core.step(state, render=render)

        trace.saturated_steps += int(saturated)
        trace.physics_steps += 1
        step += 1

        state = sim.read_drone_state()
        sample = core.sample()

        reason = hover_task.failure_reason(state, sample.position, limits)

        if reason is not None:
            log(f"Flight failed at t={core.time:.2f} s: {reason}")
            break

        core.completion.update(
            state,
            sample,
            route.reference.final_position,
            sim.dt,
            core.time,
        )

        if core.time >= route.launch_end:
            position = np.asarray(state["position"], dtype=float)
            clearance_steps += int(monitor.clearance_violated(position))

            if monitor.collided(position):
                collided = True
                reason = "collision"
                log(f"Collision at t={core.time:.2f} s, {position}")
                break

    metrics = trace.metrics(
        route.reference.path, reason, collided, core.completion
    )
    metrics["clearance_violation_rate"] = (
        clearance_steps / max(trace.physics_steps, 1)
    )

    return metrics


def make_core(sim, log=print):
    """A PathCore in a playing simulator, ready for fly_route."""
    core = PathCore(sim, log=log)
    sim.world.play()

    return core


def draw_wind(seed, profile_name):
    """One wind realization, fixed by seed for both modes."""
    return hover_task.sample_stage_wind(
        np.random.default_rng(seed),
        path_sac_config.EVAL_WIND_PROFILES[profile_name],
    )
