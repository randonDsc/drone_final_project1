"""MuJoCo geometry adapter around the original conservative 3D Dijkstra code."""
from dataclasses import replace
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from drone_code.grid_map import Box, Plane
from drone_code.mapping import SceneGeometry, build_occupancy_map
from drone_code.navigation import plan_flight, line_of_sight, WaypointFollower
from drone_code.path_reference import ReferenceSample, TimedReference, round_corners
from .references import TaskReference


def geom_bounds(sim, geom_id, *, body_frame=False):
    """World or drone-body AABB from MuJoCo's compiled local geometry bounds."""
    model, data = sim.model, sim.data
    local_center = model.geom_aabb[geom_id, :3]
    half = model.geom_aabb[geom_id, 3:]
    if body_frame:
        quat = model.geom_quat[geom_id]
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, quat)
        rotation = rotation.reshape(3, 3)
        center = model.geom_pos[geom_id] + rotation@local_center
    else:
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        center = data.geom_xpos[geom_id] + rotation@local_center
    extent = abs(rotation)@half
    return center-extent, center+extent


def read_geometry(sim):
    model = sim.model
    drone_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] == sim.body_id
                 and (model.geom_contype[i] or model.geom_conaffinity[i])]
    bounds = [geom_bounds(sim, i, body_frame=True) for i in drone_ids]
    lower = np.min([b[0] for b in bounds], axis=0)
    upper = np.max([b[1] for b in bounds], axis=0)
    extent = np.maximum(abs(lower), abs(upper))
    radius = float(np.linalg.norm(extent[:2]))
    tilt = np.deg2rad(sim.config.max_tilt_degrees)
    half_size = np.array([radius+extent[2]*np.sin(tilt), radius+extent[2]*np.sin(tilt),
                          extent[2]+radius*np.sin(tilt)])
    boxes, planes, report = [], [], []
    source_ids = {}
    for i in range(model.ngeom):
        if i in drone_ids or model.geom_bodyid[i] == sim.body_id:
            continue
        interacts = any((model.geom_contype[i] & model.geom_conaffinity[j]) or
                        (model.geom_contype[j] & model.geom_conaffinity[i]) for j in drone_ids)
        if not interacts:
            continue
        if model.body_weldid[model.geom_bodyid[i]] != 0:
            raise ValueError("Dynamic obstacle mapping is unsupported")
        path = sim.geom_sources.get(i, model.geom(i).name or f"geom_{i}")
        source_ids[path] = i
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            normal = sim.data.geom_xmat[i].reshape(3, 3)[:, 2].copy()
            axis = int(np.argmax(abs(normal)))
            if not np.isclose(abs(normal[axis]), 1, atol=1e-7):
                raise ValueError(f"Tilted infinite plane unsupported: {path}")
            normal = np.eye(3)[axis]*np.sign(normal[axis])
            plane = Plane(normal, float(normal@sim.data.geom_xpos[i]), path)
            planes.append(plane)
            report.append(f"PLANE {path}: normal={normal}, offset={plane.offset}")
        else:
            a, b = geom_bounds(sim, i)
            boxes.append(Box(a, b, path))
            report.append(f"BOX {path}: min={a}, max={b}")
    return SceneGeometry(boxes, planes, half_size, float(lower[2]), report), source_ids


class FeedbackWaypointReference:
    """Source WaypointFollower; target-error RMSE is feedback-dependent."""
    def __init__(self, points, hold, duration):
        self.path = np.asarray(points)
        self.final_position = self.path[-1].copy()
        self.follower = WaypointFollower(self.path)
        self.duration, self.hold = duration, hold
        self.motion_end = float("inf")
        self.current = ReferenceSample(self.path[0].copy(), np.zeros(3), np.zeros(3), False)

    def advance(self, state, time, dt):
        if time < self.hold:
            return
        target = self.follower.update(state["position"], state["linear_velocity"], dt)
        if self.follower.done and not np.isfinite(self.motion_end):
            self.motion_end = time
        self.current = ReferenceSample(target, np.zeros(3), np.zeros(3), self.follower.done)

    def sample(self, time):
        return self.current


class NavigationMission:
    def __init__(self, sim, config, start):
        self.sim, self.config = sim, config
        settings = config.navigation
        self.geometry, self.source_ids = read_geometry(sim)
        if not self.geometry.boxes:
            raise ValueError("Dijkstra obstacle experiment needs collision obstacles; select blocks or warehouse")
        began = time.perf_counter()
        self.occupancy = build_occupancy_map(self.geometry, start, config.task.goal,
                    bounds=settings.bounds, resolution=settings.resolution,
                    safety_margin=settings.safety_margin, max_cells=settings.max_grid_cells)
        self.has_launch = bool(self.occupancy.grid[self.occupancy.index(start)])
        self.plan = plan_flight(self.occupancy, self.geometry, start, config.task.goal,
                    max_expanded=settings.max_expanded_nodes, safety_margin=settings.safety_margin,
                    support_roots=settings.support_paths)
        self.planning_seconds = time.perf_counter()-began
        points = self.plan.waypoints
        self.reference_mode = settings.reference_mode
        if self.reference_mode == "isaac_timed":
            points = round_corners(points, settings.corner_radius, spacing=settings.spacing,
                       is_safe=lambda a, b: line_of_sight(self.occupancy, a, b),
                       locked=(1,) if self.has_launch else ())
            self.reference = TimedReference(points, config.task.speed, config.task.acceleration,
                       settings.lateral_acceleration, settings.vertical_speed,
                       start_hold=config.task.start_hold, end_hold=config.task.end_hold,
                       spacing=settings.spacing)
            self.launch_end = self.reference.time_at_point(self.plan.waypoints[1]) if self.has_launch else 0
        elif self.reference_mode == "quintic_stops":
            # Respect vertical as well as total speed on each analytic segment.
            task = replace(config.task, kind="waypoints", waypoints=tuple(map(tuple, points)))
            self.reference = TaskReference(task, vertical_speed=settings.vertical_speed)
            self.launch_end = float(self.reference.times[1]) if self.has_launch else 0
        else:
            self.reference = FeedbackWaypointReference(points, config.task.start_hold, config.max_seconds)
            self.launch_end = None if self.has_launch else 0
        # Both reference modes that interpolate geometry keep every chord safe.
        if self.reference_mode != "isaac_waypoints":
            for i, (a, b) in enumerate(zip(self.reference.path[:-1], self.reference.path[1:])):
                if self.has_launch:
                    entry = self.plan.waypoints[1]
                    launch_chord = (np.allclose(a[:2], points[0, :2]) and np.allclose(b[:2], points[0, :2])
                                    and min(a[2], b[2]) >= self.plan.waypoints[0, 2]-1e-8
                                    and max(a[2], b[2]) <= entry[2]+1e-8)
                    if launch_chord:
                        continue  # The entire departure was independently checked.
                if not line_of_sight(self.occupancy, a, b):
                    raise ValueError("Reference contains an unvalidated collision/grid segment")
        self.raw_lower = np.array([b.lower for b in self.geometry.boxes])
        self.raw_upper = np.array([b.upper for b in self.geometry.boxes])
        half = self.geometry.drone_half_size
        self.body_lower, self.body_upper = self.raw_lower-half, self.raw_upper+half
        self.support_ids = self._launch_supports(np.asarray(start)) if self.has_launch else set()
        self.snapshot = {}

    def _launch_supports(self, start):
        roots = self.config.navigation.support_paths
        result = set()
        for box in self.geometry.boxes:
            named = any(box.path == root or box.path.startswith(root.rstrip("/")+"/") for root in roots)
            if named and np.all(start[:2] >= box.lower[:2]) and np.all(start[:2] <= box.upper[:2]) and box.upper[2] <= start[2]+1e-8:
                result.add(self.source_ids[box.path])
        for plane in self.geometry.planes:
            if plane.normal[2] > .99 and plane.offset <= start[2]:
                result.add(self.source_ids[plane.path])
        return result

    def launch_active(self, time):
        if not self.has_launch:
            return False
        if self.reference_mode == "isaac_waypoints":
            return self.reference.follower.index < 2
        return time < self.launch_end-1e-9

    def before_step(self, state, time, dt):
        if self.reference_mode == "isaac_waypoints":
            self.reference.advance(state, time, dt)
            if self.has_launch and self.reference.follower.index >= 2 and self.launch_end is None:
                self.launch_end = time

    def observe(self, state, time):
        position = state["position"]
        delta = np.maximum(self.body_lower-position, position-self.body_upper)
        signed = np.linalg.norm(np.maximum(delta, 0), axis=1)+np.minimum(np.max(delta, axis=1), 0)
        clearance = float(signed.min())
        for plane in self.geometry.planes:
            clearance = min(clearance, float(position@plane.normal-plane.offset-abs(plane.normal)@self.geometry.drone_half_size))
        out_of_bounds = False
        try:
            voxel_violation = bool(self.occupancy.grid[self.occupancy.index(position)])
        except ValueError:
            voxel_violation, out_of_bounds = True, True
        launch = self.launch_active(time)
        tilt = np.rad2deg(np.arccos(np.clip(self.sim.data.xmat[self.sim.body_id].reshape(3, 3)[2, 2], -1, 1)))
        self.snapshot = dict(clearance_m=clearance,
                  clearance_violated=clearance <= self.config.navigation.safety_margin,
                  occupancy_violated=voxel_violation, launch_active=launch,
                  out_of_bounds=out_of_bounds,
                  envelope_tilt_exceeded=bool(tilt > self.sim.config.max_tilt_degrees+1e-6))
        return self.snapshot

    def failure_reference(self, state, sample):
        if self.reference_mode != "isaac_waypoints":
            return sample.position
        # A distant active waypoint is a valid command. Limit geometric route
        # deviation here; treating its target error as timed lag would fail at
        # every ordinary waypoint transition.
        points = self.plan.waypoints
        if len(points) < 2:
            return points[0]
        delta = np.diff(points, axis=0)
        fraction = np.clip(np.sum((state["position"]-points[:-1])*delta, axis=1)/np.maximum(np.sum(delta**2, axis=1), 1e-12), 0, 1)
        nearest = points[:-1]+fraction[:, None]*delta
        return nearest[np.argmin(np.linalg.norm(nearest-state["position"], axis=1))]

    def info(self):
        s = self.config.navigation
        return dict(algorithm="source 26-connected Dijkstra, no corner cutting, farthest-visible shortening",
                    reference_mode=self.reference_mode,
                    position_error_semantics="active feedback target" if self.reference_mode == "isaac_waypoints" else "time-indexed reference",
                    planning_seconds=self.planning_seconds, expanded_nodes=self.plan.expanded,
                    grid_distance_m=self.plan.grid_distance, shortened_distance_m=self.plan.distance,
                    grid_shape=list(self.occupancy.grid.shape), grid_resolution_m=s.resolution,
                    blocked_cells=int(np.count_nonzero(self.occupancy.grid)),
                    vehicle_half_size_m=self.geometry.drone_half_size.tolist(),
                    tracking_reserve_m=s.safety_margin, boxes=len(self.geometry.boxes), planes=len(self.geometry.planes),
                    has_launch=self.has_launch, launch_end_s=self.launch_end,
                    reference_duration_s=self.reference.duration,
                    start=self.plan.waypoints[0].tolist(), goal=self.plan.waypoints[-1].tolist())

    def save(self, output):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/"occupancy_grid.npz", grid=self.occupancy.grid,
                    origin=self.occupancy.origin, cell_size=self.occupancy.resolution,
                    drone_half_size=self.geometry.drone_half_size, safety_margin=self.config.navigation.safety_margin,
                    start=self.plan.waypoints[0], goal=self.plan.waypoints[-1])
        np.savetxt(output/"path_cells.csv", self.plan.cells, fmt="%d", delimiter=",", header="ix,iy,iz", comments="")
        np.savetxt(output/"waypoints_world.csv", self.plan.waypoints, delimiter=",", header="x_m,y_m,z_m", comments="")
        (output/"occupancy_obstacles.txt").write_text("\n".join(self.geometry.report)+"\n")
        raw = dict(boxes=[dict(path=b.path, lower=b.lower.tolist(), upper=b.upper.tolist()) for b in self.geometry.boxes],
                   planes=[dict(path=p.path, normal=p.normal.tolist(), offset=p.offset) for p in self.geometry.planes])
        (output/"collision_geometry.json").write_text(json.dumps(raw, indent=2)+"\n")
        (output/"plan.json").write_text(json.dumps(self.info(), indent=2, allow_nan=False)+"\n")
