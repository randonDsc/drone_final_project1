"""Geometry, original-protocol fidelity and actual obstacle flight checks."""
from dataclasses import replace
import json
import numpy as np
import pytest

from drone_code import hover_task, path_sac_config
from drone_code.navigation import line_of_sight, WaypointFollower
from drone_code.path_reference import sample_reference
from mujoco_drone.config import ExperimentConfig, SceneConfig, TaskConfig, WindConfig
from mujoco_drone.env import DroneEnv
from mujoco_drone.experiments import plan_experiment, run_experiment
from mujoco_drone.physics import DroneSimulation
from mujoco_drone.planning import FeedbackWaypointReference, read_geometry
from mujoco_drone.scenes import load_manifest, navigation_preset


@pytest.mark.parametrize("mode", ["quintic_stops", "isaac_timed"])
def test_actual_obstacle_flight_completes_with_clearance(tmp_path, mode):
    config = navigation_preset("blocks")
    config.navigation = replace(config.navigation, reference_mode=mode)
    result = run_experiment(config, tmp_path/mode)
    assert result["completed"] and not result["failed"]
    assert result["position_rmse"] < .05
    assert result["minimum_clearance_m"] > config.navigation.safety_margin
    assert result["clearance_violation_rate"] == 0
    assert result["occupancy_violation_rate"] == 0
    assert result["envelope_tilt_exceeded_rate"] == 0


def test_nominal_route_checks_every_interpolated_chord_and_rejects_blocked_goal():
    env = DroneEnv(navigation_preset("blocks"))
    try:
        env.reset(seed=7)
        mission = env.navigation
        assert mission.plan.distance > np.linalg.norm(mission.plan.waypoints[-1]-mission.plan.waypoints[0])
        assert all(line_of_sight(mission.occupancy, a, b)
                   for a, b in zip(mission.reference.path[:-1], mission.reference.path[1:]))
        env.config.task = replace(env.config.task, goal=(2.5, 0, 1.5))
        with pytest.raises(ValueError, match="blocked"):
            env.reset(seed=7)
    finally:
        env.close()


def test_geometry_uses_compiled_x2_and_only_static_collision_geoms():
    sim = DroneSimulation(scene=SceneConfig(name="blocks"))
    try:
        geometry, sources = read_geometry(sim)
        assert len(geometry.boxes) == 3 and len(geometry.planes) == 1
        assert sim.model.nq == 7 and sim.model.nv == 6 and sim.model.nu == 4
        assert sim.total_mass == pytest.approx(1.325)
        assert geometry.drone_half_size[0] > .4  # Yaw and tilt envelope, not Iris constants.
        assert geometry.drone_half_size[2] > .15
        for box in geometry.boxes:
            source = next(b for b in sim.scene_manifest["data"]["boxes"] if b["path"] == box.path)
            np.testing.assert_allclose(box.lower, source["lower"], atol=1e-10)
            np.testing.assert_allclose(box.upper, source["upper"], atol=1e-10)
            assert sources[box.path] in sim.geom_sources
    finally:
        sim.close()


def test_warehouse_manifest_retains_source_counts_and_audited_frozen_support():
    manifest = load_manifest(SceneConfig(name="warehouse"))["data"]
    assert len(manifest["boxes"]) == 3491 and len(manifest["planes"]) == 1
    assert manifest["source"]["frozen_dynamic_launch_supports"] == ["/World/Cube"]
    sim = DroneSimulation(scene=SceneConfig(name="warehouse"))
    try:
        geometry, _ = read_geometry(sim)
        assert len(geometry.boxes) == 3491
        assert sim.model.nbody == 2  # World and one free drone; stand is explicitly static.
        for expected, actual in zip(manifest["boxes"], geometry.boxes):
            np.testing.assert_allclose(actual.lower, expected["lower"], atol=1e-10)
            np.testing.assert_allclose(actual.upper, expected["upper"], atol=1e-10)
    finally:
        sim.close()


def test_native_contacts_and_conservative_envelope_are_distinct():
    sim = DroneSimulation(scene=SceneConfig(name="blocks"))
    try:
        sim.reset([2.5, 0, 1.5], [1, 0, 0, 0], np.zeros(3), np.zeros(3))
        assert sim.contacts() and sim.collided()
        wall = next(i for i, path in sim.geom_sources.items() if path.endswith("wall_a"))
        other = next(i for i, path in sim.geom_sources.items() if path.endswith("wall_b"))
        assert sim.collided({other})  # Exempting one support cannot hide another collision.
        assert not sim.collided({wall})
        sim.reset([1.7, 0, 1.5], [1, 0, 0, 0], np.zeros(3), np.zeros(3))
        assert not sim.collided()
    finally:
        sim.close()


def test_warehouse_launch_has_only_verified_supports_and_zero_residual():
    env = DroneEnv(navigation_preset("warehouse"))
    try:
        env.reset(seed=17)
        mission = env.navigation
        assert mission.has_launch and mission.launch_active(0)
        paths = {env.sim.geom_sources[i] for i in mission.support_ids}
        assert "/World/Cube" in paths and len(paths) == 2  # Named stand and its floor.
        env.step(np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(env.core.action, 0)
        np.testing.assert_array_equal(env.core.correction, 0)
        assert env.failure is None
    finally:
        env.close()


@pytest.mark.parametrize("profile", ["training_range", "unseen_gusts", "outside_range"])
def test_original_eval_route_wind_and_initial_rng_order_match(profile):
    seed = 300003
    rng = np.random.default_rng(seed)
    settings = path_sac_config.EVAL_WIND_PROFILES[profile]
    wind = hover_task.sample_stage_wind(rng, settings)
    stage = path_sac_config.stage_by_name(path_sac_config.EVAL_ROUTE_STAGE)
    kind, reference = sample_reference(rng, stage["route"], path_sac_config.EPISODE, "circle")
    initial = hover_task.sample_initial_state(rng, dict(path_sac_config.EPISODE, target_position=reference.path[0]))
    env = DroneEnv(ExperimentConfig(protocol="isaac_path", task=TaskConfig(kind="circle"), wind=WindConfig(scenario=profile)))
    try:
        _, info = env.reset(seed=seed)
        np.testing.assert_array_equal(env.core.reference.path, reference.path)
        assert env.route_type == kind and env.core.reference.duration == reference.duration
        for key in initial:
            np.testing.assert_array_equal(env.initial[key], initial[key])
        assert info["wind_seed"] == wind["seed"]
        np.testing.assert_array_equal(env.sim.wind.mean_wind, wind["mean_wind"])
        np.testing.assert_allclose(env.sim.wind.gust_std, np.array(path_sac_config.BASE_WIND["gust_std"])*wind["gust_scale"])
        assert env.sim.wind.gust_timescale == settings.get("gust_timescale", 2.0)
        assert env.core.direction_rate == wind["direction_rate"]
    finally:
        env.close()


def test_feedback_adapter_preserves_original_waypoint_commands():
    points = np.array([[0., 0., 1.5], [2, 2, 3], [3, 4, 3]])
    reference = FeedbackWaypointReference(points, hold=1, duration=60)
    source = WaypointFollower(points)
    state = dict(position=points[0].copy(), linear_velocity=np.zeros(3))
    reference.advance(state, .5, .1)
    np.testing.assert_array_equal(reference.sample(.5).position, points[0])
    for i in range(30):
        state["position"] = points[0]+np.array([.01, .01, 0])*max(i-10, 0)
        expected = source.update(state["position"], state["linear_velocity"], .1)
        reference.advance(state, 1+i*.1, .1)
        np.testing.assert_array_equal(reference.sample(1+i*.1).position, expected)
        assert reference.follower.index == source.index


def test_map_and_search_resource_limits_are_explicit():
    config = navigation_preset("blocks")
    for field, value, message in [("max_grid_cells", 100, "cells"), ("max_expanded_nodes", 1, "budget")]:
        candidate = replace(config, navigation=replace(config.navigation, **{field: value}))
        env = DroneEnv(candidate)
        try:
            with pytest.raises((ValueError, RuntimeError), match=message):
                env.reset(seed=0)
        finally:
            env.close()


def test_plan_only_saves_map_but_never_claims_flight_results(tmp_path):
    output = tmp_path/"plan"
    result = plan_experiment(navigation_preset("blocks"), output)
    assert result["expanded_nodes"] > 0
    assert (output/"navigation"/"occupancy_grid.npz").exists()
    assert (output/"navigation"/"waypoints_world.csv").exists()
    assert not (output/"metrics.json").exists() and not (output/"flight.csv").exists()
    assert "planning only" in json.loads((output/"metadata.json").read_text())["execution"]


def test_navigation_resets_reject_training_and_truncation_is_not_completion(tmp_path):
    config = navigation_preset("blocks")
    with pytest.raises(ValueError, match="open space"):
        DroneEnv(config, training=True)
    result = run_experiment(replace(config, max_seconds=.2), tmp_path/"short")
    assert not result["completed"] and not result["failed"]
    assert result["seconds"] == pytest.approx(.2)


def test_malformed_scene_fails_with_recorded_execution_error(tmp_path):
    scene = tmp_path/"invalid.json"
    scene.write_text(json.dumps(dict(schema_version=1, units="metres", up_axis="Z", boxes=[],
                    planes=[dict(path="floor", normal=[1, 0, 1], offset=0)])))
    config = ExperimentConfig(scene=SceneConfig(name="custom", manifest=str(scene)))
    with pytest.raises(ValueError, match="unit vectors"):
        run_experiment(config, tmp_path/"failed")
    assert json.loads((tmp_path/"failed"/"error.json").read_text())["type"] == "ValueError"
