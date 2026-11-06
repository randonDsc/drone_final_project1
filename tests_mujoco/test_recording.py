import csv
import json
import os

import mujoco
import numpy as np
import pytest

from mujoco_drone.config import ExperimentConfig, SensorConfig, TaskConfig
from mujoco_drone.experiments import run_experiment
from mujoco_drone.metrics import recovery_time
from mujoco_drone.physics import DroneSimulation
from mujoco_drone.sensors import CameraCapture, read_imu


def test_logged_timestamps_units_sensors_and_metadata(tmp_path):
    path = tmp_path/"flight"
    result = run_experiment(ExperimentConfig(task=TaskConfig(hover_seconds=.2)), path)
    with (path/"flight.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 24
    for row in rows:
        time = float(row["time_s"])
        assert time == float(row["control_interval_end_s"])
        assert time-float(row["control_interval_start_s"]) == pytest.approx(1/120)
        assert all(np.isfinite(float(v)) for v in row.values())
    metadata = json.loads((path/"metadata.json").read_text())
    assert metadata["state_source"] == "ground_truth_body_origin"
    assert metadata["model_audit"]["mass_kg"] == 1.325
    assert metadata["source"]["sha256"]
    assert not result["completed"]  # Cannot meet the 0.5 s dwell in 0.2 s.
    with pytest.raises(FileExistsError):
        run_experiment(ExperimentConfig(), path)


def test_imu_specific_force_is_distinct_from_ground_truth_acceleration():
    sim = DroneSimulation()
    sim.data.ctrl[:] = 0
    mujoco.mj_forward(sim.model, sim.data)
    np.testing.assert_allclose(read_imu(sim)["specific_force_sensor_m_s2"], 0, atol=1e-9)
    np.testing.assert_allclose(sim.read_drone_state()["com_acceleration"], [0, 0, -9.81], atol=1e-9)


def test_recovery_requires_dwell_and_reports_unobserved():
    times = np.arange(0, 6, .1)
    errors = np.where(times < 3, .5, .05)
    speed = np.zeros_like(times)
    assert recovery_time(times, errors, speed, gust_end=2) == pytest.approx(1)
    assert recovery_time(times[:35], errors[:35], speed[:35], gust_end=2) is None


@pytest.mark.skipif(os.environ.get("MUJOCO_TEST_RENDER") != "1", reason="Set MUJOCO_TEST_RENDER=1 to explicitly test platform OpenGL")
def test_camera_rgb_depth_and_known_plane_distance(tmp_path):
    sim = DroneSimulation()
    camera_id = sim.model.camera("drone_camera").id
    sim.model.cam_pos[camera_id] = [0, 0, -.1]
    sim.model.cam_quat[camera_id] = [1, 0, 0, 0]  # optical axis -Z, down to floor
    mujoco.mj_forward(sim.model, sim.data)
    camera = CameraCapture(sim, SensorConfig(width=64, height=48), tmp_path/"camera")
    try:
        camera.capture()
        with np.load(tmp_path/"camera"/"frame_000000.npz") as frame:
            assert frame["rgb"].shape == (48, 64, 3)
            assert frame["rgb"].dtype == np.uint8
            assert frame["depth_m"].shape == (48, 64)
            assert frame["depth_m"][24, 32] == pytest.approx(1.4, abs=.002)
            assert float(frame["time_s"]) == 0
    finally:
        camera.close()
