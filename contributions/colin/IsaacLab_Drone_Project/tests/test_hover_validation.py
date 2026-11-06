"""Regression checks for validation scoring, especially incomplete/reset runs."""

from types import SimpleNamespace

import torch

from scripts.validate_hover import initial_conditions, summarize


def trajectory(error=0.0, clipped=False):
    rows = []
    for step in range(501):
        row = {
            "time_s": step * 0.02,
            "error_m": error,
            "speed_m_s": 0.0,
            "tilt_deg": 0.0,
            "yaw_error_deg": 0.0,
            "position_w": [error, 0, 1.5],
        }
        if step < 500:
            row["saturated"] = clipped
        rows.append(row)
    return rows


def test_complete_stable_trajectory_passes():
    assert summarize(trajectory(), 0.02, 500, None)["passed"]


def test_timeout_reset_and_incomplete_trajectory_cannot_pass():
    assert not summarize(trajectory()[:100], 0.02, 500, "unexpected_env_termination_or_reset")["passed"]
    assert not summarize(trajectory()[:100], 0.02, 500, None)["passed"]


def test_error_and_clipping_fail_despite_completion():
    assert not summarize(trajectory(error=0.1), 0.02, 500, None)["passed"]
    assert not summarize(trajectory(clipped=True), 0.02, 500, None)["passed"]


def test_early_excursion_fails_even_with_good_final_state():
    rows = trajectory()
    rows[5]["tilt_deg"] = 60.0
    assert not summarize(rows, 0.02, 500, None)["passed"]


def test_command_logging_cannot_mutate_matched_start():
    actuator = SimpleNamespace(tau_inc_s=torch.ones(1, 4), tau_dec_s=torch.ones(1, 4), thrust_const=torch.ones(1, 4))
    row = trajectory()[0]
    reference = initial_conditions(row, actuator)
    row["motor_command"] = [3.0] * 4
    row["position_w"][0] = 10.0
    assert "motor_command" not in reference["state"]
    assert reference["state"]["position_w"][0] == 0.0
