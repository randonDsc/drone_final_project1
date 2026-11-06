"""Simulator-free checks for force conventions, feedback, and controller state."""

import torch

from IsaacLab_Drone_Project.controllers.cascaded_pid import CascadedPID


def setup_controller():
    return CascadedPID(torch.tensor([1.24]), torch.tensor([[0.0, 0.0, -9.81]]), 0.02)


def state():
    return (torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([[1.0, 0, 0, 0]]), torch.zeros(1, 3))


def test_hover_and_force_direction():
    controller = setup_controller()
    wrench = controller.compute(torch.zeros(1, 4), *state())
    torch.testing.assert_close(wrench, torch.tensor([[0, 0, 1.24 * 9.81, 0, 0, 0]]))
    target = torch.tensor([[0.2, 0, 0.1, 0]])
    wrench = controller.compute(target, *state())
    assert wrench[0, 2] > 1.24 * 9.81
    assert wrench[0, 4] > 0  # Positive pitch points body thrust toward world +X.


def test_integral_freezes_on_clipping_and_resets():
    controller = setup_controller()
    controller.compute(torch.ones(1, 4), *state())
    before = controller.velocity_integral.clone()
    controller.allocation_feedback(torch.full((1, 4), 8.0), torch.full((1, 4), 6.0))
    controller.compute(torch.ones(1, 4), *state())
    torch.testing.assert_close(controller.velocity_integral, before)
    controller.reset()
    assert not controller.velocity_integral.any()
    assert not controller.rate_integral.any()
    assert controller.previous_velocity is None


def test_yaw_wrap_uses_short_rotation():
    controller = setup_controller()
    position, velocity, _, angular_velocity = state()
    yaw = torch.tensor(3.13)
    quaternion = torch.tensor([[torch.cos(yaw / 2), 0, 0, torch.sin(yaw / 2)]])
    wrench = controller.compute(torch.tensor([[0, 0, 0, -3.13]]), position, velocity, quaternion, angular_velocity)
    assert 0 < wrench[0, 5] < 0.02
