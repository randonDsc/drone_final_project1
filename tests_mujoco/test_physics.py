import numpy as np
import pytest

from drone_code.hover_task import euler_to_quaternion
from drone_code.wind import quaternion_to_matrix
from mujoco_drone.config import WindConfig
from mujoco_drone.physics import DroneSimulation


def test_compiled_model_and_hover_force_balance():
    sim = DroneSimulation(wind=WindConfig(enabled=False))
    audit = sim.audit()
    assert audit["mass_kg"] == pytest.approx(1.325)
    assert np.linalg.eigvalsh(audit["inertia_body_kg_m2"]).min() > 0
    np.testing.assert_allclose(audit["hover_controls_N"], 3.2495625)
    assert sim.model.opt.density == 0 and sim.model.opt.viscosity == 0
    for _ in range(120):
        sim.step_command(sim.read_drone_state(), sim.total_mass*9.81, 0, 0, 0)
    np.testing.assert_allclose(sim.read_drone_state()["position"], [0, 0, 1.5], atol=1e-10)
    np.testing.assert_allclose(sim.data.sensor("body_linacc").data, [0, 0, 9.81], atol=1e-9)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_allocation_torque_sign_and_magnitude(axis):
    sim = DroneSimulation(wind=WindConfig(enabled=False))
    command = np.array([sim.total_mass*9.81, 0, 0, 0])
    command[axis+1] = 0.01
    controls = np.linalg.solve(sim.allocation_matrix, command)
    sim.data.ctrl[:] = controls
    import mujoco
    mujoco.mj_forward(sim.model, sim.data)
    inertia = np.array(sim.audit()["inertia_body_kg_m2"])
    np.testing.assert_allclose(inertia @ sim.data.qacc[3:], command[1:], atol=1e-10)


def test_freejoint_world_body_and_com_frames():
    sim = DroneSimulation(wind=WindConfig(enabled=False))
    q = euler_to_quaternion(.1, .2, .7)
    omega = np.array([.1, .7, .2])
    sim.reset(orientation=q, angular_velocity=omega, linear_velocity=[1, 2, 3])
    state = sim.read_drone_state()
    np.testing.assert_allclose(state["angular_velocity"], omega)
    np.testing.assert_allclose(state["linear_velocity"], [1, 2, 3])
    np.testing.assert_allclose(state["com_velocity"], np.array([1, 2, 3])+np.cross(omega, quaternion_to_matrix(q)@sim.body_com))


def test_com_wrench_and_saturation():
    sim = DroneSimulation(wind=WindConfig(scenario="constant", mean_velocity=(2, 0, 0)))
    sim.step_command(sim.read_drone_state(), sim.total_mass*9.81, 0, 0, 0)
    expected = .5*1.225*.03*2**2
    assert sim.last["force"][0] == pytest.approx(expected)
    arm = sim.wind.offsets-sim.body_com
    np.testing.assert_allclose(sim.last["torque"], np.cross(arm, sim.last["point_forces"]).sum(axis=0))
    _, clipped, saturated = sim.mix([1000, 10, 10, 10])
    assert saturated and np.all((0 <= clipped) & (clipped <= 13))
    with pytest.raises(ValueError):
        sim.mix([np.nan, 0, 0, 0])
