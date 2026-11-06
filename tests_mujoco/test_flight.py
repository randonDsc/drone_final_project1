import numpy as np
import pytest

from mujoco_drone.config import ExperimentConfig, PhysicsConfig, TaskConfig, WindConfig
from mujoco_drone.core import FlightCore
from mujoco_drone.env import DroneEnv
from mujoco_drone.physics import DroneSimulation
from mujoco_drone.references import make_reference


def test_long_perturbed_calm_hover_has_no_secular_drift():
    env = DroneEnv(ExperimentConfig(task=TaskConfig(hover_seconds=120), seed=17))
    env.reset(seed=17)
    errors = []
    while not env.needs_reset:
        _, _, terminated, _, _ = env.step(np.zeros(4))
        assert not terminated
        errors.append(np.linalg.norm(env.sim.read_drone_state()["position"]-env.core.sample().position))
    assert max(errors) < .25
    assert max(errors[-100:]) < .005
    assert env.core.completion.completed
    env.close()


def test_seeded_observations_weather_and_actions_repeat_exactly():
    config = ExperimentConfig(task=TaskConfig(hover_seconds=1), wind=WindConfig(scenario="turbulence"))
    def flight():
        env = DroneEnv(config)
        observation, _ = env.reset(seed=71)
        result = [observation]
        rng = np.random.default_rng(9)
        while not env.needs_reset:
            observation, reward, _, _, _ = env.step(rng.uniform(-1, 1, 4))
            result.append(np.r_[observation, reward])
        env.close()
        return result
    for a, b in zip(flight(), flight()):
        np.testing.assert_array_equal(a, b)


def test_gymnasium_contract_and_partial_final_decision():
    from stable_baselines3.common.env_checker import check_env
    env = DroneEnv(ExperimentConfig(task=TaskConfig(hover_seconds=.225)))
    check_env(env, warn=True)
    observation, _ = env.reset(seed=1)
    assert observation.shape == (128,) and observation.dtype == np.float32
    count = 0
    while not env.needs_reset:
        observation, reward, terminated, truncated, info = env.step(np.zeros(4))
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        count += info["physics_steps"]
    assert count == 27
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step(np.zeros(4))
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step([0, 0, np.nan, 0])
    env.close()


def test_x2_residuals_fit_exact_allocation_budget():
    core = FlightCore(DroneSimulation())
    assert 0 < core.residual_scale < 1
    assert core.rotor_residual_bound.max() <= core.residual_budget+1e-12


@pytest.mark.parametrize("kind", ["straight", "waypoints", "circle", "figure_eight"])
def test_reference_velocity_and_acceleration_are_true_derivatives(kind):
    reference = make_reference(TaskConfig(kind=kind))
    dt = 1e-4
    for time in np.linspace(3.2, reference.motion_end-.2, 15):
        before, current, after = [reference.sample(time+x) for x in (-dt, 0, dt)]
        np.testing.assert_allclose((after.position-before.position)/(2*dt), current.velocity, atol=1e-6)
        np.testing.assert_allclose((after.velocity-before.velocity)/(2*dt), current.acceleration, atol=1e-6)
    assert reference.sample(reference.duration).finished


def test_curriculum_stages_reset_and_early_failures():
    env = DroneEnv(ExperimentConfig(wind=WindConfig(scenario="curriculum")), training=True)
    for stage in ["short_slow", "turns", "climbs_loops", "full"]:
        observation, info = env.reset(seed=4, options={"stage": stage})
        assert info["stage"] == stage
        assert np.all(np.isfinite(observation))
        for _ in range(3):
            _, reward, terminated, _, _ = env.step(np.zeros(4))
            assert np.isfinite(reward) and not terminated
    env.sim.data.qpos[2] = .1
    import mujoco
    mujoco.mj_forward(env.sim.model, env.sim.data)
    _, reward, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and not truncated and info["failure_reason"] is not None
    assert reward <= -100
    env.close()
