import numpy as np
import pytest

from mujoco_drone.config import WindConfig
from mujoco_drone.weather import ScenarioWind


POINTS = np.array([[-.14, -.18, .05], [-.14, .18, .05], [.14, .18, .08], [.14, -.18, .08]])


def test_quadratic_drag_zero_relative_velocity_and_passivity():
    wind = ScenarioWind(WindConfig(scenario="constant", mean_velocity=(2, 0, 0)), POINTS, 5)
    wind.step(.01)
    _, forces, _ = wind.point_forces([0, 0, 0], [1, 0, 0, 0], [2, 0, 0], [0, 0, 0])
    np.testing.assert_allclose(forces, 0)
    _, f2, _ = wind.point_forces([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
    wind.mean_wind *= 2
    wind.point_winds = wind._compose()
    _, f4, _ = wind.point_forces([0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0])
    np.testing.assert_allclose(f4, 4*f2)
    relative = np.array([3, -1, 2])-wind.point_winds
    _, force, _ = wind.point_forces([0, 0, 0], [1, 0, 0, 0], [3, -1, 2], [0, 0, 0])
    assert np.all(np.sum(force*relative, axis=1) <= 0)


def test_seeded_ou_and_reset():
    config = WindConfig(scenario="turbulence", mean_velocity=(0, 0, 0))
    a, b = [ScenarioWind(config, POINTS, 91) for _ in range(2)]
    first = [a.step(.01).copy() for _ in range(40)]
    second = [b.step(.01).copy() for _ in range(40)]
    np.testing.assert_array_equal(first, second)
    a.reset()
    np.testing.assert_array_equal(first, [a.step(.01).copy() for _ in range(40)])


def test_gust_program_and_random_direction():
    wind = ScenarioWind(WindConfig(scenario="gust", gust_start=1, gust_duration=2), POINTS, 4)
    np.testing.assert_array_equal(wind.programmed_mean(.9), 0)
    np.testing.assert_array_equal(wind.programmed_mean(1), [0, 4, 0])
    np.testing.assert_array_equal(wind.programmed_mean(3), 0)
    a = ScenarioWind(WindConfig(scenario="randomized"), POINTS, 4)
    b = ScenarioWind(WindConfig(scenario="randomized"), POINTS, 4)
    np.testing.assert_array_equal(a.requested_mean, b.requested_mean)
    assert 0 <= np.linalg.norm(a.requested_mean) <= 6


@pytest.mark.parametrize("kwargs", [{"gust_std": (np.nan, 0, 0)}, {"effective_area": -1}, {"speed_range": (4, 1)}, {"gust_timescale": 0}])
def test_invalid_weather_rejected(kwargs):
    with pytest.raises(ValueError):
        WindConfig(**kwargs)
