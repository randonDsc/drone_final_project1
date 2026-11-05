"""PID and SAC environment checks: run with Isaac Sim's python, from drone_code:

    C:\\isaacsim\\python.bat -m unittest discover -s tests -p test_hover_sac.py -v

No SimulationApp needed: the env runs against test_hover_task's stand-in.
"""
import unittest

import numpy as np

import hover_task
import model_contract
import hover_sac_config
import hover_sac_policy

from controller import PIDController
from hover_sac_env import HoverSacEnv
from test_hover_task import FakeSim


LEVEL = (1.0, 0.0, 0.0, 0.0)
STILL = (0.0, 0.0, 0.0)
TARGET = {
    "height": 2.0, "xy": (0.0, 0.0), "mass": 1.5,
    "max_xy_speed": 5.0, "max_z_speed": 5.0,
    "max_total_thrust": 28.0,
}


def command(pid, position, dt=0.0, saturated=False):
    return pid.update(
        position, LEVEL, STILL, STILL, TARGET,
        dt=dt, actuators_saturated=saturated,
    )


class Pid(unittest.TestCase):
    def test_on_target_at_rest_is_hover_thrust(self):
        np.testing.assert_allclose(
            command(PIDController(), (0.0, 0.0, 2.0), 0.01),
            (1.5 * 9.81, 0.0, 0.0, 0.0),
        )

    def test_zero_dt_changes_nothing(self):
        pid = PIDController()

        for _ in range(5):
            command(pid, (0.3, 0.0, 1.8))

        self.assertEqual(pid.height_integral, 0.0)
        np.testing.assert_array_equal(pid.xy_integral, 0.0)

    def test_integral_scales_with_dt(self):
        fine, coarse = PIDController(), PIDController()

        for _ in range(10):
            command(fine, (0.3, 0.0, 1.8), 0.01)

        command(coarse, (0.3, 0.0, 1.8), 0.1)

        self.assertAlmostEqual(fine.height_integral, coarse.height_integral)
        np.testing.assert_allclose(fine.xy_integral, coarse.xy_integral)

    def test_integral_stops_at_its_limit(self):
        pid = PIDController()

        for _ in range(5000):
            command(pid, (0.3, 0.0, 1.8), 0.1)

        limit = pid.gains["height"]["integral_limit"]
        self.assertAlmostEqual(pid.height_integral, limit)

    def test_error_outside_the_zone_does_not_wind_up(self):
        pid = PIDController()
        command(pid, (30.0, 0.0, 2.0), 0.1)

        np.testing.assert_array_equal(pid.xy_integral, 0.0)

    def test_saturation_stops_growth_but_allows_shrinking(self):
        pid = PIDController()
        command(pid, (0.0, 0.0, 1.8), 0.1)
        held = pid.height_integral

        command(pid, (0.0, 0.0, 1.8), 0.1, saturated=True)
        self.assertEqual(pid.height_integral, held)

        command(pid, (0.0, 0.0, 2.2), 0.1, saturated=True)
        self.assertLess(pid.height_integral, held)

    def test_reset_clears_every_integrator(self):
        pid = PIDController()
        command(pid, (0.3, 0.3, 1.8), 0.1)
        pid.reset()

        self.assertEqual(pid.height_integral, 0.0)
        np.testing.assert_array_equal(pid.xy_integral, 0.0)
        np.testing.assert_array_equal(pid.attitude_integral, 0.0)

    def test_negative_dt_is_rejected(self):
        with self.assertRaises(ValueError):
            command(PIDController(), (0.0, 0.0, 2.0), -0.1)


class Observation(unittest.TestCase):
    def test_attitude_error_has_no_jump_across_plus_minus_pi(self):
        low = hover_sac_policy.attitude_error(
            hover_task.euler_to_quaternion(0.0, 0.0, 3.13), -3.13
        )

        self.assertAlmostEqual(low[2], 3.13 * 2 - 2 * np.pi, places=6)

    def test_quaternion_sign_does_not_matter(self):
        q = hover_task.euler_to_quaternion(0.1, -0.2, 0.3)

        np.testing.assert_allclose(
            hover_sac_policy.attitude_error(q, 0.0),
            hover_sac_policy.attitude_error(-q, 0.0),
        )

    def test_first_frame_fills_the_history_oldest_first(self):
        history = hover_sac_policy.ObservationHistory()
        first = np.full(hover_sac_policy.FRAME_SIZE, 1.0, dtype=np.float32)
        second = np.full(hover_sac_policy.FRAME_SIZE, 2.0, dtype=np.float32)

        self.assertEqual(
            history.push(first).size, hover_sac_policy.OBSERVATION_SIZE
        )

        stacked = history.push(second)

        self.assertEqual(stacked[0], 1.0)
        self.assertEqual(stacked[-1], 2.0)


class Env(unittest.TestCase):
    def make(self, stage="gusty"):
        return HoverSacEnv(FakeSim(), wind_stage=stage)

    def test_runs_ten_decisions_per_simulated_second(self):
        env = self.make()
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertEqual(info["physics_steps"], 12)

    def test_correction_is_added_once_and_held(self):
        env = self.make("light")
        env.reset(seed=1)
        env.step(np.zeros(4, dtype=np.float32))
        zero = np.array(env.sim.commands)

        env = self.make("light")
        env.reset(seed=1)
        env.step(np.array([1, 0, 0, 0], dtype=np.float32))
        pushed = np.array(env.sim.commands)

        self.assertAlmostEqual(
            pushed[0, 0] - zero[0, 0],
            hover_sac_config.RESIDUAL_LIMITS["thrust"],
        )

    def test_reset_clears_integrators_and_history(self):
        env = self.make()
        env.reset(seed=1)

        for _ in range(3):
            env.step(np.zeros(4, dtype=np.float32))

        self.assertNotEqual(env.pid.height_integral, 0.0)

        observation, _ = env.reset(seed=1)

        self.assertEqual(env.pid.height_integral, 0.0)
        np.testing.assert_array_equal(
            observation[: hover_sac_policy.FRAME_SIZE],
            observation[-hover_sac_policy.FRAME_SIZE:],
        )

    def test_same_seed_gives_the_same_wind_for_either_policy(self):
        env = self.make()
        _, first = env.reset(seed=4)
        env.step(np.ones(4, dtype=np.float32))
        _, second = env.reset(seed=4)

        np.testing.assert_array_equal(first["wind_mean"], second["wind_mean"])
        self.assertEqual(first["wind_seed"], second["wind_seed"])

    def test_stages_follow_training_progress(self):
        env = HoverSacEnv(FakeSim())

        self.assertEqual(env.stage["name"], "light")

        env.set_training_step(10_000_000)

        self.assertEqual(env.stage["name"], "strong")

    def test_time_limit_is_truncation_and_failure_is_termination(self):
        env = self.make()
        env.reset(seed=1)
        env.decisions = env.max_decisions - 1
        _, _, terminated, truncated, _ = env.step(
            np.zeros(4, dtype=np.float32)
        )

        self.assertFalse(terminated)
        self.assertTrue(truncated)

        env.reset(seed=1)
        env.sim.state["position"] = env.target_position + (5.0, 0.0, 0.0)
        _, reward, terminated, truncated, _ = env.step(
            np.zeros(4, dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertLess(reward, -10.0)

    def test_persistent_saturation_is_penalized(self):
        env = self.make()
        env.reset(seed=1)
        state = env.sim.read_drone_state()
        action = np.zeros(4, dtype=np.float32)

        for _ in range(hover_sac_config.PERSISTENT_SATURATION_DECISIONS):
            costs = env._step_costs(state, action, 12, 12)

        self.assertGreater(costs["persistent_saturation"], 0.0)

        costs = env._step_costs(state, action, 0, 12)

        self.assertEqual(costs["persistent_saturation"], 0.0)


class Contracts(unittest.TestCase):
    def test_settings_change_is_refused(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "final_model.zip"

            self.assertIn(
                "PPO models",
                model_contract.sac_incompatibility(
                    model, hover_sac_config.contract()
                ),
            )

            model_contract.write_contract(folder, hover_sac_config.contract())

            self.assertIsNone(model_contract.sac_incompatibility(
                model, hover_sac_config.contract()
            ))

            changed = dict(hover_sac_config.contract(), rl_rate_hz=20)

            self.assertIn(
                "rl_rate_hz",
                model_contract.sac_incompatibility(model, changed),
            )

    def test_replay_buffer_names_match_what_sb3_saves(self):
        path = hover_sac_config.replay_buffer_path_for

        self.assertEqual(
            path("a/final_model.zip").as_posix(),
            "a/final_replay_buffer.pkl",
        )
        self.assertEqual(
            path("a/sac_100000_steps.zip").as_posix(),
            "a/sac_replay_buffer_100000_steps.pkl",
        )


if __name__ == "__main__":
    unittest.main()
