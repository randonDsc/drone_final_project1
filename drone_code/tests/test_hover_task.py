"""Hover task and env checks: run with Isaac Sim's python, from drone_code:

    C:\\isaacsim\\python.bat -m unittest discover -s tests -p test_hover_task.py -v

No SimulationApp needed: the env runs against a small point-mass stand-in, so
these check the task logic, not the physics. smoke_ppo_hover.py covers Isaac.
"""
import unittest

import numpy as np

import hover_task
import model_contract
import ppo_config
import sim_config

from drone_diagnostics import rotate_vector
from hover_env import DroneHoverEnv, load_ppo_policy
from motor import MotorModel
from wind import GustingWind, quaternion_to_matrix


class FakeSim:
    """Point-mass stand-in with DroneSim's interface; attitude stays level."""

    def __init__(self):
        self.total_mass = 1.6
        self.dt = 1.0 / 120.0
        self.motors = MotorModel(**sim_config.MOTOR)
        self.wind = GustingWind(**dict(sim_config.WIND, enabled=True))
        self.running = True
        self.commands = []
        self.state = None
        self.imu_mount_rotation = np.eye(3)
        self.imu_mount_offset = np.zeros(3)
        self.imu_force = np.array([0.0, 0.0, 9.81])

    def read_imu(self):
        return np.zeros(3), self.imu_force.copy()

    def base_target(self):
        return {
            "mass": self.total_mass,
            "max_xy_speed": 5.0,
            "max_z_speed": 5.0,
            "max_tilt_degrees": 10.0,
            "max_total_thrust": 28.0,
        }

    def teleport(self, position, orientation, linear_velocity,
                 angular_velocity, motor_speeds):
        self.state = {
            "position": np.array(position, dtype=float),
            "orientation": np.array(orientation, dtype=float),
            "linear_velocity": np.array(linear_velocity, dtype=float),
            "angular_velocity": np.array(angular_velocity, dtype=float),
        }

    def read_drone_state(self):
        return {key: value.copy() for key, value in self.state.items()}

    def step_command(self, state, thrust, roll, pitch, yaw, render=True):
        self.commands.append((thrust, roll, pitch, yaw))
        self.wind.step(self.dt)
        acceleration = thrust / self.total_mass - 9.81
        self.state["linear_velocity"][2] += acceleration * self.dt
        self.state["position"] += self.state["linear_velocity"] * self.dt
        self.imu_force = np.array([0.0, 0.0, thrust / self.total_mass])
        return np.zeros(4), False, np.zeros(3), np.zeros(3)

    def is_running(self):
        return self.running

    def close(self):
        pass


def make_env(wind_profile="weak", state_source="ground_truth", **overrides):
    return DroneHoverEnv(
        FakeSim(),
        wind_profile=wind_profile,
        state_source=state_source,
        **overrides,
    )


def hover_state(**overrides):
    state = {
        "position": np.array(ppo_config.HOVER["target_position"], float),
        "orientation": np.array([1.0, 0.0, 0.0, 0.0]),
        "linear_velocity": np.zeros(3),
        "angular_velocity": np.zeros(3),
    }
    state.update(overrides)
    return state


class Geometry(unittest.TestCase):
    def test_euler_matches_controller_convention(self):
        for angles in ((0.1, -0.2, 0.3), (-0.3, 0.1, -1.0)):
            R = quaternion_to_matrix(hover_task.euler_to_quaternion(*angles))
            np.testing.assert_allclose(
                hover_task.roll_pitch(R), angles[:2], atol=1e-12
            )
            self.assertAlmostEqual(np.arctan2(R[1, 0], R[0, 0]), angles[2])

    def test_batched_rotation_matches_matrix(self):
        rng = np.random.default_rng(0)
        q = rng.normal(size=(5, 4))
        v = rng.normal(size=(5, 3))
        rotated = rotate_vector(q, v)

        for i in range(5):
            np.testing.assert_allclose(
                rotated[i], quaternion_to_matrix(q[i]) @ v[i]
            )
            np.testing.assert_allclose(rotate_vector(q[i], v[i]), rotated[i])


class Costs(unittest.TestCase):
    weights = ppo_config.REWARD
    target = np.array(ppo_config.HOVER["target_position"], float)

    def costs(self, state, action=np.zeros(4), previous=np.zeros(4)):
        return hover_task.step_costs(
            state, self.target, action, previous, self.weights
        )

    def test_perfect_hover_costs_nothing(self):
        self.assertEqual(sum(self.costs(hover_state()).values()), 0.0)

    def test_weights_and_scales(self):
        costs = self.costs(hover_state(
            position=self.target + [1.0, 0.0, 0.0],
            linear_velocity=np.array([0.0, 1.0, 0.0]),
            angular_velocity=np.array([0.0, 0.0, 1.0]),
            orientation=hover_task.euler_to_quaternion(
                np.deg2rad(20), 0.0, 0.0
            ),
        ), action=np.ones(4), previous=np.zeros(4))

        self.assertAlmostEqual(costs["position"], 1.0)
        self.assertAlmostEqual(costs["speed"], 0.2)
        self.assertAlmostEqual(costs["attitude"], 0.2)
        self.assertAlmostEqual(costs["angular_rate"], 0.05)
        self.assertAlmostEqual(costs["action"], 0.04)
        self.assertAlmostEqual(costs["action_rate"], 0.2)


class Failure(unittest.TestCase):
    limits = ppo_config.FAILURE
    target = np.array(ppo_config.HOVER["target_position"], float)

    def reason(self, **state):
        return hover_task.failure_reason(
            hover_state(**state), self.target, self.limits
        )

    def test_each_limit(self):
        self.assertIsNone(self.reason())
        self.assertEqual(
            self.reason(position=np.array([0.0, 0.0, 0.3])), "too_low"
        )
        self.assertEqual(
            self.reason(position=self.target + [3.0, 0.0, 0.0]),
            "out_of_bounds",
        )
        self.assertEqual(
            self.reason(orientation=hover_task.euler_to_quaternion(
                np.deg2rad(50), 0.0, 0.0
            )),
            "excessive_tilt",
        )
        self.assertEqual(
            self.reason(linear_velocity=np.array([4.0, 0.0, 0.0])),
            "too_fast",
        )
        self.assertEqual(
            self.reason(angular_velocity=np.array([6.0, 0.0, 0.0])),
            "spinning",
        )
        self.assertEqual(
            self.reason(position=np.array([np.nan, 0.0, 2.0])),
            "invalid_state",
        )

    def test_step_cost_bound_holds_for_random_healthy_states(self):
        rng = np.random.default_rng(0)
        bound = hover_task.max_step_cost(ppo_config.REWARD, self.limits)
        checked = 0

        for _ in range(20000):
            state = hover_state(
                position=self.target + rng.uniform(-2, 2, 3) * [1, 1, 0.7],
                linear_velocity=rng.uniform(-2, 2, 3),
                angular_velocity=rng.uniform(-3, 3, 3),
                orientation=hover_task.euler_to_quaternion(
                    *rng.uniform(-0.8, 0.8, 3)
                ),
            )

            if hover_task.failure_reason(state, self.target, self.limits):
                continue

            total = sum(hover_task.step_costs(
                state, self.target, rng.uniform(-1, 1, 4),
                rng.uniform(-1, 1, 4), ppo_config.REWARD,
            ).values())

            self.assertLessEqual(total, bound)
            checked += 1

        self.assertGreater(checked, 1000)

    def test_crashing_never_beats_the_worst_survival(self):
        gamma = ppo_config.PPO["gamma"]
        cost = hover_task.max_step_cost(ppo_config.REWARD, self.limits)
        penalty = hover_task.failure_penalty(
            ppo_config.REWARD, self.limits, gamma
        )

        # Worst survival: maximum cost every decision for ever.
        survive = cost * (1 - gamma**100000) / (1 - gamma)
        self.assertGreaterEqual(penalty, survive)


class Wind(unittest.TestCase):
    def test_episode_scaling_does_not_compound(self):
        wind = FakeSim().wind
        base = wind.gust_std.copy()
        wind.reset_episode(1, (1, 0, 0), gust_scale=0.5)
        wind.reset_episode(2, (1, 0, 0), gust_scale=0.5)
        np.testing.assert_allclose(wind.gust_std, 0.5 * base)

    def test_plain_reset_still_repeats_configured_sequence(self):
        wind = FakeSim().wind
        first = wind.step(0.01).copy()
        wind.reset()
        np.testing.assert_array_equal(wind.step(0.01), first)


class Env(unittest.TestCase):
    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env

        check_env(make_env(), skip_render_check=True)

    def test_spaces_and_types(self):
        env = make_env()
        observation, info = env.reset(seed=1)

        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertIn(observation, env.observation_space)
        self.assertEqual(env.action_space.dtype, np.float32)

        result = env.step(env.action_space.sample())
        self.assertEqual(len(result), 5)
        self.assertIsInstance(result[1], float)

    def test_decimation_holds_residual_and_recomputes_pd(self):
        env = make_env()
        env.reset(seed=1)
        env.sim.commands.clear()

        _, _, _, _, info = env.step(np.ones(4, dtype=np.float32))
        commands = np.array(env.sim.commands)

        self.assertEqual(len(commands), 4)
        self.assertEqual(info["physics_steps"], 4)

        # Torques are PID + constant, and PID sees a level drone here, so
        # thrust must change between substeps as PID reacts to the motion.
        self.assertGreater(np.ptp(commands[:, 0]), 0.0)

    def test_action_clipped_to_limits(self):
        env = make_env()
        env.reset(seed=1)
        _, _, _, _, info = env.step(np.full(4, 5.0, dtype=np.float32))

        np.testing.assert_allclose(
            info["residual"],
            hover_task.residual_limits_array(ppo_config.RESIDUAL_LIMITS),
        )

    def test_invalid_action_rejected(self):
        env = make_env()
        env.reset(seed=1)

        with self.assertRaises(ValueError):
            env.step(np.array([np.nan, 0, 0, 0], dtype=np.float32))

    def test_time_limit_truncates(self):
        env = make_env("none")
        env.reset(seed=1)

        for decision in range(ppo_config.HOVER["max_decisions"]):
            _, _, terminated, truncated, info = env.step(
                np.zeros(4, dtype=np.float32)
            )
            self.assertFalse(terminated)

        self.assertTrue(truncated)
        self.assertEqual(info["failure_penalty"], 0.0)

    def test_failure_terminates_with_penalty(self):
        env = make_env("none")
        env.reset(seed=1)
        env.sim.state["position"][0] += 5.0

        _, reward, terminated, truncated, info = env.step(
            np.zeros(4, dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["failure_reason"], "out_of_bounds")
        self.assertLess(
            reward,
            -ppo_config.REWARD["scale"] * info["failure_penalty"],
        )

    def test_invalid_state_gives_finite_output(self):
        env = make_env("none")
        env.reset(seed=1)
        env.sim.state["position"][0] = np.nan

        observation, reward, terminated, _, info = env.step(
            np.zeros(4, dtype=np.float32)
        )

        self.assertTrue(terminated)
        self.assertEqual(info["failure_reason"], "invalid_state")
        self.assertEqual(info["physics_steps"], 0)
        self.assertTrue(np.all(np.isfinite(observation)))
        self.assertTrue(np.isfinite(reward))

    def test_step_after_end_needs_reset(self):
        env = make_env("none")

        with self.assertRaises(RuntimeError):
            env.step(np.zeros(4, dtype=np.float32))

    def test_reset_clears_action_history(self):
        env = make_env()
        env.reset(seed=1)
        env.step(np.ones(4, dtype=np.float32))

        observation, _ = env.reset(seed=1)
        np.testing.assert_array_equal(observation[19:23], np.zeros(4))
        self.assertEqual(env.decisions, 0)

    def test_observation_hides_wind(self):
        a, b = make_env(), make_env()
        observation_a, _ = a.reset(seed=3)
        observation_b, _ = b.reset(seed=3)
        b.sim.wind.reset_episode(9, (9.0, 0.0, 0.0))

        np.testing.assert_array_equal(
            observation_a,
            b._observation(b.sim.read_drone_state()),
        )

    def test_seed_reproduces_wind_and_start(self):
        env = make_env()
        _, first = env.reset(seed=7)
        start = env.sim.read_drone_state()
        gust = env.sim.wind.point_winds.copy()
        _, second = env.reset(seed=7)

        np.testing.assert_array_equal(first["wind_mean"], second["wind_mean"])
        self.assertEqual(first["wind_seed"], second["wind_seed"])
        np.testing.assert_array_equal(
            start["position"], env.sim.read_drone_state()["position"]
        )
        np.testing.assert_array_equal(gust, env.sim.wind.point_winds)

    def test_training_resets_draw_new_wind(self):
        env = make_env()
        env.reset(seed=7)
        seeds = {env.reset()[1]["wind_seed"] for _ in range(20)}

        self.assertEqual(len(seeds), 20)

    def test_fixed_wind_is_never_resampled(self):
        settings = {
            "enabled": True,
            "mean_wind": (1.5, -2.5, 0.25),
            "gust_std": (0.5, 0.5, 0.2),
            "local_gust_std": (0.15, 0.15, 0.08),
        }
        env = make_env(wind_profile=settings)
        base_gust_std = env.sim.wind.gust_std.copy()
        env.reset(seed=3)
        seeds = set()

        for _ in range(10):
            _, info = env.reset()
            seeds.add(info["wind_seed"])

            np.testing.assert_array_equal(
                info["wind_mean"], settings["mean_wind"]
            )
            np.testing.assert_array_equal(
                env.sim.wind.mean_wind, settings["mean_wind"]
            )
            np.testing.assert_array_equal(
                env.sim.wind.gust_std, base_gust_std
            )

        self.assertEqual(len(seeds), 10)

    def test_gusts_fluctuate_around_fixed_mean(self):
        settings = dict(
            sim_config.WIND,
            mean_wind=(1.5, -2.5, 0.25),
            gust_std=(0.5, 0.5, 0.2),
            gust_timescale=2.0,
            local_gust_std=(0.0, 0.0, 0.0),
        )
        wind = GustingWind(**settings)
        wind.reset_episode(11, settings["mean_wind"], 1.0)

        samples = np.array([
            wind.step(1 / 30).mean(axis=0) for _ in range(600_000)
        ])

        np.testing.assert_allclose(
            samples.mean(axis=0), settings["mean_wind"], atol=0.03
        )
        np.testing.assert_allclose(
            samples.std(axis=0), settings["gust_std"], atol=0.03
        )

    def test_configured_target_reaches_controller_and_start(self):
        target = (3.0, -2.0, 1.5)
        env = make_env(task=dict(ppo_config.HOVER, target_position=target))
        env.reset(seed=1)

        np.testing.assert_array_equal(env.target_position, target)
        np.testing.assert_array_equal(env.pid_target["xy"], target[:2])
        self.assertEqual(env.pid_target["height"], target[2])
        self.assertLess(
            np.max(np.abs(env.sim.read_drone_state()["position"] - target)),
            0.2 + 1e-9,
        )

    def test_vecnormalize_paths_match_saved_names(self):
        path = ppo_config.vecnormalize_path_for

        self.assertEqual(
            path("a/final_model.zip").as_posix(),
            "a/final_vecnormalize.pkl",
        )
        self.assertEqual(
            path("a/hover_500000_steps.zip").as_posix(),
            "a/hover_vecnormalize_500000_steps.pkl",
        )

    def test_closed_window_raises(self):
        from hover_env import SimulationClosed

        env = make_env()
        env.reset(seed=1)
        env.sim.running = False

        with self.assertRaises(SimulationClosed):
            env.step(np.zeros(4, dtype=np.float32))


class TruthIsolation(unittest.TestCase):
    """Truth may only reach reward, failure, physics, and metrics."""

    @staticmethod
    def fly(position_shift, velocity_shift, tilt):
        """Fly with corrupted truth readings but the same IMU stream."""
        env = make_env(state_source="imu", wind_profile="none")
        env.sim.total_mass = 1.6
        reset_truth = env.sim.read_drone_state

        def shifted():
            state = reset_truth()
            state["position"] = state["position"] + position_shift
            state["linear_velocity"] = state["linear_velocity"] + velocity_shift
            state["orientation"] = hover_task.euler_to_quaternion(tilt, 0.0, 0.0)
            state["angular_velocity"] = state["angular_velocity"] + tilt
            return state

        env.sim.read_drone_state = shifted

        # Keep the IMU stream independent of the commands under test.
        env.sim.read_imu = lambda: (
            np.array([0.01, -0.02, 0.0]), np.array([0.05, 0.0, 9.81])
        )

        observations = [env.reset(seed=5)[0]]

        for _ in range(40):
            observations.append(
                env.step(np.array([0.3, -0.2, 0.1, 0.0], dtype=np.float32))[0]
            )

        return np.array(observations), np.array(env.sim.commands)

    def test_truth_cannot_reach_observation_or_commands(self):
        clean = self.fly(np.zeros(3), np.zeros(3), 0.0)
        corrupted = self.fly(
            np.array([0.6, -0.5, 0.4]),
            np.array([0.7, 0.4, -0.3]),
            np.deg2rad(20.0),
        )

        np.testing.assert_array_equal(clean[0], corrupted[0])
        np.testing.assert_array_equal(clean[1], corrupted[1])

    def test_ground_truth_mode_does_follow_truth(self):
        env = make_env()
        observation, _ = env.reset(seed=5)
        np.testing.assert_allclose(
            observation[:3],
            env.target_position - env.sim.read_drone_state()["position"],
            atol=1e-6,
        )

    def test_imu_start_hides_the_sampled_state(self):
        env = make_env(state_source="imu")
        observation, _ = env.reset(seed=5)

        np.testing.assert_array_equal(observation[:6], np.zeros(6))

    def test_failure_still_uses_truth(self):
        env = make_env(state_source="imu", wind_profile="none")
        env.reset(seed=5)
        env.sim.state["position"] = env.target_position + (3.0, 0.0, 0.0)

        _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertTrue(terminated)
        self.assertEqual(info["failure_reason"], "out_of_bounds")

    def test_dead_imu_stops_instead_of_flying_on_a_stale_estimate(self):
        env = make_env(state_source="imu", wind_profile="none")
        env.reset(seed=5)
        env.sim.read_imu = lambda: None

        with self.assertRaisesRegex(RuntimeError, "IMU stopped"):
            for _ in range(50):
                env.step(np.zeros(4, dtype=np.float32))

    def test_same_seed_repeats_the_imu_noise(self):
        def run():
            env = make_env(state_source="imu")
            env.reset(seed=9)

            return [
                env.step(np.zeros(4, dtype=np.float32))[0] for _ in range(20)
            ]

        np.testing.assert_array_equal(run(), run())


class Contract(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.run_dir = Path(self.folder.name) / "run"
        self.run_dir.mkdir()
        self.model = self.run_dir / "final_model.zip"

    def write(self, source):
        model_contract.write_contract(self.run_dir, model_contract.make_contract(
            source, {"a": 1}, {"b": 2}, hover_task.OBSERVATION_SIZE
        ))

    def test_old_model_is_refused_on_estimated_state(self):
        problem = model_contract.incompatibility(self.model, "imu")

        self.assertIn("ground_truth", problem)
        self.assertIn("different meaning", problem)

    def test_old_pd_model_is_refused_on_ground_truth_too(self):
        self.assertIn(
            "PID baseline",
            model_contract.incompatibility(self.model, "ground_truth"),
        )

    def test_imu_model_is_accepted_and_refused_on_truth(self):
        self.write("imu")

        self.assertIsNone(model_contract.incompatibility(self.model, "imu"))
        self.assertIsNotNone(
            model_contract.incompatibility(self.model, "ground_truth")
        )

    def test_checkpoints_use_the_run_contract(self):
        self.write("imu")
        checkpoint = self.run_dir / "checkpoints" / "hover_50000_steps.zip"

        self.assertIsNone(model_contract.incompatibility(checkpoint, "imu"))

    def test_changed_sensor_settings_are_reported(self):
        self.write("imu")

        self.assertEqual(
            model_contract.setting_differences(self.model, {"a": 9}, {"b": 2}),
            ["imu_noise.a: trained 1, now 9"],
        )

    def test_load_refuses_before_touching_any_file(self):
        env = make_env(state_source="imu")

        with self.assertRaisesRegex(RuntimeError, "different meaning"):
            load_ppo_policy(env, self.model, self.run_dir / "stats.pkl")


class Residual(unittest.TestCase):
    def test_full_residual_stays_within_rotor_headroom_at_hover(self):
        positions = np.array(sim_config.ROTOR_POSITIONS)
        motors = MotorModel(**sim_config.MOTOR)

        allocation = np.vstack([
            np.ones(4),
            positions[:, 1],
            -positions[:, 0],
            motors.reaction_signs * motors.km / motors.kf,
        ])

        limits = hover_task.residual_limits_array(ppo_config.RESIDUAL_LIMITS)
        hover = 1.618615 * hover_task.GRAVITY / 4
        headroom = min(
            hover, motors.kf * motors.max_speed**2 - hover
        )

        worst = np.abs(np.linalg.inv(allocation)) @ limits

        self.assertLess(worst.max(), 0.25 * headroom)


if __name__ == "__main__":
    unittest.main()
