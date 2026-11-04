"""Short Isaac Sim validation, to run before any long training.

    C:\\isaacsim\\python.bat drone_code\\smoke_ppo_hover.py

Takes no arguments: settings are in run_config.SMOKE_PPO.

Checks reset/step behavior, zero-residual operation, bounded nonzero actions,
reset leakage, matching wind between policies, failure handling, and one small
PPO rollout/update. Prints PASS or FAIL per check, then the simulation rate.
Exit code is 1 if anything failed.
"""
import sys
import tempfile
import time
import traceback
from pathlib import Path

from isaacsim import SimulationApp

import ppo_config
import run_config


run = run_config.SMOKE_PPO

if run["state_source"] not in ppo_config.STATE_SOURCES:
    raise SystemExit(f"state_source must be one of {ppo_config.STATE_SOURCES}.")

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": not run["gui"]})


def main():
    import numpy as np

    import hover_task

    from controller import PIDController
    from drone_diagnostics import cross
    from hover_env import make_hover_env
    from wind import quaternion_to_matrix

    env = make_hover_env(
        simulation_app,
        wind_profile="weak",
        render_sim=run["gui"],
        state_source=run["state_source"],
    )
    sim = env.sim

    results = []
    zeros = np.zeros(4, dtype=np.float32)

    # Reset-to-reset physics differences below this are solver noise [m].
    tolerance = 1e-3

    def fly(seed, decisions, action=zeros):
        """Positions after each decision, starting from reset(seed)."""
        observation, _ = env.reset(seed=seed)
        positions = []

        for _ in range(decisions):
            observation, _, terminated, truncated, _ = env.step(action)
            positions.append(sim.read_drone_state()["position"].copy())

            if terminated or truncated:
                break

        return np.array(positions, dtype=float)

    def check(function):
        name = function.__name__

        try:
            detail = function()
            results.append(True)
            print(f"PASS {name}: {detail}", flush=True)
        except Exception:
            results.append(False)
            print(f"FAIL {name}:\n{traceback.format_exc()}", flush=True)

    def reset_and_step_contract():
        observation, info = env.reset(seed=1)
        assert observation.dtype == np.float32
        assert env.observation_space.contains(observation), observation
        assert np.all(np.isfinite(observation))

        state = sim.read_drone_state()
        assert state["position"][2] > 1.5, state["position"]

        result = env.step(env.action_space.sample())
        assert len(result) == 5
        assert isinstance(result[1], float)
        assert result[4]["physics_steps"] == 4
        assert not result[2] and not result[3]

        return f"obs shape {observation.shape}, wind {info['wind_mean']}"

    def timing_is_120hz_and_30hz():
        assert abs(sim.world.get_physics_dt() - 1 / 120) < 1e-9

        env.reset(seed=1)
        start = sim.world.current_time
        env.step(zeros)
        elapsed = sim.world.current_time - start
        assert abs(elapsed - 1 / 30) < 1e-6, elapsed

        return f"decision takes {elapsed:.5f} s"

    def reset_has_no_kick():
        # Level, motionless start: the body must not pick up rotation just
        # from the rotor restart. The limit sits above the 0.03-0.05 rad/s
        # that the rotor drives induce even in steady hover, and below the
        # 0.15 rad/s of a restart above PROPELLER_SETTLED_SPEED.
        task = dict(env.task)
        task.update(
            initial_position_range=(0, 0, 0),
            initial_velocity_range=(0, 0, 0),
            initial_angle_range_degrees=(0, 0, 0),
            initial_angular_velocity_range=(0, 0, 0),
        )
        saved = env.task, env.wind_profile
        env.task = task
        env.wind_profile = ppo_config.WIND_PROFILES["none"]
        enabled = sim.wind.enabled
        sim.wind.enabled = False

        try:
            env.reset(seed=1)
            rates = []

            for _ in range(5):
                env.step(zeros)
                rates.append(
                    np.linalg.norm(sim.read_drone_state()["angular_velocity"])
                )
        finally:
            env.task, env.wind_profile = saved
            sim.wind.enabled = enabled

        assert max(rates) < 0.08, rates
        return f"peak body rate {max(rates):.5f} rad/s"

    def zero_residual_matches_plain_pid():
        decisions = 60
        through_env = fly(11, decisions)

        # Same reset, flown directly on the backend with no env in the loop.
        env.reset(seed=11)
        target = env.pid_target
        pid = PIDController()
        saturated = False
        direct = []

        for step in range(decisions * env.decimation):
            state = sim.read_drone_state()
            command = pid.update(
                positions=state["position"],
                orientations=state["orientation"],
                velocities=state["linear_velocity"],
                angular_velocities=state["angular_velocity"],
                target=target,
                dt=sim.dt,
                actuators_saturated=saturated,
            )
            _, saturated, _, _ = sim.step_command(
                state, *command, render=False
            )

            if (step + 1) % env.decimation == 0:
                direct.append(sim.read_drone_state()["position"].copy())

        gap = np.abs(through_env - np.array(direct, dtype=float)).max()
        assert gap < tolerance, gap

        return f"largest position gap {gap:.2e} m"

    def imu_agrees_with_true_motion():
        # The first run on Isaac's real IMU: checks axes, signs, and mounting.
        env.reset(seed=14)

        for _ in range(30):
            env.step(zeros)

        gyro, accel = sim.read_imu()
        state = sim.read_drone_state()

        R = quaternion_to_matrix(state["orientation"])
        mount = sim.imu_mount_rotation

        rate_gap = np.linalg.norm(
            mount @ gyro - R.T @ state["angular_velocity"]
        )
        force_gap = np.linalg.norm(
            mount @ accel - R.T @ np.array([0.0, 0.0, hover_task.GRAVITY])
        )

        assert rate_gap < 0.2, rate_gap
        assert force_gap < 2.0, force_gap

        return f"gyro gap {rate_gap:.3f} rad/s, force gap {force_gap:.3f} m/s^2"

    def estimator_flies_one_episode():
        env.reset(seed=15)
        steps = 0
        info = {"failure_reason": None}

        for _ in range(150):
            _, _, terminated, truncated, info = env.step(zeros)
            steps += 1

            if terminated or truncated:
                break

        state = sim.read_drone_state()
        estimate = env.estimator.state()

        tilt = np.degrees(np.arccos(np.clip(
            quaternion_to_matrix(state["orientation"])[:, 2]
            @ quaternion_to_matrix(estimate["orientation"])[:, 2],
            -1.0,
            1.0,
        )))

        assert env.estimator.aligned
        assert env.missed_imu_samples == 0
        assert tilt < 8.0, tilt

        return (
            f"{steps} decisions, tilt error {tilt:.2f} deg, position error "
            f"{info['estimate_error']:.3f} m, {info['failure_reason']}"
        )

    def bounded_nonzero_actions():
        limits = env.residual_limits
        env.reset(seed=2)
        rng = np.random.default_rng(0)
        moved = 0.0

        for _ in range(30):
            action = rng.uniform(-3, 3, 4).astype(np.float32)
            _, _, terminated, _, info = env.step(action)

            assert np.all(np.abs(info["residual"]) <= limits + 1e-9)
            assert info["saturated_steps"] == 0, info
            moved = max(moved, np.abs(info["residual"]).max())

            if terminated:
                raise AssertionError(info["failure_reason"])

        return f"largest applied residual {moved:.3f}, no saturation"

    def residual_changes_motion():
        zero = fly(3, 30)
        pushed = fly(3, 30, np.array([1, 0, 0, 0], dtype=np.float32))
        climb = pushed[-1, 2] - zero[-1, 2]
        assert climb > 0.02, climb

        return f"+thrust climbs {climb:.3f} m more in 1 s"

    def same_seed_is_repeatable():
        first = fly(4, 90)
        second = fly(4, 90)
        gap = np.abs(first - second).max()
        assert gap < tolerance, gap

        return f"largest position gap {gap:.2e} m"

    def nothing_leaks_between_episodes():
        clean = fly(5, 60)

        # A violent episode with a saturating action, then the same seed.
        fly(6, 120, np.ones(4, dtype=np.float32))
        after = fly(5, 60)

        gap = np.abs(clean - after).max()
        assert gap < tolerance, gap

        return f"largest position gap {gap:.2e} m"

    def wind_is_independent_of_policy():
        def wind_sequence(action):
            env.reset(seed=8)
            winds = []

            for _ in range(20):
                env.step(action)
                winds.append(sim.wind.point_winds.copy())

            return np.array(winds)

        calm = wind_sequence(zeros)
        pushed = wind_sequence(np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(calm, pushed)

        return "identical point winds under different actions"

    def wind_differs_between_episodes():
        env.reset(seed=9)
        seen = {env.reset()[1]["wind_seed"] for _ in range(5)}
        assert len(seen) == 5

        return f"{len(seen)} distinct gust seeds"

    def failure_terminates():
        env.reset(seed=10)

        # Knock it over: a real sideways teleport well past the tilt limit.
        sim.teleport(
            env.target_position,
            hover_task.euler_to_quaternion(np.deg2rad(60), 0.0, 0.0),
            np.zeros(3),
            np.zeros(3),
            np.full(4, env.hover_motor_speed),
        )

        _, reward, terminated, truncated, info = env.step(zeros)
        assert terminated and not truncated
        assert info["failure_reason"] == "excessive_tilt", info
        assert reward < -100.0 * ppo_config.REWARD["scale"], reward

        return f"reason {info['failure_reason']}, reward {reward:.1f}"

    def com_matches_per_body_calculation():
        env.reset(seed=12)
        env.step(zeros)

        # Old per-body formula, written out, against the single-view result.
        from drone_diagnostics import (
            read_total_com_in_body_frame,
            rotate_vector,
        )

        view = sim.vehicle_bodies
        masses = np.asarray(view.get_masses(), dtype=float).reshape(-1)
        positions, orientations = view.get_world_poses()
        local_coms, _ = view.get_coms()
        local_coms = np.asarray(local_coms).reshape(-1, 3)

        weighted = np.zeros(3)

        for i, mass in enumerate(masses):
            weighted += mass * (
                np.asarray(positions[i], dtype=float)
                + rotate_vector(
                    np.asarray(orientations[i], dtype=float),
                    local_coms[i],
                )
            )

        body_position, body_orientation = sim.body.get_world_poses()
        inverse = np.array(body_orientation[0], dtype=float)
        inverse[1:] *= -1

        expected = rotate_vector(
            inverse,
            weighted / masses.sum() - body_position[0],
        )

        fast = read_total_com_in_body_frame(sim.body, view)
        np.testing.assert_allclose(fast, expected, atol=1e-6)
        np.testing.assert_allclose(
            cross(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])),
            np.cross([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        )

        return f"vehicle COM offset {np.round(fast, 5)} m"

    def small_ppo_update():
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            VecNormalize,
        )

        check_env(env, warn=True)

        venv = VecNormalize(
            DummyVecEnv([lambda: Monitor(env)]),
            norm_obs=True,
            norm_reward=False,
            clip_obs=ppo_config.TRAINING["observation_clip"],
        )

        settings = dict(ppo_config.PPO, n_steps=128, batch_size=64, n_epochs=2)
        model = PPO("MlpPolicy", venv, seed=0, verbose=0, **settings)

        before = [p.detach().clone() for p in model.policy.parameters()]
        model.learn(total_timesteps=256)

        changed = any(
            not (a == b).all()
            for a, b in zip(before, model.policy.parameters())
        )
        assert changed, "PPO update did not change the policy."

        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "model"
            stats_path = Path(folder) / "stats.pkl"
            model.save(str(model_path))
            venv.save(str(stats_path))

            loaded = VecNormalize.load(str(stats_path), venv.venv)
            np.testing.assert_array_equal(
                loaded.obs_rms.mean, venv.obs_rms.mean
            )
            PPO.load(str(model_path), device="cpu")

        return f"{model.num_timesteps} steps, policy changed, stats reloaded"

    def report_rate():
        env.reset(seed=13)
        start = time.perf_counter()
        steps = 0

        for _ in range(120):
            _, _, terminated, truncated, info = env.step(zeros)
            steps += info["physics_steps"]

            if terminated or truncated:
                env.reset()

        elapsed = time.perf_counter() - start

        return (
            f"{steps / elapsed:.0f} physics steps/s, "
            f"{120 / elapsed:.0f} decisions/s, "
            f"{3600 * 120 / elapsed / 1e6:.2f} M decisions/hour"
        )

    # Flying plain PID on truth only matches the env in ground-truth mode.
    source_checks = (
        (imu_agrees_with_true_motion, estimator_flies_one_episode)
        if run["state_source"] == "imu"
        else (zero_residual_matches_plain_pid,)
    )

    try:
        for function in (
            reset_and_step_contract,
            timing_is_120hz_and_30hz,
            reset_has_no_kick,
            *source_checks,
            bounded_nonzero_actions,
            residual_changes_motion,
            same_seed_is_repeatable,
            nothing_leaks_between_episodes,
            wind_is_independent_of_policy,
            wind_differs_between_episodes,
            failure_terminates,
            com_matches_per_body_calculation,
            small_ppo_update,
            report_rate,
        ):
            check(function)

    finally:
        env.close()

    print(f"\n{sum(results)}/{len(results)} checks passed.", flush=True)

    return all(results)


passed = False

try:
    passed = main()
finally:
    simulation_app.close()

sys.exit(0 if passed else 1)
