"""Watch the drone fly in the Isaac Sim window. No training happens here.

    C:\\isaacsim\\python.bat drone_code\\watch_hover_ppo.py

Takes no arguments: every setting is in watch_config.py. Repeats 10 s hover
episodes in real time until the window is closed, and prints the active settings
at startup. The green sphere marks the hover target; it is hidden inside the
drone when on it.

Pause -> Play resumes. Stop -> Play restarts the current episode.
"""
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

import model_contract
import hover_ppo_config
import sensor_config
import watch_config


project_dir = Path(__file__).resolve().parents[1]
model_path = project_dir / watch_config.MODEL_PATH
stats_path = hover_ppo_config.vecnormalize_path_for(model_path)

# Report a bad setting now, not after Isaac Sim has spent a minute starting.
if len(watch_config.TARGET_POSITION) != 3:
    sys.exit("watch_config.TARGET_POSITION must be [x, y, z].")

if watch_config.TARGET_POSITION[2] < (
    hover_ppo_config.FAILURE["min_height"]
    + hover_ppo_config.HOVER["initial_position_range"][2]
):
    sys.exit("watch_config.TARGET_POSITION z is too low; it must be >= 0.7 m.")

if watch_config.PLAYBACK_SPEED <= 0:
    sys.exit("watch_config.PLAYBACK_SPEED must be positive.")

if watch_config.USE_PPO:
    for path in (model_path, stats_path):
        if not path.is_file():
            sys.exit(f"File not found: {path}\nCheck watch_config.MODEL_PATH.")

    problem = model_contract.incompatibility(
        model_path, watch_config.STATE_SOURCE
    )

    if problem:
        sys.exit(problem)

if watch_config.STATE_SOURCE not in hover_ppo_config.STATE_SOURCES:
    sys.exit(f"watch_config.STATE_SOURCE must be one of {hover_ppo_config.STATE_SOURCES}.")

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": False})


def show_hover_target(env):
    """Point the view at the target and mark it. In-memory stage only."""
    from isaacsim.core.rendering_manager import ViewportManager
    from pxr import Gf, UsdGeom

    target = env.target_position

    marker = UsdGeom.Sphere.Define(env.sim.stage, "/World/HoverTarget")
    marker.CreateRadiusAttr(0.05)
    marker.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])
    marker.AddTranslateOp().Set(Gf.Vec3d(*target.tolist()))

    ViewportManager.set_camera_view(
        "/OmniverseKit_Persp",
        eye=(target + (2.5, -2.5, 1.0)).tolist(),
        target=target.tolist(),
    )


def print_active_settings(env):
    """Report what the running simulation holds, not what the file says."""
    import numpy as np

    wind = env.sim.wind
    pid_target = env.pid_target

    print("\n========== Active settings (watch_config.py) ==========")

    if watch_config.USE_PPO:
        print("Controller:      PID + PPO (deterministic, no training)")
        print(f"Model:           {model_path}")
        print(f"Normalization:   {stats_path}")
    else:
        print("Controller:      PID only")

    if env.estimator is None:
        print("Control state:   GROUND TRUTH (perfect state, no sensors)")
    else:
        print(f"Control state:   IMU + estimator, {sensor_config.IMU_PATH}")
        print(f"  IMU noise      {sensor_config.IMU_NOISE}")
        print(f"  estimator      {sensor_config.ESTIMATOR}")
        print(
            "  No position or heading reference: estimates drift, and the "
            "RMSE below is the true error."
        )

    print(f"Target position: {env.target_position.tolist()} m")
    print(
        f"PID target:      xy {pid_target['xy'].tolist()} m, "
        f"height {pid_target['height']} m"
    )

    if wind.enabled:
        print("Wind:            enabled")
        print(
            f"  mean           {wind.mean_wind.tolist()} m/s "
            f"(speed {np.linalg.norm(wind.mean_wind):.2f})"
        )
        print(
            f"  gust std       {wind.gust_std.tolist()} m/s "
            f"over {wind.gust_timescale} s"
        )
        print(
            f"  local gust std {wind.local_gust_std.tolist()} m/s "
            f"over {wind.local_gust_timescale} s"
        )
    else:
        print("Wind:            disabled (no wind force)")

    print(
        f"Episodes:        from seed {watch_config.FIRST_SEED}, "
        f"{watch_config.EPISODES or 'until the window is closed'}, "
        f"speed x{watch_config.PLAYBACK_SPEED}"
    )
    print("=======================================================")

    if watch_config.USE_PPO:
        for difference in model_contract.setting_differences(
            model_path, sensor_config.IMU_NOISE, sensor_config.ESTIMATOR
        ):
            print(f"WARNING: differs from training: {difference}")

    limit = hover_ppo_config.WIND_PROFILES[
        hover_ppo_config.DEFAULT_WIND_PROFILE
    ]["mean_speed_range"][1]

    if (
        watch_config.USE_PPO
        and wind.enabled
        and np.linalg.norm(wind.mean_wind) > limit
    ):
        print(
            f"WARNING: mean wind is above the {limit} m/s of the default "
            "training profile; PPO has not been tested there."
        )

    print(flush=True)


def wait_for_play(sim):
    """Idle through Pause or Stop; return True if the timeline was stopped."""
    stopped = False

    while sim.is_running() and not sim.world.is_playing():
        stopped = stopped or sim.world.is_stopped()
        simulation_app.update()

    return stopped


def fly_episode(env, seed, policy):
    """Fly one paced episode; None means Stop was pressed, so run it again."""
    import numpy as np

    from hover_env import SimulationClosed

    sim = env.sim
    observation, reset_info = env.reset(seed=seed)

    period = (
        env.decimation * sim.physics_dt / watch_config.PLAYBACK_SPEED
    )
    next_frame = time.perf_counter() + period

    errors = []
    actions = []
    estimate_error = float("nan")

    while True:
        if wait_for_play(sim):
            sim.reset_after_stop()
            sim.world.play()
            return None

        observation, _, terminated, truncated, info = env.step(
            policy(observation)
        )

        errors.append(info["position_error"])
        actions.append(info["residual_norm"])
        estimate_error = info["estimate_error"]

        delay = next_frame - time.perf_counter()

        if delay > 0:
            time.sleep(delay)
            next_frame += period
        else:
            # Behind real time, or back from a pause: do not rush to catch up.
            next_frame = time.perf_counter() + period

        if terminated or truncated:
            break

    # While the window closes, physics readings are not a real flight.
    if not sim.is_running():
        raise SimulationClosed("Window closed during an episode.")

    errors = np.array(errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    return {
        "wind": (
            f"mean wind {reset_info['wind_mean'].tolist()} m/s"
            if sim.wind.enabled
            else "wind off"
        ),
        "seconds": len(actions) * env.decimation * sim.physics_dt,
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_error": float(np.max(errors)),
        "estimate_error": estimate_error,
        "action": float(np.mean(actions)),
        "failure": info["failure_reason"],
    }


def main():
    import numpy as np

    from hover_env import (
        SimulationClosed,
        load_ppo_policy,
        make_hover_env,
    )

    env = make_hover_env(
        simulation_app,
        wind_profile=watch_config.WIND,
        render_sim=True,
        task=dict(
            hover_ppo_config.HOVER,
            target_position=watch_config.TARGET_POSITION,
        ),
        state_source=watch_config.STATE_SOURCE,
    )

    try:
        show_hover_target(env)

        if watch_config.USE_PPO:
            policy = load_ppo_policy(env, model_path, stats_path)
        else:
            policy = lambda observation: np.zeros(4, dtype=np.float32)

        print_active_settings(env)

        finished = 0

        while (
            watch_config.EPISODES == 0
            or finished < watch_config.EPISODES
        ):
            seed = watch_config.FIRST_SEED + finished
            row = fly_episode(env, seed, policy)

            if row is None:
                print(f"Stopped; restarting seed {seed}.", flush=True)
                continue

            print(
                f"seed {seed}: {row['wind']}, "
                f"{row['seconds']:.1f} s, RMSE {row['rmse']:.3f} m, "
                f"max {row['max_error']:.3f} m, "
                f"estimate off by {row['estimate_error']:.3f} m, "
                f"|a| {row['action']:.3f}, "
                f"{row['failure'] or 'no failure'}",
                flush=True,
            )
            finished += 1

    except SimulationClosed:
        print("Window closed.", flush=True)

    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close()
