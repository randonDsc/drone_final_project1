"""Compare PID-only with PID+PPO on matching wind. Run with Isaac Sim's python.bat.

    C:\\isaacsim\\python.bat drone_code\\eval_ppo_hover.py

Takes no arguments: set run_config.EVAL_PPO first.

Each seed in ppo_config.EVAL_SEEDS gives one mean wind, gust sequence, and start
state, and both controllers fly every one of them. Writes per-seed rows and
paired statistics to runs/ppo_hover/<run_name>/eval_<state>_<wind>.json, with the
same results in eval_<state>_<wind>_episodes.csv and _summary.csv.

Metrics: position RMSE, maximum position error, failure rate, residual
magnitude, motor saturation rate, and estimate error. All come from simulator
ground truth, which never reaches the controller or policy during flight.
"""
import csv
import json
from pathlib import Path

from isaacsim import SimulationApp

import model_contract
import ppo_config
import run_config


run = run_config.EVAL_PPO

project_dir = Path(__file__).resolve().parents[1]
run_dir = project_dir / ppo_config.RUN_DIR / run["run_name"]
model_path = project_dir / (run["model"] or run_dir / "final_model.zip")

if run["state_source"] not in ppo_config.STATE_SOURCES:
    raise SystemExit(f"state_source must be one of {ppo_config.STATE_SOURCES}.")

if not run["pid_only"] and model_path.is_file():
    problem = model_contract.incompatibility(model_path, run["state_source"])

    if problem:
        raise SystemExit(problem)

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": not run["gui"]})


def run_episode(env, seed, policy):
    """Fly one episode; policy(observation) returns a normalized action."""
    import numpy as np

    observation, reset_info = env.reset(seed=seed)

    errors = []
    estimate_errors = []
    residual_norms = []
    residual_physical = []
    saturated = 0
    physics_steps = 0
    total_cost = 0.0
    reason = None

    while True:
        observation, _, terminated, truncated, info = env.step(
            policy(observation)
        )

        errors.append(info["position_error"])
        estimate_errors.append(info["estimate_error"])
        residual_norms.append(info["residual_norm"])
        residual_physical.append(np.abs(info["residual"]))
        saturated += info["saturated_steps"]
        physics_steps += info["physics_steps"]
        total_cost += info["total_cost"]
        reason = info["failure_reason"]

        if terminated or truncated:
            break

    errors = np.array(errors, dtype=float)
    finite = errors[np.isfinite(errors)]
    mean_residual = np.mean(residual_physical, axis=0)

    return {
        "seed": int(seed),
        "wind_mean": [float(v) for v in reset_info["wind_mean"]],
        "wind_seed": reset_info["wind_seed"],
        "decisions": len(errors),
        "failed": reason is not None,
        "failure_reason": reason,
        "position_rmse": float(np.sqrt(np.mean(finite**2))),
        "max_position_error": float(np.max(finite)),
        "final_estimate_error": float(
            np.array(estimate_errors, dtype=float)[-1]
        ),
        "residual_norm": float(np.mean(residual_norms)),
        "mean_abs_thrust_residual": float(mean_residual[0]),
        "mean_abs_torque_residual": [float(v) for v in mean_residual[1:]],
        "saturation_rate": saturated / max(physics_steps, 1),
        "total_cost": float(total_cost),
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    import numpy as np

    return {
        "episodes": len(rows),
        "position_rmse": float(np.mean([r["position_rmse"] for r in rows])),
        "max_position_error": float(
            np.max([r["max_position_error"] for r in rows])
        ),
        "mean_max_position_error": float(
            np.mean([r["max_position_error"] for r in rows])
        ),
        "failure_rate": float(np.mean([r["failed"] for r in rows])),
        "final_estimate_error": float(
            np.nanmean([r["final_estimate_error"] for r in rows])
        ),
        "residual_norm": float(np.mean([r["residual_norm"] for r in rows])),
        "saturation_rate": float(
            np.mean([r["saturation_rate"] for r in rows])
        ),
        "total_cost": float(np.mean([r["total_cost"] for r in rows])),
    }


def main():
    import numpy as np

    from hover_env import load_ppo_policy, make_hover_env

    wind_profile = run["wind_profile"]

    seeds = ppo_config.EVAL_SEEDS

    if run["episodes"] is not None:
        seeds = seeds[:run["episodes"]]

    env = make_hover_env(
        simulation_app,
        wind_profile=wind_profile,
        render_sim=run["gui"],
        state_source=run["state_source"],
    )

    try:
        policies = {
            "pid_only": lambda observation: np.zeros(4, dtype=np.float32),
        }

        if not run["pid_only"]:
            policies["pid_ppo"] = load_ppo_policy(
                env,
                model_path,
                project_dir / run["vecnormalize"]
                if run["vecnormalize"]
                else ppo_config.vecnormalize_path_for(model_path),
            )

        results = {}

        for name, policy in policies.items():
            results[name] = [
                run_episode(env, seed, policy)
                for seed in seeds
            ]

        # Both controllers must have flown the same realizations.
        for rows in results.values():
            for row, reference in zip(rows, results["pid_only"]):
                if (row["wind_mean"], row["wind_seed"]) != (
                    reference["wind_mean"], reference["wind_seed"]
                ):
                    raise RuntimeError(
                        f"Wind differs on seed {row['seed']}."
                    )

        summary = {
            name: summarize(rows)
            for name, rows in results.items()
        }

        print(
            f"\nState source: {run['state_source']}; "
            f"wind profile: {wind_profile}; seeds: {len(seeds)}"
        )
        print(
            f"{'':10}{'RMSE [m]':>10}{'max [m]':>10}{'fail':>8}"
            f"{'|a|':>8}{'sat':>8}{'cost':>9}{'est err':>9}"
        )

        for name, s in summary.items():
            print(
                f"{name:10}{s['position_rmse']:10.3f}"
                f"{s['max_position_error']:10.3f}"
                f"{s['failure_rate']:8.2f}{s['residual_norm']:8.3f}"
                f"{s['saturation_rate']:8.4f}{s['total_cost']:9.1f}"
                f"{s['final_estimate_error']:9.3f}"
            )

        if "pid_ppo" in results:
            better = sum(
                ppo["position_rmse"] < pid["position_rmse"]
                for ppo, pid in zip(results["pid_ppo"], results["pid_only"])
            )
            print(f"PID+PPO has lower RMSE on {better}/{len(seeds)} seeds.")

        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / f"eval_{run['state_source']}_{wind_profile}.json"
        output.write_text(
            json.dumps(
                {
                    "state_source": run["state_source"],
                    "summary": summary,
                    "episodes": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_csv(
            output.with_name(f"{output.stem}_episodes.csv"),
            [
                {"controller": name, **row}
                for name, rows in results.items()
                for row in rows
            ],
        )
        write_csv(
            output.with_name(f"{output.stem}_summary.csv"),
            [{"controller": name, **s} for name, s in summary.items()],
        )

        print(f"Saved {output} and CSV files beside it", flush=True)

    finally:
        env.close()


try:
    main()
finally:
    simulation_app.close()
