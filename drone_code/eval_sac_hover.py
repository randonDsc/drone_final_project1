"""Compare PID-only with PID + SAC on matching wind. Run with Isaac Sim's python.bat.

    C:\\isaacsim\\python.bat drone_code\\eval_sac_hover.py

Takes no arguments: set run_config.EVAL_SAC first.

Each seed in sac_config.EVAL_SEEDS gives one wind and start state, and both modes
fly every one of them. Writes eval_<stage>.json, eval_<stage>_episodes.csv, and
eval_<stage>_summary.csv in runs/sac_hover/<run_name>/.

Metrics: position RMSE over the episode and over its second half, maximum error,
failure rate, residual magnitude, and motor saturation rate.
"""
import csv
import json
from pathlib import Path

from isaacsim import SimulationApp

import model_contract
import run_config
import sac_config


run = run_config.EVAL_SAC

project_dir = Path(__file__).resolve().parents[1]
run_dir = project_dir / sac_config.RUN_DIR / run["run_name"]
model_path = project_dir / (run["model"] or run_dir / "final_model.zip")

# Report these now, not after Isaac Sim has started.
try:
    sac_config.stage_by_name(run["wind_stage"])
except ValueError as error:
    raise SystemExit(str(error))

if not run["pid_only"]:
    if not model_path.is_file():
        raise SystemExit(
            f"Model not found: {model_path}\n"
            "Set pid_only in run_config.EVAL_SAC for the baseline alone."
        )

    problem = model_contract.sac_incompatibility(
        model_path, sac_config.contract()
    )

    if problem:
        raise SystemExit(problem)

# SimulationApp must exist before any omni, pxr, or isaacsim.core import.
simulation_app = SimulationApp({"headless": not run["gui"]})


def run_episode(env, seed, policy):
    """Fly one episode; policy(observation) returns a normalized action."""
    import numpy as np

    observation, reset_info = env.reset(seed=seed)

    errors = []
    residual_norms = []
    saturated = 0
    physics_steps = 0
    total_cost = 0.0
    reason = None

    while True:
        observation, _, terminated, truncated, info = env.step(
            policy(observation)
        )

        errors.append(info["position_error"])
        residual_norms.append(info["residual_norm"])
        saturated += info["saturated_steps"]
        physics_steps += info["physics_steps"]
        total_cost += info["total_cost"]
        reason = info["failure_reason"]

        if terminated or truncated:
            break

    errors = np.array(errors, dtype=float)
    finite = errors[np.isfinite(errors)]
    late = finite[len(finite) // 2:]

    return {
        "seed": int(seed),
        "wind_speed": float(np.linalg.norm(reset_info["wind_mean"])),
        "wind_seed": reset_info["wind_seed"],
        "decisions": len(errors),
        "failed": reason is not None,
        "failure_reason": reason,
        "position_rmse": float(np.sqrt(np.mean(finite**2))),
        "late_position_rmse": float(np.sqrt(np.mean(late**2))),
        "max_position_error": float(np.max(finite)),
        "residual_norm": float(np.mean(residual_norms)),
        "saturation_rate": saturated / max(physics_steps, 1),
        "total_cost": float(total_cost),
    }


def summarize(rows):
    import numpy as np

    def mean(name):
        return float(np.mean([row[name] for row in rows]))

    return {
        "episodes": len(rows),
        "position_rmse": mean("position_rmse"),
        "late_position_rmse": mean("late_position_rmse"),
        "max_position_error": float(
            np.max([row["max_position_error"] for row in rows])
        ),
        "failure_rate": mean("failed"),
        "residual_norm": mean("residual_norm"),
        "saturation_rate": mean("saturation_rate"),
        "total_cost": mean("total_cost"),
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    import numpy as np

    from sac_hover_env import make_sac_env
    from sac_policy import load_sac_policy

    seeds = sac_config.EVAL_SEEDS

    if run["episodes"] is not None:
        seeds = seeds[:run["episodes"]]

    env = make_sac_env(
        simulation_app,
        render_sim=run["gui"],
        wind_stage=run["wind_stage"],
    )

    try:
        policies = {
            "pid_only": lambda observation: np.zeros(4, dtype=np.float32),
        }

        if not run["pid_only"]:
            policies["pid_sac"] = load_sac_policy(model_path)

        results = {
            name: [run_episode(env, seed, policy) for seed in seeds]
            for name, policy in policies.items()
        }

        # Both modes must have flown the same realizations.
        for rows in results.values():
            for row, reference in zip(rows, results["pid_only"]):
                if (row["wind_speed"], row["wind_seed"]) != (
                    reference["wind_speed"], reference["wind_seed"]
                ):
                    raise RuntimeError(f"Wind differs on seed {row['seed']}.")

        summary = {name: summarize(rows) for name, rows in results.items()}

        print(f"\nWind stage: {run['wind_stage']}; seeds: {len(seeds)}")
        print(
            f"{'':10}{'RMSE [m]':>10}{'late [m]':>10}{'max [m]':>10}"
            f"{'fail':>8}{'|a|':>8}{'sat':>8}{'cost':>9}"
        )

        for name, s in summary.items():
            print(
                f"{name:10}{s['position_rmse']:10.3f}"
                f"{s['late_position_rmse']:10.3f}"
                f"{s['max_position_error']:10.3f}"
                f"{s['failure_rate']:8.2f}{s['residual_norm']:8.3f}"
                f"{s['saturation_rate']:8.4f}{s['total_cost']:9.1f}"
            )

        if "pid_sac" in results:
            better = sum(
                sac["position_rmse"] < pid["position_rmse"]
                for sac, pid in zip(results["pid_sac"], results["pid_only"])
            )
            print(f"PID + SAC has lower RMSE on {better}/{len(seeds)} seeds.")

        run_dir.mkdir(parents=True, exist_ok=True)
        output = run_dir / f"eval_{run['wind_stage']}.json"
        output.write_text(
            json.dumps(
                {
                    "wind_stage": run["wind_stage"],
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
