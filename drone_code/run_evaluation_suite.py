"""Run a fixed wind benchmark against named path-SAC checkpoints.

--model NAME=PATH may be repeated; first model is the comparison baseline.
--resume OUTPUT resumes completed batches and preserves interrupted attempts.
"""
import argparse
import contextlib
import csv
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "drone_code/evaluation_suites"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suite_config(path):
    suite = read(path)
    from wind_events import SCENARIOS
    import path_sac_config as cfg
    for key, allowed in (("routes", {"hover", "straight", "waypoints"}), ("scenarios", set(SCENARIOS))):
        values = suite[key]
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError(f"Invalid or duplicate {key}")
    seeds = suite["seeds"]
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique nonnegative integers")
    if not 7 <= suite["duration_s"] <= cfg.EPISODE["max_seconds"]:
        raise ValueError("Duration must allow hover event at 5 s plus 2 s follow-up")
    return suite


def source_hashes():
    # Include imported simulator/controller modules, not just the evaluator.
    return {p.name: digest(p) for p in sorted((ROOT / "drone_code").glob("*.py"))}


def model_entries(specifications):
    import path_sac_config as cfg
    from model_contract import run_folder
    entries = []
    for spec in specifications:
        name, separator, filename = spec.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name.lower() == "pid":
            raise ValueError("Use --model NAME=PATH; names contain letters, digits, _ or -, and cannot be PID")
        if name.lower() in {m["name"].lower() for m in entries}:
            raise ValueError("Model names must be unique")
        path = Path(filename)
        path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise ValueError(f"Missing checkpoint: {path}")
        problem = cfg.incompatibility(path)
        if problem:
            raise ValueError(problem)
        contract = run_folder(path) / cfg.CONTRACT_FILE
        entries.append(dict(name=name, source=str(path), sha256=digest(path), contract=str(contract), contract_sha256=digest(contract)))
    return entries


def check_batch(folder, plan, model, route):
    from summarize_wind_events import analyze
    manifest = read(folder / "manifest.json")
    suite = plan["suite"]
    expected = dict(route=route, seeds=suite["seeds"], scenarios=suite["scenarios"], duration_s=suite["duration_s"], model_sha256=model["sha256"])
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Batch {folder.name} differs from plan: {key}")
    snapshots = list((folder / "source").glob("*.py"))
    if not snapshots or any(plan["sources"].get(p.name) != digest(p) for p in snapshots):
        raise ValueError("Evaluation source differs from saved pipeline plan")
    with contextlib.redirect_stdout(io.StringIO()):
        rows = analyze(folder, plot=True)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output, plan):
    import numpy as np
    summaries, pairs, comparisons = [], [], []
    models = plan["models"]
    rng = np.random.default_rng(73001)
    for route in plan["suite"]["routes"]:
        baseline = output / "batches" / f'{route}__{models[0]["name"]}'
        base_flights = {(f["scenario"], f["seed"], f["controller"]): f for f in read(baseline / "flights.json")}
        for model in models:
            folder = output / "batches" / f'{route}__{model["name"]}'
            for row in check_batch(folder, plan, model, route):
                if row["controller"] == "pid_only" and model != models[0]:
                    continue
                summaries.append(dict(route=route, model="PID" if row["controller"] == "pid_only" else model["name"], **row))
            for seed in plan["suite"]["seeds"]:
                if read(folder / f"reference_{seed}.json") != read(baseline / f"reference_{seed}.json"):
                    raise ValueError("Reference mismatch across models")
            flights = {(f["scenario"], f["seed"], f["controller"]): f for f in read(folder / "flights.json")}
            for scenario in plan["suite"]["scenarios"]:
                deltas = []
                for seed in plan["suite"]["seeds"]:
                    pid_csv = f"{scenario}_{seed}_pid_only.csv"
                    if digest(folder / pid_csv) != digest(baseline / pid_csv):
                        raise ValueError("Repeated PID traces differ; comparison rejected")
                    old, new = [records[(scenario, seed, "pid_sac")] for records in (base_flights, flights)]
                    for key in ("wind_seed", "heading_rad", "event_s"):
                        if old[key] != new[key]:
                            raise ValueError(f"Unmatched {key}")
                    delta = None
                    if not old["failed"] and not new["failed"] and old["position_rmse_m"] is not None and new["position_rmse_m"] is not None:
                        delta = new["position_rmse_m"] - old["position_rmse_m"]
                        deltas.append(delta)
                    if model != models[0]:
                        pairs.append(dict(route=route, scenario=scenario, seed=seed, baseline=models[0]["name"], model=model["name"], baseline_rmse_m=old["position_rmse_m"], model_rmse_m=new["position_rmse_m"], model_minus_baseline_rmse_m=delta, baseline_failed=old["failed"], model_failed=new["failed"]))
                if model != models[0]:
                    interval = np.quantile(rng.choice(deltas, size=(50000, len(deltas))).mean(axis=1), [.025, .975]).tolist() if len(deltas) >= 2 else [None, None]
                    comparisons.append(dict(route=route, scenario=scenario, baseline=models[0]["name"], model=model["name"], successful_pairs=len(deltas), model_lower_count=sum(d < 0 for d in deltas), mean_delta_m=float(np.mean(deltas)) if deltas else None, ci95_low_m=interval[0], ci95_high_m=interval[1]))
    write_csv(output / "metrics.csv", summaries)
    write_csv(output / "paired_models.csv", pairs)
    write_csv(output / "comparisons.csv", comparisons)
    write(output / "report.json", dict(completed=True, models=models, suite=plan["suite"], metrics=summaries, comparisons=comparisons, limitations="Per-condition seed bootstrap intervals are descriptive, not adjusted for multiple comparisons. Failed pairs excluded from deltas and reported separately. Fixed-duration segments, not full-route completion. One checkpoint per model; environment seeds do not measure training variability."))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    routes, scenarios = plan["suite"]["routes"], plan["suite"]["scenarios"]
    fig, axes = plt.subplots(len(routes), 1, figsize=(10, 4 * len(routes)), squeeze=False)
    labels = ["PID"] + [m["name"] for m in models]
    for route, ax in zip(routes, axes[:, 0]):
        width = .8 / len(labels)
        for index, label in enumerate(labels):
            rows = {r["scenario"]: r for r in summaries if r["route"] == route and r["model"] == label}
            values = [rows[s]["mean_finite_episode_rmse_m"] for s in scenarios]
            ax.bar(np.arange(len(scenarios)) + (index - (len(labels)-1)/2)*width, [v if v is not None else np.nan for v in values], width, label=label)
        ax.set_xticks(np.arange(len(scenarios)), scenarios)
        ax.set(title=route, ylabel="Mean finite flight RMSE (m)")
        ax.legend()
    fig.suptitle("Wind benchmark — inspect failure counts alongside tracking error")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--isaac-python", default=r"C:\isaacsim\python.bat")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.resume:
        if args.model or args.output or args.suite:
            parser.error("--resume uses the saved model list and output folder")
        output = args.resume.resolve()
        plan = read(output / "pipeline.json")
        if plan["sources"] != source_hashes():
            raise ValueError("Code changed since launch; use a new output to avoid mixing implementations")
    else:
        if args.report_only or not args.model:
            parser.error("Provide --model NAME=PATH, or --resume OUTPUT")
        output = (args.output or ROOT / "runs/benchmarks" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
        plan = dict(version=1, suite=suite_config(args.suite or SUITES / "wind_standard.json"), models=model_entries(args.model), sources=source_hashes())
    flights = len(plan["models"]) * len(plan["suite"]["routes"]) * len(plan["suite"]["scenarios"]) * len(plan["suite"]["seeds"]) * 2
    print(f"Planned {flights} flights (includes repeated PID checks). Output: {output}", flush=True)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
        for model in plan["models"]:
            folder = output / "models" / model["name"]
            folder.mkdir(parents=True)
            shutil.copy2(model["source"], folder / "final_model.zip")
            shutil.copy2(model["contract"], folder / "flight_contract.json")
        write(output / "pipeline.json", plan)
    for model in plan["models"]:
        folder = output / "models" / model["name"]
        if digest(folder / "final_model.zip") != model["sha256"] or digest(folder / "flight_contract.json") != model["contract_sha256"]:
            raise ValueError("Frozen checkpoint or contract changed")
    if not args.report_only:
        for route in plan["suite"]["routes"]:
            for model in plan["models"]:
                name = f'{route}__{model["name"]}'
                folder = output / "batches" / name
                manifest = folder / "manifest.json"
                if manifest.is_file() and read(manifest).get("completed"):
                    check_batch(folder, plan, model, route)
                    print(f"Validated existing {name}; skipping simulation", flush=True)
                    continue
                if folder.exists():
                    folder.rename(folder.with_name(name + "_interrupted_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
                folder.parent.mkdir(exist_ok=True)
                suite = plan["suite"]
                command = [args.isaac_python, "-u", "-B", str(ROOT / "drone_code/eval_wind_events.py"), "--model", str(output / "models" / model["name"] / "final_model.zip"), "--route", route, "--duration", str(suite["duration_s"]), "--seeds", *map(str, suite["seeds"]), "--scenarios", *suite["scenarios"], "--output", str(folder)]
                log = output / f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.log'
                write(output / "status.json", dict(state="running", batch=name, log=str(log)))
                print(f"Running {name}; log: {log}", flush=True)
                try:
                    with log.open("w", encoding="utf-8") as handle:
                        subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True)
                except (subprocess.CalledProcessError, OSError, KeyboardInterrupt) as error:
                    write(output / "status.json", dict(state="interrupted_or_failed", batch=name, log=str(log), error=str(error)))
                    raise
                check_batch(folder, plan, model, route)
                print(f"Completed and validated {name}", flush=True)
    report(output, plan)
    write(output / "status.json", dict(state="completed", report=str(output / "report.json")))
    print(f"Benchmark complete: {output / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
