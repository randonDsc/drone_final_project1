"""SECONDARY learning curve on VALIDATION seeds only (M4 item 6; NOT part of the frozen protocol, not frozen itself).

Reuses protocol_stats (frozen) on runs flown by the frozen run_protocol.py fly mode:
  PID reference:  fly --controller pid --seeds 400000-400009 --out residual_ppo/results/eval/eval-v1/pid_validation
  each model:     fly --controller pid_ppo --model <checkpoint or final zip> --seeds 400000-400009
                      --reference-realizations residual_ppo/results/eval/eval-v1/pid_validation/realizations.json
                      --out residual_ppo/results/eval/eval-v1/ppo_validation/<label>

    # Directory: <repo root>
    & "<Isaac Sim>\\python.bat" residual_ppo\\evaluation\\learning_curve.py `
        --run ppo_150000=residual_ppo/results/eval/eval-v1/ppo_validation/ppo_150000 ... `
        --out residual_ppo/results/eval/eval-v1/learning_curve

Descriptive only: no checkpoint selection. The primary PPO result is the final model on the EVAL seeds through the
frozen report mode (E6, docs/milestone3_plan.md). Refuses any run that contains an eval seed. --dry-run allows
subset runs (cells present in every run), labelled DRY RUN.
"""
import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))

import eval_protocol_v1 as protocol  # noqa: E402
import protocol_stats as ps  # noqa: E402
import run_common as rc  # noqa: E402

DEFAULT_PID_RUN = "residual_ppo/results/eval/eval-v1/pid_validation"
FROZEN_DOC = RESIDUAL_DIR / "docs" / "eval_protocol_v1.md"
HEADER = ("SECONDARY learning curve: validation seeds 400000-400009 only; descriptive; no checkpoint selection. "
          "The primary PPO result is the final model on the eval seeds (frozen report mode).")


class Refused(SystemExit):
    pass


def repo_rel(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        return f"<outside repo>/{p.name}"


def frozen_blobs():
    doc = FROZEN_DOC.read_text(encoding="utf-8")
    found = dict(re.findall(r"\| `residual_ppo/evaluation/([a-z_0-9]+\.py)` \| `([0-9a-f]{40})` \|", doc))
    if len(found) != 5:
        raise Refused(f"could not read the 5 frozen blob IDs from {repo_rel(FROZEN_DOC)}")
    return found


def load_run(path):
    d = Path(path)
    d = d if d.is_absolute() else (REPO_DIR / d)
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    real = json.loads((d / "realizations.json").read_text(encoding="utf-8"))
    return d, res, real


def run_problems(name, d, res, frozen):
    """Checks shared by the PID and controller runs: protocol hash, frozen code, validation seeds only."""
    meta = res["meta"]
    problems = []
    if meta["protocol"]["protocol_file_sha256"] != rc.EXPECTED_PROTOCOL_SHA:
        problems.append(f"{name}: protocol hash {meta['protocol']['protocol_file_sha256'][:12]} != frozen")
    blobs = {k.split("/")[-1]: v for k, v in meta.get("code_blobs", {}).items()}
    bad = [f for f, b in frozen.items() if blobs.get(f) != b]
    if bad:
        problems.append(f"{name}: code_blobs differ from the frozen IDs for {bad}")
    seeds = {int(f["seed"]) for f in res["flights"]}
    eval_seeds = sorted(seeds & set(protocol.EVAL_SEEDS))
    if eval_seeds:
        problems.append(f"{name}: contains EVAL seeds {eval_seeds[:5]} (learning curve is validation-only)")
    other = sorted(seeds - set(protocol.VALIDATION_SEEDS) - set(protocol.EVAL_SEEDS))
    if other:
        problems.append(f"{name}: seeds outside the validation set {other[:5]}")
    return problems


def main(argv=None):
    p = argparse.ArgumentParser(description="SECONDARY validation learning curve (not frozen)")
    p.add_argument("--pid-run", default=DEFAULT_PID_RUN)
    p.add_argument("--run", action="append", required=True, help="label=run_dir (one per model)")
    p.add_argument("--out", required=True, help="output path without extension (.json, .txt, .csv)")
    p.add_argument("--dry-run", default=None, help="note; allows subset runs (cells present in every run)")
    args = p.parse_args(argv)

    frozen = frozen_blobs()
    pdir, pres, preal = load_run(args.pid_run)
    problems = run_problems("pid", pdir, pres, frozen)
    if pres["meta"]["controller"]["name"] != "pid":
        problems.append("pid run: controller is not pid")
    pid_real_sha = rc.file_sha256(pdir / "realizations.json")
    scenarios, profiles, seeds = tuple(protocol.SCENARIOS), tuple(protocol.PROFILES), tuple(protocol.VALIDATION_SEEDS)
    if not args.dry_run:
        cons = ps.consistency_check(pres["flights"], scenarios, profiles, seeds, protocol.STILL_AIR_NAME)
        if not cons["pass"] or len(pres["flights"]) != len(scenarios) * len(profiles) * len(seeds):
            problems.append(f"pid run: not the complete validation grid ({len(pres['flights'])} flights; "
                            f"{cons['problems'][:2]})")

    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise Refused(f"--run must be label=run_dir, got {spec!r}")
        label, path = spec.split("=", 1)
        d, res, real = load_run(path)
        meta = res["meta"]
        problems += run_problems(label, d, res, frozen)
        if meta["controller"]["name"] == "pid":
            problems.append(f"{label}: controller is pid (expected a learned controller)")
        ref = meta.get("reference_realizations") or {}
        if ref.get("sha256_lf") != pid_real_sha:
            problems.append(f"{label}: reference_realizations is not {repo_rel(pdir / 'realizations.json')} "
                            f"(sha {str(ref.get('sha256_lf'))[:12]} != {pid_real_sha[:12]})")
        guard = meta.get("realization_guard") or {}
        if not guard.get("all_matched") or guard.get("resets_checked") != len(res["flights"]):
            problems.append(f"{label}: realization guard did not check every reset ({guard})")
        missing = [k for k in real["flights"] if k not in preal["flights"]]
        differ = [k for k, v in real["flights"].items() if k in preal["flights"]
                  and ps.row_differences(preal["flights"][k], v)]
        if missing or differ or real["key"] != preal["key"]:
            problems.append(f"{label}: realizations differ from the PID validation run (missing {len(missing)}, "
                            f"differing {len(differ)})")
        if not args.dry_run:
            cons = ps.consistency_check(res["flights"], scenarios, profiles, seeds, protocol.STILL_AIR_NAME)
            if not cons["pass"]:
                problems.append(f"{label}: not the complete validation grid ({cons['problems'][:2]})")
        runs.append({"label": label, "dir": d, "res": res,
                     "num_timesteps": meta["controller"].get("num_timesteps"),
                     "model": meta["controller"].get("model"), "model_sha256": meta["controller"].get("model_sha256")})
    if problems:
        raise Refused("learning curve refused: " + "; ".join(problems))

    if args.dry_run:  # cells present in every run (and in the PID run)
        keys = set(ps.index_flights(pres["flights"]))
        for r in runs:
            keys &= set(ps.index_flights(r["res"]["flights"]))
        scenarios = tuple(s for s in protocol.SCENARIOS if any(k[0] == s for k in keys))
        profiles = tuple(pf for pf in protocol.PROFILES if any(k[1] == pf for k in keys))
        seeds = tuple(s for s in protocol.VALIDATION_SEEDS if any(k[2] == s for k in keys))
        if len(keys) != len(scenarios) * len(profiles) * len(seeds) or not keys:
            raise Refused(f"dry run: the common cells do not form a grid ({len(keys)} cells)")

    runs.sort(key=lambda r: (r["num_timesteps"] is None, r["num_timesteps"] or 0))
    lines, table, out_rows = [HEADER], [], []
    if args.dry_run:
        lines.append(f"DRY RUN (subset: {len(scenarios)} scenarios x {len(profiles)} profiles x {len(seeds)} seeds): "
                     f"{args.dry_run}")
    lines.append(f"PID reference: {repo_rel(pdir)} (realizations sha {pid_real_sha[:12]})")
    for r in runs:
        rows = ps.controller_report(r["res"]["flights"], pres["flights"], profiles, scenarios, seeds,
                                    protocol.PRIMARY_PROFILES, protocol.SECONDARY_PROFILES,
                                    protocol.REPORTED_NOT_PRIMARY)
        lines.append("")
        lines.append(f"{r['label']}: {r['model']} (num_timesteps {r['num_timesteps']})")
        for row in rows:
            c, pd_, d = row["controller"], row["pid"], row["paired_rmse_difference"]
            fc = row["paired_failure_comparison"]
            flat = {"label": r["label"], "num_timesteps": r["num_timesteps"], "profile": row["profile"],
                    "status": row["status"], "flights": row["flights"],
                    "rmse_controller": c["position_rmse_mean_over_scenarios"],
                    "rmse_pid": pd_["position_rmse_mean_over_scenarios"], "paired_rmse_diff": d["value"],
                    "paired_rmse_diff_ci_lo": d["ci95"][0], "paired_rmse_diff_ci_hi": d["ci95"][1],
                    "failed_controller": c["failures"]["failed"], "failed_pid": pd_["failures"]["failed"],
                    "mcnemar_p": fc["mcnemar"]["p_value"], "decisions_median_controller": c["decisions"]["median"],
                    "rmse_requires_failure_comparison": row["rmse_requires_failure_comparison"]}
            out_rows.append(flat)
            flag = " [RMSE only with the failure comparison]" if row["rmse_requires_failure_comparison"] else ""
            lines.append(f"  {row['profile']:<15} RMSE {flat['rmse_controller']:.3f} vs pid {flat['rmse_pid']:.3f} m; "
                         f"paired diff {flat['paired_rmse_diff']:+.4f} [{flat['paired_rmse_diff_ci_lo']:+.4f}, "
                         f"{flat['paired_rmse_diff_ci_hi']:+.4f}]; failed {flat['failed_controller']}/{row['flights']} "
                         f"vs pid {flat['failed_pid']}/{row['flights']} (McNemar p {flat['mcnemar_p']:.3g}){flag}")
        table.append({"label": r["label"], "num_timesteps": r["num_timesteps"], "model": r["model"],
                      "model_sha256": r["model_sha256"], "run": repo_rel(r["dir"] / "results.json"),
                      "run_sha256_lf": rc.file_sha256(r["dir"] / "results.json"), "profiles": rows})

    out = Path(args.out)
    out = out if out.is_absolute() else (REPO_DIR / out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "purpose": HEADER,
              "dry_run": args.dry_run, "protocol_file_sha256": rc.EXPECTED_PROTOCOL_SHA, "frozen_blobs": frozen,
              "pid_run": repo_rel(pdir / "results.json"), "pid_run_sha256_lf": rc.file_sha256(pdir / "results.json"),
              "pid_realizations_sha256_lf": pid_real_sha, "seeds": list(seeds), "profiles": list(profiles),
              "scenarios": list(scenarios), "models": table}
    out.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("\n".join(lines))
    print(f"written: {repo_rel(out.with_suffix('.json'))}, .txt, .csv")
    print("RESULT: DONE")


if __name__ == "__main__":
    main()
