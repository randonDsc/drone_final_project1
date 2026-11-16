"""Audit a PPO run folder (M5 S3): files, contracts, counters and CSV counts. No Isaac, no unpickling.

    # Directory: <repo root>
    & "<Isaac Sim>\\python.bat" residual_ppo\\tools\\audit_ppo_run.py --run-id smoke_ppo_s0_1
    & "<Isaac Sim>\\python.bat" residual_ppo\\tools\\audit_ppo_run.py --arm direct --run-id smoke_direct_s0_1

Prints one line per check (PASS / FAIL) and `RESULT: PASS` or `RESULT: FAIL`; writes nothing.
--arm direct (M8): the same checks on runs/ppo_direct/<id>/, except that the pid_ppo adapter must REFUSE every
direct model (its contract carries the direct action mapping), instead of accepting it.
"""
import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))
sys.path.insert(2, str(RESIDUAL_DIR / "training"))

import controllers  # noqa: E402
import ppo_setup  # noqa: E402
import run_files as rf  # noqa: E402

EXPECTED_MASS = 1.618614


def zip_data(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("data").decode("utf-8"))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--expect-state", default="done")
    p.add_argument("--arm", choices=ppo_setup.ARMS, default="residual")
    args = p.parse_args(argv)
    ec, ppo_config = ppo_setup.arm_configs(args.arm)
    run = ec.run_dir(args.run_id)
    results = []

    def check(name, ok, detail=""):
        results.append(bool(ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

    meta = json.loads((run / ec.RUN_META_FILE).read_text(encoding="utf-8"))
    status = json.loads((run / ec.STATUS_FILE).read_text(encoding="utf-8"))
    settings = meta["settings"]
    n_steps = ppo_config.PPO["n_steps"]
    full = ppo_config.minibatches_per_rollout()

    check("status", status["state"] == args.expect_state,
          f"state {status['state']}, result {status['result']}, model {status['model']}")
    final = run / "final_model.zip"
    end_model = final if args.expect_state == "done" else run / Path(status["model"]).name
    check("end model + state file", end_model.is_file() and rf.state_path(end_model).is_file(), end_model.name)
    end_state = json.loads(rf.state_path(end_model).read_text(encoding="utf-8"))
    c = end_state["counters"]
    check("end counters: collected = trained_on + discarded + pending",
          c["collected"] == c["trained_on"] + c["discarded"] + end_state["pending"],
          f"{c['collected']:,} = {c['trained_on']:,} + {c['discarded']:,} + {end_state['pending']:,}")
    if args.expect_state == "done":
        check("final: pending 0 and num_timesteps >= target", end_state["pending"] == 0
              and end_state["num_timesteps"] >= settings["total_decisions"],
              f"num_timesteps {end_state['num_timesteps']:,} (target {settings['total_decisions']:,})")
    data = zip_data(end_model)
    check("zip num_timesteps / _n_updates == state", data["num_timesteps"] == end_state["num_timesteps"]
          and data["_n_updates"] == c["sb3_n_updates"],
          f"zip {data['num_timesteps']:,} / {data['_n_updates']}; state {end_state['num_timesteps']:,} / "
          f"{c['sb3_n_updates']}")

    # checkpoints
    ckpts = sorted((run / ec.CHECKPOINT_DIR).glob("ppo_*_steps.zip"), key=lambda q: int(q.stem.split("_")[1]))
    steps = [int(q.stem.split("_")[1]) for q in ckpts]
    interval = settings["checkpoint_interval"]
    expected = list(range(interval, end_state["num_timesteps"] + 1, interval))
    check("checkpoints at every interval", steps == expected, f"found {steps}, expected {expected}")
    launches = meta["launches"]
    starts = [L["start_num_timesteps"] for L in launches[1:]]
    first_resume = min(starts) if starts else None
    for q, n in zip(ckpts, steps):
        st = json.loads(rf.state_path(q).read_text(encoding="utf-8"))
        cc = st["counters"]
        ok = (zip_data(q)["num_timesteps"] == n == st["num_timesteps"] == cc["collected"]
              and cc["collected"] == cc["trained_on"] + cc["discarded"] + st["pending"])
        formula = ec.decisions_trained_on(n)
        before_resume = first_resume is None or n <= first_resume
        if before_resume:   # before any resume: the M3 formula holds
            ok = ok and cc["trained_on"] == formula and cc["discarded"] == 0
        note = f"== formula {formula:,}" if before_resume else "after a resume: formula n/a"
        check(f"checkpoint {n:,}", ok, f"trained_on {cc['trained_on']:,} ({note}), "
              f"discarded {cc['discarded']:,}, pending {st['pending']:,}, grad steps {cc['gradient_steps']:,}, "
              f"stage {st['stage']}")

    # contracts
    expected_contract = json.loads(json.dumps(ec.run_contract(settings)))
    for folder in (run, run / ec.CHECKPOINT_DIR):
        stored = json.loads((folder / ec.CONTRACT_FILE).read_text(encoding="utf-8"))
        check(f"contract {folder.name}/ == run_contract(settings)", stored == expected_contract)
    models = [*run.glob("*_model.zip"), *ckpts]
    refused, accepted = [], []
    for m in models:
        try:
            controllers.check_ppo_contract(m)
            accepted.append(m.name)
        except controllers.ControllerRefused as exc:
            refused.append(f"{m.name}: {exc}")
    if args.arm == "residual":
        check(f"adapter check_ppo_contract accepts all {len(models)} models", not refused, "; ".join(refused))
    else:   # a direct model must never be flown as a residual (pid_ppo) controller
        check(f"pid_ppo adapter check_ppo_contract REFUSES all {len(models)} direct models", not accepted and models,
              "accepted: " + ", ".join(accepted) if accepted else f"{len(refused)} refused")
        check("run_meta arm == direct", meta.get("arm") == "direct", f"arm {meta.get('arm')}")

    # per-launch CSVs and lineage sums
    log = run / ec.LOG_DIR
    csv_rows = []
    for L in launches:
        path = log / f"rollouts_{L['stamp']}.csv"
        csv_rows.append(read_csv(path) if path.is_file() else None)
    takes, unexplained = rf.lineage_row_counts(
        [{"start_rollouts": L["counters_at_start"]["rollouts"], "rows": None if r is None else len(r)}
         for L, r in zip(launches, csv_rows)], c["rollouts"])
    check("lineage: every update of the end model is found in the rollouts CSVs", unexplained == 0,
          f"rows in lineage per launch {takes}, unexplained updates {unexplained}")
    sum_gs = sum_ep = sum_early = 0
    for i, (L, rows) in enumerate(zip(launches, csv_rows)):
        stamp = L["stamp"]
        if rows is None:
            # a launch that failed (or was killed) before learn() trains nothing and writes no CSVs
            check(f"launch {L['launch']} ({L['mode']}) ended before training: recorded, no CSVs",
                  L.get("end_state") in ("failed", "killed") and L.get("end_reason"),
                  f"{L.get('end_state')}: {L.get('end_reason', '')[:140]}")
            continue
        prog = read_csv(log / f"progress_{stamp}.csv") if (log / f"progress_{stamp}.csv").is_file() else []
        mon = log / f"monitor_{stamp}.monitor.csv"
        lineage = rows[:takes[i]]
        gs = [int(r["gradient_steps"]) for r in rows]
        sum_gs += sum(int(r["gradient_steps"]) for r in lineage)
        sum_ep += sum(int(r["epochs"]) for r in lineage)
        sum_early += sum(r["early_stop"] == "True" for r in lineage)
        mass = L.get("total_mass_kg")
        started_ok = (L.get("versions") and mass is not None and abs(mass - EXPECTED_MASS) < 1e-6
                      and L["hesai"]["hesai_mass_contribution_kg"] == 0
                      and L["torch_origins"]["outside_package_dir"] == 0)
        if L.get("end_state") == "killed":
            # killed launch (Stop-Process, Ctrl+C, closed window, power loss): no end record; recoverable only
            # through a checkpoint at or below its last recorded num_timesteps
            following = next((M for M, r in zip(launches[i + 1:], csv_rows[i + 1:]) if r is not None), None)
            ok, why = rf.killed_launch_resume_ok(L, following)
            check(f"launch {L['launch']} ({L['mode']}) KILLED: resumed from a checkpoint it had reached",
                  ok and started_ok and mon.is_file() and all(g <= full for g in gs),
                  f"{why}; last heartbeat {L.get('last_heartbeat')}; rollouts.csv {len(rows)} rows of which "
                  f"{takes[i]} in the lineage ({len(rows) - takes[i]} after the resume point, excluded); "
                  f"mass {mass}")
            continue
        check(f"launch {L['launch']} ({L['mode']}) CSVs", len(prog) == len(rows) and mon.is_file()
              and all(g <= full for g in gs),
              f"rollouts.csv {len(rows)} rows ({takes[i]} in the lineage), progress.csv {len(prog)} rows, monitor "
              f"{mon.is_file()}, grad steps per rollout {sorted(set(gs))} (<= {full})")
        check(f"launch {L['launch']} meta", started_ok and L.get("end_state") is not None
              and L.get("torch_origin_clean_at_end") is True,
              f"end {L.get('end_state')} ({L.get('end_reason')}), torch {L.get('versions', {}).get('torch')}, "
              f"mass {mass}, reseed {L['reseed']}, seeded resets {L.get('seeded_resets')}")
    check("lineage sums == end counters", sum_gs == c["gradient_steps"] and sum_ep == c["sb3_n_updates"]
          and sum_early == c["early_stops"],
          f"gradient steps {sum_gs:,} / {c['gradient_steps']:,}; epochs {sum_ep} / {c['sb3_n_updates']}; early "
          f"stops {sum_early} / {c['early_stops']}")
    check("trained_on == n_steps x lineage updates", c["trained_on"] == n_steps * c["rollouts"],
          f"{c['trained_on']:,} = {n_steps} x {c['rollouts']}")
    leftovers = [q.name for q in run.rglob("*_partial*")]
    check("no _partial files left", not leftovers, ", ".join(leftovers))
    summary = (run / ec.SUMMARY_FILE).read_text(encoding="utf-8") if (run / ec.SUMMARY_FILE).is_file() else ""
    check("training_summary.txt", f"(state: {args.expect_state};" in summary, summary.splitlines()[0] if summary
          else "missing")
    ok = all(results)
    print(f"{sum(results)}/{len(results)} checks passed")
    print("RESULT: PASS" if ok else "RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
