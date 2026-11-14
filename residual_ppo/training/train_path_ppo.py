"""Train the residual PPO on top of PID (M4; design approved by Claire 2026-11-14, docs/milestone4_plan.md).

Run with Isaac's python.bat, in its own PowerShell window for long runs (README / plan section 7):

    # Directory: <repo root>
    & "<Isaac Sim>\\python.bat" residual_ppo\\training\\train_path_ppo.py --run-id smoke_ppo_s0_1 --mode fresh
    & "<Isaac Sim>\\python.bat" residual_ppo\\training\\train_path_ppo.py --run-id smoke_ppo_s0_1 --mode resume `
        --resume-model residual_ppo/runs/ppo_path/smoke_ppo_s0_1/interrupted_model.zip

Run ids (D1): ppo_s0_1p5m (main) or smoke_ppo_s0_<n> (smoke override: 10,240 decisions, checkpoints every 4,096).
Stop a run ONLY by creating runs/ppo_path/<id>/STOP (D5): it saves interrupted_model.zip and ends cleanly.
Ctrl+C or closing the window KILLS the process at once and saves nothing (M5 S1b); so does a power loss. Recovery
is then: resume from the last checkpoint (M5b S7). The next launch records the dead launch as "killed".
A run folder whose status.json says starting/running with a live pid is refused (live-run lock).
Success = the console log ends with RESULT: DONE, status.json says "done" and final_model.zip exists. Never judge a run
by its exit code (Isaac exits 0 even after an exception).
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RESIDUAL_DIR.parent
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))
sys.path.insert(2, str(RESIDUAL_DIR / "training"))

import numpy as np  # noqa: E402

import ppo_setup  # noqa: E402  (gymnasium only at import; torch is imported by import_torch_first())
import run_common as rc  # noqa: E402
import run_files as rf  # noqa: E402

ROLLOUT_COLUMNS = ["launch", "rollout", "num_timesteps", "stage", "collect_s", "update_s", "gradient_steps", "epochs",
                   "early_stop", "approx_kl", "clip_fraction", "value_loss", "policy_gradient_loss", "entropy_loss",
                   "explained_variance", "loss", "std_mean", "std_0", "std_1", "std_2", "std_3",
                   "action_clip_component", "action_clip_action", "episodes", "failures"]


def say(msg=""):
    print(f"[M4.train] {msg}", flush=True)


def refuse(msg):
    say(f"REFUSED: {msg}")
    say("RESULT: REFUSED")
    raise SystemExit(2)


def rel(path):
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_DIR.resolve()).as_posix()
    except ValueError:
        return f"<outside repo>/{p.name}"


def repo_path(path):
    p = Path(path)
    return p if p.is_absolute() else (REPO_DIR / p)


class NonFiniteError(RuntimeError):
    """NaN guard (4): a parameter or a train/* value is not finite after an update."""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run-id", required=True)
    p.add_argument("--mode", choices=("fresh", "resume"), required=True)
    p.add_argument("--resume-model", default=None, help="model zip inside the same run folder (D2)")
    p.add_argument("--curriculum-offset", type=int, default=0, help="smoke only (D8)")
    p.add_argument("--inject-nan-at", type=int, default=None, help="smoke only (D9)")
    p.add_argument("--preflight-only", action="store_true", help="run every check before Isaac, then stop")
    return p.parse_args(argv)


# ==================== before SimulationApp: every refusal happens here ====================

def pre_isaac(args):
    check_line = rc.preflight(say)
    from config import experiment_config as ec
    from config import ppo_config

    import controllers
    import path_sac_config

    if path_sac_config.sweep_variant() is not None:
        refuse(f"sweep variant {path_sac_config.sweep_variant()!r} is set ({path_sac_config.SWEEP_VARIANT_VARIABLE})")
    try:
        settings = ec.run_settings(args.run_id, args.curriculum_offset)
    except ValueError as exc:
        refuse(str(exc))
    if args.inject_nan_at is not None and not settings["smoke"]:
        refuse("--inject-nan-at is smoke-only (D9)")
    run_dir = ec.run_dir(args.run_id)
    ckpt_dir = run_dir / ec.CHECKPOINT_DIR
    contract = json.loads(json.dumps(ec.run_contract(settings)))
    forbidden = ec.forbidden_reset_seeds()

    # D12: the main run needs committed code under residual_ppo/ (runs/ and results/ ignored).
    blocking = rf.blocking_git_entries(rf.git_porcelain(REPO_DIR))
    if not settings["smoke"]:
        if blocking:
            refuse("main run needs committed code under residual_ppo/ (D12); blocking: "
                   + "; ".join(f"{b['status']}: {b['path']}" for b in blocking))
        say("git check (D12): no modified tracked files and no untracked .py under residual_ppo/ "
            "(runs/ and results/ ignored)")
    else:
        say(f"git check (D12) skipped for smoke runs ({len(blocking)} entries would block the main run)")

    # Live-run lock (M5b): never start on a folder whose run is still alive (fresh and resume, smoke and main).
    status_path = run_dir / ec.STATUS_FILE
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else None
    live = rf.live_run_reason(status)
    if live:
        refuse(live)
    if status and status.get("state") in rf.LIVE_STATES:
        say(f"status.json says {status['state']!r} but pid {status.get('pid')} is not alive (last heartbeat "
            f"{status.get('updated_at')} at {status.get('num_timesteps')} decisions): treated as a killed launch")

    ctx = {"args": args, "settings": settings, "run_dir": run_dir, "ckpt_dir": ckpt_dir, "contract": contract,
           "forbidden": forbidden, "check_line": check_line, "ec": ec, "ppo_config": ppo_config,
           "git_blocking": blocking}
    if args.mode == "fresh":
        if args.resume_model:
            refuse("--resume-model is only for --mode resume")
        existing = sorted([*run_dir.glob("*_model.zip"), *ckpt_dir.glob("ppo_*_steps.zip")])
        if existing:
            refuse(f"{rel(run_dir)} already holds a model ({', '.join(p.name for p in existing[:3])}); use a new run "
                   "id or --mode resume")
        ctx.update(source=None, state=None, counters=rf.new_counters(), stage_stats={}, reseed=None,
                   start_timesteps=0)
    else:
        if not args.resume_model:
            refuse("--mode resume needs --resume-model")
        src = repo_path(args.resume_model).resolve()
        if run_dir.resolve() not in src.parents:
            refuse(f"--resume-model must be inside {rel(run_dir)} (D2); got {rel(src)}")
        if not src.is_file():
            refuse(f"model not found: {rel(src)}")
        sp = rf.state_path(src)
        if not sp.is_file():
            refuse(f"no state file {sp.name} next to the model")
        state = json.loads(sp.read_text(encoding="utf-8"))
        if state.get("nonfinite"):
            refuse(f"{rel(src)} has non-finite parameters (NaN guard); resume from the last good checkpoint "
                   f"({state.get('last_checkpoint')})")
        info = controllers.saved_model_info(src)
        if info["num_timesteps"] != state["counters"]["collected"]:
            refuse(f"num_timesteps in the zip ({info['num_timesteps']}) != state file "
                   f"({state['counters']['collected']})")
        for folder in {src.parent, run_dir}:
            cpath = folder / ec.CONTRACT_FILE
            if not cpath.is_file():
                refuse(f"no {ec.CONTRACT_FILE} in {rel(folder)}")
            stored = json.loads(cpath.read_text(encoding="utf-8"))
            if stored != contract:
                diff = sorted(f"{g}.{k}" for g in set(contract) | set(stored)
                              for k in set(contract.get(g, {})) | set(stored.get(g, {}))
                              if contract.get(g, {}).get(k) != stored.get(g, {}).get(k))
                refuse(f"{rel(cpath)} differs from the current contract: {diff}")
        n = int(state["counters"]["collected"])
        reseed = ppo_config.PPO["seed"] + n + 1
        if reseed in forbidden:
            refuse(f"resume reseed {reseed} (= seed + {n} + 1) is an eval/validation seed (D3); resume from the "
                   "previous checkpoint instead")
        if n >= settings["total_decisions"]:
            say(f"{rel(src)} is already at {n:,} >= {settings['total_decisions']:,} decisions; nothing to do")
            say("RESULT: DONE (nothing to do)")
            raise SystemExit(0)
        ctx.update(source=src, state=state, counters=rf.counters_after_resume(state["counters"]),
                   stage_stats=state["stage_stats"], reseed=reseed, start_timesteps=n)
        if not (run_dir / ec.RUN_META_FILE).is_file():
            refuse(f"no {ec.RUN_META_FILE} in {rel(run_dir)}")
        for leftover in rf.leftover_partials(run_dir):   # a kill during a save leaves <name>_partial files
            if args.preflight_only:
                say(f"leftover partial file (would be removed by a real resume): {rel(leftover)}")
            else:
                leftover.unlink()
                say(f"removed leftover partial file: {rel(leftover)}")
    say(f"preflight OK: run {args.run_id} ({'smoke' if settings['smoke'] else 'MAIN'}), mode {args.mode}, target "
        f"{settings['total_decisions']:,} decisions, checkpoints every {settings['checkpoint_interval']:,}"
        + (f", resume from {rel(ctx['source'])} at {ctx['start_timesteps']:,} (reseed {ctx['reseed']})"
           if args.mode == "resume" else ""))
    return ctx


# ==================== run-folder writers ====================

class RunFiles:
    def __init__(self, ctx):
        self.ctx = ctx
        ec = ctx["ec"]
        self.ec = ec
        self.run_dir, self.ckpt_dir = ctx["run_dir"], ctx["ckpt_dir"]
        self.log_dir = self.run_dir / ec.LOG_DIR
        self.status_path = self.run_dir / ec.STATUS_FILE
        self.meta_path = self.run_dir / ec.RUN_META_FILE
        self.summary_path = self.run_dir / ec.SUMMARY_FILE
        self.stop_path = self.run_dir / ec.STOP_FILE
        self.stamp = time.strftime("%Y%m%d_%H%M%S")
        st = ctx["state"] or {}
        self.last_checkpoint = st.get("model") if st.get("kind") == "checkpoint" else st.get("last_checkpoint")
        self.t0 = time.perf_counter()
        self.pid_start = rf.process_start_time(os.getpid())

    def start_launch(self):
        ctx, ec = self.ctx, self.ec
        for d in (self.run_dir, self.ckpt_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)
        if ctx["args"].mode == "fresh":
            for folder in (self.run_dir, self.ckpt_dir):
                rf.atomic_write_json(folder / ec.CONTRACT_FILE, ctx["contract"])
        meta = (json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.is_file() else
                {"run_id": ctx["args"].run_id, "created_at": rf.now_iso(), "settings": ctx["settings"],
                 "ppo": ctx["ppo_config"].PPO, "policy_kwargs": ctx["ppo_config"].policy_kwargs_record(),
                 "launches": []})
        status = (json.loads(self.status_path.read_text(encoding="utf-8")) if self.status_path.is_file()
                  else None)
        for number in rf.mark_killed_launches(meta, status):   # the live-run lock already passed: it is dead
            killed = meta["launches"][number - 1]
            say(f"launch {number} has no end record: recorded as KILLED (last heartbeat "
                f"{killed['last_heartbeat']} at {killed['last_num_timesteps']} decisions)")
        self.launch = len(meta["launches"]) + 1
        a = ctx["args"]
        meta["launches"].append({
            "launch": self.launch, "mode": a.mode, "stamp": self.stamp, "started_at": rf.now_iso(),
            "argv": sys.argv[1:], "source_model": None if ctx["source"] is None else rel(ctx["source"]),
            "start_num_timesteps": ctx["start_timesteps"], "reseed": ctx["reseed"],
            "counters_at_start": dict(ctx["counters"]), "curriculum_offset": ctx["settings"]["curriculum_offset"],
            "inject_nan_at": a.inject_nan_at, "checks": ctx["check_line"], "git": rc.git_info(),
            "git_blocking_entries": ctx["git_blocking"], "isaac_sim": rc.isaac_version(), "hardware": rc.hardware(),
            "pid": os.getpid(), "end_state": None})
        self.meta = meta
        rf.atomic_write_json(self.meta_path, meta)
        self.write_status("starting", ctx["counters"], ctx["start_timesteps"])

    def update_launch(self, **fields):
        self.meta["launches"][-1].update(fields)
        rf.atomic_write_json(self.meta_path, self.meta)

    def write_status(self, state, counters, num_timesteps, model=None, error=None, result=None):
        rf.atomic_write_json(self.status_path, {
            "run_id": self.ctx["args"].run_id, "state": state, "launch": getattr(self, "launch", None),
            "pid": os.getpid(), "pid_start": self.pid_start, "num_timesteps": int(num_timesteps),
            "counters": dict(counters),
            "pending": rf.pending(counters), "last_checkpoint": self.last_checkpoint, "model": model,
            "updated_at": rf.now_iso(), "launch_wall_clock_s": time.perf_counter() - self.t0,
            "error": error, "result": result})

    def write_summary(self, state, counters, stage_stats):
        import path_sac_config

        order = [s["name"] for s in path_sac_config.CURRICULUM]
        extra = [f"launch {getattr(self, 'launch', '?')}; last checkpoint {self.last_checkpoint}; "
                 f"target {self.ctx['settings']['total_decisions']:,} decisions"]
        rf.atomic_write_text(self.summary_path, rf.summary_text(self.ctx["args"].run_id, state, counters,
                                                                stage_stats, order, extra))

    def save_model(self, model, path, kind, counters, stage_stats, nonfinite=False, reason=None):
        import path_sac_config

        counters["collected"] = int(model.num_timesteps)
        rf.atomic_save_model(model, path)
        n = int(model.num_timesteps)
        rf.atomic_write_json(rf.state_path(path), {
            "kind": kind, "model": rel(path), "num_timesteps": n, "counters": dict(counters),
            "pending": rf.pending(counters), "decisions_trained_on": counters["trained_on"],
            "stage": path_sac_config.stage_for_step(n + self.ctx["settings"]["curriculum_offset"])["name"],
            "stage_stats": stage_stats, "launch": self.launch, "saved_at": rf.now_iso(), "nonfinite": nonfinite,
            "reason": reason, "last_checkpoint": self.last_checkpoint})
        return n


# ==================== training ====================

def make_callback(ctx, files, env, grad, say_fn):
    from stable_baselines3.common.callbacks import BaseCallback

    import torch

    ec, ppo_config = ctx["ec"], ctx["ppo_config"]
    settings = ctx["settings"]
    interval, offset = settings["checkpoint_interval"], settings["curriculum_offset"]
    full_update = ppo_config.minibatches_per_rollout()
    counters, stage_stats = ctx["counters"], ctx["stage_stats"]
    csv_path = files.log_dir / f"rollouts_{files.stamp}.csv"

    class TrainingCallback(BaseCallback):
        """Curriculum, team PathCallback metrics, per-rollout stats + rollouts CSV, NaN guard, heartbeat,
        checkpoints, stop file, smoke NaN injection."""

        def __init__(self):
            super().__init__()
            self.stop_requested = False
            self.pending_row = None
            self.injected = False
            self.rollout_in_launch = 0
            self.csv_file = open(csv_path, "w", newline="", encoding="utf-8")
            self.csv = csv.DictWriter(self.csv_file, fieldnames=ROLLOUT_COLUMNS)
            self.csv.writeheader()
            self._reset_rollout()

        def _reset_rollout(self):
            self.t_collect0 = time.perf_counter()
            self.roll_stages, self.roll_episodes, self.roll_failures = [], 0, 0

        def _on_training_start(self):
            self._reset_rollout()

        def _on_rollout_start(self):
            self._finish_update()
            self._reset_rollout()

        def _on_step(self):
            n = self.num_timesteps
            counters["collected"] = n
            env.set_training_step(n + offset)
            info, done = self.locals["infos"][0], bool(self.locals["dones"][0])
            stage = info["stage"]
            lg = self.logger
            for name, value in info["costs"].items():
                lg.record_mean(f"cost/{name}", value)
            pe = info["position_error"]
            finite_pe = pe is not None and math.isfinite(pe)
            if finite_pe:
                lg.record_mean("flight/position_error", pe)
            else:  # D7
                counters["invalid_steps"] += 1
            lg.record("flight/invalid_steps", counters["invalid_steps"])
            lg.record_mean("flight/residual_norm", info["residual_norm"])
            lg.record_mean("flight/saturation_rate", info["saturated_steps"] / max(info["physics_steps"], 1))
            lg.record("flight/stage", stage)
            failed = done and info["failure_reason"] is not None
            if done:
                lg.record_mean("flight/failure_rate", float(failed))
                lg.record_mean("flight/completion_rate", float(info["completed"]))
                lg.record_mean(f"route_failure/{info['route_type']}", float(failed))
                self.roll_episodes += 1
                self.roll_failures += int(failed)
            rf.add_step(stage_stats, stage, pe if finite_pe else None, done, failed)
            self.roll_stages.append(stage)
            if n % interval == 0:
                self.save_checkpoint(n)
            if files.stop_path.exists():
                self.stop_requested = True
                say_fn(f"stop file found at {n:,} decisions (D5): stopping")
                return False
            return True

        def _on_rollout_end(self):
            now = time.perf_counter()
            m = self.model
            a = m.rollout_buffer.actions.reshape(m.n_steps, -1)
            out = np.abs(a) > 1.0
            stages = np.array(self.roll_stages[-m.n_steps:])
            for st in set(stages.tolist()):
                mask = stages == st
                rf.add_clip(stage_stats, st, int(out[mask].sum()), int(mask.sum()) * out.shape[1])
            self.rollout_in_launch += 1
            self.pending_row = {
                "launch": files.launch, "rollout": counters["rollouts"] + 1, "num_timesteps": int(m.num_timesteps),
                "stage": self.roll_stages[-1], "collect_s": now - self.t_collect0,
                "action_clip_component": float(out.mean()), "action_clip_action": float(out.any(axis=1).mean()),
                "episodes": self.roll_episodes, "failures": self.roll_failures}
            self.grad0, self.epochs0, self.t_update0 = grad["steps"], m._n_updates, now
            nan_at = ctx["args"].inject_nan_at
            if nan_at is not None and not self.injected and m.num_timesteps >= nan_at:
                with torch.no_grad():
                    next(m.policy.parameters()).view(-1)[0] = float("nan")
                self.injected = True
                say_fn(f"SMOKE HOOK (D9): set one policy parameter to NaN at {m.num_timesteps:,} decisions, "
                       "before this rollout's update")

        def _on_training_end(self):
            self._finish_update()
            self.csv_file.close()

        def _finish_update(self):
            if self.pending_row is None:
                return
            m, row = self.model, self.pending_row
            self.pending_row = None
            row["update_s"] = time.perf_counter() - self.t_update0
            gs, ep = grad["steps"] - self.grad0, m._n_updates - self.epochs0
            row.update(gradient_steps=gs, epochs=ep, early_stop=gs < full_update)
            vals = m.logger.name_to_value
            for k in ("approx_kl", "clip_fraction", "value_loss", "policy_gradient_loss", "entropy_loss",
                      "explained_variance", "loss"):
                row[k] = float(vals[f"train/{k}"]) if f"train/{k}" in vals else float("nan")
            std = torch.exp(m.policy.log_std.detach()).cpu().numpy().astype(float)
            row["std_mean"] = float(std.mean())
            for i, s in enumerate(std):
                row[f"std_{i}"] = float(s)
            counters["trained_on"] += m.n_steps
            counters["gradient_steps"] += gs
            counters["sb3_n_updates"] = int(m._n_updates)
            counters["early_stops"] += int(row["early_stop"])
            counters["rollouts"] += 1
            rf.add_rollout(stage_stats, row["stage"], row["std_mean"], row["early_stop"], gs)
            self.csv.writerow(row)
            self.csv_file.flush()
            checked = [row[k] for k in ("approx_kl", "value_loss", "policy_gradient_loss", "entropy_loss", "loss",
                                        "std_mean")]
            finite = ppo_setup.all_parameters_finite(m) and all(math.isfinite(v) for v in checked)
            files.write_status("running", counters, m.num_timesteps)  # D6 heartbeat
            say_fn(f"rollout {row['rollout']} @ {row['num_timesteps']:,} [{row['stage']}]: collect "
                   f"{row['collect_s']:.1f} s, update {row['update_s']:.2f} s, {gs} grad steps"
                   f"{' (EARLY STOP)' if row['early_stop'] else ''}, kl {row['approx_kl']:.4f}, vloss "
                   f"{row['value_loss']:.4g}, std {row['std_mean']:.4f}, clip {row['action_clip_component']:.2%}, "
                   f"episodes {row['episodes']} ({row['failures']} failed)")
            if not finite:
                raise NonFiniteError(f"non-finite parameter or train/* value after the update at "
                                     f"{row['num_timesteps']:,} decisions: {dict(zip(['approx_kl', 'value_loss', 'pg_loss', 'entropy_loss', 'loss', 'std'], checked))}")

        def save_checkpoint(self, n):
            path = ctx["ckpt_dir"] / f"{ec.checkpoint_name(n)}.zip"
            files.save_model(self.model, path, "checkpoint", counters, stage_stats)
            files.last_checkpoint = rel(path)
            files.write_status("running", counters, n)
            files.write_summary("running", counters, stage_stats)
            say_fn(f"checkpoint {rel(path)}: trained on {counters['trained_on']:,}, pending "
                   f"{rf.pending(counters):,}, gradient steps {counters['gradient_steps']:,}")

    return TrainingCallback()


def train(ctx, app, files):
    from stable_baselines3 import PPO
    from stable_baselines3.common.logger import Logger, make_output_format

    import controllers
    from hover_env import SimulationClosed

    ec, settings, counters = ctx["ec"], ctx["settings"], ctx["counters"]
    t_env0 = time.perf_counter()
    env, guard, venv = ppo_setup.build_training_env(app, ctx["forbidden"],
                                                    monitor_path=files.log_dir / f"monitor_{files.stamp}")
    if ctx["source"] is None:
        model = ppo_setup.build_model(venv)
    else:
        model = PPO.load(str(ctx["source"]), env=venv, device="cpu")
        model.set_random_seed(ctx["reseed"])   # D3 (team SAC rule): the next reset uses this seed
        if int(model._n_updates) != int(ctx["state"]["counters"]["sb3_n_updates"]):
            raise RuntimeError(f"loaded _n_updates {model._n_updates} != state {ctx['state']['counters']}")
    controllers.check_spaces(model.observation_space, model.action_space, env.observation_space, env.action_space)
    grad = ppo_setup.count_gradient_steps(model.policy.optimizer)
    env.set_training_step(model.num_timesteps + settings["curriculum_offset"])
    origins = ppo_setup.torch_origin_report()
    files.update_launch(versions=ppo_setup.versions(), torch_origins=origins, model_device=str(model.device),
                        total_mass_kg=float(env.sim.total_mass), hesai=rc.hesai_report(env.sim),
                        env_build_s=time.perf_counter() - t_env0, loaded_num_timesteps=int(model.num_timesteps))
    if origins["outside_package_dir"] or origins["modules_from_ml_archive_pip_prebundle"]:
        raise RuntimeError(f"torch origin check failed: {origins}")
    model.set_logger(Logger(folder=str(files.log_dir),
                            output_formats=[make_output_format("csv", str(files.log_dir), f"_{files.stamp}")]))
    callback = make_callback(ctx, files, env, grad, say)
    files.write_status("running", counters, model.num_timesteps)
    target = settings["total_decisions"]
    say(f"training from {model.num_timesteps:,} to >= {target:,} decisions (launch {files.launch}); "
        f"torch {origins['torch_package_dir']}; mass {env.sim.total_mass:.6f} kg")
    end = {}
    try:
        model.learn(total_timesteps=target - model.num_timesteps, callback=callback,
                    reset_num_timesteps=ctx["source"] is None)
        if callback.stop_requested:
            end = {"kind": "interrupted", "reason": "stop_file"}
        else:
            end = {"kind": "final", "reason": "completed"}
    except (KeyboardInterrupt, SimulationClosed) as stop:
        # Kept for a closed simulation. A console Ctrl+C does NOT arrive here under Kit on Windows: it kills the
        # process (M5 S1b). Use the STOP file.
        end = {"kind": "interrupted", "reason": type(stop).__name__}
    except Exception as exc:
        end = {"kind": "crashed", "reason": f"{type(exc).__name__}: {exc}", "exc": exc,
               "traceback": traceback.format_exc()}
    finally:
        if not callback.csv_file.closed:
            callback.csv_file.close()

    nonfinite = end["kind"] == "crashed" and (isinstance(end.get("exc"), NonFiniteError)
                                              or not ppo_setup.all_parameters_finite(model))
    path = files.run_dir / f"{end['kind']}_model.zip"
    files.save_model(model, path, end["kind"], counters, ctx["stage_stats"], nonfinite=nonfinite,
                     reason=end["reason"])
    if callback.stop_requested and files.stop_path.exists():
        files.stop_path.unlink()   # consumed, so a resume does not stop at once
    state = {"final": "done", "interrupted": "interrupted", "crashed": "failed"}[end["kind"]]
    result = {"final": "RESULT: DONE", "interrupted": "RESULT: INTERRUPTED", "crashed": "RESULT: FAIL"}[end["kind"]]
    files.write_status(state, counters, model.num_timesteps, model=rel(path),
                       error=None if end["kind"] != "crashed" else end["reason"], result=result)
    files.write_summary(state, counters, ctx["stage_stats"])
    origins_end = ppo_setup.torch_origin_report()   # lazy torch imports during training/saving land here
    origin_clean = not (origins_end["outside_package_dir"] or origins_end["modules_from_ml_archive_pip_prebundle"])
    if not origin_clean:
        say(f"WARNING: torch modules from outside the kit torch at the end of the launch: {origins_end}")
    files.update_launch(torch_origins_end=origins_end, torch_origin_clean_at_end=origin_clean,
                        end_state=state, end_reason=end["reason"], ended_at=rf.now_iso(),
                        end_num_timesteps=int(model.num_timesteps), end_model=rel(path), nonfinite=nonfinite,
                        counters_at_end=dict(counters), wall_clock_s=time.perf_counter() - files.t0,
                        seeded_resets=guard.seeded_resets, unseeded_resets=guard.unseeded_resets,
                        error_traceback=end.get("traceback"))
    say(f"saved {rel(path)} ({end['kind']}: {end['reason']}); {model.num_timesteps:,} decisions collected, trained on "
        f"{counters['trained_on']:,}, discarded {counters['discarded']:,}, pending {rf.pending(counters):,}; "
        f"gradient steps {counters['gradient_steps']:,}, SB3 epochs {counters['sb3_n_updates']:,}, early stops "
        f"{counters['early_stops']:,}; seeded resets {guard.seeded_resets}"
        + (f"; NONFINITE parameters" if nonfinite else ""))
    venv.close()
    if end["kind"] == "crashed":
        print(end["traceback"], flush=True)
    say(result)


def main():
    args = parse_args()
    ctx = pre_isaac(args)
    if args.preflight_only:
        say("RESULT: PREFLIGHT OK (no Isaac started)")
        return
    files = RunFiles(ctx)
    files.start_launch()
    ppo_setup.import_torch_first()  # BEFORE SimulationApp (M2.1 / M3 V2)
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        pin = ppo_setup.pin_torch_imports()   # torch.* lazy imports must not resolve to ml_archive (M5 finding)
        files.update_launch(torch_import_pin=pin)
        if not pin["probe_inside_kit_torch"]:
            raise RuntimeError(f"torch import pin failed: {pin}")
        train(ctx, app, files)
    except Exception as exc:  # failures before learn() (env build, load): record, then show the real error
        files.write_status("failed", ctx["counters"], ctx["start_timesteps"], error=f"{type(exc).__name__}: {exc}",
                           result="RESULT: FAIL")
        files.update_launch(end_state="failed", end_reason=f"{type(exc).__name__}: {exc}", ended_at=rf.now_iso())
        traceback.print_exc()
        say("RESULT: FAIL")
    finally:
        app.close()


if __name__ == "__main__":
    main()
