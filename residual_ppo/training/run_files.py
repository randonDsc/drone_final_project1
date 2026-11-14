"""Run-folder bookkeeping for PPO training (M4): atomic writes, the D12 git check, counters, per-stage statistics and
training_summary.txt. No Isaac, no torch, no gymnasium (unit-tested in the venv: tests/test_ppo_training.py).

Counters (cumulative over a run's lineage):
    collected = trained_on + discarded + pending
collected = num_timesteps; trained_on = decisions whose data went into an update; discarded = partial-rollout
decisions lost at an interruption (added on resume); pending = collected in the current rollout, not yet trained on.
"""
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# D12: ignored by the main run's clean-git check (the run's own status/meta/CSV files and evaluation outputs).
GIT_IGNORED_PREFIXES = ("residual_ppo/runs/", "residual_ppo/results/")
COUNTER_KEYS = ("collected", "trained_on", "discarded", "gradient_steps", "sb3_n_updates", "early_stops", "rollouts",
                "invalid_steps")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------- atomic writes ----------

def partial_path(path):
    """<stem>_partial<suffix> next to the target (same folder, so os.replace stays on one volume)."""
    path = Path(path)
    return path.with_name(f"{path.stem}_partial{path.suffix}")


def atomic_write_text(path, text):
    path = Path(path)
    tmp = partial_path(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, default=str) + "\n")


def atomic_save_model(model, path):
    """SB3 model.save to <stem>_partial.zip, then rename onto <stem>.zip."""
    path = Path(path)
    tmp = partial_path(path)
    model.save(str(tmp))
    os.replace(tmp, path)


def state_path(model_path):
    """The state file stored next to a model: <stem>_state.json."""
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_state.json")


# ---------- D12: clean-git check for the main run ----------

def blocking_git_entries(porcelain_lines):
    """Entries of `git status --porcelain=v1 --untracked-files=all -- residual_ppo` that block the main run.

    Blocks: any tracked file with changes (worktree or index, including staged new files) and any untracked .py file
    under residual_ppo/. Ignores paths under residual_ppo/runs/ and residual_ppo/results/, and untracked non-.py files.
    """
    blocking = []
    for line in porcelain_lines:
        if len(line) < 4 or not line.strip():
            continue
        code, path = line[:2], line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip('"')
        if not path.startswith("residual_ppo/") or path.startswith(GIT_IGNORED_PREFIXES):
            continue
        if code == "!!":
            continue
        if code == "??":
            if path.endswith(".py"):
                blocking.append({"status": "untracked .py", "path": path})
        else:
            blocking.append({"status": f"changed tracked ({code.strip() or code})", "path": path})
    return blocking


def git_porcelain(repo_dir):
    out = subprocess.run(["git", "-C", str(repo_dir), "status", "--porcelain=v1", "--untracked-files=all", "--",
                          "residual_ppo"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()


# ---------- counters ----------

def new_counters():
    return {k: 0 for k in COUNTER_KEYS}


def pending(counters):
    return counters["collected"] - counters["trained_on"] - counters["discarded"]


def counters_after_resume(state_counters):
    """Loaded state -> counters of the resumed lineage: the loaded model's pending decisions are lost."""
    c = {k: int(state_counters.get(k, 0)) for k in COUNTER_KEYS}
    c["discarded"] += pending(c)
    return c


# ---------- per-stage statistics ----------

def _stage(stats, stage):
    return stats.setdefault(stage, {"decisions": 0, "episodes": 0, "failures": 0, "pos_err_sum": 0.0, "pos_err_n": 0,
                                    "rollouts": 0, "std_sum": 0.0, "early_stops": 0, "gradient_steps": 0,
                                    "clip_components": 0, "components": 0})


def add_step(stats, stage, position_error, done, failed):
    s = _stage(stats, stage)
    s["decisions"] += 1
    if position_error is not None and math.isfinite(position_error):
        s["pos_err_sum"] += float(position_error)
        s["pos_err_n"] += 1
    if done:
        s["episodes"] += 1
        s["failures"] += int(bool(failed))


def add_rollout(stats, stage, std_mean, early_stop, gradient_steps):
    """Per-update quantities are assigned to the stage of the rollout's last decision."""
    s = _stage(stats, stage)
    s["rollouts"] += 1
    s["std_sum"] += float(std_mean)
    s["early_stops"] += int(bool(early_stop))
    s["gradient_steps"] += int(gradient_steps)


def add_clip(stats, stage, clipped_components, components):
    s = _stage(stats, stage)
    s["clip_components"] += int(clipped_components)
    s["components"] += int(components)


def stage_rows(stats, order):
    rows = []
    for name in list(order) + sorted(set(stats) - set(order)):
        if name not in stats:
            continue
        s = stats[name]
        rows.append({
            "stage": name, "decisions": s["decisions"], "episodes": s["episodes"],
            "failure_rate": s["failures"] / s["episodes"] if s["episodes"] else None,
            "mean_position_error": s["pos_err_sum"] / s["pos_err_n"] if s["pos_err_n"] else None,
            "std": s["std_sum"] / s["rollouts"] if s["rollouts"] else None,
            "rollouts": s["rollouts"], "early_stops": s["early_stops"],
            "action_clip_fraction": s["clip_components"] / s["components"] if s["components"] else None,
            "gradient_steps": s["gradient_steps"],
        })
    return rows


def _f(v, fmt):
    return "-" if v is None else format(v, fmt)


def summary_text(run_id, state, counters, stats, order, extra=()):
    """training_summary.txt: header, counters, per-stage table."""
    lines = [f"training summary: {run_id}  (state: {state}; written {now_iso()})",
             f"decisions collected {counters['collected']:,} = trained on {counters['trained_on']:,} + discarded "
             f"{counters['discarded']:,} + pending {pending(counters):,}",
             f"rollouts {counters['rollouts']:,}; gradient steps {counters['gradient_steps']:,}; SB3 epochs "
             f"{counters['sb3_n_updates']:,}; early stops {counters['early_stops']:,}; invalid steps "
             f"{counters['invalid_steps']:,}", *extra, "",
             f"{'stage':<13}{'decisions':>11}{'episodes':>10}{'fail rate':>10}{'pos err m':>10}{'std':>8}"
             f"{'rollouts':>9}{'early st':>9}{'clip %':>8}{'grad steps':>11}"]
    for r in stage_rows(stats, order):
        clip = None if r["action_clip_fraction"] is None else 100 * r["action_clip_fraction"]
        lines.append(f"{r['stage']:<13}{r['decisions']:>11,}{r['episodes']:>10,}{_f(r['failure_rate'], '.3f'):>10}"
                     f"{_f(r['mean_position_error'], '.3f'):>10}{_f(r['std'], '.4f'):>8}{r['rollouts']:>9,}"
                     f"{r['early_stops']:>9,}{_f(clip, '.2f'):>8}{r['gradient_steps']:>11,}")
    lines.append("(std, early stops, gradient steps and rollouts are assigned to the stage of each rollout's last "
                 "decision; action clip = share of sampled action components outside [-1, 1])")
    return "\n".join(lines) + "\n"


# ---------- live-run lock (M5b item 3) ----------

LIVE_STATES = ("starting", "running")


def process_start_time(pid):
    """Creation time of the LIVE process with this pid, or None if no live process has it.

    Windows: OpenProcess + WaitForSingleObject(handle, 0) (still running <=> WAIT_TIMEOUT) + GetProcessTimes; the
    creation FILETIME is returned as an int. 0 means "alive, start time unknown" (access denied). The start time
    tells a live run from an unrelated process that reused the pid after a reboot. Other systems: os.kill(pid, 0).
    """
    pid = int(pid)
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return None
        except PermissionError:
            pass
        return 0
    import ctypes
    from ctypes import wintypes

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    k.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = k.OpenProcess(0x1000 | 0x00100000, False, pid)   # QUERY_LIMITED_INFORMATION | SYNCHRONIZE
    if not handle:
        return 0 if ctypes.get_last_error() == 5 else None    # 5 = access denied: the process exists
    try:
        if k.WaitForSingleObject(handle, 0) != 0x102:          # not WAIT_TIMEOUT: it has exited
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not k.GetProcessTimes(handle, *map(ctypes.byref, times)):
            return 0
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        k.CloseHandle(handle)


def live_run_reason(status, start_time_of=process_start_time):
    """Why a launch on this run folder must be refused (its run is live), or None.

    Live = status.json says "starting"/"running" AND its pid is alive AND (when both start times are known) the
    process is the one that wrote the status. A dead pid (killed run, power loss) never blocks."""
    if not status or status.get("state") not in LIVE_STATES or status.get("pid") is None:
        return None
    started = start_time_of(int(status["pid"]))
    if started is None:
        return None
    recorded = status.get("pid_start")
    if recorded and started and recorded != started:
        return None   # the pid now belongs to another process
    return (f"the run is live: status.json says {status['state']!r} (launch {status.get('launch')}, "
            f"{status.get('num_timesteps')} decisions, updated {status.get('updated_at')}) and pid {status['pid']} "
            "is alive; stop it with the STOP file first")


# ---------- killed launches and lineage (M5b item 2) ----------

def mark_killed_launches(meta, status):
    """A launch without an end record was killed (Stop-Process, Ctrl+C, closed window, power loss). Called at the
    start of the next launch, after the live-run lock passed. Returns the launch numbers marked."""
    marked = []
    for launch in meta.get("launches", []):
        if launch.get("end_state") is None:
            same = bool(status) and status.get("launch") == launch.get("launch")
            launch.update(end_state="killed",
                          end_reason="no end record: the process was killed or lost power (inferred at the next "
                                     "launch)",
                          killed_inferred_at=now_iso(),
                          last_heartbeat=status.get("updated_at") if same else None,
                          last_num_timesteps=status.get("num_timesteps") if same else None,
                          last_status_state=status.get("state") if same else None)
            marked.append(launch.get("launch"))
    return marked


def leftover_partials(run_dir):
    return sorted(p for p in Path(run_dir).rglob("*_partial*") if p.is_file())


def lineage_row_counts(launches, end_rollouts):
    """How many leading rows of each launch's rollouts CSV belong to the final model's lineage.

    launches: [{"start_rollouts": lineage updates at the launch's start, "rows": CSV rows or None (no CSV)}], in
    launch order. Walks backwards from the end model: a launch contributes the updates between its own start and the
    point the next lineage launch resumed from; rows past that point (after the checkpoint a later launch resumed
    from) stay out. Returns (counts, unexplained) where unexplained must be 0."""
    need = int(end_rollouts)
    counts = [0] * len(launches)
    for i in range(len(launches) - 1, -1, -1):
        rows, start = launches[i]["rows"], int(launches[i]["start_rollouts"])
        if rows is None:
            continue
        counts[i] = max(0, min(int(rows), need - start))
        need = min(need, start)
    return counts, need


def killed_launch_resume_ok(killed, following):
    """A killed launch is recoverable only through a checkpoint at or below its last recorded num_timesteps."""
    if following is None:
        return False, "no later launch that trained"
    src = following.get("source_model") or ""
    last = killed.get("last_num_timesteps")
    ok = (following.get("mode") == "resume" and "/checkpoints/" in src and last is not None
          and int(following["start_num_timesteps"]) <= int(last))
    return ok, (f"next training launch {following.get('launch')} resumed from {src or None} at "
                f"{following.get('start_num_timesteps')} (killed launch last recorded {last})")
