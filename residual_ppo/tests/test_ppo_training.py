"""M4 unit tests (no Isaac): D12 git filter, run settings / contracts, counters, summary, learning-curve refusals.

    # Directory: <repo root>
    & "<python>" -m unittest discover -s residual_ppo\\tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

RESIDUAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESIDUAL_DIR))
sys.path.insert(1, str(RESIDUAL_DIR / "evaluation"))
sys.path.insert(2, str(RESIDUAL_DIR / "training"))

import eval_protocol_v1 as protocol  # noqa: E402
import learning_curve as lc  # noqa: E402
import run_common as rc  # noqa: E402
import run_files as rf  # noqa: E402
from config import experiment_config as ec  # noqa: E402
from config import ppo_config  # noqa: E402


class GitCheckTest(unittest.TestCase):
    """D12: block modified tracked files and untracked .py under residual_ppo/; ignore runs/ and results/."""

    def test_blocks_modified_tracked_and_untracked_py(self):
        lines = [" M residual_ppo/config/ppo_config.py", "M  residual_ppo/training/ppo_setup.py",
                 "A  residual_ppo/training/new_module.py", "?? residual_ppo/training/train_path_ppo.py",
                 "R  residual_ppo/old.py -> residual_ppo/tools/renamed.py"]
        got = [b["path"] for b in rf.blocking_git_entries(lines)]
        self.assertEqual(got, ["residual_ppo/config/ppo_config.py", "residual_ppo/training/ppo_setup.py",
                               "residual_ppo/training/new_module.py", "residual_ppo/training/train_path_ppo.py",
                               "residual_ppo/tools/renamed.py"])

    def test_ignores_runs_results_and_untracked_non_py(self):
        lines = [" M residual_ppo/runs/ppo_path/ppo_s0_1p5m/status.json",
                 "?? residual_ppo/runs/ppo_path/ppo_s0_1p5m/run_meta.json",
                 "?? residual_ppo/runs/ppo_path/ppo_s0_1p5m/logs/progress_20261114.csv",
                 "?? residual_ppo/runs/ppo_path/x/helper.py",
                 " M residual_ppo/results/m4/something.json", "?? residual_ppo/results/m5/new.py",
                 "?? residual_ppo/docs/notes.txt", "!! residual_ppo/scratch/model.zip", ""]
        self.assertEqual(rf.blocking_git_entries(lines), [])


class RunSettingsTest(unittest.TestCase):
    def test_allowed_ids(self):
        main = ec.run_settings("ppo_s0_1p5m")
        self.assertEqual((main["smoke"], main["total_decisions"], main["checkpoint_interval"]),
                         (False, 1_500_000, 50_000))
        smoke = ec.run_settings("smoke_ppo_s0_7", curriculum_offset=145_000)
        self.assertEqual((smoke["smoke"], smoke["total_decisions"], smoke["checkpoint_interval"],
                          smoke["curriculum_offset"]), (True, 10_240, 4_096, 145_000))
        for bad in ("ppo_s1_1p5m", "smoke_ppo_s0_", "smoke_ppo_s0_x", "m3_timing_probe_s0_8k", ""):
            with self.assertRaises(ValueError):
                ec.run_settings(bad)
        with self.assertRaises(ValueError):
            ec.run_settings("ppo_s0_1p5m", curriculum_offset=1)

    def test_smoke_contract_differs_only_by_override(self):
        main = ec.run_contract(ec.run_settings("ppo_s0_1p5m"))
        smoke = ec.run_contract(ec.run_settings("smoke_ppo_s0_1"))
        self.assertEqual(main["policy"], smoke["policy"])
        self.assertNotIn("smoke_override", main["training"])
        extra = dict(smoke["training"])
        self.assertEqual(extra.pop("smoke_override")["total_decisions"], 10_240)
        self.assertEqual(json.loads(json.dumps(extra)), json.loads(json.dumps(main["training"])))
        self.assertEqual(json.loads(json.dumps(main)), json.loads(json.dumps(ppo_config.contract())))


class CountersTest(unittest.TestCase):
    def test_resume_discards_pending(self):
        c = rf.new_counters()
        c.update(collected=6_001, trained_on=4_096, discarded=0)
        self.assertEqual(rf.pending(c), 1_905)
        r = rf.counters_after_resume(c)
        self.assertEqual((r["collected"], r["trained_on"], r["discarded"], rf.pending(r)), (6_001, 4_096, 1_905, 0))

    def test_trained_on_formula(self):
        self.assertEqual([ec.decisions_trained_on(n) for n in (4_096, 8_192, 150_000)], [2_048, 6_144, 149_504])


class SummaryTest(unittest.TestCase):
    def test_stage_table(self):
        stats = {}
        for i in range(10):
            rf.add_step(stats, "short_slow", 0.1 * i if i != 3 else float("nan"), done=(i % 5 == 4), failed=(i == 9))
        rf.add_rollout(stats, "short_slow", 0.36, early_stop=True, gradient_steps=40)
        rf.add_clip(stats, "short_slow", 2, 40)
        rows = rf.stage_rows(stats, ["short_slow", "turns"])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual((r["decisions"], r["episodes"], r["failure_rate"], r["early_stops"], r["gradient_steps"]),
                         (10, 2, 0.5, 1, 40))
        self.assertAlmostEqual(r["mean_position_error"], sum(0.1 * i for i in range(10) if i != 3) / 9)
        self.assertAlmostEqual(r["action_clip_fraction"], 0.05)
        c = rf.new_counters()
        c.update(collected=10, trained_on=0)
        text = rf.summary_text("smoke_ppo_s0_1", "running", c, stats, ["short_slow"])
        self.assertIn("(state: running;", text)
        self.assertIn("short_slow", text)

    def test_atomic_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "status.json"
            rf.atomic_write_json(p, {"a": 1})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), {"a": 1})
            self.assertEqual([q.name for q in Path(d).iterdir()], ["status.json"])


class LiveRunLockTest(unittest.TestCase):
    """M5b item 3: refuse only when status is starting/running AND that process is alive."""

    def test_decision(self):
        alive, dead = (lambda pid: 111), (lambda pid: None)
        for state in ("starting", "running"):
            st = {"state": state, "pid": 42, "pid_start": 111, "launch": 1}
            self.assertIn("live", rf.live_run_reason(st, alive))
            self.assertIsNone(rf.live_run_reason(st, dead))                    # killed run: proceed
        for state in ("done", "interrupted", "failed"):
            self.assertIsNone(rf.live_run_reason({"state": state, "pid": 42, "pid_start": 111}, alive))
        self.assertIsNone(rf.live_run_reason(None, alive))                     # no status.json
        self.assertIsNone(rf.live_run_reason({"state": "running", "pid": None}, alive))
        reused = {"state": "running", "pid": 42, "pid_start": 999}
        self.assertIsNone(rf.live_run_reason(reused, alive))                   # pid reused by another process
        unknown = {"state": "running", "pid": 42}                              # old status without pid_start
        self.assertIn("live", rf.live_run_reason(unknown, alive))
        self.assertIn("live", rf.live_run_reason({"state": "running", "pid": 42, "pid_start": 111}, lambda pid: 0))

    def test_pid_check_on_this_machine(self):
        import os
        import subprocess

        self.assertTrue(rf.process_start_time(os.getpid()))
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            started = rf.process_start_time(child.pid)
            self.assertIsNotNone(started)
            st = {"state": "running", "pid": child.pid, "pid_start": started}
            self.assertIn("live", rf.live_run_reason(st))
        finally:
            child.kill()
            child.wait()
        self.assertIsNone(rf.process_start_time(child.pid))
        self.assertIsNone(rf.live_run_reason(st))


class KilledLaunchTest(unittest.TestCase):
    """M5b item 2: killed launches in run_meta, the audit's killed branch and the lineage rule."""

    def test_mark_killed(self):
        meta = {"launches": [{"launch": 1, "end_state": "interrupted"}, {"launch": 2, "end_state": None}]}
        status = {"state": "running", "launch": 2, "num_timesteps": 4096, "updated_at": "2026-11-14T20:00:00+00:00"}
        self.assertEqual(rf.mark_killed_launches(meta, status), [2])
        k = meta["launches"][1]
        self.assertEqual((k["end_state"], k["last_num_timesteps"], k["last_heartbeat"], k["last_status_state"]),
                         ("killed", 4096, "2026-11-14T20:00:00+00:00", "running"))
        self.assertEqual(meta["launches"][0]["end_state"], "interrupted")
        self.assertEqual(rf.mark_killed_launches(meta, status), [])            # idempotent
        other = {"launches": [{"launch": 1, "end_state": None}]}
        rf.mark_killed_launches(other, {"launch": 7, "num_timesteps": 1})       # status of another launch
        self.assertIsNone(other["launches"][0]["last_num_timesteps"])

    def test_killed_branch(self):
        killed = {"launch": 1, "end_state": "killed", "last_num_timesteps": 4096}
        ck = "residual_ppo/runs/ppo_path/smoke_ppo_s0_6/checkpoints/ppo_4096_steps.zip"
        ok = {"launch": 2, "mode": "resume", "source_model": ck, "start_num_timesteps": 4096}
        self.assertTrue(rf.killed_launch_resume_ok(killed, ok)[0])
        beyond = dict(ok, start_num_timesteps=8192)                             # a checkpoint it never recorded
        self.assertFalse(rf.killed_launch_resume_ok(killed, beyond)[0])
        not_ckpt = dict(ok, source_model="residual_ppo/runs/ppo_path/x/interrupted_model.zip")
        self.assertFalse(rf.killed_launch_resume_ok(killed, not_ckpt)[0])
        self.assertFalse(rf.killed_launch_resume_ok(killed, dict(ok, mode="fresh"))[0])
        self.assertFalse(rf.killed_launch_resume_ok(killed, None)[0])
        self.assertFalse(rf.killed_launch_resume_ok({"launch": 1, "last_num_timesteps": None}, ok)[0])

    def test_lineage_rows(self):
        # S7: launch 1 wrote 2 rows, was killed; launch 2 resumed from the 4,096 checkpoint (1 update inside)
        self.assertEqual(rf.lineage_row_counts([{"start_rollouts": 0, "rows": 2}, {"start_rollouts": 1, "rows": 3}], 4),
                         ([1, 3], 0))
        # S1/S2: resume from interrupted_model keeps every row of launch 1
        self.assertEqual(rf.lineage_row_counts([{"start_rollouts": 0, "rows": 2}, {"start_rollouts": 2, "rows": 3}], 5),
                         ([2, 3], 0))
        # a launch that failed before training (no CSV) is skipped
        self.assertEqual(rf.lineage_row_counts([{"start_rollouts": 0, "rows": 2}, {"start_rollouts": 2, "rows": None},
                                                {"start_rollouts": 2, "rows": 3}], 5), ([2, 0, 3], 0))
        # a later launch resumed from BEFORE an abandoned launch's start: that launch contributes nothing
        self.assertEqual(rf.lineage_row_counts([{"start_rollouts": 0, "rows": 4}, {"start_rollouts": 4, "rows": 2},
                                                {"start_rollouts": 2, "rows": 3}], 5), ([2, 0, 3], 0))
        # missing rows are reported, not hidden
        self.assertEqual(rf.lineage_row_counts([{"start_rollouts": 2, "rows": 3}], 5), ([3], 2))

    def test_leftover_partials(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "checkpoints").mkdir()
            for name in ("status_partial.json", "checkpoints/ppo_8192_steps_partial.zip", "status.json",
                         "checkpoints/ppo_4096_steps.zip"):
                (d / name).write_text("x", encoding="utf-8")
            self.assertEqual([q.name for q in rf.leftover_partials(d)],
                             ["ppo_8192_steps_partial.zip", "status_partial.json"])


CANONICAL_PID = RESIDUAL_DIR / "results" / "eval" / "eval-v1" / "pid" / "results.json"


def _template_row():
    """A real hover/training_range flight row of the canonical PID run (all fields summarize() needs)."""
    flights = json.loads(CANONICAL_PID.read_text(encoding="utf-8"))["flights"]
    return next(f for f in flights if f["scenario"] == "hover" and f["wind_profile"] == "training_range")


def _fake_run(d, controller, seeds, ref_sha=None, blobs=None):
    d.mkdir(parents=True)
    flights, real = [], {}
    template = _template_row()
    for seed in seeds:
        flights.append({**template, "seed": seed, "controller": controller})
        real[f"hover|training_range|{seed}"] = {k: template[k] for k in ("wind_seed", "wind_speed", "route_seconds",
                                                                         "initial_obs_sha256", "max_decisions")}
    meta = {"protocol": {"protocol_file_sha256": rc.EXPECTED_PROTOCOL_SHA},
            "code_blobs": {f"residual_ppo/evaluation/{k}": v for k, v in (blobs or lc.frozen_blobs()).items()},
            "controller": {"name": controller, "num_timesteps": 10_240, "model": "m.zip", "model_sha256": "s"},
            "reference_realizations": None if ref_sha is None else {"sha256_lf": ref_sha},
            "realization_guard": None if controller == "pid" else {"resets_checked": len(seeds), "all_matched": True}}
    (d / "results.json").write_text(json.dumps({"meta": meta, "flights": flights}), encoding="utf-8")
    (d / "realizations.json").write_text(json.dumps({"note": "NOT canonical", "key": list(
        real[next(iter(real))]), "flights": real}), encoding="utf-8")
    return d


class LearningCurveTest(unittest.TestCase):
    def _run(self, tmp, ctrl_seeds, pid_seeds=(400000,), wrong_ref=False, out=True):
        pid = _fake_run(tmp / "pid", "pid", pid_seeds)
        sha = "0" * 64 if wrong_ref else rc.file_sha256(pid / "realizations.json")
        ctrl = _fake_run(tmp / "ctrl", "pid_ppo", ctrl_seeds, ref_sha=sha)
        return ["--pid-run", str(pid), "--run", f"smoke={ctrl}", "--out", str(tmp / "lc"), "--dry-run", "unit test"]

    def test_dry_run_one_cell_ok(self):
        with tempfile.TemporaryDirectory() as d:
            lc.main(self._run(Path(d), (400000,)))
            self.assertTrue((Path(d) / "lc.json").is_file())

    def test_refuses_eval_seed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                lc.main(self._run(Path(d), (protocol.EVAL_SEEDS[0],), pid_seeds=(protocol.EVAL_SEEDS[0],)))
            self.assertIn("EVAL seeds", str(cm.exception))

    def test_refuses_wrong_reference(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit) as cm:
                lc.main(self._run(Path(d), (400000,), wrong_ref=True))
            self.assertIn("reference_realizations", str(cm.exception))

    def test_full_mode_refuses_incomplete_grid(self):
        with tempfile.TemporaryDirectory() as d:
            args = self._run(Path(d), (400000,))[:-2]  # drop --dry-run
            with self.assertRaises(SystemExit) as cm:
                lc.main(args)
            self.assertIn("complete validation grid", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
