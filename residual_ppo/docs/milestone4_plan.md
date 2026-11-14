# Milestone 4: PPO training script, PID validation reference, learning curve (+ M5 smoke test)

Owner: Claire (DSC 180A). Design approved by Claire on 2026-11-14.

Labels: **Confirmed** (ran or read code), **Documented**, **Inference**, **Proposed**.

## 1. Files

| File | Purpose |
|---|---|
| `training/train_path_ppo.py` | The PPO training script, built on `training/ppo_setup.py`. |
| `training/ppo_setup.py` | M3 helpers, extended with atomic writes, run settings, the git check and the training callback. |
| `evaluation/learning_curve.py` | New, **unfrozen**. Secondary learning curve on validation seeds; reuses `protocol_stats`. |
| `tools/audit_ppo_run.py` | M5 S3 file and count audit of a run folder. No Isaac. |
| `tools/check_eval_torch_origin.py` | M5b S8: torch-origin evidence for evaluation flights, mirroring the frozen fly load order. Not frozen. |
| `results/eval/eval-v1/pid_validation/` | PID on validation seeds 400000–400009 × 6 profiles × 6 scenarios (360 flights), flown with the frozen `run_protocol.py fly`. |

## 2. Decisions (approved by Claire, 2026-11-14)

| # | Decision |
|---|---|
| D1 | Allowed run ids: `ppo_s0_1p5m` (main) and `smoke_ppo_s0_<n>`. Any other id is refused. |
| D2 | `--resume-model` must be inside the same run folder. |
| D3 | Resume reseed = `seed + num_timesteps + 1` (the team SAC's rule). A resume whose reseed would be an eval or validation seed is refused before SimulationApp; resume from the previous checkpoint instead. |
| D4 | `total_timesteps` = 1,500,000 and the actual count is recorded. An uninterrupted run ends at exactly 1,501,184. Primary result = final model of the main run (1,501,184 if uninterrupted; otherwise the recorded count). |
| D5 | Stop file `runs/ppo_path/<id>/STOP`: clean stop → `interrupted_model.zip`. **This is the only way to stop a run.** Ctrl+C or closing the window kills the process at once and saves nothing (M5 S1b, Confirmed); recovery is resume from the last checkpoint (M5b S7). |
| D6 | `status.json` is also updated after every rollout (heartbeat `updated_at`). |
| D7 | Non-finite `position_error` values (invalid states) are skipped in the `flight/position_error` mean and counted as `flight/invalid_steps`. |
| D8 | Smoke-only `--curriculum-offset` (the curriculum sees `num_timesteps + offset`), recorded in the contract's training group. |
| D9 | Smoke-only `--inject-nan-at <step>`: makes a parameter NaN before the update that follows that step, to test the NaN guard. |
| D10 | The final model is also flown on validation seeds as the learning curve's last point (descriptive only). |
| D11 | Validation flights run **after** training, never in parallel with it. |
| D12 | The main run (fresh **and** resume) refuses to start when, under `residual_ppo/`, there is a **modified tracked file** or an **untracked `.py` file**. Paths under `residual_ppo/runs/` and `residual_ppo/results/` are ignored, so the run's own status, meta and CSV files never block a resume. Smoke runs skip the check. S0 tests both cases. |
| + | `training_summary.txt` in the run folder, written at every checkpoint and at exit. It holds a per-stage table: decisions, episodes, failure rate, mean position error, std, early stops, action clip fraction, gradient steps. |

Carried over from M3: checkpoints every 50k decisions saved mid-rollout; the primary result is the final model with
no checkpoint selection; checkpoints at 150k, 400k and 800k form the secondary learning curve on validation seeds;
E7b compares training cost by gradient steps.

## 3. Training script design

### Startup order
1. **Before SimulationApp:**
   - preflight (checks 21/21, import guard);
   - id and run settings (D1, smoke override);
   - hook flags allowed only for smoke runs;
   - fresh: refuse a run folder that already holds a model;
   - resume: path inside the run folder (D2); state file present and finite; zip `num_timesteps` == state; stored
     contract == current; reseed not forbidden (D3);
   - main run: git check (D12).

   `--preflight-only` stops here with `RESULT: PREFLIGHT OK`, without starting Isaac.
2. `import_torch_first()`, then SimulationApp, then the env stack, then a fresh `PPO` or
   `PPO.load(..., env=venv, device="cpu")` + `set_random_seed(reseed)`.
3. `learn(...)`.
4. Save; update status, summary and meta; `RESULT: DONE` / `RESULT: INTERRUPTED` / `RESULT: FAIL`.

### Counters (cumulative over the run's lineage)
- `collected` = `num_timesteps`;
- `trained_on`: decisions whose data went into an update;
- `discarded`: partial-rollout decisions lost at an interruption;
- `pending`: collected in the current rollout but not yet trained on;
- `gradient_steps`: counted by the optimizer hook;
- `sb3_n_updates`: epochs;
- `early_stops`.

Invariant: `collected = trained_on + discarded + pending`. On resume, `discarded += pending` of the loaded state.

### Files

| File | Content |
|---|---|
| `ppo_contract.json` (run folder and `checkpoints/`) | `ppo_config.contract()`, with `smoke_override` in the training group for smoke runs |
| `checkpoints/ppo_<n>_steps.zip` + `_state.json` | weights, counters, stage, stage stats, launch number |
| `final_model.zip` / `interrupted_model.zip` / `crashed_model.zip` + `_state.json` | end-state model; `crashed` carries `nonfinite`, and resume refuses non-finite models |
| `status.json` | state (running / done / interrupted / failed), counters, last checkpoint, `updated_at`, pid, error |
| `run_meta.json` | run id, settings, `launches` list (mode, source model, reseed, versions, git, mass, HESAI, hardware, times, end state, counters) |
| `training_summary.txt` | per-stage table + totals (at every checkpoint and at exit) |
| `logs/progress_<stamp>.csv` | SB3 CSV: team PathCallback metrics + rollout/time keys |
| `logs/rollouts_<stamp>.csv` | one row = rollout k with its own update k (no lag) |
| `logs/monitor_<stamp>.monitor.csv` | SB3 Monitor per-episode file |
| `logs/console_*.log` | console (gitignored, `runs/**/*.log`) |

All writes are atomic: write `<name>_partial`, then `os.replace`.

## 4. M5 smoke test plan (S0–S6)

See the M4 design report. In short:
- **S0:** refusals before Isaac, including D12 both ways.
- **S1:** fresh run with a STOP file at about 6,000 decisions.
- **S1b:** Ctrl+C (result: it kills the run; see D5).
- **S2:** resume to the end.
- **S3:** audit.
- **S4:** NaN guard.
- **S5:** PID validation reference + one adapter flight of the smoke model.
- **S6:** learning-curve dry run + refusals.

## 5. Results (Confirmed, 2026-11-14; HEAD 30c7ce9 with the M4 files uncommitted)

### PID validation reference (item 6)
- Run: frozen `run_protocol.py fly --controller pid --seeds 400000-400009 --check0`, written to
  `results/eval/eval-v1/pid_validation/`. 360 flights in 13.4 min; `RESULT: DONE`.
- Checks: consistency check PASS on the full validation grid; protocol hash d3f677dc…; code_blobs == the frozen IDs;
  total mass 1.618614 kg; realizations marked "NOT canonical (subset run)", which is correct.
- Failures per profile (out of 60):

  | still_air | training_range | unseen_gusts | outside_range | strong_range | beyond_strong |
  |---|---|---|---|---|---|
  | 0 | 0 | 0 | 34 | 59 | 60 |

- `--check0` flies its fixed zero-action cell (turns / training_range / 300000), which is on an eval seed. That is a
  PID-only identity check, the same as the canonical PID run; no learned model touches eval seeds.

### M5 smoke test

| Step | Result |
|---|---|
| S0 refusals (before Isaac) | **PASS, 11/11.** Each case refused with a clear message, or passed preflight where it should; see the list below. |
| S1 STOP file | **PASS.** `smoke_ppo_s0_4`, `--curriculum-offset 145000`. Stopped at 5,977 decisions (mid-rollout 3): `interrupted_model.zip` + state saved, 4,096 trained on, 1,881 pending; `RESULT: INTERRUPTED`. The 4,096 checkpoint was trained on 2,048 (== formula). |
| S1b Ctrl+C | **Documented branch.** A real CTRL_C_EVENT (attached to the training console) **terminated the process at once; nothing was saved**. Python never saw a `KeyboardInterrupt` (Inference: Kit runs with `--/app/installSignalHandlers=0` and the default console handler ends the process). `status.json` stayed "running" with a stale `updated_at`, so the D6 heartbeat exposes it. **Rule: stop runs with the STOP file only.** |
| S2 resume | **PASS.** Reseed 5978 (= 0 + 5,977 + 1) used on the first reset; stage turns (offset hook crossed 150k); 8,192 checkpoint trained on 6,144; final at **12,121** decisions (D4 overshoot), trained on 10,240 = 5 × 2048, discarded 1,881; `RESULT: DONE`. |
| S3 audit | **PASS, 19/19** (`tools/audit_ppo_run.py`, `results/m5/s3_audit_smoke_ppo_s0_4.txt`): counters invariant, zip == state (num_timesteps, `_n_updates` 50), checkpoints at 4,096 and 8,192, contracts == `run_contract` and accepted by the adapter, CSV rows == rollouts per launch, lineage sums (400 gradient steps, 50 epochs, 0 early stops), mass, HESAI 0, torch origin clean at start and end of both launches, no `_partial` files, summary present. |
| S4 NaN guard | **PASS** (`smoke_ppo_s0_5`). Hook at 4,096 → `crashed_model.zip` marked non-finite, status failed, `RESULT: FAIL`; resume refused, naming the last good checkpoint. The trigger was torch's distribution validation raising `ValueError` inside the update (the hooked parameter was `log_std`, so the scale became NaN); the crash handler's finiteness check marked it. The post-update guard is the second net for NaN that does not reach a distribution. |
| S5 adapter flight | **PASS.** Frozen `fly --controller pid_ppo` on the smoke final model, seed 400000, training_range / hover, against the PID validation realizations: adapter accepted it (contract + spaces), 69 decisions all finite, realization guard 1/1 matched, `RESULT: DONE`. No tracking metrics reported (untrained model; raw output gitignored). |
| S6 learning curve | **PASS.** Dry run on the S5 cell written. Refused: a PID run on eval seeds; full mode with a 1-cell run; the canonical eval PID run given as a model. Nothing was written when refused. |
| Tests / checks | **68/68 tests** (56 + 12 new in `tests/test_ppo_training.py`) and **checks 21/21**, in BOTH the venv and Isaac python.bat. |

S0 cases (`results/m5/s0_refusals_part{1,2,3}.txt`):
- D1: an id that is not allowed;
- D12 case (a): the main run is refused on the current tree, listing exactly the 2 modified tracked files and 5
  untracked `.py` files;
- D12 case (b), live: untracked files under `runs/` and `results/m4`/`m5` and the untracked `.md` plan are not listed;
  also unit-tested both ways;
- D9 and D8: hooks on the main id;
- D3: forbidden reseed 300000 (crafted dummy state at 299,999);
- D2: a model in another run folder, and a `..` path that escapes the run folder;
- fresh on a folder with a model;
- a tampered contract copy (`training.ppo` differs);
- the wrong curriculum offset (`training.smoke_override` differs);
- a correct resume with `--preflight-only` → `PREFLIGHT OK`.

The first S0.8 attempt printed nothing: PowerShell 5.1 `Set-Content -Encoding utf8` wrote a BOM into the tampered JSON.
It was redone with Python (part 3). A live main-run preflight that *passes* is only possible once the M4 code is
committed; it is the first step after the commit.

### Findings fixed during M5 (Confirmed)
1. **Kit's `OmniFinder` hijacks lazy torch imports.**
   - It is first on `sys.meta_path` and maps not-yet-imported module names to extension prebundles, so a `torch.*`
     submodule first imported after SimulationApp came from the stale `extsDeprecated/omni.isaac.ml_archive` torch
     2.11, although `torch.__path__` is clean.
   - Seen twice:
     - `torch.utils.serialization` via `PPO.load` on resume (`smoke_ppo_s0_1` launch 2, caught by the start-of-launch
       origin check; the run stopped before training);
     - `torch._subclasses.schema_check_mode` in the crash path (`smoke_ppo_s0_2`, caught by the new end-of-launch
       check).
   - Fixes:
     - `import_torch_first()` now pre-imports `torch.utils.serialization(.config)`;
     - **`ppo_setup.pin_torch_imports()`**, called right after SimulationApp, puts a finder ahead of OmniFinder that
       resolves `torch.*` only through the parent package's `__path__` (the kit torch); nothing on disk changes;
     - an **end-of-launch origin check** is recorded in `run_meta` and checked by the audit.
   - The S1 saves were not affected: torch only imports that module in save for non-CPU storage (torch
     `serialization.py` line 1291), and the stale `config.py` is byte-identical anyway.
   - After the fix, S1/S2/S4 were repeated on the final code (`smoke_ppo_s0_4`, `_5`) with clean origins at start and
     end.
   - The frozen `run_protocol.py fly` has no pin and no origin check. M5b S8 (section 6) gives the evidence for it
     without changing frozen code: both controllers PASS.
2. **The callback must not take module-level `locals()`** (from M3; not repeated here).

Smoke folders (gitignored, `runs/ppo_path/smoke_*/`); their status and launch records are condensed in
`results/m5/m5_runs.json`:
- `_1`: first pass, with the refused resume;
- `_2`: NaN run before the pin;
- `_3`: Ctrl+C;
- `_4`: clean final-code run;
- `_5`: NaN run with the pin;
- `_6`: M5b S7 hard kill and resume from a checkpoint.

## 6. M5b: pre-commit hardening (Confirmed, 2026-11-14, unless labelled otherwise)

The 5 frozen evaluation files and the shared modules were not touched (working-tree blob IDs == HEAD; checks 21/21).

### 6.1 S7: hard-kill recovery (`smoke_ppo_s0_6`)
- Fresh run with `--curriculum-offset 145000`, launched as in S1. At about 5,000 decisions (19.7 s after the 4,096
  heartbeat; estimate), `Stop-Process -Force` on the Python process.
- After the kill: pid dead; `status.json` still `running` at 4,096 with a stale `updated_at`; the only model in the
  folder was `checkpoints/ppo_4096_steps.zip`; no end record.
- Resume from `checkpoints/ppo_4096_steps.zip` to the end, `RESULT: DONE`. Claire's expectation holds exactly:

  | | expected | observed |
  |---|---|---|
  | reseed | 4097 | 4097 (seeded resets [4097]) |
  | final decisions | 10,240 | 10,240 |
  | trained_on | 8,192 | 8,192 (= 2048 x 4) |
  | discarded | 2,048 | 2,048 |
  | pending | 0 | 0 |

- Audit: **RESULT: PASS, 19/19** (`results/m5/s7_audit_smoke_ppo_s0_6.txt`). Launch 1's rollouts CSV has 2 rows;
  only 1 is in the lineage. The second row is the update made after the 4,096 checkpoint was saved, so it is
  excluded from the sums (320 gradient steps, 40 epochs).
- The resume launch ran 2-3x slower than usual (collect 97-145 s per rollout, updates 5-11 s). Another heavy
  application started on the PC three seconds before the kill and ran throughout (Inference from the process start
  time). Counts are unaffected; the timings of that launch are not representative. It shows why checklist item 5
  matters for the main run.

### 6.2 Killed launches in the lineage
- At the start of every launch, after the live-run lock passed, a launch with no end record is written to `run_meta`
  as `end_state: "killed"` with `last_heartbeat`, `last_num_timesteps` and `last_status_state` from `status.json`.
- A resume removes leftover `*_partial*` files in the run folder and logs each one (`--preflight-only` only lists
  them). S7 removed 2 planted files; the kill itself left none.
- `tools/audit_ppo_run.py`:
  - killed-launch branch: passes only if the next launch that trained resumed from a file under `checkpoints/` at or
    below the killed launch's last recorded `num_timesteps`;
  - lineage rule (`run_files.lineage_row_counts`): walking back from the end model, each launch contributes the
    updates between its own start and the point the next lineage launch resumed from. Rows past that point stay out.
- Unit tests: `KilledLaunchTest` (4 tests).

### 6.3 Live-run lock
- Preflight refuses (fresh and resume, smoke and main) when `status.json` says `starting` or `running` and that
  process is alive. A dead pid never blocks.
- Pid check (`run_files.process_start_time`): `OpenProcess` + `WaitForSingleObject(handle, 0)` + `GetProcessTimes`
  through `ctypes`, no new package. `status.json` now also records `pid_start` (process creation time), so a pid
  reused by another program after a reboot does not block a resume. Access denied counts as alive.
- Unit tests: `LiveRunLockTest` (decision table, plus a real child process that is alive and then killed).
- Live refusal cases S0.12 and S0.13 (`results/m5/s7_hard_kill.txt`): fresh and resume on `smoke_ppo_s0_6` while it
  was running were both refused with "the run is live".

### 6.4 Ctrl+C wording
`train_path_ppo.py` docstring and row D5 now say: stop only with the STOP file; Ctrl+C or closing the window kills
the run and saves nothing; recovery is resume from the last checkpoint (S7).

### 6.5 `learning_curve.py` stays in `evaluation/`
- `code_blobs` appears in the frozen files only at `run_protocol.py` line 68 (`def code_blobs`), line 354 (written
  into a fly run's meta) and line 759 (written into a report's output). It is never read back or compared.
  `protocol_stats.py` does not contain it.
- `report` (lines 661-692) refuses on: protocol hash, non-canonical PID run, realizations, consistency checks.
  `compare` (lines 772-808) compares flights and realizations entries. `reproduce` (lines 429-448) uses flights only.
- So no whole-dict comparison exists, and an extra file in `evaluation/` cannot cause a refusal or mismatch in M7.
- The canonical PID run already lists 6 files, not 5: the sixth is the historic `zero_residual_check.py`. New runs
  will list 7.
- The only reader of `code_blobs` is the unfrozen `learning_curve.py`, which compares just the 5 frozen files.

### 6.6 S8: torch origin evidence for evaluation flights (no protocol v2)
- `tools/check_eval_torch_origin.py` follows the frozen fly order: preflight, `torch` + `torch._dynamo`, the frozen
  adapter loads the model, all before SimulationApp; then SimulationApp with **no pin**, env, `bind_env`, one short
  flight (60 decisions, validation seed 400000, hover / training_range; no tracking metrics).
- Results (`results/m5/s8_eval_torch_origin_pid_ppo.json`, `..._pid_sac.json`):

  | Controller | Model | Batch actions before vs after | Torch modules outside kit torch | Torch modules first imported after the app | Result |
  |---|---|---|---|---|---|
  | pid_ppo | `smoke_ppo_s0_4` final | bit-identical (64 observations) | 0 | 0 | RESULT: PASS |
  | pid_sac | team `final_model.zip` | bit-identical (64 observations) | 0 | 0 | RESULT: PASS |

- Limits (Inference): this is one short flight per controller in a separate script, not the frozen runner itself.
  It shows the load-then-predict path imports no torch module after the app; it cannot prove that for every later
  code path.

### 6.7 D3 consequence for the main run
- Of the 30 checkpoints, two cannot be resumed from, because reseed = 0 + n + 1 is a forbidden seed:

  | Checkpoint | Reseed | Forbidden as | Use instead |
  |---|---|---|---|
  | `ppo_300000_steps.zip` | 300001 | eval seed | `ppo_250000_steps.zip` |
  | `ppo_400000_steps.zip` | 400001 | validation seed | `ppo_350000_steps.zip` |

- Recovery rule: use the previous checkpoint (at most 50,000 more decisions repeated, about 18 minutes).
- Both checkpoints are still saved and still evaluated; 400k is one of the three learning-curve points.

### 6.8 Tests and checks
74/74 tests (56 + 18 in `tests/test_ppo_training.py`) and checks 21/21, in BOTH the venv and Isaac python.bat.
