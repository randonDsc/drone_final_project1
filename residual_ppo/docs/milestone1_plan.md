# Milestone 1 — Shared environment integration: step plan

Goal: a self-contained, verified copy of the team's PathSacEnv inside `residual_ppo\`, which builds,
resets and steps in Isaac Sim with the intended PID, and does not depend on `drone_team_main` at runtime.

Deliverable: `PathSacEnv` constructed from `residual_ppo\` and passing the checks in M1.5.

Each step ends with a STOP: report the results to Claire and wait for her go-ahead.
Paths below are absolute; `$ROOT = "C:\Users\tucla\Desktop\DSC180A"`.

---

## M1.0 Preflight (read-only)

**Why:** the last Isaac check printed nothing, and the branch is 2 commits behind. Both must be understood first.

1. Find Isaac Python:
   ```powershell
   # Directory: C:\Users\tucla\Desktop\DSC180A
   Get-ChildItem C:\isaacsim -Filter python.bat -Recurse -Depth 2 | Select-Object FullName
   ```
   Then run the package check with that path, using `;` separated prints so a missing package shows a traceback:
   `& "<python.bat>" -c "import sys; print(sys.version); import numpy, gymnasium, stable_baselines3, torch; print(numpy.__version__, gymnasium.__version__, stable_baselines3.__version__, torch.__version__, torch.cuda.is_available())"`.
   If a package is missing, STOP and propose `& "<python.bat>" -m pip install <pkg>`. Do not install without approval.
2. Inspect incoming commits without changing anything:
   ```powershell
   # Directory: C:\Users\tucla\Desktop\DSC180A
   git -C drone_final_project1 fetch origin
   git -C drone_final_project1 log --oneline --stat HEAD..origin/claire/drone-simulation
   ```
   If they don't touch `flight_log.txt` or the modified `.pyc` files, propose `git -C drone_final_project1 pull --ff-only`.
3. Confirm the team worktree is still at the baseline and clean in `drone_code`:
   `git -C drone_team_main rev-parse HEAD` (expect `61d2722…`) and `git -C drone_team_main status --short -- drone_code`.

**Pass:** `python.bat` path known, all 4 packages import in Isaac Python, branch up to date (or a reason it isn't), team at 61d2722.

## M1.1 Commit the current `residual_ppo\` baseline

**Why:** `residual_ppo\` is entirely untracked. Commit it before adding 16 more files, so every later step is a reviewable diff.

- Add `residual_ppo\.gitignore` containing `__pycache__/` (and later model/run outputs).
- Stage explicit paths only: `git -C drone_final_project1 add residual_ppo/controller.py residual_ppo/sim_config.py residual_ppo/README.md residual_ppo/.gitignore`.
- Show `git -C drone_final_project1 status --short` and the proposed message to Claire; commit only after approval. Don't push yet.

## M1.2 `tools\sync_shared.py` + manifest

**Why a script instead of manual copying:** it makes the copy reproducible and records which commit and which bytes
we use. `--check` can then prove later that nobody edited a shared file. That is our evidence for "same environment as SAC".

`residual_ppo\tools\sync_shared.py` (runs with the normal venv; standard library only):
- `SOURCE = <DSC180A>\drone_team_main\drone_code`, `EXPECTED_COMMIT = "61d2722"`, `MODULES =` the 18 names in CLAUDE.md.
- Refuse to run if the team HEAD doesn't start with EXPECTED_COMMIT, or if `git status --porcelain -- drone_code` is non-empty there.
- Default mode: copy each `<name>.py` with `shutil.copy2`. Refuse to overwrite a destination whose hash differs from the
  manifest unless `--force` is passed. Write `SHARED_MANIFEST.json` = {source_repo, source_commit (full SHA), copied_at, files: {name: sha256}}.
- `--check` mode: re-hash `residual_ppo\<name>.py` against the manifest AND against the team source; print a table; exit 1 on any mismatch.
- `controller.py` and `sim_config.py` already exist: they must hash-match the source (they do), and are included in the manifest.
- Read-only toward `drone_team_main` (no git writes, no file writes there).

Run:
```powershell
# Directory: C:\Users\tucla\Desktop\DSC180A\drone_final_project1\residual_ppo
Set-Location C:\Users\tucla\Desktop\DSC180A\drone_final_project1\residual_ppo
& C:\Users\tucla\Desktop\DSC180A\.venv\Scripts\python.exe tools\sync_shared.py
& C:\Users\tucla\Desktop\DSC180A\.venv\Scripts\python.exe tools\sync_shared.py --check
```
**Pass:** 18 files present, `--check` reports 18/18 OK.

## M1.3 Scene files

**Why:** the env opens `<repo root>\sim_scene\dij3d_boxes.usd`, which needs `dij3d.usd` next to it and `..\sim_asset\iris.usd`.
They live at the repo root, not in `residual_ppo\`. This is a deliberate, documented exception to "everything under residual_ppo"
(moving them would break the USD-relative references). Ask Claire before doing it.

- `Test-Path` the two files in `drone_final_project1\sim_scene\`. If missing, copy them from `drone_team_main\sim_scene\`.
- Compare `Get-FileHash` of `sim_asset\iris.usd` across both worktrees (expected identical).
- Do NOT replace `sim_scene\drone1.usd` (it differs from main but the path env does not use it).
- Add the two USDs' hashes to the manifest under `assets` (extend sync_shared.py rather than hand-editing JSON).

## M1.4 Offline tests (no Isaac)

**Why:** prove the copied pure-Python logic (PID adapter, reward, references, residual budget) behaves like the team's,
and that imports resolve to `residual_ppo\`, before paying Isaac's startup cost.

1. Copy the team's `drone_code\tests\test_path_sac.py` verbatim to `residual_ppo\tests\` (its `parents[1]` resolves to `residual_ppo`).
   Add it to the manifest's tracked files.
2. New `residual_ppo\tests\test_import_paths.py`: insert `residual_ppo` at `sys.path[0]`. For all 18 modules use
   `importlib.util.find_spec(name).origin` (this does not execute the module, so it works for `drone_sim` without Isaac) and
   assert the parent folder is `residual_ppo`. Also assert no entry of `sys.path` ends with `drone_code`. Also assert that
   `controller.py`'s hash equals the manifest.
3. Run with BOTH interpreters:
   ```powershell
   # Directory: C:\Users\tucla\Desktop\DSC180A\drone_final_project1\residual_ppo
   & C:\Users\tucla\Desktop\DSC180A\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
   & "<isaac python.bat>" -m unittest discover -s tests -p "test_*.py" -v
   ```
**Pass:** test_path_sac 15/15 and import-path tests OK in both. (Already observed 15/15 in a Linux replica: re-confirm on Windows.)

## M1.5 Isaac environment smoke test

**Why:** first real proof that the copied stack builds the simulator, loads the scene, and produces valid observations and rewards.
This is NOT PID flight validation (that is M2) and trains nothing.

New `residual_ppo\tests\isaac_env_smoke.py` (not named `test_*.py`, so unittest discovery never launches Isaac):
1. `sys.path.insert(0, residual_ppo)`; `from isaacsim import SimulationApp`; `app = SimulationApp({"headless": True})`.
2. Then `import path_sac_env, path_sac_config, controller` and print `__file__` of each plus the resolved scene path
   (`Path(hover_env.__file__).resolve().parents[1] / hover_ppo_config.SCENE`, and whether it exists). Abort if any file is outside `residual_ppo`.
3. `env = path_sac_env.make_path_sac_env(app)`; print spaces, `env.core.decimation` (expect 12), `env.sim.dt` (expect 1/120), total mass, and residual limits.
4. `obs, info = env.reset(seed=123)`: assert shape (128,), float32, finite. Print info (wind_mean, wind_seed, route_type, stage, route_seconds).
5. Step `np.zeros(4, np.float32)` for 30 decisions or until the episode ends. Each step: assert finite obs/reward,
   print reward, terminated/truncated, failure_reason, position_error, saturated_steps/physics_steps.
6. Curriculum check: for step in (0, 149_999, 150_000, 400_000, 800_000, 1_500_000): `env.set_training_step(step)`; reset; print `info["stage"]`.
7. Determinism check: two `reset(seed=7)` calls must give identical `wind_seed`, `route_type`, `route_seconds` and initial obs.
8. `finally:` `env.close()` then `app.close()`. Must not write anything under `runs\`.

Run from the repo root (mirrors how the team runs scripts):
```powershell
# Directory: C:\Users\tucla\Desktop\DSC180A\drone_final_project1
Set-Location C:\Users\tucla\Desktop\DSC180A\drone_final_project1
& "<isaac python.bat>" residual_ppo\tests\isaac_env_smoke.py 2>&1 | Tee-Object -FilePath residual_ppo\results\m1_env_smoke.log
```
**Pass:** env builds, all module paths inside `residual_ppo`, obs/reward finite, stage mapping
short_slow/short_slow/turns/climbs_loops/full/full, reset deterministic, process exits cleanly.
With zero action under PID, an early failure would be a red flag to investigate in M2, not something to "fix" in shared code.

## M1.6 Close out

- `sync_shared.py --check` still 18/18 OK (proves nothing shared was edited while debugging).
- Commit (explicit paths): tools, manifest, shared modules, tests, docs, scene files (if approved). Don't commit the smoke log unless Claire wants it.
- Update the README in `residual_ppo` (what's shared, from which commit, how to verify) and the "Current status" in CLAUDE.md → M2.

## Explicitly out of scope for M1

PPO config/model/callbacks/training (M3–M4), PID-only flight validation and the zero-residual equivalence test (M2),
new eval wind profiles (M7), any edit to a shared module, anything in `drone_team_main` or `drone_claire_backup`.
