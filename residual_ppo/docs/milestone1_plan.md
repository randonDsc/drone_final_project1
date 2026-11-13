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

**Identity rule: a shared file's identity is its git blob ID at 61d2722, NOT its on-disk SHA-256.**
Why: `core.autocrlf=true` in `drone_final_project1` checks text files out with CRLF line endings, so the working copy's
SHA-256 (`controller.py`: 1FA0F757…) differs from the LF bytes stored in git / shown on GitHub (da20b7f6…), even though the
content is identical. The git blob ID (`controller.py`: e34bd61) is computed after line-ending normalization, so it is the same
for the team's file, our CRLF working copy, and an LF copy. It is also exactly what gets committed.

`residual_ppo\tools\sync_shared.py` (runs with the normal venv; standard library only, calls `git` via `subprocess`):
- `TEAM_REPO = <DSC180A>\drone_team_main`, `SOURCE_DIR = "drone_code"`, `EXPECTED_COMMIT = "61d2722"`,
  `DEST_REPO = <DSC180A>\drone_final_project1`, `MODULES =` the 18 names in CLAUDE.md.
- Resolve the full source SHA with `git -C <TEAM_REPO> rev-parse 61d2722^{commit}`; refuse to run if it doesn't resolve.
  Also refuse if the team HEAD doesn't start with EXPECTED_COMMIT (guards against baseline confusion). The team working-tree
  dirty check is no longer needed for correctness, because bytes come from the commit object, not the team's working tree.
- Source bytes: `git -C <TEAM_REPO> cat-file blob <full_sha>:drone_code/<name>.py`, captured as raw bytes with
  `subprocess.run(..., capture_output=True, check=True).stdout` and written with `open(dest, "wb")`. (Same bytes as
  `git show 61d2722:drone_code/<file>`; `cat-file blob` is used because it never applies textconv. Never copy via PowerShell
  `>` redirection, which re-encodes text.) Source blob ID: `git -C <TEAM_REPO> rev-parse <full_sha>:drone_code/<name>.py`.
- Default mode, per file: if the destination exists and `git -C <DEST_REPO> hash-object residual_ppo/<name>.py` equals the
  source blob ID, leave it untouched (this covers the existing CRLF `controller.py` and `sim_config.py`). If it exists with a
  different blob ID, refuse unless `--force`. Otherwise write it. After writing, re-run `hash-object` and assert it equals the source blob.
- Write `SHARED_MANIFEST.json` = {`source_repo`, `source_path`: "drone_code", `source_commit`: full 40-char SHA,
  `identity`: "git blob id (git hash-object, autocrlf-normalized)", `copied_at`, `files`: {`"<name>.py"`: full 40-char blob ID}}.
- `--check` mode: for each manifest entry, compare `git -C <DEST_REPO> hash-object residual_ppo/<file>` against the manifest
  blob ID, and also against `rev-parse <source_commit>:drone_code/<file>`. Print a table (file, manifest, local, source, OK/FAIL).
  Exit 1 on any mismatch, missing file, or missing manifest entry. `hash-object` must run inside `DEST_REPO` without
  `--no-filters`, so the autocrlf normalization applies.
- `controller.py` and `sim_config.py` already exist and already match (e34bd61, 4d5d59c; committed in 2aaf0cd); they are included in the manifest.
- Read-only toward `drone_team_main` (only `rev-parse` / `cat-file`; no git writes, no file writes there).

Run:
```powershell
# Directory: C:\Users\tucla\Desktop\DSC180A\drone_final_project1\residual_ppo
Set-Location C:\Users\tucla\Desktop\DSC180A\drone_final_project1\residual_ppo
& C:\Users\tucla\Desktop\DSC180A\.venv\Scripts\python.exe tools\sync_shared.py
& C:\Users\tucla\Desktop\DSC180A\.venv\Scripts\python.exe tools\sync_shared.py --check
```
**Pass:** 18 files present, `--check` reports 18/18 OK (blob IDs match manifest and 61d2722), and
`git -C drone_final_project1 status --short` shows only new untracked files under `residual_ppo\` (no `M` on controller.py / sim_config.py).

**Negative test (must also pass):** proves `--check` actually detects an edit and the sync refuses to hide it.
```powershell
# Directory: C:\Users\tucla\Desktop\DSC180A\drone_final_project1\residual_ppo
Set-Location C:\Users\tucla\Desktop\DSC180A\drone_final_project1\residual_ppo
$py = "C:\Users\tucla\Desktop\DSC180A\.venv\Scripts\python.exe"
Add-Content -Path wind.py -Value ""          # (1) tamper: append one blank line
& $py tools\sync_shared.py --check            # (2) must print FAIL (blob mismatch) for wind.py, 17/18, exit 1
& $py tools\sync_shared.py                    # (3) must REFUSE, write nothing, exit 1
& $py tools\sync_shared.py --force            # (4a) restores wind.py from 61d2722, exit 0
& $py tools\sync_shared.py --check            # (4b) must be 18/18 OK again, exit 0
```
Passed 2026-11-12: (1) wind.py blob d6eaf33 → 28980ed; (2) FAIL for wind.py only, exit 1; (3) REFUSING, wind.py and
manifest unchanged, exit 1; (4) wind.py OVERWRITTEN, manifest unchanged, then 18/18 OK, exit 0.

## M1.3 Scene files

**Why:** the env opens `<repo root>\sim_scene\dij3d_boxes.usd` (`hover_ppo_config.SCENE`), which references `../sim_asset/iris.usd`.
`dij3d_boxes.usd` has NO sublayers and does not load `dij3d.usd` or `drone1.usd`. Its only `authoring_layer = "./dij3d.usd"` is
metadata: `sim_scene/build_box_scene.py` generated it offline from `dij3d.usd`. The files live at the repo root, not in `residual_ppo\`.
This is a deliberate, documented exception to "everything under residual_ppo" (moving them would break the USD-relative reference).

**Evidence (Confirmed 2026-11-12, Isaac python.bat + headless SimulationApp, USD 0.25.11):**
`UsdUtils.ComputeAllDependencies(drone_team_main/sim_scene/dij3d_boxes.usd)` → 6 layers, 10 assets, 0 unresolved.
Local: `sim_scene/dij3d_boxes.usd`, `sim_asset/iris.usd`, plus `kit/mdl/core/Base/OmniPBR.mdl` from the Isaac install.
Remote (cannot be pinned by the manifest; see CLAUDE.md open issues): HESAI XT32 lidar layers (4 × https, Isaac 6.1 S3 content)
and 9 material files over http (Carpet_Gray, Linen_Blue .mdl/.png). Text export of dij3d_boxes.usd: no `drone1`, no sublayers.

Same identity rule as M1.2: git blob ID at 61d2722, not `Get-FileHash` (binary USDC, so autocrlf does not convert it, but
one rule for every shared file). Expected blobs: `sim_scene/dij3d_boxes.usd` = 8af9451d89795ebeae3b014b73f55e9828c866d9,
`sim_asset/iris.usd` = 003d97f30dfd6bb196e018a29a1743a1228528a3.

- `sync_shared.py` gets an `ASSETS` list (repo-relative paths, same path in the source commit), recorded in the manifest as
  `"assets": {path: {"blob_id": ..., "mode": "copy" | "check"}}` and covered by `--check`:
  - `sim_scene/dij3d_boxes.usd`, mode `copy`: written from `git cat-file blob 61d2722:sim_scene/dij3d_boxes.usd`.
  - `sim_asset/iris.usd`, mode `check`: already on Claire's branch; verified, never written (not even with `--force`).
- Do NOT take `sim_scene/dij3d.usd` (only an input of `build_box_scene.py`, not loaded at runtime).
- Do NOT replace `sim_scene/drone1.usd`. Claire's version (811a546) differs from main's (1ffb83e: adds a rigid-body
  `/World/Cube_01` at (-3.5, 0, 0.04), a lidar Replicator RenderProduct, and viewport changes). It is used only by Claire's old
  `drone_code/motor_physics.py`, not by the path env.

**Pass:** `--check` reports 20/20 OK (18 modules + 2 assets). Asset tamper test: append a byte to `dij3d_boxes.usd` →
`--check` FAIL for it only (19/20, exit 1) → sync refuses (exit 1) → `--force` restores → 20/20 OK.
Passed 2026-11-12 (tampered blob a7e5dfd; restored to 8af9451). `git status` shows `sim_scene/dij3d_boxes.usd` as new, no change to `drone1.usd` or `iris.usd`.

## M1.4 Offline tests (no Isaac)

**Why:** prove the copied pure-Python logic (PID adapter, reward, references, residual budget) behaves like the team's,
and that imports resolve to `residual_ppo\`, before paying Isaac's startup cost.

1. Write the team's `61d2722:drone_code/tests/test_path_sac.py` verbatim to `residual_ppo\tests\` via sync_shared.py
   (`cat-file blob`, same identity rule; its `parents[1]` resolves to `residual_ppo`). Add it to the manifest with its blob ID.
2. New `residual_ppo\tests\test_import_paths.py`: insert `residual_ppo` at `sys.path[0]`. For all 18 modules use
   `importlib.util.find_spec(name).origin` (this does not execute the module, so it works for `drone_sim` without Isaac) and
   assert the parent folder is `residual_ppo`. Also assert no entry of `sys.path` ends with `drone_code`. Also assert that
   `controller.py`'s git blob ID (`git hash-object`) equals the manifest.
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
3. `env = path_sac_env.make_path_sac_env(app)`; print spaces, `env.core.decimation` (expect 12), `env.sim.dt` (expect 1/120), and residual limits.
   **Mass report:**
   - Per-body masses from `env.sim.vehicle_bodies.prim_paths` / `get_masses()`, and `env.sim.total_mass` (their sum; this is what
     the PID hover thrust and residual limits are scaled by).
   - HESAI lidar (`/World/iris/body/HESAI_XT32_SD10`, from `sensor_config.LIDAR["path"]`): for every prim under it, report whether it has
     `UsdPhysics.RigidBodyAPI`, `CollisionAPI`, `MassAPI` (and authored mass/density). `hesai_rigid_body_mass_kg` = summed mass of
     vehicle bodies under that path (0 if the lidar is not its own rigid body). Also report `/World/iris/body`'s authored `MassAPI`
     mass: if it has none, HESAI colliders would add density-based mass to `body` itself, so flag `hesai_colliders > 0` in that case.
   - Write `residual_ppo\results\m1_env_config_snapshot.json`: source commit, manifest `--check` result, package versions + torch device,
     spaces, decimation, dt, residual limits, `total_mass_kg`, per-body masses, `hesai_rigid_body_mass_kg`, HESAI API summary, body authored mass.
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
short_slow/short_slow/turns/climbs_loops/full/full, reset deterministic, process exits cleanly,
total mass and HESAI mass contribution reported and saved in the config snapshot.
With zero action under PID, an early failure would be a red flag to investigate in M2, not something to "fix" in shared code.

## M1.6 Close out

- `sync_shared.py --check` still 20/20 OK (proves nothing shared was edited while debugging).
- Commit (explicit paths): tools, manifest, shared modules, tests, docs, config snapshot. Don't commit the smoke log unless Claire wants it.
- Update the README in `residual_ppo` (what's shared, from which commit, how to verify) and the "Current status" in CLAUDE.md → M2.

## Explicitly out of scope for M1

PPO config/model/callbacks/training (M3–M4), PID-only flight validation and the zero-residual equivalence test (M2),
new eval wind profiles (M7), any edit to a shared module, anything in `drone_team_main` or `drone_claire_backup`.
