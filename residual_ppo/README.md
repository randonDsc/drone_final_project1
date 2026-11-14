# Residual PPO for Quadrotor Path Following

PPO-based residual controller on top of PID, for wind-disturbed quadrotor trajectory tracking in NVIDIA Isaac Sim.
Compared against PID alone and PID + SAC (the team's Path SAC workflow).

**Guiding principle:** same drone, same PID, same physics, same observation/action/reward definitions and same wind
curriculum as the team's Path SAC. Only the RL algorithm differs.

## What is shared, and from where

Everything the environment needs is copied **byte-for-byte** from team `main` at commit
`61d2722b260dcb467ab487cb24bf22f61d410a65` and recorded in [`SHARED_MANIFEST.json`](SHARED_MANIFEST.json):

- **18 modules** (top level of `residual_ppo/`): the exact import closure of `PathSacEnv` (16) plus
  `hover_sac_policy` / `hover_sac_config` (only for loading a SAC checkpoint for comparison).
  They must stay at the top of `residual_ppo/`: `hover_env.py` finds the scene at `../sim_scene/`.
- **1 team test**: `tests/test_path_sac.py`.
- **2 scene assets** (repo root): `sim_scene/dij3d_boxes.usd` (copied) and `sim_asset/iris.usd` (verified only).
  The scene also loads remote content from NVIDIA's S3 (HESAI lidar, materials), which the manifest cannot pin.

**Shared files are never edited.** PPO-specific behavior goes in new files (wrappers, callbacks, configs).

Identity is the **git blob ID**, not the on-disk SHA-256: with `core.autocrlf=true`, Windows checkouts are CRLF,
so on-disk hashes differ from the LF bytes in git even when the content is identical.

## Verify

Run from `residual_ppo/` with the normal venv or Isaac's `python.bat`:

```powershell
python tools\sync_shared.py --check-local   # local files vs manifest (no team repo needed)
python tools\sync_shared.py --check         # local files vs manifest AND team commit 61d2722
python -m unittest discover -s tests -p "test_*.py" -v   # 56 tests (team test + ours), both interpreters
```

- `--check` and the source test look for the team worktree `drone_team_main` **next to** this repo.
  Anywhere else (e.g. a fresh clone), set **`SYNC_SHARED_TEAM_REPO`** to its path:
  `$env:SYNC_SHARED_TEAM_REPO = "<path>\drone_team_main"`. Without it (and no sibling folder), the source test is
  skipped and the skip reason names the variable. **If the variable is set but points to a wrong path, the test fails
  rather than skips**, so a typo cannot silently hide the source check.
- `python tools\sync_shared.py` (no flag) re-copies missing shared files; `--force` restores edited ones.

Isaac environment smoke test (needs Isaac Sim; run from the repo root with Isaac's `python.bat`):

```powershell
& "<Isaac Sim>\python.bat" residual_ppo\tests\isaac_env_smoke.py *> residual_ppo\results\m1_env_smoke.log
```

It writes `results/m1_env_config_snapshot.json` and `results/m1_env_smoke_summary.txt`. Full Isaac logs
(`results/**/*.log`) are not committed.

## Evaluation protocol v1 (frozen)

The protocol is frozen in [`docs/eval_protocol_v1.md`](docs/eval_protocol_v1.md): 30 eval seeds
(300000–300029) × 6 scenarios × 6 wind profiles = 1,080 flights per controller. Profiles: `still_air` (ours, reported,
not primary), the team's `training_range`, `unseen_gusts`, `outside_range` (**primary endpoints**), and our
evaluation-only `strong_range` / `beyond_strong` (secondary). Its identity is the protocol hash
`d3f677dc37e2d347e87c0098eaab6bbea3c28f3ee10f2bfe84ebc9a5b745278e` plus the git blob IDs of the five evaluation files
listed there; **any change to them is protocol v2** and requires re-running every controller.

Verify the frozen code (from the repo root):

```powershell
git hash-object residual_ppo/evaluation/eval_protocol_v1.py residual_ppo/evaluation/protocol_stats.py `
    residual_ppo/evaluation/run_protocol.py residual_ppo/evaluation/run_common.py residual_ppo/evaluation/controllers.py
# must print the blob IDs in docs/eval_protocol_v1.md; every results file also stores them in meta.code_blobs
```

Run the protocol (from the repo root; flying needs Isaac's `python.bat`, the analysis modes run in either interpreter):

```powershell
# 1,080 flights for one controller (PID: ~38 min on this PC); checks 21/21 run first, check 0 proves diagnostics are passive
& "<Isaac Sim>\python.bat" residual_ppo\evaluation\run_protocol.py fly --controller pid --seeds 300000-300029 `
    --check0 --out residual_ppo/results/eval/eval-v1/pid *> residual_ppo\results\eval\eval-v1\pid.log
# matched-realization check across all profiles + per-profile table
python residual_ppo\evaluation\run_protocol.py baseline-table --run residual_ppo/results/eval/eval-v1/pid --name <name>
# E5 report of a controller run vs the canonical PID run (refuses mismatched realizations or a different protocol hash)
python residual_ppo\evaluation\run_protocol.py report --controller-run residual_ppo/results/eval/eval-v1/<controller> `
    --out residual_ppo/results/eval/eval-v1/<controller>/report_vs_pid
# bit-for-bit comparison of two runs on their shared flights
python residual_ppo\evaluation\run_protocol.py compare --reference <run dir> --candidate <run dir> --name <name> --purpose "<text>"
```

The canonical PID run is `results/eval/eval-v1/pid/` (its `realizations.json` is the cross-controller reference).

**Learned controllers** (`evaluation/controllers.py`; rules in E3b of the frozen protocol):

```powershell
& "<Isaac Sim>\python.bat" residual_ppo\evaluation\run_protocol.py fly --controller pid_sac `
    --model residual_ppo/models/sac_team/flight1/final_model.zip --seeds 300000-300029 --out residual_ppo/results/eval/eval-v1/pid_sac
& "<Isaac Sim>\python.bat" residual_ppo\evaluation\run_protocol.py fly --controller pid_ppo `
    --model <run folder>\final_model.zip --seeds 300000-300029 --out residual_ppo/results/eval/eval-v1/pid_ppo
```

- The model is loaded **before** SimulationApp. A run is refused (before flying) if the contract check fails or the
  model's spaces differ from the env's, and stops at the first reset whose realization differs from the PID reference
  (`--reference-realizations`, default the canonical PID run; validation-seed runs pass a PID run on those seeds).
- `pid_sac`: `path_sac_config.incompatibility(model, check_training=True)` must be None (needs the team's
  `flight_contract.json` in the model's run folder).
- `pid_ppo`: **`ppo_contract.json` must sit next to the model file** with a `"policy"` group equal to
  `path_sac_config.policy_contract()`. **M4 note: PPO training must write `ppo_contract.json` next to every model it
  saves (final model and every checkpoint folder)**, or the evaluation refuses the model.
- **A run counts as successful only if its log contains `RESULT: DONE` and its output files exist — never by exit
  code.** Isaac runs exit with code 0 even when they raise (Inference: `SimulationApp.close()` ends the process with
  0 before the exception propagates). A failed or refused run logs `RESULT: FAIL` and writes no results files.

**M4 to-dos (from M2.5):**
- The PPO training script writes `ppo_contract.json` next to every model it saves (final and checkpoints), and an
  explicit `status.json` (`running` / `done` / `failed` + last step), because exit codes cannot be trusted.
- Before validating checkpoints (E6), fly PID once on the validation seeds 400000–400009 (all scenarios and profiles)
  as the validation reference; validation runs pass it as `--reference-realizations`.

**Reproducing earlier M2 steps.** M2.2 (`evaluation/zero_residual_check.py`) and the M2.3/M2.4 results were produced
under the earlier protocol hash `18658b9c…` (before the D4 profiles were added). **To reproduce M2.2, check out commit
`714f64f`**; on the current code its runner refuses (hash mismatch). M2.4b showed the 720 flights of the four original
profiles are bit-identical under both hashes.

## Environment

- Isaac Sim 6.2.0 standalone, Isaac Python 3.12.14: numpy 2.3.1, gymnasium 1.2.3, stable_baselines3 2.7.0,
  torch 2.14.1+cpu. **PPO runs on CPU** (`device="cpu"`).
- Record package versions from the imported modules (`torch.__version__`), not `importlib.metadata`, which reports
  a stale torch version once SimulationApp has started.

### ENV-1: our software versions differ from the team's (open; current choice: keep ours)

| Item | Team (README §3) | This branch |
|---|---|---|
| Isaac Sim | 6.1.0 | 6.2.0 |
| Python | 3.12.13 | 3.12.14 |
| stable-baselines3 | 2.9.0 | 2.7.0 |
| gymnasium | 1.4.0 | 1.2.3 |
| torch | 2.11.0 + CUDA 12.8 (Isaac's bundled copy) | 2.14.1+cpu (pip install into Isaac Python) |
| numpy | 2.3.1 | 2.3.1 |
| GPU | RTX 4070 SUPER, driver 617.14 | RTX 5070 Ti, driver 617.42 |

- **Why two torches:** installing stable-baselines3 with pip pulled torch from PyPI (the default Windows build is
  CPU-only), because pip cannot see Isaac's bundled torch 2.11.0+cu128, which only goes on the path once
  SimulationApp starts. Both copies are on disk. The install is left as it is.
- **Consequence:** scripts that build an SB3 model import `torch` (and `torch._dynamo`) **before** SimulationApp;
  otherwise the two copies mix and the optimizer import fails.
- **Comparison rule:** every number we compare (PID, the team's SAC checkpoint re-evaluated, our PPO) is produced
  in this environment with evaluation protocol v1. The team README's numbers are context only, never a comparison
  baseline. Every results file records the Isaac, SB3, gymnasium and torch versions.
- **Check before M3:** load and evaluate the team's `final_model.zip` under SB3 2.7.0. If that fails or warns, make
  one planned change (remove the extra torch, upgrade to SB3 2.9.0 and gymnasium 1.4.0, with `pip freeze` before
  and after) and re-run the M1 checks and tests. **Load check done (2026-11-13):** `SAC.load(..., device="cpu")` under
  SB3 2.7.0 succeeds with 0 warnings and predicts deterministically; the full evaluation is M7. The checkpoint comes
  from Hugging Face `pcldsc/Iris_drone_residual_correction_SAC` (provenance in `models/sac_team/SOURCE.json`; binaries
  gitignored).

## Milestone 1 results (shared environment integration)

- Checks 21/21 (18 modules + 1 test + 2 assets). Tests 23/23 in the venv (Python 3.13) and in Isaac Python (3.12),
  also in a fresh CRLF clone.
- `PathSacEnv` builds from `residual_ppo/`: decimation 12 (120 Hz physics, 10 Hz RL), obs (128,), action (4,).
- **Total mass 1.618614 kg** (body 1.5 kg + 4 rotors); the HESAI lidar adds 0 kg (no rigid body, collider or mass).
- Zero residual (PID only): no failure and no motor saturation over 30 decisions; curriculum stage mapping correct;
  seeded resets are deterministic.

## Milestone 2 results (PID-only baseline + evaluation protocol v1)

- **M2.1** headless speed: about 49 env decisions/s; eval ≈ 0.6 h per controller.
- **M2.2** zero residual = PID: bit-exact (noise floor 0; 72,000 physics steps compared, max difference 0).
- **M2.3** pilot (team's 10 seeds): reproduces the team README §2 PID-only rows (RMSE within 0.08%, failure counts
  exact).
- **M2.4** PID on 30 seeds; pilot flights and a fresh-process re-run bit-identical. **H1 rejected:** about 87% of PID's
  training_range MSE is due to wind, not reference lag (wind_share_MSE 0.868, 95% CI [0.808, 0.906]).
- **M2.4b** D4 profiles added; 1,080-flight PID baseline (protocol `d3f677dc…`):

  | profile | status | RMSE [m] | failed / 180 | median decisions |
  |---|---|---|---|---|
  | still_air | reported | 0.183 | 0 | 154 |
  | training_range | primary | 0.483 | 0 | 154 |
  | unseen_gusts | primary | 0.484 | 0 | 154 |
  | outside_range | primary | 1.641 | 110 | 62 |
  | strong_range | secondary | 1.651 (flown portion only) | 179 | 23 |
  | beyond_strong | secondary | 1.604 (flown portion only) | 180 | 18 |

- **M2.5** PPO and SAC controller adapters (contract + space checks, realization guard, loaded before SimulationApp;
  smoke-tested on a validation seed with the team SAC checkpoint and an untrained PPO), then the protocol frozen with
  the reporting rules (exact McNemar paired failure test, exact binomial failure CIs, RMSE never interpreted alone when
  there are failures); PID baseline re-flown with the final code (bit-identical); PID-vs-PID dry run of the report.

## Status

Milestones 1 and 2 complete. **Next: M3** (PPO training setup and smoke run). Step plans live in [`docs/`](docs/).
