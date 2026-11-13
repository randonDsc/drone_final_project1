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
python -m unittest discover -s tests -p "test_*.py" -v
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
(`results/*.log`) are not committed.

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
  and after) and re-run the M1 checks and tests.

## Milestone 1 results (shared environment integration)

- Checks 21/21 (18 modules + 1 test + 2 assets). Tests 23/23 in the venv (Python 3.13) and in Isaac Python (3.12),
  also in a fresh CRLF clone.
- `PathSacEnv` builds from `residual_ppo/`: decimation 12 (120 Hz physics, 10 Hz RL), obs (128,), action (4,).
- **Total mass 1.618614 kg** (body 1.5 kg + 4 rotors); the HESAI lidar adds 0 kg (no rigid body, collider or mass).
- Zero residual (PID only): no failure and no motor saturation over 30 decisions; curriculum stage mapping correct;
  seeded resets are deterministic.

## Status

Milestone 1 complete. **M2 (PID-only baseline + evaluation protocol v1)** in progress: M2.1 (headless speed) done,
about 49 env decisions/s. Next: M2.2 (zero residual = PID). No PPO training code yet.
Step plans live in [`docs/`](docs/).
