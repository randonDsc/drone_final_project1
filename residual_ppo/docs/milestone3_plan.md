# Milestone 3: PPO training setup, verification and timing probe

Owner: Claire (DSC 180A). Design confirmed by Claire on 2026-11-14. ent_coef and target_kl remain provisional.
Guiding principle: the drone, PID, physics, observation/action/reward definitions and wind curriculum are the same
as the team's Path SAC. Only the RL algorithm differs.

Labels: **Confirmed** (ran or read code), **Documented**, **Inference**, **Proposed**.

## 1. Files

| File | Purpose |
|---|---|
| `config/ppo_config.py` | All SB3 PPO kwargs, policy_kwargs and budget. Asserts gamma == `path_sac_config.SAC["gamma"]` and n_steps % batch_size == 0 at import. Contract = `{"policy": policy_contract(), "training": {...}}`. No torch at import. |
| `config/experiment_config.py` | Run layout, run ids, checkpoint rule, `decisions_trained_on()`, forbidden reset seeds (eval + validation). |
| `training/ppo_setup.py` | Shared build helpers, reused by the M4 training script: torch-first import, `SeedGuard`, env stack, model, curriculum callback, gradient-step counter, parameter counts, versions, torch-origin report. |
| `training/verify_ppo_setup.py` | `--mode verify` runs V1–V8 (one rollout, no update). `--mode probe` runs the M3 timing probe. |

## 2. PPO decisions (P1–P6, confirmed)

SB3 2.7.0 `PPO("MlpPolicy", venv, **ppo_config.model_kwargs())`:

| Key | Value | Note |
|---|---|---|
| n_steps / batch_size / n_epochs | 2048 / 256 / 10 | 8 minibatches per epoch; at most 80 gradient steps per rollout |
| learning_rate | 3e-4, constant | same as SAC |
| gamma | 0.99 | asserted equal to SAC gamma; the failure penalty is cost_cap/(1−γ) |
| gae_lambda / clip_range | 0.95 / 0.2 | |
| ent_coef (P1) | 0.001, **provisional** | |
| target_kl (P2) | 0.02, **provisional** | SB3 stops a rollout's epochs once approx_kl > 0.03; gradient steps are counted per rollout |
| P3 defaults, written out | normalize_advantage True, vf_coef 0.5, max_grad_norm 0.5, clip_range_vf None, use_sde False, ortho_init True | a future SB3 default change cannot alter the run |
| device / seed | cpu / 0 | |
| budget | TOTAL_DECISIONS 1,500,000 → 733 rollouts = **1,501,184 decisions** | at most 58,640 gradient steps |
| net_arch (P4) | pi [256,256], vf [256,256], separate | actor 99,848, critic 99,073 parameters |
| activation (P5) | ReLU | matches the team SAC actor; SB3's PPO default is Tanh |
| log_std_init (P6) | −1.0 (σ = 0.368) | expected clip rate: 0.66% of components, 2.6% of actions (σ = 1: 31.7% / 78.3%) |

**Correction found while implementing (Confirmed, SB3 2.7.0 `ppo.py`):** SB3's `_n_updates` (logged as `train/n_updates`)
counts **epochs**, not minibatch gradient steps (`_n_updates += 1` once per epoch). Runs therefore record both:
`sb3_n_updates` (epochs, at most 7,330 for 1.5M) and `gradient_steps`, counted by an optimizer step hook (at most
58,640).

**E7b addition (decided by Claire, 2026-11-14):** training cost is compared by **gradient steps**:
- PPO's count comes from the optimizer step hook (`ppo_setup.count_gradient_steps`);
- SAC's is about 1.495M (`_n_updates` 1,494,999 in the checkpoint);
- SB3 epochs are listed alongside (PPO `sb3_n_updates`).

This adds to the frozen E7b in `eval_protocol_v1.md`, which still requires the exact training decisions and the
wall-clock hours with hardware and device. The frozen doc is not edited.

## 3. Run layout and checkpoint rule (confirmed)

- Runs go in `residual_ppo/runs/ppo_path/<run_id>/`. Model zips are gitignored (`runs/**/*.zip`); small JSON and CSV
  files can be committed.
  - Main run id: `ppo_s0_1p5m`.
  - M5 smoke runs: `smoke_ppo_s0_<n>`.
- Each run folder holds:
  - `final_model.zip` + `ppo_contract.json`;
  - `checkpoints/ppo_<steps>_steps.zip` + `checkpoints/ppo_contract.json`;
  - `status.json` (running / done / failed + last step);
  - `run_meta.json` (versions, device, total_mass_kg, HESAI mass, hardware, wall clock, decisions, gradient steps,
    SB3 epochs, early stops);
  - `logs/` (SB3 `progress.csv`, `monitor.csv`).
- **Checkpoints:** every 50,000 decisions, saved when `num_timesteps % 50_000 == 0`. That gives 30 checkpoints,
  including exactly 150k, 400k, 800k and 1.5M, the same decision counts as SAC's. Saves happen mid-rollout, so each
  checkpoint records `num_timesteps`, `sb3_n_updates`, `gradient_steps` and `decisions_trained_on`.
  - `decisions_trained_on = ((n − 1) // 2048) × 2048`.
  - Values at the SAC decision counts: 150k → 149,504; 400k → 399,360; 800k → 798,720; 1.5M → 1,499,136.
  - The design report's 147,456 for 150k was off by one rollout; 149,504 is correct.
  - The final model at 1,501,184 is trained on all its decisions.
- **Env stack:** `PathSacEnv → SeedGuard → Monitor → DummyVecEnv`, no VecNormalize.
  - `SeedGuard` refuses eval seeds (300000–300029), validation seeds (400000–400009) and any reset options.
  - Only the first reset is seeded (0); later resets continue the env RNG.
- **Curriculum:** a callback calls `env.set_training_step(num_timesteps)` after every step, as the team SAC does.

### M4 / E6: primary result (decided by Claire, 2026-11-14)

**Primary PPO result = final model at 1,501,184 decisions (like the team SAC final model; no checkpoint selection).
Checkpoints 150k/400k/800k are evaluated on validation seeds 400000–400009 as a secondary learning curve only.**

### Test runs (decided by Claire, 2026-11-14)

- **M3 timing probe** (`m3_timing_probe_s0_8k`): 4 rollouts = 8,192 decisions with updates, run only to measure
  `t_update`. The model is discarded (saved only to a temporary folder to time `save`, then deleted) and never
  evaluated. It is not the M5 smoke test.
- **M5 smoke test:** stays the end-to-end test of the full M4 training script.

## 4. M3 verification V1–V8: results (Confirmed, 2026-11-14)

- Run: `training/verify_ppo_setup.py --mode verify` at HEAD d62207e with the M3 files uncommitted. It ended with
  `RESULT: DONE`, and `results/m3/ppo_setup_check.{json,txt}` were written.
- Setup: Isaac 6.2.0, Python 3.12.14, numpy 2.3.1, gymnasium 1.2.3, SB3 2.7.0, torch 2.14.1+cpu (8 threads, CUDA
  False), device cpu.
- total_mass_kg 1.618614; HESAI contribution 0 kg (as in M1.5).

| # | Check | Result |
|---|---|---|
| V1 | Preflight | PASS. Checks 21/21; all 18 shared modules and both configs resolve inside residual_ppo; gamma 0.99 == SAC; no sweep variant; contract policy == `policy_contract()`; contract file name == the adapter's; 40 forbidden seeds; 733 rollouts / 1,501,184 decisions / ≤ 58,640 gradient steps; 30 checkpoints including the 4 SAC counts. |
| V2 | Import order | PASS. torch + `torch._dynamo` imported before SimulationApp; PPO built after the app and env with **no ImportError**; optimizer Adam; torch 2.14.1+cpu from `<ISAAC_ROOT>\kit\python\Lib\site-packages\torch`; 1,044 torch modules with a file, 0 outside that folder; 0 modules from the ml_archive `pip_prebundle`. The only loaded path containing "ml_archive" is the Isaac extension package `omni.isaac.ml_archive` itself. |
| V3 | Spaces | PASS. Env obs Box(−10, 10, (128,), float32) and act Box(−1, 1, (4,), float32); the model's and VecEnv's spaces equal the env's. |
| V4 | Parameters | PASS. Actor **99,848**, critic **99,073**, features extractor 0, total 198,921; hidden activations ReLU; log_std [−1, −1, −1, −1]. |
| V5 | Initial mean | PASS. Mean \|μ\| per component 0.0010–0.0022, max \|μ\| 0.0068 (flag threshold 0.1); σ = 0.3679. |
| V6 | Action clip fraction | PASS. 0.696% of components (expected 0.656%), 2.78% of actions (expected 2.60%); max \|a\| 1.43. |
| V7 | No update | PASS. `collect_rollouts(2048)` OK; state_dict SHA-256 identical before and after (c44dcdbd…); 0 optimizer steps; empty Adam state; SB3 `_n_updates` 0. |
| V8 | Rollout contents | PASS. See the details below. |

V8 details:
- rewards, values, returns, advantages and log_probs are all finite;
- 24 episodes ended, all truncated, 0 terminated (no failures with an untrained σ = 0.37 residual in the short_slow stage); mean length 83 decisions;
- all 24 truncations bootstrapped: buffer reward − raw reward = γ·V(terminal_obs) to within 1.9e-8; 0 non-truncated rewards changed;
- stage short_slow throughout;
- resets: seeded [0] then 24 unseeded;
- 46.6 decisions/s including the policy forward pass (M2.1, zero action: 49.1).

Fixes made during the verification:
1. **Crash after the checks passed.** At module level `locals()` is `globals()`, so passing it to
   `callback.on_training_start` let SB3's `update_locals` overwrite the script's `env`. The script now passes fresh
   dicts. The first run's V-results were identical.
2. **False V2 failure.** `torch.ops` and `torch.classes` carry placeholder `__file__` values (`"_ops.py"`,
   `"_classes.py"`, set in torch's own source) that resolved against the working directory. The check now skips
   non-absolute `__file__` values (and lists them) and counts modules from the ml_archive `pip_prebundle`.

Re-run determinism: two runs gave the same state_dict hash and the same clip and rollout numbers (Confirmed).

## 5. M3 timing probe: results (Confirmed, 2026-11-14)

- Run: `--mode probe`, seed 0, 8,192 decisions, all in the short_slow stage. It ended with `RESULT: DONE`, and
  `results/m3/timing_probe.{json,txt}` and `timing_probe/progress.csv` were written. V1–V4 passed again.
- "PPO clip frac" is SB3's `train/clip_fraction`: the share of samples where the PPO probability ratio was clipped.
  "action clip" is the share of sampled actions outside [−1, 1].

| Rollout | Collect (s) | Update (s) | Grad steps | value_loss | PPO clip frac | approx_kl | std | explained var | Action clip (comp / action) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 44.1 | 0.21 | 80 | 1.997 | 0.100 | 0.0118 | 0.3665 | −0.03 | 0.70% / 2.78% |
| 2 | 44.1 | 0.18 | 80 | 0.963 | 0.083 | 0.0126 | 0.3654 | 0.24 | 0.65% / 2.54% |
| 3 | 44.1 | 0.18 | 80 | 0.765 | 0.140 | 0.0148 | 0.3627 | 0.37 | 0.61% / 2.44% |
| 4 | 44.2 | 0.19 | 80 | 0.496 | 0.137 | 0.0152 | 0.3590 | 0.51 | 0.60% / 2.34% |

- **t_update is about 0.19 s per rollout** (80 gradient steps, 2.4 ms each). No early stop: approx_kl stayed at or
  below 0.015, under the 0.03 threshold. SB3 counted 40 epochs.
- Collection runs at 46.4 decisions/s. Updates are about 0.4% of the wall clock, so training is **env-bound**.
- **Estimate for 1.5M** (733 rollouts):
  - collection 8.99 h at the probe rate (8.49 h at M2.1's 49.1/s);
  - updates ≤ 0.04 h;
  - 31 saves at 0.02 s each, plus 8 s startup;
  - **≈ 9.0 h in total** (Inference: the probe ran only short_slow; M2.1 measured a similar per-decision rate in the
    full-stage training_range wind, so later stages should cost about the same per decision).
- This wall-clock figure is not a head-to-head comparison with SAC's documented ~30 h, which used a different device
  and one gradient step per decision.
- Healthy early signs, all at these 4 rollouts only:
  - value loss fell 2.0 → 0.50 and explained variance rose −0.03 → 0.51;
  - PPO clip fraction 0.08–0.14;
  - std drifted down 0.368 → 0.359 (Inference: entropy shrinks slowly even with ent_coef 0.001; watch it in M4/M5).
- The CSV logger writes each update's `train/*` values on the next row, so the last update is missing from
  `progress.csv`. The JSON records all 4.

## 6. Next

- M3 is done once these files are committed (pending Claire's approval).
- M4: write the training script on top of `training/ppo_setup.py`. It must:
  - write `ppo_contract.json` next to every model;
  - write `status.json` and `run_meta.json`;
  - save checkpoints with `decisions_trained_on`;
  - log gradient steps and early stops;
  - run PID once on validation seeds 400000–400009 as the validation reference.
- M5: end-to-end smoke test of that script.
