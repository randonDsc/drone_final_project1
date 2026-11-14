# Evaluation protocol v1 — FROZEN (2026-11-13, M2.5)

Frozen copy of section E of `docs/milestone2_plan.md` (as of M2.5, including the D4 profiles and the M2.5 reporting
rules). It applies unchanged to **PID, PID + SAC and PID + PPO**.

**Identity of the frozen protocol**

- Protocol version: `eval-v1`
- Protocol file hash (`protocol_file_sha256`, LF-normalized SHA-256 of `evaluation/eval_protocol_v1.py`, recorded in
  every results file and checked by the runner as `run_common.EXPECTED_PROTOCOL_SHA`):
  **`d3f677dc37e2d347e87c0098eaab6bbea3c28f3ee10f2bfe84ebc9a5b745278e`**
- Git blob IDs of the frozen code (verify with `git -C drone_final_project1 hash-object <path>`, which applies the
  repository's line-ending normalization; on-disk SHA-256 differs under `core.autocrlf=true`):

| File | Git blob ID |
|---|---|
| `residual_ppo/evaluation/eval_protocol_v1.py` | `5b490ddde319d347e56ba399e974fc30b2c0d3bc` |
| `residual_ppo/evaluation/protocol_stats.py` | `328fcbdc8ec328f1756f4eaf83455b8f49b6ea3a` |
| `residual_ppo/evaluation/run_protocol.py` | `ce3a0da5c89a32cff3621ae6d53626bb8c3edce3` |
| `residual_ppo/evaluation/run_common.py` | `1fb19f777db67e5aec040ca660a521e891d6d7b3` |
| `residual_ppo/evaluation/controllers.py` | `8dae58a9069813e0ad629a4cac9cc0e0b7e18150` |

**Change rule.** Any change to the five files above, or to anything written below, makes a new protocol version
(**v2**). A new version requires re-running every controller on it (PID included) and is recorded with its own hash and
blob IDs. Results produced under different protocol hashes are never compared directly.

**Earlier hashes (history, not v1 results).** M2.1 `speed.json` recorded `69c3c95c…` (comment-only difference).
M2.2–M2.4 results recorded `18658b9c…` (before the D4 profiles); M2.4b proved the 720 flights of the 4 original profiles
bit-identical under both hashes (`results/m2/m2_4b_equality.json`). To reproduce M2.2 exactly, check out commit
`714f64f` (its runner expects `18658b9c…`).

**Canonical PID baseline under this protocol:** `results/eval/eval-v1/pid/` (1,080 flights; its `realizations.json`
is the cross-controller reference). PID-vs-PID dry run of the reporting rules: `results/m2/m2_5_report_dryrun.txt`.

---

Applies unchanged to **PID, PID + SAC and PID + PPO**. Changing anything below makes a new version (v2) and
requires re-running every controller on it.

**E1. Realizations.** Eval seeds `300000..300029` (D3, decided 2026-11-13), disjoint from the training seeds and
from the validation seeds (E6).
A seed fixes route, start state and wind. Route stage `full` for every scenario. Scenarios (route_type):
hover, straight, turns (waypoints), altitude, circle, figure_eight. Every controller flies every
(scenario, profile, seed). The runner asserts identical `wind_seed`, `wind_speed`, `route_seconds` and initial
observation across controllers.

**E2. Wind profiles.** Team: `training_range`, `unseen_gusts`, `outside_range` (verbatim from 61d2722). Ours:
`still_air`, defined as **the `training_range` ranges with `no_wind_probability = 1.0`**:
`{"no_wind_probability": 1.0, "mean_speed_range": (2.0, 7.0), "gust_scale_range": (0.5, 1.0),
"direction_rate_range": (0.0, 0.2)}`. Every draw is still made (so routes and starts match the windy profiles), then
mean wind, gust scale and turn rate are set to 0. Drag from the drone's own motion remains. Registered in memory
(D1); this definition is copied verbatim into every snapshot and results file.
**`strong_range` and `beyond_strong` (D4, decided 2026-11-13; OUR evaluation-only profiles, never used in training).**
Each is an exact copy of the team's `outside_range` with only `mean_speed_range` changed (built in code from
`outside_range`, asserting that `mean_speed_range` is the only differing key):
- `strong_range`: `{"no_wind_probability": 0.0, "mean_speed_range": (10.0, 14.0), "gust_scale_range": (0.5, 1.0),
  "direction_rate_range": (0.0, 0.2)}`
- `beyond_strong`: `{"no_wind_probability": 0.0, "mean_speed_range": (14.0, 16.0), "gust_scale_range": (0.5, 1.0),
  "direction_rate_range": (0.0, 0.2)}`

Registered in memory like `still_air` (D1); definitions copied into every results file (`our_profiles`). Because
`sample_stage_wind` makes the same draws for any range, a seed gives these profiles the same direction, gust scale,
turn rate and gust seed as `outside_range`; only the mean speed differs (`low + (high − low)·u` with the same `u`).
Evaluation order: still_air, training_range, unseen_gusts, outside_range, strong_range, beyond_strong
(6 × 6 × 30 = 1,080 flights per controller).

**E3. Episode rules.** Unchanged env: 60 s limit, settle 3 s, failure limits from `path_sac_config.FAILURE`,
completion from `COMPLETION`. Policies are frozen and **deterministic** (SAC: actor mean; PPO:
`predict(deterministic=True)`). Each flight in a fresh `reset`; no learning during evaluation.

**E3b. Controllers and model checks (M2.5 (a), `evaluation/controllers.py`).** One controller per run; `act` is
deterministic and returns float32[4]. Models are loaded **before SimulationApp** (two torch installs, M2.1).
- `pid`: zero residual (team PID-only; M2.2 bit-exact equal to a pure PID loop).
- `pid_ppo`: refused unless **`ppo_contract.json` next to the model file** has a `"policy"` group equal (after JSON
  normalization) to `path_sac_config.policy_contract()`; then `PPO.load(model, device="cpu")`;
  act = `predict(obs, deterministic=True)[0]`.
- `pid_sac`: refused unless `path_sac_config.incompatibility(model, check_training=True)` is None; act through the
  team's `path_sac_policy.load_path_policy` (deterministic actor mean); the model is also loaded once with
  `SAC.load(device="cpu")` for the space check and the record.
- Both learned controllers: refused unless the model's observation and action spaces equal the env's (checked after
  the env is built, before any flight). `record()` stores the repo-relative model path, model sha256, saved and runtime
  SB3 versions, num_timesteps, contract path + sha256, device.
- **Realization guard:** every reset of a non-PID run must reproduce the reference PID realization (`wind_seed`,
  `wind_speed`, `route_seconds`, `initial_obs_sha256`, `max_decisions`) exactly; the run stops at the first difference
  or missing entry. Default reference: the canonical `results/eval/eval-v1/pid/realizations.json`; validation-seed
  runs (E6) pass a PID run on those seeds. Every run records an action check (all actions finite, shape (4,)).

**E4. Metrics per flight** (`PathTrace.metrics`, unchanged): `position_rmse` (time-indexed), `cross_track_rmse`
and `cross_track_max` (geometric), `max_deviation`, `completed`, `completion_time`, `failed` +
`failure_reason`, `collided` (too_low), `saturation_rate`, `residual_norm`, `residual_change`
(0 for PID). Failed flights are **never dropped**; their metrics cover what was flown.
Diagnostics (not endpoints), recorded for every controller: `accel_cap_binding_frac`, `tilt_cap_binding_frac`,
`along_track_lag_mean`, `along_track_lag_rms`, `cross_track_component_rms` (definitions in the pre-registration block above M2.3).

**E5. Summaries.**
- **Primary endpoint (pre-registered):** for each wind profile, the **mean position RMSE across the 6 scenarios**
  (unweighted mean of per-scenario means), reported as the **paired difference controller − PID** over seeds, with a
  95% percentile bootstrap CI. The bootstrap resamples **seeds** (each resample keeps a seed's flights in all 6
  scenarios together), 10,000 resamples, RNG seed 0. One primary result per profile; `still_air` is reported but is
  not a primary endpoint. A controller "improves on PID" in a profile only if the CI is entirely below 0.
  **Primary profiles = the 3 team profiles only** (`training_range`, `unseen_gusts`, `outside_range`).
  `strong_range` and `beyond_strong` (D4) are **secondary**: the same paired statistic is reported for them, labelled
  secondary, and they cannot overturn or add to a primary result. `still_air` is reported prominently (nominal
  tracking; see CLAUDE.md M7 notes) but is not primary.
- **Secondary:** every per-cell (scenario × profile) result below, and all other metrics. Secondary results are
  descriptive; they cannot overturn the primary endpoint.
- Per (scenario, profile, controller): `path_sac_policy.summarize` (team means, failed included), plus median and
  IQR of `position_rmse` and `cross_track_rmse`, and failure-reason counts.
- **Paired comparisons** (controller − PID, same seeds): mean per-seed difference with a 95% percentile bootstrap
  CI (10,000 resamples over seeds, fixed RNG seed 0). "Better" is claimed only if the CI excludes 0.
  Failure and completion rates are compared as counts with their exact binomial CIs.
- Aggregates across scenarios: unweighted mean of the per-scenario means (each scenario counts once), reported
  separately per profile. No single overall score mixes profiles.
- Tables always show n (flights) and failures next to every mean.
- **Reporting rules added in M2.5 (Claire, 2026-11-13; decided before any learned controller is evaluated):**
  1. **Paired failure comparison, every profile.** Flights are paired by (scenario, seed) within a profile
     (6 × 30 = 180 pairs). McNemar **exact** test on the paired `failed` flags (controller vs PID): with
     b = pairs where only the controller failed and c = pairs where only PID failed, two-sided
     p = min(1, 2·P(X ≤ min(b, c))), X ~ Binomial(b + c, 0.5); p = 1 if b + c = 0. Reported with b, c and the
     both/neither counts. Plus each controller's failure count out of 180 with an exact (Clopper–Pearson) 95% binomial
     CI. Computed with the binomial distribution in `protocol_stats.py` (no new packages).
  2. **RMSE never alone when there are failures.** In any profile where either controller (the evaluated one or PID)
     has at least one failed flight, position RMSE is reported and interpreted only together with the paired failure
     comparison (1), never on its own.
  3. **strong_range and beyond_strong** (secondary): the descriptive endpoints, in this order, are (a) failure count
     (exact CI), (b) the paired McNemar test, (c) median decisions flown (IQR; NumPy default linear percentiles, over
     all 180 flights), (d) position RMSE labelled **"over flown portion only"**.
  Implemented in `protocol_stats.py` (`mcnemar_exact`, `clopper_pearson`, `paired_failure_comparison`,
  `decisions_median_iqr`, `controller_report`) and `run_protocol.py report` (which also refuses a controller run whose
  realizations differ from the canonical PID ones).

**E6. Model selection.** Each training run is evaluated at its **final** checkpoint and at pre-declared checkpoints
(150k, 400k, 800k, end). Choosing a checkpoint by the eval seeds is not allowed. Any checkpoint selection (for PPO and,
if retrained, SAC) uses the **validation seed set `400000..400009`**, which is disjoint from the eval seeds (300000..300029)
and from the training seeds (training resets are drawn from the run's own RNG stream started at the run seed; the
runner asserts no validation/eval seed is ever passed to a training reset). The selection rule (e.g. lowest
validation primary endpoint on `training_range`) is written down before validation runs. The selected checkpoint is
then evaluated once on the eval seeds.

**E7. Records.** Each results file stores: protocol version + SHA of the protocol file, controller name, model path +
contract, source commit and `SHARED_MANIFEST` check result, package versions from imported modules, device,
`total_mass_kg`, wall-clock time and hardware, **and the registered still-air definition (E2)**. Wall-clock time is
reported per controller and is not a head-to-head comparison.

**E7b. PPO vs SAC: training cost.** Every PPO-vs-SAC comparison reports, for each controller, **both** the number of
env decisions used in training (exact count, from the run metadata) **and** the wall-clock training hours with the
hardware and device (e.g. PPO: CPU torch on this PC; SAC: teammate's RTX 4070 SUPER, or ours if retrained). Results are
compared at equal decisions where checkpoints allow (150k / 400k / 800k / final). Wall-clock is context, not a
head-to-head score.

**E8. Comparison with the team's published numbers.** Team README §2 rows are compared only for the open scene;
**their Dijkstra (warehouse) rows are dropped** (D2). Team numbers use 10 seeds; ours on 30 seeds are not directly
comparable, so the comparison uses our results restricted to seeds 300000–300009.

**E9. Teammate's SAC numbers: check (2026-11-12).** `drone_team_main\runs\` does not exist locally (gitignored) and no
`results.json` / `flights*.csv` / `summary.csv` / `final_model.zip` is anywhere in the team worktree. So it cannot be
verified here. What the docs say (**Documented**, 61d2722): README §2 "evaluated by `eval_path_sac.py` on 10 seeds per
scenario"; README §7 reproduction sets `episodes = None` (all `EVAL_SEEDS`); the open-scene percentages are consistent
with 60 flights per profile (e.g. 5% = 3/60) but that does not prove it. **Questions for the teammate:**
1. Was the README §2 table produced with `PATH_EVAL.episodes = None` (all 10 seeds 300000–300009)?
2. Which model: `final_model.zip` after 1.5M steps (README, PATH_SAC.md) or after more? The committed config has
   `total_timesteps = 2_000_000` with `mode = "resume"`.
3. Can they share `runs/flight_sac/flight1/` (`final_model.zip`, `flight_contract.json`, `evaluation/final_model/*`)?
4. How was "Max deviation" aggregated in the README table (max over flights, or mean of per-flight maxima)?
   `summarize()` gives the max; worth confirming.

**E9 status (2026-11-13):**
- Q1: *supported, not asked*: the M2.3 reproduction on all 10 seeds matches README §2 (RMSE within 0.08%, completed and
  failed counts exact out of 60 per profile).
- **Q2 resolved (Confirmed):** the Hugging Face `final_model.zip` (README §3) has `num_timesteps` 1,500,000 and passes
  `incompatibility()` for policy + training (`models\sac_team\SOURCE.json`).
- **Q3 resolved:** files obtained from Hugging Face `pcldsc/Iris_drone_residual_correction_SAC` @ 7dd1f14b (model +
  contract; `evaluation/final_model/*` is not in that repo).
- **Q4 resolved (Inference):** README max deviation = the mean over the 6 scenarios of the per-scenario maxima
  (`summarize()`'s max per scenario, then averaged); matches to 0.003 m on all three profiles, while the overall maximum
  does not (2.44 / 2.10 / 3.15 m).
