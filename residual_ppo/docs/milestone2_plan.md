# Milestone 2 — PID-only baseline and evaluation protocol: step plan (approved by Claire 2026-11-12, decisions D1–D6 below)

Goal: know exactly what PID alone does on every evaluation scenario and wind profile, prove that "PID + zero
residual" through `PathSacEnv` *is* PID, measure simulator speed, and freeze one evaluation protocol that every
later PPO and SAC result must use.

Deliverables: (1) zero-residual equivalence evidence, (2) still-air vs wind PID table, (3) full PID baseline
table, (4) speed measurement + wall-clock estimate, (5) `docs/eval_protocol_v1.md` (frozen protocol, section E below).

**No training in M2.** No PPO model, no SB3 `learn()`. Shared modules are not edited. All new code goes in
`residual_ppo\evaluation\` and `residual_ppo\tools\`; all outputs in `residual_ppo\results\m2\` (summaries and
JSON/CSV committed, full Isaac logs not). Each step ends with a STOP and a report.

---

## What the code already defines (Confirmed by reading 61d2722; cited so the plan can be checked)

- **Team evaluation** (`drone_code/eval_path_sac.py`, not in our closure): for each scenario × wind profile ×
  seed, `env.reset(seed, options={route_stage: "full", route_type, wind_profile})`, then step until
  terminated/truncated, then `env.trace.metrics(...)`; groups summarized with `path_sac_policy.summarize`
  (means over all flights, failed ones included). It asserts both controllers flew the same realization
  (`wind_speed`, `wind_seed`, `route_seconds`).
- **The team's "PID only" baseline is PID + zero action through `PathSacEnv`** (`np.zeros(4)` when policy is None).
  So the team never compared it against an independent PID loop. That is what M2.2 adds.
- `path_sac_config`: `EVAL_SEEDS = 300000..300009` (10), `EVAL_SCENARIOS` = hover, straight, turns (waypoints),
  altitude, circle, figure_eight; `EVAL_ROUTE_STAGE = "full"`; `EVAL_WIND_PROFILES` = training_range,
  unseen_gusts, outside_range; `PATH_TRAIN.total_timesteps = 2_000_000`; comment: hover SAC ran ~17 decisions/s
  on the teammate's RTX 4070 SUPER.
- The residual enters at one line, `final = command + self.correction` (`PathCore.step`). Zero action gives
  `correction = 0.0`, so the sum is the PID command exactly (IEEE `x + 0.0 == x`).
- At every decision the env also calls `core.observe()` → `pid.read()` (PID `update` with `dt=0`). Reading
  `controller.py`, integrators change by `ki * error * dt`, so `dt=0` should leave PID state untouched.
  **Not yet proven**; M2.2 tests it.
- `hover_task.sample_stage_wind` always makes the same random draws and zeroes wind afterwards when calm. With
  the same seed, **route and start state are identical across all wind profiles**. `gust_scale = 0` zeroes both
  gust components (`wind.py`). "Still air" = zero wind; aerodynamic drag from the drone's own motion remains.
- `env.reset(options={"wind_profile": name})` accepts only names in `path_sac_config.EVAL_WIND_PROFILES`.

---

## M2.0 Decisions (Claire, 2026-11-12)

| # | Decision | Outcome |
|---|---|---|
| D1 | Add `still_air` without editing shared files | **Yes.** Our evaluation module registers it **in memory at startup** (`EVAL_WIND_PROFILES.setdefault`, asserting no existing key is overwritten). The still-air definition is written into the protocol (E2) **and into every snapshot/results file**. |
| D2 | Dijkstra scenario | **Excluded from protocol v1** (needs `path_navigation.py` + another scene, outside the M1 closure). **Any comparison with the team's published results drops their Dijkstra (warehouse) rows.** |
| D3 | Seed count | **Provisional: 30 seeds `300000..300029`** (first 10 = team `EVAL_SEEDS`). **Confirm or reduce after M2.1**, using its eval wall-clock estimate. Never fewer than the team's 10. |
| D4 | `strong_range` / `beyond_strong` | **Deferred until after M2.4.** Not part of protocol v1 for now; if added later they are labelled as our eval-only profiles. |
| D5 | Wall-clock estimates | **Yes:** training estimate for 2,000,000 decisions, **plus eval wall-clock per 720 flights**. (Budget open issue: team docs say SAC trained 1.5M, config says 2M; see E9.) |
| D6 | Equivalence tolerance | **Pre-registered:** tolerance = the M2.2 noise floor; **bit-exact required if the floor is 0**. The floor is measured and the tolerance written to `results\m2\zero_residual_tolerance.json` **before** the A-vs-B comparison runs. The comparison script refuses to run without that file. Never loosened afterwards. |

---

## M2.1 Headless speed measurement (no training)

**Why first:** it sizes every later step (the full baseline is 720 flights per controller) and gives the
2M-decision estimate needed to plan M4. It also settles D3.

`residual_ppo\tools\measure_speed.py` (Isaac `python.bat`, headless):
1. Build the env once (startup time recorded separately). Zero action, `training_range` and `still_air`, route
   stage `full`, fixed seeds.
2. Warm-up: 200 decisions, discarded. Then 3 repeats × 2,000 decisions, timed with `time.perf_counter`,
   with resets inside the timed window as in training (episodes end naturally).
3. Report: decisions/s, physics steps/s (= decisions × 12), reset time (mean), process startup time, mean ± spread
   over repeats, plus CPU model, GPU, and package versions (from imported modules).
4. Policy overhead: time one forward pass of an untrained SB3-style 256×256 MLP on CPU (inference only, no
   gradient step) per decision. This is not training.
5. **Estimate for 2,000,000 decisions:** `2e6 / decisions_per_s` (env) + `977 rollouts × t_update` (PPO, n_steps 2048:
   977 × 2048 = 2,000,896 decisions). `t_update` cannot be measured without a gradient step, so M2 reports the env
   part and leaves `t_update` as a placeholder, measured in the M3 smoke run. Reported with hardware; this is **not**
   a PPO-vs-SAC speed comparison.
6. **Eval wall-clock per 720 flights (D5):** reset every (scenario, seed) of the 30 eval seeds once (profile-independent
   by the RNG argument) and read each episode's decision cap `ceil(min(route_seconds, 60) × 10)`. Upper bound =
   4 profiles × Σ caps / decisions_per_s + 720 × reset time (failures only shorten flights). Also given per 10 seeds
   and per seed, to settle D3.

Outputs: `results\m2\speed.json`, `results\m2\speed_summary.txt`.
**Pass:** 3 repeats within ±10% of each other; both estimates written down with their assumptions; D3 re-decided.

## M2.2 Zero residual equals PID (identical seeds)

**Why:** every residual-RL result is "PID + residual" minus "PID". If zero residual through the env is not exactly
PID (e.g. `observe()` mutating PID state, or the env's step loop differing from a plain PID loop), that bias would
be blamed on, or credited to, the learner.

`residual_ppo\evaluation\zero_residual_check.py` (Isaac `python.bat`). Two controllers on identical seeds:
- **A, env path:** `PathSacEnv.reset(seed, options)` then `env.step(np.zeros(4))` until it ends (what the team calls PID-only).
- **B, pure PID loop:** the same `env.reset(seed, options)` for an identical start (route, wind, teleport). Then a
  loop that drives `env.core` / `env.sim` directly, per physics step: wind direction update, `pid.read`,
  `desired_attitude`, `pid.integrate`, then `sim.step_command(state, *command)` with **no residual term and no
  `observe()` calls**, and the same validity, failure and truncation rules at decision boundaries. It mirrors
  `PathCore.step` line by line and is written in our file (no shared edits).
- Instrumentation: wrap `env.sim.step_command` on the instance (our file only) to record, per physics step, the
  command sent to the mixer and the resulting state (position, orientation, velocities, motor speeds).

Runs (in this order; D6):
1. **Noise floor:** A twice on the same seed (same process), and A in two separate processes, on the same cells as run 2.
   Write the floor and the resulting tolerance to `results\m2\zero_residual_tolerance.json` (bit-exact if the floor is 0).
   STOP and report before run 2.
2. **Equivalence:** A vs B on 6 scenarios × {still_air, training_range} × 3 seeds (300000–300002), full episodes.
   Report per run: physics steps compared, max \|Δposition\| [m], max \|Δorientation\|, max \|Δcommand\|, first step
   where anything differs, same end reason and step.
3. **Guard test:** inject a known tiny residual (action = 1e-3 on thrust) into A, and confirm the comparison
   reports a difference. This proves the check can fail.

Outputs: `results\m2\zero_residual.json` and a summary.
**Pass:** the pre-registered D6 tolerance met on every run; the guard test differs; same termination step and reason everywhere.

## M2.3 Still air vs wind: separate tracking lag from wind effect

**Why:** PID error in wind = error PID already has in still air on that route (lag, cornering, drag from its own
speed) + error caused by wind. Only the second is what a residual can reasonably be credited with removing.

Using the protocol runner (section E), as a **pilot on the team's 10 seeds (300000–300009)** before the full run:
for every scenario and seed, fly PID in `still_air` and in each wind profile (training_range, unseen_gusts,
outside_range; D4 deferred). This also shakes out the runner on a smaller run; M2.4 repeats the analysis on
all seeds (D3).
- Check first (fails the step if violated): for each seed, `route_seconds` and the initial observation are identical
  across all profiles (confirms the RNG argument above).
- Report per scenario × profile: mean of each metric, and the **paired per-seed difference wind − still_air**
  (RMSE, cross-track RMSE, max deviation), with 95% bootstrap CI (section E).
- The still-air rows alone are the "tracking lag" table.

Outputs: `results\m2\still_vs_wind.{json,csv}` and summary.
**Pass:** consistency check passes; table complete (no missing cells; failed flights included and counted).

## M2.4 Full PID-only baseline table

The protocol (section E) run end to end with controller `pid` (= env path A, proven equal to pure PID in M2.2):
6 scenarios × (still_air + 3 team profiles) × 30 seeds = 720 flights (fewer if D3 reduces the seeds).
D4 is decided after this step.
M2.3's still-air vs wind table is then recomputed on all 30 seeds from these same flights (nothing flown twice).
Output exactly in the format later PPO/SAC runs will use: `results\eval\eval-v1\pid\{results.json, flights.csv,
summary.csv, summary.txt}` with the protocol hash.
**Pass:** all flights present; the matched-realization check passes; the re-run of 1 cell gives identical numbers.

Important: PID failing under `outside_range` (or stronger) is a **result, not a bug**. Do not tune PID, wind or
limits: they are shared with SAC.

## M2.5 Close out

Extract section E to `docs/eval_protocol_v1.md` (frozen). Checks `--check` 21/21, tests 23/23 in both interpreters.
Commit with explicit paths (with approval). Update README and CLAUDE.md status → M3.

---

## E. Evaluation protocol v1 (to be frozen as `docs/eval_protocol_v1.md`)

Applies unchanged to **PID, PID + SAC and PID + PPO**. Changing anything below makes a new version (v2) and
requires re-running every controller on it.

**E1. Realizations.** Eval seeds `300000..300029` (D3, provisional until M2.1), disjoint from the training seeds and
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
(D1); this definition is copied verbatim into every snapshot and results file. `strong_range` / `beyond_strong`:
deferred (D4).

**E3. Episode rules.** Unchanged env: 60 s limit, settle 3 s, failure limits from `path_sac_config.FAILURE`,
completion from `COMPLETION`. Policies are frozen and **deterministic** (SAC: actor mean; PPO:
`predict(deterministic=True)`). Each flight in a fresh `reset`; no learning during evaluation.

**E4. Metrics per flight** (`PathTrace.metrics`, unchanged): `position_rmse` (time-indexed), `cross_track_rmse`
and `cross_track_max` (geometric), `max_deviation`, `completed`, `completion_time`, `failed` +
`failure_reason`, `collided` (too_low), `saturation_rate`, `residual_norm`, `residual_change`
(0 for PID). Failed flights are **never dropped**; their metrics cover what was flown.

**E5. Summaries.**
- **Primary endpoint (pre-registered):** for each wind profile, the **mean position RMSE across the 6 scenarios**
  (unweighted mean of per-scenario means), reported as the **paired difference controller − PID** over seeds, with a
  95% percentile bootstrap CI. The bootstrap resamples **seeds** (each resample keeps a seed's flights in all 6
  scenarios together), 10,000 resamples, RNG seed 0. One primary result per profile; `still_air` is reported but is
  not a primary endpoint. A controller "improves on PID" in a profile only if the CI is entirely below 0.
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

---

## Explicitly out of scope for M2

Any training or gradient update; PPO model, callbacks or config (M3+); SAC checkpoint loading or retraining (M7);
editing shared modules or the PID; the Dijkstra scenario (unless D2 changes).
