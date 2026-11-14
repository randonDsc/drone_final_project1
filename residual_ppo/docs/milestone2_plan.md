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
| D3 | Seed count | **Decided (2026-11-13): 30 seeds `300000..300029`** (first 10 = team `EVAL_SEEDS`). D3 was deferred to M2.1 **only for cost**; M2.1 measured ≤ 0.66 h per controller for 720 flights, so cost does not justify a change and the pre-registered 30 stand. Not increased either (that would be a post-hoc change). |
| D4 | `strong_range` / `beyond_strong` | ~~Deferred until after M2.4.~~ **Decided (Claire, 2026-11-13, after M2.4): added as OUR evaluation-only, SECONDARY profiles.** Each is an exact copy of the team's `outside_range` with only `mean_speed_range` changed: `strong_range` (10, 14) m/s, `beyond_strong` (14, 16) m/s. Registered in memory like `still_air` (D1; no shared edits). Primary endpoints stay the 3 team profiles; `still_air` stays reported, not primary. Definitions in E2, endpoint status in E5. Adding them changes the protocol file hash (see M2.4b). |
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

**M2.1 run 1 finding (2026-11-12, Confirmed):** throughput completed, then building SB3's `ActorCriticPolicy` failed:
its Adam optimizer imports `torch._dynamo`, which after SimulationApp mixes in the stale torch from
`extsDeprecated\omni.isaac.ml_archive\pip_prebundle` (`ImportError: _EvalFrameOverride`). A probe showed that
importing `torch` + `torch._dynamo` **before** `SimulationApp` avoids it. M2.1 now times a plain torch MLP of the
same actor shape (forward only, a lower bound for SB3 `predict`). **M3 requirement:** the training script imports
torch (and `torch._dynamo`) before SimulationApp and the M3 smoke run confirms an SB3 PPO update works.

**M2.1 result (run 2, 2026-11-12, Confirmed): PASS.** Checks 21/21 before Isaac. Zero residual, route stage `full`:
training_range 49.11 decisions/s (48.94–49.19), still_air 49.19 (49.01–49.33), ≈ 590 physics steps/s; all repeats
within ±10% (in fact < 1%); 0 failures in 12,000 timed decisions; reset 2.6–4.0 ms; startup 7.2 s (SimulationApp) +
1.3 s (env). Policy forward 0.027 ms/decision (plain MLP, lower bound); SB3 `get_device("auto")` = cpu.
Eval caps (30 seeds × 6 scenarios): 28,920 decisions per profile; still_air and training_range had identical
route/start/caps and zero still-air wind on all 180 cases.
**Estimates:** training 2M decisions = env 11.3 h + inference 0.02 h + 977 × `t_update` (M3). Eval 720 flights
≤ 0.66 h per controller (0.22 h per 10 seeds). → D3 decided: 30 seeds (see M2.0).
Files: `results\m2\speed.json`, `results\m2\speed_summary.txt`.

**Protocol hash note:** `speed.json` records `protocol_file_sha256` = `69c3c95c26576e2bf51d543de9e19052c3a58510e3b0737c8e936ee11f148d46`
(`eval_protocol_v1.py` as committed in b738103). Afterwards, only the line-14 comment was changed ("provisional 30 (D3,
re-decided after M2.1)" → "30 (D3, decided 2026-11-13 after M2.1)"); no constant changed. The new hash is
`18658b9ce0801677369f6ee16d5d49467965fe96888a091d6d7762fd7376d0dc`, and every run from M2.2 on records that one.
`speed.json` is not re-run for this (M2.1 measured speed only, no protocol results).

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

**M2.2 result (2026-11-13, Confirmed): PASS, bit-exact.** Runner `evaluation\zero_residual_check.py` (modes `floor`,
`tolerance`, `compare`, `supplementary`). Checks 21/21 before every Isaac run; all 18 shared modules resolved inside
`residual_ppo\`; protocol hash `18658b9c…` recorded in every results file; total_mass_kg 1.618614, HESAI 0 kg;
Isaac 6.2.0-rc.14, numpy 2.3.1, gymnasium 1.2.3, SB3 2.7.0, torch 2.14.1+cpu (torch imported before SimulationApp).
- *Compared at every physics step* (instance wrapper on `env.sim.step_command`): pre-step position, orientation,
  linear and angular velocity; PID integrators (height, xy, attitude) after `integrate`; wrench to the mixer; rotor
  speeds from the mixer; saturation flag; wind force/torque; mean wind; plus the post-episode state and the episode
  fields (physics steps, decisions, end reason, completion, route_seconds, wind_seed). Path B stubs `core.observe` on
  the instance during its reset, so its PID is never read with dt=0.
- *Run 1, noise floor:* A twice per cell (A,A) on the 36 run-2 cells, in two separate processes. All 4 pairs
  (within p1, within p2, across processes ×2) bitwise identical on 36/36 cells; one trace hash `e0b25e6d…` for all four
  runs. **Floor = 0 for every quantity → tolerance bit-exact (D6).** `results\m2\zero_residual_tolerance.json`.
- *Run 2, equivalence:* A vs B (A,B per cell), 36 cells, 72,000 physics steps: **max |diff| = 0 for every quantity,
  bitwise identical on 36/36, same end (all truncated) and step everywhere.** A called `observe()` 1 + decisions times
  per flight, B never. This proves (empirically) that `pid.read` with dt=0 leaves PID state untouched.
- *Run 3, guard:* thrust action 1e-6 and 1e-3 at decision 0 only, 2 cells: detected 4/4, first difference at physics
  step 0. Even 1e-6 (+8e-7 N) changes the physics (position up to ~3e-5 m, rotor speeds ~1e-3 rad/s), not only the command.
- *Supplementary, added after the pre-registered run, same bit-exact tolerance* (Claire, 2026-11-13; run 2 had no
  failed flight, so the failure-termination branch was untested): A vs B on `outside_range` × 6 scenarios × seeds
  300000–300002, 18 cells, 15,780 physics steps. **12/18 flights failed (all `out_of_bounds`); every one has the same end
  reason and decision in A and B and is bitwise identical; all 18 cells pass bit-exact.** Guards with action 1e-6 on
  roll torque (straight/still_air/300000) and yaw torque (turns/training_range/300000) at decision 0: detected 2/2.
  `results\m2\zero_residual_supplementary.json`. No tracking metric is computed from any M2.2 trace (seeds overlap the
  H1 pilot); failure counts here are not compared with anything.
- *Process note:* the plan said STOP after run 1. Claire's instruction for the session asked for all three runs in one
  go, so runs 2–3 followed without a stop. The pre-registration still holds: the tolerance file was written
  (2026-11-13T22:49:00Z) before run 2 started, the comparison refuses to run without it, and `zero_residual.json`
  records its hash (`sha256_lf` 45f9ff4d…).
- *Path redaction:* committed JSONs store the Isaac install prefix in `package_files` as `<ISAAC_ROOT>`. In the two
  files written before that rule, the prefix was replaced afterwards (only those strings changed). The tolerance file's
  LF hash is a945d764… after redaction; putting the prefix `C:\isaacsim\isaac-sim-standalone-6.2.0-windows-x86_64` back
  restores 45f9ff4d…, the hash recorded at run time. The supplementary JSON was written already redacted.
  **`<ISAAC_ROOT>` applies from M2.2 on.** Earlier committed files (the M1.5 `results\m1_env_config_snapshot.json`)
  keep their absolute Isaac paths, by decision (Claire, 2026-11-13).
- Raw per-step traces (`results\m2\zero_residual_traces\`, 86 MB+) and Isaac logs are gitignored; their hashes are in the JSON.

## Pre-registered before any M2.3 run (Claire, 2026-11-12)

**H1.** *On the team's open-scene routes, PID-only RMSE in still air is close to its RMSE in `training_range`, i.e.
most of PID's error is reference-tracking lag, not wind.*

- **Motivation (Documented, team README §2, open scene, 10 seeds):** PID-only position RMSE is 0.502 m in
  `training_range` and 0.497 m in `unseen_gusts`, almost identical although `unseen_gusts` has 1.2–1.5× gusts that
  are also faster and turn more quickly. That suggests gust strength explains little of PID's error. It is indirect:
  the team never flew still air, and `outside_range` (8–10 m/s mean) does hurt PID (1.653 m, 62% failures).
- **Primary test: wind share on mean squared error (approved by Claire, 2026-11-12).** Per flight,
  `MSE = position_rmse²` (time-indexed, over what was flown). Per seed `s` and profile `p`:
  `M_p(s)` = mean of flight MSE across the 6 scenarios. For a set of seeds `S`:

      wind_share_MSE(S) = Σ_s [M_training_range(s) − M_still_air(s)] / Σ_s M_training_range(s)

  i.e. paired on seeds, computed as a ratio of sums over seeds (more stable than averaging per-seed ratios when one
  seed's MSE is small). 95% percentile bootstrap CI: resample seeds with replacement (each seed keeps all 6 scenarios
  in both profiles), recompute the statistic, 10,000 resamples, RNG seed 0.
- **Decision rule (approved):** H1 is *supported* if the upper 95% CI bound of `wind_share_MSE` is below 0.5
  (wind explains less than half of PID's training_range MSE), *rejected* if the lower bound is above 0.5,
  *inconclusive* otherwise.
- **Failures (approved).** List every failed flight in `still_air` or `training_range` (scenario, seed, profile,
  `failure_reason`, decisions flown). The **primary verdict uses all flights** (failed ones included, MSE over what was
  flown). **Sensitivity analysis:** recompute `wind_share_MSE` and its CI after excluding every seed that has a failure
  in *either* profile, from *both* profiles (keeps the pairing). Reported next to the primary result; it does not
  replace it. If the two verdicts differ, say so.
- **Secondary only:** the same share on RMSE, `(RMSE_tr − RMSE_still) / RMSE_tr` (same pairing and bootstrap),
  **per-scenario MSE wind shares** (same formula restricted to one scenario, ratio of sums over seeds, seed bootstrap
  CI), and the paired RMSE difference. They cannot change the H1 verdict.
- **Pre-registered sanity check (hover).** The `hover` reference is a fixed point, so in still air PID only corrects
  the small random start offset (±0.1 m, ±3° roll/pitch, ±5° yaw): its still-air MSE should be near 0 and its MSE
  wind share near 1. **If hover's MSE wind share is below 0.8, the measurement is flagged as suspect, and H1 is not
  interpreted** until the cause is found (e.g. still air not actually calm, or a start transient dominating).
  Reported: hover still-air MSE, training_range MSE and share, with CI.
- **Interpretation notes (fixed in advance):**
  - RMSE-based shares *understate* wind's contribution: independent error sources add in squares
    (MSE_total ≈ MSE_lag + MSE_wind), so square roots compress the wind part. Example: lag 0.4 m RMS plus
    independent wind 0.3 m RMS gives 0.5 m total; the RMSE share is 0.2 but the MSE share is 0.36. That is why the
    primary statistic uses MSE.
  - Interaction counts as wind: any extra error from wind *combined with* lag (e.g. a headwind that increases
    lag on a fast segment) is part of `M_training_range − M_still_air`, so it is attributed to wind.
  - Negative values are kept as they are (never clipped to 0). Wind can reduce error (e.g. a tailwind offsetting
    lag), and clipping would bias the share upward.
- Run on the M2.3 pilot (team's 10 seeds) and confirmed on all seeds in M2.4. The M2.4 result is the one reported.

**Per-flight diagnostics** (M2.3 and M2.4; instrumentation in our runner only, no shared-file edits). Recorded at
every physics step by wrapping `env.sim.step_command` on the instance (as in M2.2), using `env.core.desired` and the
PID part of the command (`final − core.correction`, which is the whole command for PID):
- `accel_cap_binding_frac`: fraction of physics steps where the PID's horizontal demand is at the cap
  (`ReferencePID.acceleration_capped(...)`, the shared helper, ≥ 98% of the cap) **and** the cap in force is
  `PID_ACCELERATION_CAP` (1.5 m/s²).
- `tilt_cap_binding_frac`: same, but the cap in force is the tilt term `vertical_accel × tan(MAX_TILT_DEGREES)` (< 1.5).
  The two fractions do not overlap; their sum is the helper's "capped" fraction.
- `along_track_lag_mean` [m]: mean of `(p_ref − p) · v_ref / |v_ref|` over physics steps with `|v_ref| > 0.05 m/s`
  (positive = drone behind the reference), plus the number of steps it was computed on.
  Also `along_track_lag_rms` and `cross_track_component_rms` (the part of `p_ref − p` perpendicular to `v_ref`),
  so RMSE can be split into "behind" vs "beside" the reference.
These are diagnostics, not endpoints: they explain H1 and are not used to rank controllers.

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

### M2.3 runner design (approved by Claire 2026-11-13, with additions a–d; written before any M2.3 flight)

Code: `evaluation\controllers.py` (controller interface, no Isaac), `evaluation\run_protocol.py` (Isaac runner),
`evaluation\protocol_stats.py` (NumPy only: summaries, aggregation, reproduction comparison; later bootstrap/H1),
venv tests in `tests\test_protocol_stats.py`.
- **Controller interface:** `name`, `act(observation) -> float32[4]` (deterministic), `record() -> dict`. Only `pid`
  (`np.zeros(4, float32)`, as the team's PID-only) is implemented; `pid_ppo` (`predict(deterministic=True)`) and
  `pid_sac` (team `load_path_policy`, after `path_sac_config.incompatibility`) raise NotImplementedError until M3/M7.
  One controller per invocation.
- **Flight loop and metrics** mirror `drone_team_main\drone_code\eval_path_sac.py` `fly()` exactly: `env.reset(seed,
  options={route_stage, route_type, wind_profile})`; step until terminated/truncated; `reason` from the last step's
  info; row = scenario, wind_profile, seed, `wind_speed = norm(info["wind_mean"])`, wind_seed, route_seconds,
  plus `env.trace.metrics(env.core.reference.path, reason, reason == "too_low", env.core.completion)`.
  Per (scenario, profile): `path_sac_policy.summarize(rows)`, plus median/IQR and failure-reason counts.
- **Matched realizations:** every flight row stores `initial_obs_sha256` and `max_decisions`. Within a run, for each
  (scenario, seed): route_seconds, max_decisions and initial observation identical across profiles; still_air
  wind_mean = 0 (the M2.3 consistency check). Across controllers: a non-PID run loads the PID `realizations.json`
  (key: wind_seed, wind_speed, route_seconds, initial_obs_sha256) and refuses on any mismatch. Seeds are asserted to be
  eval (or validation) seeds.
  **(d) The canonical `realizations.json` is the one written by the M2.4 30-seed PID run.** Pilot runs write their own
  copy, which is never used as the cross-controller reference.
- **Diagnostics** (read-only instance wrapper on `env.sim.step_command`, as in M2.2): per physics step, `p_ref`,
  `v_ref` from `core.sample()` (before `core.time` advances), PID command, `core.desired`; cap fractions from the shared
  `acceleration_capped(...)`, split by which cap is in force; along-track lag mean/RMS and cross-track component RMS over
  steps with |v_ref| > 0.05 (counts recorded; hover gives none).
  **(c) The PID command is taken as `final − core.correction`. That is exact only for zero correction (PID):** for
  PPO/SAC it is the PID part reconstructed from the sum (floating-point rounding), so their diagnostics are labelled
  "PID part = final − correction (reconstructed)" in every file.
- **Check 0 (before the pilot):** one cell flown with diagnostics on and off must give identical metrics and identical
  per-decision trace (proves the wrapper does not change the flight).
- **Outputs:** M2.4 and later protocol runs → `results\eval\eval-v1\<controller>\{results.json, flights.csv,
  summary.csv, summary.txt, realizations.json}`. M2.3 pilot → `results\m2\pilot_pid\` (same file set), plus
  `results\m2\team_reproduction.{json,txt}`; later `results\m2\still_vs_wind.{json,csv}`. Every file records E7 items and
  uses `<ISAAC_ROOT>`.

### M2.3 first check: reproduce the team README §2 open-scene PID-only rows (pre-registered before any flight)

PID on seeds 300000–300009 × {training_range, unseen_gusts, outside_range} × 6 scenarios (180 flights, flown inside
the pilot). Aggregation as README §2 ("averaged over 6 route types"): per profile, the mean over the 6 scenarios of
`summarize()` values. Targets (README §2, 61d2722):

| profile | RMSE [m] | max deviation [m] | completed | failed |
|---|---|---|---|---|
| training_range | 0.502 | 1.77 | 5% | 0% |
| unseen_gusts | 0.497 | 1.51 | 5% | 0% |
| outside_range | 1.653 | 3.11 | 0% | 62% |

**(a) Pass range:** per profile, position RMSE within ±5% relative of the README value; completed and failed flights
each within ±3 of the README count out of 60 flights (README % × 60, e.g. 5% → 3, 62% → 37.2). Max deviation is
**descriptive only**, reported in both aggregations (mean of per-scenario maxima, and the maximum over all 60 flights).
**(b) If any profile is outside the range,** investigate in this order before M2.4: (1) when README §2 last changed
versus the code commits (`git log` in `drone_team_main`, read-only), i.e. whether the table was produced by older code;
(2) Isaac version (6.2.0 vs the team's 6.1.0) and GPU/PhysX differences. **Never tune anything** (PID, wind, limits,
seeds) to match. Report the differences either way; STOP after the reproduction report, before the consistency check
and any H1 statistic.

### M2.3 result (pilot, seeds 300000–300009, 2026-11-13, Confirmed unless labelled): PASS

- *Check 0:* one cell (turns / training_range / 300000) with diagnostics on and off: identical metrics and per-decision
  trace. Pilot: 240 flights (4 profiles × 6 scenarios × 10 seeds), 11.9 min. Files `results\m2\pilot_pid\` (its
  `realizations.json` is a pilot copy, not canonical).
- *Reproduction of README §2 (pre-registered range): PASS.* RMSE 0.50241 / 0.49682 / 1.65268 m (training_range /
  unseen_gusts / outside_range; every |relative diff| ≤ 0.08%); completed 3 / 3 / 0 and failed 0 / 0 / 37 of 60 (exact).
  Max deviation (descriptive): mean of per-scenario maxima 1.768 / 1.507 / 3.112 m, overall max 2.44 / 2.10 / 3.15 m.
  Files `results\m2\team_reproduction.{json,txt}`.
- *Consistency check: PASS* (240/240 flights, 60 (scenario, seed) groups; route_seconds, max_decisions, initial
  observation hash identical across the 4 profiles; still_air wind zero).
- *Still air vs wind* (`results\m2\still_vs_wind.{json,csv}`, `still_vs_wind_summary.txt`): still-air RMSE 0.115–0.244 m;
  training_range adds +0.30 to +0.42 m RMSE per scenario (every 95% CI above 0). Along-track lag mean is slightly
  negative in still air (−0.007 to −0.057 m: the drone is marginally ahead of the reference) and +0.09 to +0.30 m in
  training_range.
- *Cap-binding diagnostics:* accel cap (1.5 m/s²) binds on 0–4.6% of physics steps in still air, 0–9.4% in
  training_range and unseen_gusts, and 55–67% in outside_range on moving routes (38% on hover). The tilt cap never binds
  (0% everywhere).
- **H1 pilot verdict: REJECTED.** Primary `wind_share_MSE` = **0.900, 95% CI [0.768, 0.948]** (mean MSE training_range
  0.3297 m², still_air 0.0329 m²); lower bound > 0.5.
  - Hover sanity check: OK. Still-air MSE 0.01433 m², training_range 0.20241 m², share 0.929 [0.845, 0.963] (≥ 0.8).
  - Failures: 0 failed flights in still_air or training_range, so the sensitivity set equals the primary set (same
    verdict).
  - Secondary (cannot change the verdict): RMSE share 0.655 [0.508, 0.753]; paired RMSE difference +0.329 m
    [+0.185, +0.514]; per-scenario MSE shares 0.834 (turns) to 0.951 (straight), every lower bound > 0.6.
  - Reading: in training_range about 90% of PID's MSE is attributable to wind (including wind × lag interaction, as
    fixed in advance), not to reference-tracking lag. The pilot verdict is not the reported one; M2.4 (30 seeds) is.
- **Why the H1 motivation pointed the wrong way (Inference, not tested):** `unseen_gusts` has stronger, faster gusts
  but a lower mean wind speed (2–6 m/s vs 2–7 m/s in training_range). If mean-wind drag dominates PID's error, the
  weaker mean offsets the stronger gusts, so near-equal RMSE in README §2 says little about whether wind matters.

## M2.4 Full PID-only baseline table

The protocol (section E) run end to end with controller `pid` (= env path A, proven equal to pure PID in M2.2):
6 scenarios × (still_air + 3 team profiles) × 30 seeds = 720 flights (D3).
D4 is decided after this step.
M2.3's still-air vs wind table is then recomputed on all 30 seeds from these same flights (nothing flown twice).
Output exactly in the format later PPO/SAC runs will use: `results\eval\eval-v1\pid\{results.json, flights.csv,
summary.csv, summary.txt}` with the protocol hash.
**Pass:** all flights present; the matched-realization check passes; the re-run of 1 cell gives identical numbers.

**M2.4 run details (fixed 2026-11-13, before the M2.4 flights):**
- Run: `run_protocol.py fly --controller pid --seeds 300000-300029 --check0 --out residual_ppo/results/eval/eval-v1/pid`.
  Its `realizations.json` is marked CANONICAL; `meta.code_blobs` records the git blob IDs of the evaluation code run.
- *Pilot equality (added by Claire):* every pilot flight (seeds 300000–300009, 240 flights) must equal the same
  (scenario, profile, seed) flight of the M2.4 run bit-for-bit on every row field (metrics, realization fields,
  diagnostics), and the pilot `realizations.json` entries must equal the canonical ones. Any difference is reported.
  → `results\m2\m2_4_pilot_equality.{json,txt}` (`run_protocol.py compare`).
- *1-cell re-run:* cell **figure_eight / outside_range / 300029**, flown in a fresh process (diagnostics on, no check 0).
  It is the last flight of the M2.4 run, so it has the longest history of earlier flights; the re-run has none. Must
  equal the M2.4 row bit-for-bit. → `results\m2\m2_4_rerun\` (1-flight run) + `results\m2\m2_4_rerun_check.{json,txt}`.
- *Matched realizations + still-vs-wind + H1 (reported):* `run_protocol.py still-vs-wind --name still_vs_wind_m2_4`
  → `results\m2\still_vs_wind_m2_4.{json,csv}`, `still_vs_wind_m2_4_summary.txt`. Pilot files stay unchanged.

**M2.4 result (2026-11-13, Confirmed): PASS.** Checks 21/21 and import-path guard before Isaac; protocol hash
`18658b9c…`; total_mass_kg 1.618614, HESAI 0 kg; Isaac 6.2.0-rc.14, numpy 2.3.1, gymnasium 1.2.3, SB3 2.7.0, torch
2.14.1+cpu. Flown from git HEAD 501999b plus uncommitted evaluation edits; `meta.code_blobs` records the exact blob IDs
(run_protocol.py cab3d6f0, protocol_stats.py 0539d779, controllers.py 9fa925b4, run_common.py 4b7374c9,
eval_protocol_v1.py 57718499).
- Check 0 PASS; **720/720 flights in 35.8 min**; files `results\eval\eval-v1\pid\` (`realizations.json` CANONICAL).
- **Matched realizations (consistency check): PASS** (720/720, 180 (scenario, seed) groups).
- **Pilot equality: PASS, bit-for-bit.** 240/240 pilot flights identical on all 32 row fields; 240/240 realizations
  entries identical. So adding seeds 300010–300029 changed nothing on the first 10, and the run order/history does not
  matter.
- **1-cell re-run: PASS, bit-for-bit** (figure_eight / outside_range / 300029, fresh process, 32/32 fields).
- **PID-only baseline, per profile** (mean over the 6 scenarios of `summarize()`, 180 flights per profile):

  | profile | RMSE [m] | cross-track RMSE [m] | max dev (mean of scenario maxima / overall) [m] | completed | failed |
  |---|---|---|---|---|---|
  | still_air | 0.183 | 0.124 | 0.571 / 0.798 | 116 (64%) | 0 |
  | training_range | 0.483 | 0.356 | 1.768 / 2.441 | 24 (13%) | 0 |
  | unseen_gusts | 0.484 | 0.366 | 1.531 / 2.104 | 16 (9%) | 0 |
  | outside_range | 1.641 | 1.172 | 3.113 / 3.151 | 0 | 110 (61%), all out_of_bounds |

- *Still air vs wind (30 seeds):* still-air RMSE 0.108 (hover) to 0.279 m (turns); training_range adds +0.27 to
  +0.35 m RMSE per scenario (every CI above 0). Along-track lag: −0.004 to −0.059 m in still air, +0.09 to +0.18 m in
  training_range. Accel cap binding: 0–11.2% in still air (turns highest), 0–14.0% in training_range, 53.9–62.0% in
  outside_range on moving routes (38.5% hover). Tilt cap: 0% everywhere.
- **H1 (reported verdict, 30 seeds): REJECTED.** Primary `wind_share_MSE` = **0.868, 95% CI [0.808, 0.906]** (mean MSE
  training_range 0.2918 m², still_air 0.0384 m²).
  - Hover sanity: OK. Still-air MSE 0.01294 m², training_range 0.18815 m², share 0.931 [0.892, 0.954].
  - Failures in still_air/training_range: 0, so sensitivity = primary (same verdict).
  - Secondary: RMSE share 0.621 [0.544, 0.679]; paired RMSE difference +0.300 m [+0.220, +0.384]; per-scenario MSE
    shares 0.773 (turns) to 0.933 (straight), every lower bound ≥ 0.68.
  - Same verdict as the pilot (0.900 [0.768, 0.948]); the CI is narrower with 30 seeds.
- D4 (`strong_range` / `beyond_strong`) is to be decided now, after M2.4 (not decided here). → Decided 2026-11-13, see D4
  and M2.4b.

## M2.4b PID baseline re-flown with the D4 profiles (fixed 2026-11-13, before flying)

D4 adds two profiles to `evaluation\eval_protocol_v1.py`, so the protocol file hash changes (M2.2–M2.4 results
record `18658b9c…`; from M2.4b on, the new hash, also set as `run_common.EXPECTED_PROTOCOL_SHA`). The version name stays
`eval-v1`: no controller other than PID has been evaluated, and the protocol is frozen only in M2.5.
- The 4-profile M2.4 run is kept for reference, moved (`git mv`, content unchanged) from `results\eval\eval-v1\pid\` to
  `results\m2\m2_4_pid_4profiles\`. Committed M2.4 check files that name the old path as `source_run`/`candidate_run`
  refer to this moved copy (their recorded sha256 identifies it). Its `realizations.json` says CANONICAL but is
  **superseded** by the M2.4b one.
- Run: `run_protocol.py fly --controller pid --seeds 300000-300029 --check0 --out residual_ppo/results/eval/eval-v1/pid`
  with all 6 profiles → 1,080 flights; its `realizations.json` is the canonical cross-controller reference.
- **Check 1, bit-identity:** the 720 flights of the original 4 profiles must equal the M2.4 run bit-for-bit (all 32
  row fields + realizations entries; `run_protocol.py compare`, reference = `m2_4_pid_4profiles`)
  → `results\m2\m2_4b_equality.{json,txt}`.
- **Check 2, matched realizations across all 6 profiles:** route_seconds, max_decisions, initial observation hash
  identical per (scenario, seed); still_air wind zero → `results\m2\m2_4b_baseline.{json,txt}`, with the per-profile
  table (primary / secondary / reported-not-primary labels, failure counts and reasons per scenario).
- The H1 verdict is not recomputed (it uses only still_air and training_range, which check 1 proves unchanged).

**M2.4b result (2026-11-13, Confirmed): PASS.** Protocol hash `d3f677dc…`; checks 21/21; total_mass_kg 1.618614,
HESAI 0 kg; same package versions as M2.4. Flown from HEAD 3352061 plus the uncommitted D4 edits; `meta.code_blobs`:
eval_protocol_v1.py 5b490ddd, run_common.py 1fb19f77, run_protocol.py 68fe3361 (others unchanged from M2.4).
- Check 0 PASS; **1,080/1,080 flights in 38.2 min** → `results\eval\eval-v1\pid\` (canonical `realizations.json`).
- **Check 1: PASS, bit-for-bit.** 720/720 flights of the original 4 profiles identical to M2.4 on all 32 fields;
  720/720 realizations entries identical.
- **Check 2 (matched realizations, 6 profiles): PASS** (1,080/1,080; 180 groups).
- **PID-only baseline, protocol eval-v1 (`d3f677dc…`)**, mean over 6 scenarios of `summarize()`, 180 flights per profile:

  | profile | status | RMSE [m] | cross [m] | max dev (mean of maxima / all) [m] | completed | failed | reasons | mean decisions | accel cap |
  |---|---|---|---|---|---|---|---|---|---|
  | still_air | reported, not primary | 0.183 | 0.124 | 0.571 / 0.798 | 116 | 0 | – | 161 | 2.8% |
  | training_range | PRIMARY | 0.483 | 0.356 | 1.768 / 2.441 | 24 | 0 | – | 161 | 5.3% |
  | unseen_gusts | PRIMARY | 0.484 | 0.366 | 1.531 / 2.104 | 16 | 0 | – | 161 | 4.6% |
  | outside_range | PRIMARY | 1.641 | 1.172 | 3.113 / 3.151 | 0 | 110 (61%) | out_of_bounds 110 | 82 | 54.8% |
  | strong_range | secondary (D4) | 1.651 | 1.191 | 3.228 / 3.251 | 0 | 179 (99%) | out_of_bounds 179 | 27 | 71.6% |
  | beyond_strong | secondary (D4) | 1.604 | 1.168 | 3.302 / 3.329 | 0 | 180 (100%) | out_of_bounds 180 | 18 | 74.2% |

  Failed per scenario (hover/straight/turns/altitude/circle/figure_eight): outside_range 5/19/24/14/21/27;
  strong_range 29/30/30/30/30/30 (the one survivor: hover, seed 300009, mean wind 10.66 m/s); beyond_strong 30 in every
  scenario. Sampled mean wind speeds fall inside the definitions (strong 10.14–13.96, beyond 14.07–15.98 m/s).
- **Reading note (descriptive):** in strong_range / beyond_strong PID leaves the 3 m bound (`out_of_bounds`) after a
  median of 23 / 18 decisions (2.3 / 1.8 s). RMSE covers only what was flown and is capped in practice by the 3 m bound,
  so it does **not** rise with wind once almost every flight fails (1.65 / 1.60 m vs 1.64 m in outside_range). For
  these profiles failure counts and decisions flown carry the information; RMSE comparisons there need the flight
  length next to them. (Not a protocol change; to be considered for how M7 tables present the secondary profiles.)

Important: PID failing under `outside_range` (or stronger) is a **result, not a bug**. Do not tune PID, wind or
limits: they are shared with SAC.

## M2.5 Close out

Extract section E to `docs/eval_protocol_v1.md` (frozen). Checks `--check` 21/21, tests 23/23 in both interpreters.
Commit with explicit paths (with approval). Update README and CLAUDE.md status → M3.

**M2.5 result (2026-11-13, Confirmed).**
- E5 reporting rules (1)–(3) added before any learned controller was evaluated; implemented in `protocol_stats.py`
  (`binom_cdf`, `clopper_pearson`, `mcnemar_exact`, `failure_count`, `paired_failure_comparison`,
  `decisions_median_iqr`, `controller_report`) and `run_protocol.py report`. Venv tests for exact values (binomial CDF
  176/1024; McNemar (3, 7) p = 0.34375, (10, 0) p = 2/1024, capped at 1; Clopper–Pearson 5/10 = [0.187086, 0.812914],
  0/10 upper = 1 − 0.025^0.1, tail probabilities = 0.025 at the bounds), the rules (RMSE flag, secondary endpoint order
  and label) and PID-vs-itself = null.
- Dry run (`results\m2\m2_5_report_dryrun.{json,txt}`): PID vs the canonical PID run; every paired RMSE difference 0
  [0, 0], every McNemar p = 1, failure counts with exact CIs (outside_range 110/180, rate CI [0.536, 0.683];
  strong_range 179/180 [0.969, 1.000]; beyond_strong 180/180 [0.980, 1.000]; 0/180 → [0, 0.020]); median decisions
  (IQR) 154 (84–214) in still_air / training_range / unseen_gusts, 62 (39–95) outside_range, 23 (21–29) strong_range,
  18 (17–19) beyond_strong. Negative test: a controller run with one realization entry altered (+1e-9 s) is refused.
- Frozen: `docs\eval_protocol_v1.md` (section E verbatim + protocol hash `d3f677dc…` + blob IDs: eval_protocol_v1.py
  5b490ddd, protocol_stats.py 328fcbdc, run_protocol.py cff95f85, run_common.py 1fb19f77, controllers.py 9fa925b4).
- Checks 21/21 and tests **50/50** in both the venv (3.13.15, numpy 2.5.4) and Isaac `python.bat` (3.12.14, numpy
  2.3.1); no test skipped. *(Superseded by M2.5 (a) below: the freeze was redone after adding the adapters.)*

**M2.5 (a): controller adapters before freezing (Claire chose option (a), 2026-11-13).** `controllers.py` was in the
frozen set while `pid_ppo` / `pid_sac` were stubs, so the adapters were implemented first (E3b):
- `controllers.py`: `PidPpoController`, `PidSacController`, `check_ppo_contract`, `check_sac_contract`,
  `check_spaces`, `saved_model_info` (zip read without unpickling), `ControllerRefused`. Contract checks run before any
  SB3 load. `run_protocol.py fly`: `--model`, `--reference-realizations`; controller built before SimulationApp;
  `bind_env` space check; realization guard; `meta.action_check`, `meta.realization_guard`.
- Venv tests (56 in total): refusals (no model, missing file, pid with a model, missing/mismatched/extra-key/no-policy
  PPO contract, SAC contract missing, SAC training group differing → `check_training` refuses; all before SB3 reads the
  file), space mismatch (obs and action), realization guard (1-ulp wind_speed, hash, missing key), and positive loads
  (tiny PPO and SAC on a spaces-only env act deterministically; SAC through the team loader).
- Isaac smoke (`results\m2\m2_5_adapter_smoke\smoke_summary.{json,txt}`; raw runs gitignored), cell turns /
  training_range / validation seed 400000: PID (reference), team SAC checkpoint (sha256 2d9e5cd0…, saved SB3 2.9.0,
  loaded under 2.7.0, 1,500,000 timesteps, contract 8b9f6283…), untrained 256×256 PPO (`tools\make_untrained_ppo.py`,
  model in gitignored `scratch\`): each **flew, actions finite, realizations matched**; spaces equal the env's; loaded
  before SimulationApp without the torch._dynamo error. Negative: SAC against the canonical reference (no seed 400000)
  was refused at the first reset and wrote nothing. The process exit code is 0 even for a refused/failed run
  (Inference: SimulationApp.close() ends the process before the exception propagates); failures show as `RESULT: FAIL` and missing results files.
  **Rule (Claire):** a run succeeds only if its log has `RESULT: DONE` and its outputs exist; never judged by exit code.
  `run_protocol.py` is not changed for this.
- **PID re-flown with the final code:** M2.4b run moved (`git mv`, unchanged) to `results\m2\m2_4b_pid\`; new run
  `results\eval\eval-v1\pid\` (check 0 PASS, 1,080 flights in 38.1 min, 109,631 decisions, all actions finite,
  `RESULT: DONE`). **Bit-for-bit equal to M2.4b on all 1,080 flights (32 fields) and all 1,080 realization entries**
  (`results\m2\m2_5_equality.{json,txt}`); matched realizations across 6 profiles PASS
  (`results\m2\m2_5_baseline.{json,txt}`); PID-vs-PID dry run regenerated on this run (all differences 0 [0, 0], all
  McNemar p = 1). Its `meta.code_blobs` equal the frozen blob IDs.
- **Final freeze** (`docs\eval_protocol_v1.md`, regenerated, includes E3b): protocol hash `d3f677dc…` (unchanged);
  blob IDs eval_protocol_v1.py 5b490ddd, protocol_stats.py 328fcbdc, **run_protocol.py ce3a0da5**, run_common.py
  1fb19f77, **controllers.py 8dae58a9** (the M2.5 "Frozen" bullet above is superseded).
- M4 to-dos: training writes `ppo_contract.json` next to every saved model and an explicit `status.json`
  (running/done/failed + last step); before validating checkpoints, fly PID once on validation seeds 400000–400009 as
  the validation reference.

---

## E. Evaluation protocol v1 (FROZEN 2026-11-13 as `docs/eval_protocol_v1.md`; edits here do not change v1)

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

---

## Explicitly out of scope for M2

Any training or gradient update; PPO model, callbacks or config (M3+); SAC checkpoint loading or retraining (M7);
editing shared modules or the PID; the Dijkstra scenario (unless D2 changes).
