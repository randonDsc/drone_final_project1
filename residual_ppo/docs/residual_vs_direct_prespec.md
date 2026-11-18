# Pre-specification: PID + PPO residual vs PPO direct control

Labels: **Confirmed** (ran or read it), **Documented** (written in a plan or README), **Inference**, **Proposed**.

## 0. When this was written (Confirmed)

- Written and committed on 2026-11-15 at repository HEAD **5c8bcb8** (`claire/drone-simulation`), the commit that
  recorded the finished PPO residual run `ppo_s0_1p5m`.
- At that point **no direct-control environment, policy, training script, model or run exists**: there is no file,
  config, checkpoint or result for it anywhere under `residual_ppo/`.
- **No learned controller had yet been flown on the eval seeds 300000–300029**: `results/eval/eval-v1/` held only the
  PID runs (`pid/`, `pid_validation/`). The only learned-controller flights so far were smoke tests on validation
  seed 400000 (M2.5 adapter smoke, M5 S5); none of them produced tracking metrics that were reported.
- So everything below is fixed before either side of the comparison has an eval result.

## 1. Context and the ablation line (decided by Claire, 2026-11-15)

Primary ablation line, owned end to end: **PID → PID + PPO residual → PPO direct control → continual PPO.**
This document pre-specifies only the step **PID + PPO residual vs PPO direct control**. Continual PPO needs its own
pre-specification, written before its model exists. (CLAUDE.md recorded continual PPO / CBP as out of scope on
2026-11-12; including it in the ablation line is Claire's 2026-11-15 decision.)

**Team SAC (PID + SAC residual) stays in as descriptive context only, with no cross-algorithm inferential claim.**
It differs from our PPO in algorithm, trainer (team script, SB3 2.9.0, CUDA torch) and parameter count. The
primary comparison uses **trainable** parameters:
- PPO: **198,921 trainable** = actor 99,848 + critic 99,073 (Confirmed, M3 V-checks).
- SAC: **301,066 trainable** = actor 100,872 + two critics 2 × 100,097. That is **501,260 including the two frozen
  target critics** (2 × 100,097), which follow the critics by Polyak averaging (tau 0.005) and are never updated by
  gradient. (Inference: arithmetic from `net_arch` pi/qf [256, 256], obs 128, action 4, in
  `models/sac_team/SOURCE.json`.)

SAC vs PID is reported through the frozen `report` mode in its own, clearly separate descriptive section. No
PPO-vs-SAC test of any kind is computed or reported.

## 2. Why this pre-specification is needed

Protocol v1 (`docs/eval_protocol_v1.md`, frozen) defines only **controller-vs-PID** comparisons: E5's primary
endpoint is "the paired difference controller − PID", and `run_protocol.py report` compares one controller run
against the canonical PID run. A residual-vs-direct comparison is therefore not covered by v1. It is added here as
a **separate analysis** that reuses v1's seeds, profiles, episode rules, metrics and statistics. It does not edit
any frozen file and does not change v1.

## 3. What is compared

- **Residual:** PID + PPO residual, the **final model** of `ppo_s0_1p5m` (`runs/ppo_path/ppo_s0_1p5m/final_model.zip`,
  sha256 `6b9e199310d1a93b4ada7c3e7b99c4341d89de3b085e0720e4c7eb29b4756234`), flown with the frozen
  `run_protocol.py fly --controller pid_ppo` on eval seeds 300000–300029.
- **Direct:** PPO direct control, its **final model**, flown on the same eval seeds.
- **Final models only, no checkpoint selection** (same rule as the PPO residual primary result,
  `docs/milestone3_plan.md`, "M4 / E6: primary result"). Checkpoints may form a descriptive learning curve on the
  validation seeds 400000–400009 only.
- Both controllers deterministic (`predict(obs, deterministic=True)`), as in E3.

## 4. Endpoints and statistics

**Primary endpoint (per profile).** For each wind profile, the paired difference **direct − residual** in position
RMSE, paired by (scenario, profile, seed): per seed, the mean over the 6 scenarios of
`position_rmse(direct) − position_rmse(residual)`; reported as the mean over the 30 eval seeds with a **95% percentile
bootstrap CI over seeds** (each resample keeps a seed's 6 scenarios together), **10,000 resamples, RNG seed 0**,
exactly the E5 settings. Negative = direct control tracks better. Direct control "improves on the residual" in a
profile only if the CI is entirely below 0; "is worse" only if it is entirely above 0.

**Paired failure comparison (every profile).** Pairs = (scenario, seed) within a profile (6 × 30 = 180). McNemar
**exact** test on the paired `failed` flags, with b = pairs where only direct failed and c = pairs where only the
residual failed; two-sided p = min(1, 2·P(X ≤ min(b, c))), X ~ Binomial(b + c, 0.5); p = 1 if b + c = 0. Reported with
b, c and the both/neither counts, plus each controller's failure count out of 180 with an exact **Clopper–Pearson**
95% CI.

**Implementation: reuse `evaluation/protocol_stats.py` unchanged.** `controller_report(ctrl_flights, pid_flights, …)`
is generic in its two flight lists: called with the direct flights as `ctrl_flights` and the residual flights in the
`pid_flights` slot, it computes exactly the statistics above (`bootstrap_mean`, `paired_failure_comparison`,
`mcnemar_exact`, `clopper_pearson`, `decisions_median_iqr`) with the E5 rules. A small new script (not frozen)
calls it and relabels the output "residual" instead of "PID". Before computing anything, it refuses unless both runs
have identical realizations (`wind_seed`, `wind_speed`, `route_seconds`, `max_decisions`) for every
(scenario, profile, seed) and both match the canonical PID `realizations.json`.

**E5 rule 2, carried over verbatim** (`eval_protocol_v1.md`, E5 rule 2; "the evaluated one or PID" reads here as
"direct or residual"):
> **RMSE never alone when there are failures.** In any profile where either controller (the evaluated one or PID)
> has at least one failed flight, position RMSE is reported and interpreted only together with the paired failure
> comparison (1), never on its own.

**E5 rule 3 applies unchanged to `strong_range` and `beyond_strong`:** endpoints in the order (a) failure count
(exact CI), (b) paired McNemar test, (c) median decisions flown (IQR), (d) position RMSE labelled "over flown portion
only".

**Ceiling fallback: decisions survived (added by amendment A3, section 10).**
- **When it applies:** only in a profile where **both** controllers fail **all 180** flights. Then every pair is
  "both failed", there are no discordant pairs (b = c = 0, McNemar p = 1), and the failure test cannot tell the
  controllers apart.
- **Statistic:** the paired difference **direct − residual in decisions survived per flight**, using the existing
  per-flight `decisions` field (decisions flown before the episode ended), paired by (scenario, profile, seed). Per seed
  it is the mean over the 6 scenarios of `decisions(direct) − decisions(residual)`
  (`protocol_stats.per_seed_mean` with value `f["decisions"]`). It is reported as the mean over the 30 eval seeds with
  `protocol_stats.bootstrap_mean`: 95% percentile seed bootstrap, each resample keeping a seed's 6 scenarios together,
  **10,000 resamples, RNG seed 0**, exactly the E5 settings. Each controller's `protocol_stats.decisions_median_iqr`
  is reported next to it. `protocol_stats.py` is used unchanged.
- **Sign:** positive means direct control survives longer. This is the opposite of the RMSE difference, where
  negative is better.
- **Reading:** direct "survives longer" only if the CI is entirely above 0, and "survives less long" only if it is
  entirely below 0. Otherwise the result is inconclusive.
- **Pairing is valid:** the realization guard (section 4, implementation) makes `route_seconds` and `max_decisions`
  identical for both controllers in every pair, so both have the same maximum possible number of decisions.
- **Scope, stated as a rule:** this is a **graded fallback for ceiling cases only**. It **never replaces an
  informative failure test**: when at least one controller fails fewer than 180 of 180 flights in a profile, the
  McNemar comparison decides there and the fallback is not used, not even as a tie-breaker. It is not computed for the
  primary profiles' endpoints, and it does not change E5 rules 2 or 3. E5 rule 3 (c), median decisions, is still
  reported as before.
- **Not covered:** b = c = 0 *without* a full double ceiling, where both controllers fail exactly the same flights but
  not all of them. That case is not a ceiling. It is reported as the McNemar result (p = 1) and the fallback does not
  apply.

**Undertraining check (added by A5; descriptive, not an endpoint, never changes a verdict).**
- Both arms train with target_kl 0.02 (parity). Because a direct policy starts from "hover in place" rather than from a
  working PID, the same KL throttle may undertrain it. This check makes that visible; it does not correct for it.
- **Reported for BOTH arms:**
  - total gradient steps (optimizer-hook count; residual arm 10,305) with SB3 epochs alongside;
  - the change from the **800k checkpoint to the final model** on the **validation seeds 400000–400009** (never the
    eval seeds), in the same quantities `evaluation/learning_curve.py` reports for each profile: mean position RMSE,
    paired difference vs PID, failures.
- **Flag rule:** an arm is flagged **"possibly undertrained"** if its mean validation `training_range` position RMSE
  improves by **more than 0.02 m** from the 800k checkpoint to the final model, i.e.
  RMSE(800k) − RMSE(final) > 0.02 m. If either model fails in `training_range`, the failure counts are reported with it
  (E5 rule 2).
- **Known before writing (Confirmed, `results/eval/eval-v1/learning_curve.txt`):** the residual arm moves from 0.122 to
  0.097 m, a change of **0.0250 m** (paired diff vs PID −0.3523 → −0.3773, same PID flights). It **will be flagged**.
  That is the intended behaviour: the flag says "still improving at the end of the budget", for either arm.

## 5. Profile roles (quoted from `eval_protocol_v1.md`, E5)

> One primary result per profile; `still_air` is reported but is not a primary endpoint. […]
> **Primary profiles = the 3 team profiles only** (`training_range`, `unseen_gusts`, `outside_range`).
> `strong_range` and `beyond_strong` (D4) are **secondary**: the same paired statistic is reported for them, labelled
> secondary, and they cannot overturn or add to a primary result. `still_air` is reported prominently (nominal
> tracking; see CLAUDE.md M7 notes) but is not primary.

There is one primary result per primary profile and no correction for multiple profiles, the same as E5.

## 6. Pre-registered prediction (Claire, 2026-11-15; P1 replaced by P1′ on 2026-11-16, A4; P2 retired and P1′ (c) restated, A5)

**Basis (Confirmed, from `ppo_s0_1p5m`):**
- Both residual controllers (PPO and the team SAC) share the residual limit
  **L = [0.8 N thrust, 0.05 N·m roll, 0.03 N·m pitch, 0.02 N·m yaw]** (`path_sac_config.RESIDUAL_LIMITS`).
- In the `full` training stage the PPO run clipped **24.79%** of sampled action components to [−1, 1].
  (These are stochastic, exploration-noise actions, std about 0.21. They show the policy mean pressing against L;
  they do not prove the deterministic policy sits at L.)
- **Rotor saturation stayed at 0.0** in every training rollout (`flight/saturation_rate`, all 733 rollouts), and in
  every canonical PID eval flight. So the motors had headroom that the residual could not use.
- The mean `residual_norm` of training flights rose with the curriculum: 0.685 (short_slow), 0.672 (turns),
  0.933 (climbs_loops), 1.224 (full; per-rollout max 1.318). `residual_norm` is the mean over decisions of the norm of
  the normalized residual action after it is clipped to [−1, 1] per component (`PathCore.apply_action`,
  `PathTrace.record`), so its ceiling is 2.0.

**P1 is RETIRED (amendment A4, 2026-11-16, section 10).** P1 predicted that direct control, "not bounded by L",
would fail *less* than the residual in strong wind. A4 records, as a code fact, that the direct arm is the one with
**less** horizontal authority, so P1 would fail by construction. Its text up to A3 is in commit c3198fa. It is
replaced by P1′ below.

**Prediction (A4).** Direct control enters the control cascade **upstream** of the PID's 1.5 m/s² horizontal
acceleration cap; the residual enters **downstream** of the whole cascade and is not subject to that cap (it is bounded
by L instead). So direct control is **comparable to or better than** the residual where the cap does not bind, and
**fails more** where drag exceeds the cap. Stated as checkable outcomes:

- **P1′ (a) (where the cap does not bind; primary profiles `training_range` and `unseen_gusts`).** Direct control is
  comparable to or better than the residual in position RMSE (primary endpoint, section 4, direct − residual).
  E5 rule 2 applies first if either controller has a failure in the profile.
- **P1′ (b) (strong wind; secondary profiles, E5 rule 3 order).** Direct control **fails more** than the residual,
  because its horizontal authority is capped below the drag.
  - **Decided by `strong_range`:** b > c (b = pairs where only direct failed) with McNemar exact p < 0.05.
  - **`beyond_strong` counts only under the A1 ceiling rule:** it is evaluated the same way (b > c, McNemar exact p)
    only if at least one controller fails fewer than 180 of 180, and is then reported as corroborating (b > c) or
    contradicting (c > b) P1′ (b). Otherwise it is uninformative through the failure test.
  - **At a double ceiling (A3):** the ceiling fallback in section 4 (decisions survived, direct − residual) is
    reported as a **graded** result. In `strong_range` it decides P1′ (b), labelled "graded (ceiling fallback)": holds
    (graded) if the CI is entirely **below** 0 (direct survives less long), falsified (graded) if it is entirely above
    0, inconclusive otherwise. In `beyond_strong` it is graded corroboration (CI entirely below 0) or graded
    contradiction (CI entirely above 0), otherwise inconclusive. A graded verdict is always reported as such.
  - Known when A4 was written (Confirmed, `results/eval/eval-v1/pid_ppo/report_vs_pid.txt`): the residual fails
    **16/180** in `strong_range` and **107/180** in `beyond_strong`. Neither profile can therefore reach a double
    ceiling, so the A3 fallback cannot apply in either; both are decided by McNemar.
  - McNemar exact p < 0.05 needs at least 6 discordant pairs, all in one direction (b + c = 6, min(b, c) = 0 gives
    p = 2 · 0.5^6 = 0.031; b + c = 5 gives 0.0625).
- **P1′ (c) (mechanism evidence; reported either way, not a test of (b); restated by A5, section 10).** In each of
  `strong_range` and `beyond_strong`, the direct arm's mean `accel_cap_binding_frac` (mean over the profile's 180 eval
  flights, as in the A4 tables) is **within 0.15 of PID's** and **strictly exceeds the residual arm's**:
  - `strong_range`: direct **≥ 0.57** (PID 0.716465 − 0.15 = 0.5665, rounded to two decimals as written by Claire)
    **and > 0.128161** (residual, `pid_ppo`);
  - `beyond_strong`: direct **≥ 0.59** (PID 0.742279 − 0.15 = 0.5923, rounded the same way) **and > 0.539473**
    (residual).
  - **Supported** if both conditions hold in both profiles. **Falsified** if the direct arm's mean is at or below the
    residual arm's in **both** profiles. Any other outcome is reported as **not supported** (mixed). If (c) is not
    supported, the authority explanation is not supported, whatever (b) shows.
  - The two-decimal thresholds 0.57 and 0.59 are the operative ones; they are fixed here so the rounding cannot be
    chosen after the result. "Within 0.15" is a lower bound only: a direct mean above PID's also satisfies it.
  - **Recorded at writing (Confirmed, A4 step 1 tables, section 10):** `accel_cap_binding_frac` is computed by the
    frozen `Diagnostics` (`evaluation/run_protocol.py:137-149`) on the **PID part** of the command (final − correction).
    For the residual arm it therefore says how often the PID part is capped *before* the residual is added; it does not
    measure a limit on the residual itself. PID, which is wholly subject to the cap, averages 0.716 and 0.742 there and
    no PID flight exceeds 0.917, which is why A4's "approaches 1.0" was wrong (A5).
  - **How the direct runner computes it:** with the same frozen `Diagnostics` class, unchanged. The direct env exposes
    `core.correction` = 0 (so the "PID part" is the whole inner-loop command), `core.desired` = the inner loop's desired
    attitude, and `core.pid.acceleration_capped` = the shared `ReferencePID.acceleration_capped`
    (`path_sac_policy.py:212-233`). This is to be verified with `--check0`-style diagnostics-off-vs-on equality before
    the direct eval flights.
- **P2 is RETIRED (A5), superseded by P1′ (a).** It read: "Little or no RMSE advantage for direct in
  `training_range`: the lower bound of the direct − residual 95% CI is ≥ −0.02 m." It existed only to operationalise
  the retired P1's claim that direct control would not help in `training_range`. P1′ (a) predicts direct control is
  comparable to *or better than* the residual there, so keeping both would make the pre-specification
  self-contradictory. P1′ (a)'s own falsifier (section 7) stands.
- **P3 (mechanism; descriptive, not a test).** In the residual run, the mean `residual_norm` (ceiling 2.0 = all four
  components at their limits) is higher in `strong_range` and
  `beyond_strong` than in `training_range`, and rotor `saturation_rate` stays at or near 0 in those profiles.

`outside_range` and `still_air` are reported with no prediction attached. (Before A4 this sentence also listed
`unseen_gusts`, which P1′ (a) now covers.)

## 7. What would falsify the prediction

- **P1 is retired (A4).** Its falsifiers no longer apply.
- **P1′ (a) is falsified** if the direct − residual paired RMSE 95% CI lies **entirely above +0.02 m** in
  `training_range` **or** in `unseen_gusts`. Otherwise it holds. If either controller fails in that profile, E5 rule 2
  applies first.
- **P1′ (b) is judged on `strong_range`:**
  - **holds** if b > c with McNemar exact p < 0.05;
  - is **not supported** (inconclusive, not falsified) if b > c with p ≥ 0.05, or b = c, or c > b with p ≥ 0.05;
  - is **falsified** if c > b (direct fails fewer) with McNemar exact p < 0.05 in `strong_range`, **or** in
    `beyond_strong` when that profile is informative (at least one controller below 180/180);
  - at a double ceiling only, it is judged by the graded fallback (section 4, A3) with the signs in section 6. Per
    the residual's known failure counts (16/180, 107/180), this cannot occur in either profile.
- In any profile at the double ceiling (both controllers fail 180/180), the failure test is uninformative. That
  profile is judged by the **ceiling fallback** (section 4, A3): the paired difference in decisions survived per
  flight with its seed-bootstrap CI, labelled "graded (ceiling fallback)". Median decisions flown (E5 rule 3 (c)) is
  reported as well. Whenever at least one controller is below 180/180, the fallback is not used.
- **P1′ (c)** (restated by A5) is reported in every case: supported if, in both `strong_range` and `beyond_strong`,
  the direct mean `accel_cap_binding_frac` is ≥ 0.57 / ≥ 0.59 and > 0.128161 / > 0.539473; **falsified** if the direct
  mean is at or below the residual's in both profiles; otherwise not supported (mixed). If it is not supported, the
  authority explanation is not supported, even if (b) holds.
- **P2 is retired (A5)**; its falsifier no longer applies.
- **P3** stays a descriptive report on the residual arm. It was written as the mechanism for the retired P1, and
  under A4 it no longer bears on the P1′ mechanism, which is (c). Its values are already known from M7 (Confirmed,
  `pid_ppo`): mean `residual_norm` 1.137 in `training_range`, 1.464 in `strong_range`, 1.660 in `beyond_strong`; rotor
  saturation 0 in every profile.
- The bullet "the whole prediction is falsified if direct control is worse than the residual in strong wind" is
  retired with P1 (A4). Under P1′ that outcome is the prediction (b).

Any outcome is reported, whatever its direction.

## 8. Not fixed by this document (must be fixed in a committed addendum BEFORE the direct-control model is trained)

The frozen `run_protocol.py fly` can only build the residual env (`path_sac_env.make_path_sac_env`) and the
controllers `pid`, `pid_ppo`, `pid_sac` (frozen `controllers.py`), so it cannot fly a direct-control policy. Still open:

1. **The direct-control env:** action space and its mapping to rotor commands or thrust/torques, the action limits,
   the observation (same 128-dim observation, or not), and whether the PID is removed entirely.
   The reward, curriculum, failure rules and 10 Hz decision rate are kept from the shared env.
2. **How it is flown:** either (a) protocol v2 (new controller and env in the frozen files; every controller re-flown),
   or (b) a separate, unfrozen runner that reuses `eval_protocol_v1` (seeds, profiles, scenarios, E3 episode rules),
   `PathTrace.metrics`, and the realization guard against the canonical PID `realizations.json`. With (b),
   `initial_obs_sha256` is checked only if the observation is identical.
3. **The training budget and hyperparameters:** Proposed default is the same as the residual run (SB3 PPO, MlpPolicy
   256×256, the same hyperparameters, 1.5M decisions = 733 rollouts, seed 0, same curriculum). Any difference must be
   written down with its reason.
4. ~~(Added by A4.) The numeric criterion for P1′ (c) and how the direct runner computes `accel_cap_binding_frac`.~~
   **Fixed by A5** (section 6, P1′ (c)).
5. ~~(Added by A4.) Whether P2 stays.~~ **Fixed by A5: P2 retired.**
6. **(A5.) Inner loop: D-env-1 option (a).** Direct control uses the shared, capped inner loop **unchanged**: the
   shared `PIDController.update` (1.5 m/s² cap included), driven by target substitution through its public target
   interface, the same technique `ReferencePID` uses (`path_sac_policy.py:112-122`). No fork or reimplementation of
   `controller.py`.

Changing anything in sections 3–7 after the direct-control model exists invalidates this pre-specification.

## 9. Today's write-up (Claire, 2026-11-15)

- **Primary result:** the frozen `run_protocol.py report` of **PID + PPO residual vs PID** on the eval seeds.
- **Separate descriptive section:** team **SAC vs PID**, through the same frozen report, as context only. It contains
  no PPO-vs-SAC test or paired statistic of any kind.
- The residual-vs-direct comparison above is future work until the direct-control model exists.

## 10. Amendments (A1–A5 made before any direct-control code existed; A6–A10 after the direct-control env code was written and checked, but before any direct-control model, training run or eval result existed)

**A1 and A2 (Claire, 2026-11-15; committed on top of 3218b53, the original pre-specification).** State when they
were made (Confirmed): HEAD 3218b53; no direct-control env, policy, script, config, model or run under `residual_ppo/`
(a search for direct-control code found nothing; `runs/ppo_path/` holds only the residual run, its killed first
launch and the smoke runs; `models/` holds only `sac_team`); no flight of any learned controller on the eval seeds
(`results/eval/eval-v1/` holds only `pid/` and `pid_validation/`); no Isaac process running. The M7 flights of the
residual controller had not started.

- **A1: P1 rewritten (sections 6 and 7).** Before: P1 required c > b in **both** `strong_range` and `beyond_strong`,
  with p < 0.05 in at least one; b ≥ c in both falsified it. After: P1 is decided by `strong_range` (c > b, McNemar
  exact p < 0.05); `beyond_strong` counts only if at least one controller fails fewer than 180/180, and is otherwise
  reported as uninformative due to a ceiling effect.
  **Why:** the canonical PID fails **180/180** flights in `beyond_strong` and 179/180 in `strong_range` (Confirmed,
  `results/eval/eval-v1/pid/`). If both of Claire's controllers also hit that ceiling, there are no discordant pairs
  (b = c = 0). Under the original wording that would have counted as "b ≥ c", so a prediction that could not be
  evaluated would have been scored as false. A ceiling says nothing about which controller has more authority, so it
  must be reported as uninformative, not as a result. The same ceiling logic is applied to `strong_range` (P1 is
  unevaluable there if both controllers fail all 180 flights).
- **A2: parameter counts (section 1).** The primary comparison is trainable parameters: SAC 301,066 vs PPO 198,921.
  SAC's 501,260 is given only as "including the two frozen target critics".

Because A1 and A2 were made before the direct-control model existed and before any learned eval flight, they keep
the pre-specification valid. They do not touch sections 3–5 (what is compared, endpoints, statistics, profile roles).

**A3 (Claire, 2026-11-15; committed on top of 5fefb29): ceiling fallback, written BLIND.**
- **What:** section 4 gains the ceiling fallback, the paired difference direct − residual in decisions survived per
  flight. It uses the E5 bootstrap settings and the existing `decisions` field, through `per_seed_mean`,
  `bootstrap_mean` and `decisions_median_iqr` in `protocol_stats.py`. P1 (section 6) and section 7 now use it at a
  double ceiling.
  - Before A3, a double ceiling made `strong_range` "unevaluable" and `beyond_strong` "uninformative", and median
    decisions was only a descriptive contrast.
  - The fallback is a **graded fallback for ceiling cases only**. It **never replaces an informative failure test**:
    if at least one controller fails fewer than 180/180, McNemar decides and the fallback is not used. Sections 3, 5
    and the primary endpoint are unchanged.
- **When it was made (Confirmed):**
  - HEAD **5fefb29**.
  - **No direct-control env, policy, script, config, model or run exists**: a search of `residual_ppo/` (outside
    `results/`) for direct-control code finds only this document; `runs/` holds only `ppo_path/` (the residual run,
    its killed first launch, the smoke runs); `models/` holds only `sac_team`. No Isaac process was running.
  - **Claire had not seen any `pid_ppo` or `pid_sac` result.** The six M7 flight runs had finished (12:04:44 to
    15:10:53 local, as reported by Claire). Their output folders exist on disk: `results/eval/eval-v1/pid_ppo/`,
    `pid_sac/`, and `ppo_validation/ppo_150000`, `ppo_400000`, `ppo_800000`, `ppo_final`. Only the folders' existence
    was checked; **no `results.json`, report, log or failure count in them was opened or summarised**, by Claire or
    by Claude, before this amendment was committed and pushed.
- **Why:**
  - A1 made a double ceiling "unevaluable", which is honest but throws away information. Two controllers that both
    fail every flight can still differ a lot in how long they stay in the air.
  - Decisions survived is a graded, already-recorded per-flight quantity. Paired by (scenario, profile, seed), it is
    measured against identical realizations and identical `max_decisions`.
  - It is fixed now, before any result is seen, so that using it cannot be a choice made after looking at the
    numbers. Limiting it to the double ceiling keeps the failure test as the primary strong-wind endpoint wherever
    that test can say anything.

**A4 (Claire, 2026-11-16; committed on top of 1efd663): P1 retired, replaced by P1′.**
- **What:** section 6 retires P1 and adds P1′ (a), (b), (c). Section 7 replaces P1's falsifiers with P1′'s and
  retires the "whole prediction is falsified" bullet. Section 8 gains items 4 (numeric criterion for (c), and how
  the direct runner computes the diagnostic) and 5 (whether P2 stays). Sections 3–5 (what is compared, endpoints,
  statistics, profile roles), P2 and P3 are unchanged.
- **Reason, as a code fact (Confirmed by reading the code; not an outcome):**
  - Direct control enters the cascade **upstream** of a **1.5 m/s² horizontal acceleration cap**. The cap is
    `min(1.5, a_vert · tan(max_tilt))`, with 1.5 a literal in the shared `controller.py:213-216`. At our mass of
    **1.618614 kg** it is worth 1.5 × 1.618614 = **2.43 N**. At hover a_vert · tan 10° = 1.73 m/s² > 1.5, so the 1.5
    is the binding term.
  - The residual enters **downstream of the whole cascade**: `final = command + self.correction`, added once before
    the mixer (`path_sac_policy.py:575-576`). It is not subject to the cap; it is bounded by L instead.
  - Drag is 0.5 · ρ · Cd · A · v² with ρ = 1.225 kg/m³ and Cd · A = 1.0 × 0.03 = **0.03 m²** (`sim_config.py:245-250`;
    `wind.py:89-90`), i.e. 0.018375 · v² N. It reaches 2.43 N at **v ≈ 11.5 m/s** (Inference, arithmetic: steady
    airspeed equal to the mean wind, gusts and the drone's own motion ignored). That is inside `strong_range`
    (10–14 m/s) and below all of `beyond_strong` (14–16 m/s).
  - The shared gains carry the same warning (Documented, `sim_config.py:192-194`): "The limit is about 10 m/s of
    steady wind by the drag model, and controller.py caps the total at 1.5 m/s^2." That comment does not state how
    its 10 m/s was computed; the 11.5 m/s above uses the configured Cd·A and our measured mass.
  - P1 assumed direct control was the arm with **more** authority. The opposite is true, so P1 would fail by
    construction.
- **Verification that decided A4 (Claire's condition: write A4 only if PID's `accel_cap_binding_frac` is high in
  `outside_range`, `strong_range` and `beyond_strong`).**
  - Computed from the **committed** `results/eval/eval-v1/{pid,pid_ppo,pid_sac}/flights.csv` (HEAD 1efd663; working
    copies checked equal to the HEAD blobs), 180 flights per cell. The value per flight is the fraction of its
    physics steps on which the frozen `Diagnostics` (`evaluation/run_protocol.py:137-149`) found the PID part's
    horizontal demand at the cap (≥ 0.98 × cap, `path_sac_policy.py:212-233`), split into the 1.5 cap and the tilt
    cap.
  - **`accel_cap_binding_frac`: mean, and flights above 0.9 (of 180).**

    | Profile | PID mean | PID > 0.9 | PID max | PID+PPO mean | > 0.9 | PID+SAC mean | > 0.9 |
    |---|---|---|---|---|---|---|---|
    | training_range | 0.053 | 0 | 0.353 | 0.000 | 0 | 0.000 | 0 |
    | unseen_gusts | 0.046 | 0 | 0.349 | 0.000 | 0 | 0.000 | 0 |
    | outside_range | **0.548** | 0 | 0.878 | 0.005 | 0 | 0.002 | 0 |
    | still_air | 0.028 | 0 | 0.432 | 0.000 | 0 | 0.000 | 0 |
    | strong_range | **0.716** | 1 | 0.917 | 0.128 | 0 | 0.119 | 0 |
    | beyond_strong | **0.742** | 0 | 0.810 | 0.539 | 3 | 0.569 | 1 |

  - **`tilt_cap_binding_frac`: 0.000 in every flight** of all three controllers and all six profiles (0 flights
    above 0.9). The tilt cap never binds; only the 1.5 m/s² cap does.
  - Supporting detail (Confirmed): in `outside_range`, PID's failed flights have mean 0.684 (n = 110) and its
    completed flights 0.334 (n = 70). Per scenario, PID's strong_range and beyond_strong means are 0.705–0.746 in all
    six scenarios, hover included. Step-weighted PID fractions are 0.470 / 0.702 / 0.742.
  - **Reading (Inference; "high" had no numeric threshold before this was computed):** the cap binds on the
    **majority** of PID's physics steps in the three profiles (55%, 72%, 74%), against 3–5% in `still_air`,
    `training_range` and `unseen_gusts`. The condition was judged met on that basis. It does **not** bind on nearly
    every step: almost no PID flight is above 0.9 (0, 1 and 0 of 180).
  - Script (not committed): session scratchpad `cap_binding.py`.
- **When it was made (Confirmed):**
  - HEAD **1efd663**.
  - **No direct-control env, policy, script, config, model or result exists:** a search of `residual_ppo/` (outside
    `results/`) for direct-control code finds nothing; `runs/` holds only `ppo_path/` (the residual run, its killed
    first launch, smoke runs); `models/` holds only `sac_team`; `results/eval/eval-v1/` holds only `pid`, `pid_ppo`,
    `pid_sac`, `pid_validation` and `ppo_validation`. No Isaac (kit) process was running.
  - **Unlike A1–A3, the residual arm's eval results HAD been seen** (M7, committed 0ec9121, write-up 1efd663),
    including its failure counts (16/180 `strong_range`, 107/180 `beyond_strong`) and, in step 1, its
    `accel_cap_binding_frac`. P1′ is a prediction about the direct arm only, whose results do not exist, but its
    comparator is known. That is why the residual-arm values are written into P1′ (b) and (c) explicitly.

**A5 (Claire, 2026-11-16; committed on top of 9fb18b9): P1′ (c) restated, P2 retired, undertraining check, D-env-1.**
- **(a) P1′ (c) restated (section 6, section 7).** A4's "approaches 1.0" was wrong: PID, wholly subject to the cap,
  averages only 0.716 and 0.742 in `strong_range` and `beyond_strong`. New criterion: the direct arm's mean is within
  0.15 of PID's (≥ 0.57 / ≥ 0.59) **and** strictly exceeds the residual arm's (0.128161 / 0.539473), in each strong
  profile. Falsified if direct is at or below the residual in both. The direct runner computes the quantity with the
  frozen `Diagnostics` class unchanged.
- **(b) P2 retired, superseded by P1′ (a).** P2 only existed to operationalise the retired P1's claim that direct
  control would not help in `training_range`; keeping it beside P1′ (a), which allows direct to be better there, would
  make the pre-specification self-contradictory. P1′ (a)'s own falsifier stands.
- **(c) Undertraining check (section 4).** Reported for both arms: gradient steps, and the 800k → final change on the
  validation seeds; an arm is flagged "possibly undertrained" if validation `training_range` RMSE improves by more than
  0.02 m over that interval. The residual arm moves 0.0250 m and will be flagged, as intended.
- **D-env-1 confirmed as option (a)** (section 8, item 6): direct control uses the shared capped inner loop unchanged,
  through target substitution; no fork of `controller.py`.
- **When it was made (Confirmed):**
  - HEAD **9fb18b9**, before any direct-control code was written: a search of `residual_ppo/` (outside `results/`) for
    direct-control code finds nothing; `runs/` holds only `ppo_path/`; `models/` holds only `sac_team`;
    `results/eval/eval-v1/` holds no direct-control result. No Isaac (kit) process was running.
  - As for A4, the residual arm's eval results and validation learning curve had been seen; the residual-arm numbers
    used above are quoted from the committed files. No direct-control model or result exists.

**A6–A10 (Claire, 2026-11-16; committed on top of b684b81, with the direct-control code): env checks and the
inner-loop findings.**
- **When they were made (Confirmed):**
  - The direct-control env, configs, check and diagnostic scripts were written and checked: `direct_control/`,
    `config/direct_*`, `tools/check_direct_env.py`, `tools/hold_diagnostic.py`, `tests/test_direct_control.py`.
  - **No direct-control model, training run (smoke or main), or eval / validation result exists.** `runs/` holds
    only `ppo_path/`; `results/eval/eval-v1/` holds no direct-control result.
  - All Isaac evidence below comes from zero-command and constant-command flights on the **validation seeds
    400000–400009**, with no model, run by Claire from her own window. Files are in `results/m8/`: gate runs 1
    (01:04) and 2 (01:33); hold diagnostic v1 (01:35); m8b suite (03:05–03:10).
- **None of A6–A10 changes sections 3–7** (what is compared, endpoints, statistics, profile roles, P1′, P3). They
  record env facts and what the comparison may and may not claim.

**A6 — the H2/H3 specification error (code-correctness gate, not an endpoint).**
- The first gate (`tools/check_direct_env.py`, Part H) required a zero velocity command in still air to settle:
  "drift after 3 s ≤ 0.10 m" and "final speed ≤ 0.05 m/s". Those were guesses from an ideal 0.83 s velocity loop.
  Runs 1 and 2 failed them identically: max drift 0.2118 m, max final speed 0.1387 m/s.
- **Still air was correctly delivered by the env (Confirmed from the code and the logged wind).**
  - A hypothesis that the test had run in wind was raised: the log prints "gust std=[1. 1. 0.35]". It was refuted
    from `wind.py`.
  - That line is printed once, when the sim is built, from the base config (`drone_sim.py:301-314`).
  - Every still_air reset draws calm wind with gust scale 0 (`hover_task.py:223-224`), and
    `GustingWind.reset_episode` multiplies both gust strengths by it (`wind.py:196-197`).
  - The m8b run logged the point wind every decision: exactly 0.000 m/s in every still-air flight.
- **The thresholds were wrong because the plant has a bounded limit cycle (A7), not because the test was wrong.**
- **Replacement:** H2/H3 became regression gates on the limit cycle, with bounds set from the m8b T1 measurements
  plus stated margins. Derivation in the `check_direct_env.py` docstring:
  - "limit-cycle amplitude within the measured band": 110–120 s RMS amplitude in [0.0539, 0.0874] m/s. Measured
    0.06739–0.07287, margin 20% relative. Logged wind must be exactly 0.
  - "limit-cycle amplitude ratio stable": amplitude(110–120 s) / amplitude(20–30 s) in [0.8265, 1.0977]. Measured
    0.9265–0.9977, margin 0.10 absolute. No early stop allowed.
  - The seeds are validation seeds, never the 300000-series eval seeds.

**A7 — the limit cycle in the shared inner loop (Confirmed measurements; mechanism partly open).**
- **Measured:** hover, zero command (direct) or zero residual (residual arm = PID alone), validation seeds
  400000–400009, no saturated physics step, no integrator activity on xy or height.

  | | Direct arm (T1: still air, 120 s) | Residual arm (T3: still air, 30 s) |
  |---|---|---|
  | Present in | 10/10 seeds | 10/10 seeds |
  | Period | 2.61 s | 2.82–2.88 s |
  | Velocity RMS amplitude | 0.068–0.075 m/s | 0.087–0.092 m/s |
  | Tilt gain (actual / commanded) at its own period | pitch 1.99, roll 2.02, phase −88° | pitch 1.79, roll 1.87, phase −76° / −70° |
  | Max point wind | 0.000 m/s | 0.000 m/s |

  - **Bounded over 120 s (T1):** R = amplitude(110–120 s) / amplitude(20–30 s) = 0.926–0.998 in all 10 seeds.
    The amplitude converges to the same level from below (v1 zero-command runs) and from above (v1 vx = 1 m/s runs).
  - **With gusts it is unchanged:** 3/3 seeds hold in T2 (direct, 120 s); 7 hold, 1 grows, 2 decay in T4 (residual,
    30 s; gust noise in 10 s windows).
- **The 10 Hz outer loop is not the cause.** In T1 and T3 the action is constant, so no policy feedback acts. T3
  shows the mode in the pure 120 Hz PID cascade with the residual held at exactly 0.
- **Consequence, stated plainly:**
  - Neither arm's still-air RMSE is clean "nominal tracking". Both arms fly through an inner loop that oscillates on
    its own.
  - The residual PPO's still-air improvement (validation 0.176 → 0.060 m RMSE, `results/eval/eval-v1/
    learning_curve.txt`) is therefore a **hypothesis** that the residual damps this mode, **not a demonstrated
    result**, unless it is tested. A test would be the same diagnostic with the trained residual flying.
  - The M7 write-up's still-air section must be read with this caveat.
- **Open question (Inference): the ~18% damping deficit.**
  - Units are N·m/rad (`controller.py:6`, `sim_config.py:207`, and the mixer's moment arms in metres).
  - The composite inertia about the vehicle COM is Ixx 0.0349 and Iyy 0.0312 kg·m², from the inertias the run logged.
  - An ideal rigid-body PD attitude loop (kp 0.2, kd 0.05) predicts a 2.62 s roll period, matching the measured
    2.61 s. kp / ω² = 0.0345 at 2.61 s, against the measured roll inertia 0.0349.
  - **Routh** on that ideal model says both arms are stable:
    - direct: kd 0.05 > 1.2·I = 0.0418 (roll) / 0.0375 (pitch);
    - residual: 0.0120 > 0.0115 (roll) / 0.0105 (pitch).
  - The measured tilt gain is about 20% above the ideal model in both arms (direct 1.99 / 2.02 vs 1.64 / 1.66;
    residual 1.79 / 1.87 vs 1.66 / 1.75). That implies an effective damping of about 0.041 N·m·s/rad, which sits on
    the direct arm's stability boundary.
  - What removes about 18% of the attitude damping in the sim, and what caps the amplitude, is not identified.

**A8 — the steady bias (Confirmed measurements; source unidentified).**
- **Measured:** the direct arm drifts at a steady velocity of about 0.02 m/s in still air (T1, t ≥ 10 s: mean vx
  +0.012 to +0.013, vy +0.014 to +0.015 m/s), with the same sign on all ten seeds.
- **Explained by an attitude offset:** actual − commanded tilt is pitch +0.081 to +0.091° and roll −0.096 to
  −0.109° (T1; v1 30 s: +0.035 to +0.128° and −0.116 to −0.143°), same sign on all ten seeds.
  - It follows because roll and pitch have no integrator (ki = 0), so a steady disturbance torque holds a steady
    attitude error.
  - That error is turned into velocity through g·tan(offset) / kd.
  - The same-signed offset is present in the residual arm (T3: pitch +0.019 to +0.089°, roll −0.020 to −0.159°),
    where the position loop absorbs it (mean velocity about 0).
- **Refuted with numbers (Confirmed):**
  - **Rotor-mass asymmetry:** rotors 0/1 (0.029236 kg) and 2/3 (0.030071 kg) are diagonal pairs. The COM shift is
    (+0.0003, −0.0082) mm, giving m·g·d = 4.0e-6 (pitch) and 1.3e-4 (roll) N·m. The offsets imply 2.8–3.8e-4 and
    4.5–4.9e-4 N·m: about 100× too small in pitch and 3.6× in roll.
  - **The mixer residual:** the mixer reads the live vehicle COM (`calibrated_com_offset` stays None, because
    `connect_imu` is only called by `DroneHoverEnv` in IMU mode). Allocation and thrust application use the same
    `ROTOR_POSITIONS`. A NumPy replay of the code path, including the float32 cast, leaves 2.5e-10 to 7.3e-10 N·m.
  - **A force-at-COM mismatch:** it would give 2.6e-3 / 8.6e-3 N·m, 6–20× too large and with the wrong pitch-to-roll
    ratio.
- **Untested candidate (Inference):** the visual propeller spin drives (`PROPELLER_SPIN`, drive torque limit 0.1 N·m,
  rotor links clamped by PhysX near 100 rad/s). A slight tilt of the joint axes would leak roll/pitch torque of
  about the measured size.
- **Consequence:** the direct policy must cancel about 0.02 m/s of steady drift with its own commands, which it can,
  since it observes velocity error. The residual arm's PID cancels it through position feedback. This is one of the
  "interface" differences between the arms, recorded here.

**A9 — `initial_obs_sha256` cannot match between arms; physical identity is proven instead (Confirmed).**
- Observation features 18–20 (attitude error vs the desired attitude) and 24–27 (command) are controller-derived.
  In the direct arm they come from the inner loop under the policy's command; in the residual arm, from the PID.
  No shadow PID is allowed anywhere (plan 3b). So the initial-observation hash differs in every cell by design.
- **Physical identity is established by the 360-cell reset comparison** in `tools/check_direct_env.py` (run 2,
  01:33, all PASS). The shared `PathSacEnv` and `DirectEnv` were reset on the same sim for every validation cell:
  - R1: `PathSacEnv` reproduces all 5 reference fields;
  - R2: `DirectEnv` reproduces `wind_seed`, `wind_speed`, `route_seconds`, `max_decisions` (360/360);
  - R3: drone state and mean wind are bit-identical (360/360);
  - R4: every controller-independent feature is bit-identical (360/360);
  - R5: seed + 1 is refused.
- **The future direct eval runner** guards every reset on the remaining four realization fields, plus that one-time
  check. It must not reconstruct a PID observation to recreate the hash.

**A10 — integrator rationale (Confirmed from the code).**
- The shared controller's xy and height integrals integrate **position error** (`controller.py:245-253, 278-286`).
  They belong to the position loop that direct control replaces. In the direct arm they are therefore **inactive by
  design**: the integrate call targets the current position, so their error input is exactly 0
  (`direct_control/direct_env.py`, `VelocityPID.integrate`).
- The yaw attitude integral (ki 0.02) is unchanged: it integrates the true yaw error, as in the residual arm. Roll
  and pitch have ki = 0 in both arms.
- Integrating at the substituted target instead would wind the xy integral to its 1.2 m/s² limit (80% of the cap)
  within 4–8 s at route speeds (plan section 8.1). Integrating on the reference would be a shadow PID.

**A11 (Claire, 2026-11-17): a descriptive demo of the direct model ran BEFORE its protocol evaluation. A note, not
an amendment: nothing in sections 3–9 changes.**
- **What ran (Confirmed).** The M9 visual demo (`residual_ppo/demo/`, results `results/m9_demo/`) flew the final
  direct model (`ppo_direct_s0_1p5m`), PID and PID + PPO residual (`ppo_s0_1p5m` final) on 6 hand-placed tracks in
  two scenes with ACTIVE obstacles (the training warehouse `dij3d_boxes.usd` and the team's `forest.usd`), at 3 fixed
  winds (calm, 3.5 and 7 m/s mean; one direction per track), one flight per controller and case: 54 flights.
- **Not protocol eval-v1 and not this pre-specification's evaluation.** Different scenes (obstacles active), fixed
  hand-placed tracks instead of the eval scenarios, fixed winds instead of the profiles, n = 1 per cell, no
  statistics. Seeds were 600000–600017 (cases), 600100–600102 (winds) and 600900–600901 (in-scene checks); **no
  eval seed (300000–300029) and no validation seed (400000–400009) was used by the direct model**, so the
  evaluation's flights remain unseen. The demo's numbers are not inputs to P1′ or to any endpoint in section 4.
- **What has now been seen (stated so the record is complete).** In the demo, the direct model flew all 18 cases
  without a failure or an obstacle contact, with a larger position RMSE than the residual in all 18 cases and a
  smaller one than PID in 17 of 18 (the exception: F1 north edge, calm, 0.125 vs 0.118 m). That is descriptive, on different tasks and conditions, and does not bear on P1′ (a)–(c):
  the demo has no strong_range or beyond_strong wind, and it did not compute `accel_cap_binding_frac`.
- **No change to P1′, the endpoints, the profile roles, the falsification rules (sections 6–7), A1–A10 or the
  frozen protocol files.** The direct adapter written for the demo (`demo/direct_controller.py`: contract checked
  before loading, DirectEnv required, deterministic) is meant to be reused by the direct eval runner (section 8 item
  2, option (b)), which does not exist yet.
