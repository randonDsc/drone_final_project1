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

## 5. Profile roles (quoted from `eval_protocol_v1.md`, E5)

> One primary result per profile; `still_air` is reported but is not a primary endpoint. […]
> **Primary profiles = the 3 team profiles only** (`training_range`, `unseen_gusts`, `outside_range`).
> `strong_range` and `beyond_strong` (D4) are **secondary**: the same paired statistic is reported for them, labelled
> secondary, and they cannot overturn or add to a primary result. `still_air` is reported prominently (nominal
> tracking; see CLAUDE.md M7 notes) but is not primary.

There is one primary result per primary profile and no correction for multiple profiles, the same as E5.

## 6. Pre-registered prediction (Claire, 2026-11-15; P1 replaced by P1′ on 2026-11-16, A4)

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
- **P1′ (c) (mechanism evidence; reported either way, not a test of (b)).** In `strong_range` and `beyond_strong`, the
  direct arm's `accel_cap_binding_frac` should approach 1.0, while the residual's stays low. A failure here means the
  authority explanation is not supported, whatever (b) shows.
  - **Recorded at writing (Confirmed, A4 step 1 tables, section 10):** `accel_cap_binding_frac` is computed by the
    frozen `Diagnostics` (`evaluation/run_protocol.py:137-149`) on the **PID part** of the command (final − correction).
    For the residual arm it therefore says how often the PID part is capped *before* the residual is added; it does not
    measure a limit on the residual itself.
  - Its residual-arm values are already known: mean **0.128** in `strong_range` and **0.539** in `beyond_strong`
    (`pid_ppo`). PID, which is fully subject to the cap, has mean 0.716 and 0.742 there, and no PID flight exceeds 0.917.
  - **"Approaches 1.0" and "stays low" have no numeric criterion yet.** That criterion, and how the direct runner
    computes `accel_cap_binding_frac`, must be fixed in the addendum before any direct-control code is written
    (section 8, item 4). Until then (c) is descriptive.
- **P2 (nominal wind; primary profile `training_range`).** Little or no RMSE advantage for direct: the lower bound of
  the direct − residual 95% CI is ≥ −0.02 m, i.e. direct is not better by more than 2 cm with confidence.
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
- **P1′ (c)** is reported in every case. If it fails, the authority explanation is not supported, even if (b) holds.
  Its numeric criterion is open (section 8, item 4).
- **P2 is falsified** if the upper bound of the `training_range` CI is below −0.02 m (direct is clearly better by more
  than 2 cm in nominal wind). Then any advantage is not specific to strong wind. If failures occur in
  `training_range`, E5 rule 2 applies first.
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
4. **(Added by A4.) The numeric criterion for P1′ (c)** ("approaches 1.0", "stays low"; for example a threshold on the
   mean or a paired direct − residual seed-bootstrap CI), and **how the direct runner computes
   `accel_cap_binding_frac`**: on the whole inner-loop command (the direct arm has no correction) against the inner
   loop's desired attitude, with the same `acceleration_capped` test (`path_sac_policy.py:212-233`).
5. **(Added by A4.) Whether P2 stays.** P2 predicts that direct is *not* better than the residual by more than 2 cm in
   `training_range`, and it was justified by the retired P1 premise. P1′ (a) allows direct to be better there. A4 does
   not change P2; Claire decides whether to keep or retire it.

Changing anything in sections 3–7 after the direct-control model exists invalidates this pre-specification.

## 9. Today's write-up (Claire, 2026-11-15)

- **Primary result:** the frozen `run_protocol.py report` of **PID + PPO residual vs PID** on the eval seeds.
- **Separate descriptive section:** team **SAC vs PID**, through the same frozen report, as context only. It contains
  no PPO-vs-SAC test or paired statistic of any kind.
- The residual-vs-direct comparison above is future work until the direct-control model exists.

## 10. Amendments (each made before any direct-control code, model or result existed)

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
