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

## 5. Profile roles (quoted from `eval_protocol_v1.md`, E5)

> One primary result per profile; `still_air` is reported but is not a primary endpoint. […]
> **Primary profiles = the 3 team profiles only** (`training_range`, `unseen_gusts`, `outside_range`).
> `strong_range` and `beyond_strong` (D4) are **secondary**: the same paired statistic is reported for them, labelled
> secondary, and they cannot overturn or add to a primary result. `still_air` is reported prominently (nominal
> tracking; see CLAUDE.md M7 notes) but is not primary.

There is one primary result per primary profile and no correction for multiple profiles, the same as E5.

## 6. Pre-registered prediction (Claire, 2026-11-15)

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

**Prediction.** The residual controller is **authority-limited in `strong_range` and `beyond_strong`**. Direct
control, which is not bounded by L, shows its **largest advantage there**, and **little or no advantage in
`training_range`**. Stated as checkable outcomes:

- **P1 (strong wind; secondary profiles, E5 rule 3 order; amended 2026-11-15, see section 10, A1).**
  - **Decided by `strong_range`:** direct fails less than the residual there, c > b with McNemar exact p < 0.05.
  - **`beyond_strong` counts toward P1 only if at least one of the two controllers fails fewer than 180 of 180.**
    It is then evaluated the same way (c > b, McNemar exact p) and reported as corroborating (c > b) or contradicting
    (b > c) P1. Otherwise (both fail 180/180) it is reported as **uninformative due to a ceiling effect**, and it
    neither supports nor falsifies P1.
  - McNemar exact p < 0.05 needs at least 6 discordant pairs, all in one direction (b + c = 6, min(b, c) = 0 gives
    p = 2 · 0.5^6 = 0.031; b + c = 5 gives 0.0625).
- **P2 (nominal wind; primary profile `training_range`).** Little or no RMSE advantage for direct: the lower bound of
  the direct − residual 95% CI is ≥ −0.02 m, i.e. direct is not better by more than 2 cm with confidence.
- **P3 (mechanism; descriptive, not a test).** In the residual run, the mean `residual_norm` (ceiling 2.0 = all four
  components at their limits) is higher in `strong_range` and
  `beyond_strong` than in `training_range`, and rotor `saturation_rate` stays at or near 0 in those profiles.

`unseen_gusts`, `outside_range` and `still_air` are reported with no prediction attached.

## 7. What would falsify the prediction

- **P1 (amended, A1) is judged on `strong_range`:**
  - **holds** if c > b with McNemar exact p < 0.05;
  - is **not supported** (inconclusive, not falsified) if c > b with p ≥ 0.05;
  - is **falsified** if b ≥ c, provided at least one controller fails fewer than 180 of 180 there;
  - is **unevaluable (ceiling effect)** if both controllers fail all 180 `strong_range` flights. This is the same
    ceiling logic as for `beyond_strong`. The canonical PID already fails 179/180 there.
- **`beyond_strong`** follows the rule in P1: it informs P1 only when at least one controller fails fewer than
  180/180; otherwise it is reported as uninformative due to a ceiling effect.
- In any profile at the ceiling, median decisions flown (E5 rule 3 (c)) is reported as the only descriptive contrast.
- **P2 is falsified** if the upper bound of the `training_range` CI is below −0.02 m (direct is clearly better by more
  than 2 cm in nominal wind). Then any advantage is not specific to strong wind. If failures occur in
  `training_range`, E5 rule 2 applies first.
- **The authority explanation is not supported** (even if P1 holds) if P3 fails: the residual's mean `residual_norm`
  in `strong_range` / `beyond_strong` is not higher than in `training_range`, or rotor saturation is substantial for
  the residual in those profiles (then the motors, not L, would be the binding limit).
- **The whole prediction is falsified** if direct control is worse than the residual in strong wind: b > c with
  McNemar p < 0.05 in `strong_range`, or in `beyond_strong` when it is informative (at least one controller below
  180/180).

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

Changing anything in sections 3–7 after the direct-control model exists invalidates this pre-specification.

## 9. Today's write-up (Claire, 2026-11-15)

- **Primary result:** the frozen `run_protocol.py report` of **PID + PPO residual vs PID** on the eval seeds.
- **Separate descriptive section:** team **SAC vs PID**, through the same frozen report, as context only. It contains
  no PPO-vs-SAC test or paired statistic of any kind.
- The residual-vs-direct comparison above is future work until the direct-control model exists.

## 10. Amendments (each made before any direct-control code or model existed)

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
