# Milestone 7 — Evaluation results: PID + PPO residual vs PID (protocol eval-v1)

Labels: **Confirmed** (ran or read it), **Documented** (written in a plan, README or checkpoint metadata),
**Inference** (reasoned, not tested), **Proposed**.

Status: draft for Claire's review (2026-11-15), not yet shared with the team.

## 0. Summary

- **Primary result (Confirmed, frozen `report` mode):** PID + PPO residual improves on PID in all three primary
  profiles. The paired RMSE difference is
  - training_range −0.383 m [−0.456, −0.314],
  - unseen_gusts −0.381 m [−0.441, −0.324],
  - outside_range −1.389 m [−1.440, −1.335].

  In outside_range PID fails 110 of 180 flights and PID + PPO fails none (McNemar exact p = 1.5e−33). The RMSE
  difference there is read together with that failure result (E5 rule 2).
- **Secondary profiles (Confirmed):**
  - strong_range: 16/180 failures vs PID's 179/180.
  - beyond_strong: 107/180 vs PID's 180/180.
  - Both McNemar p < 1e−20.
- **Still air (Confirmed):**
  - With zero wind the residual is large: mean `residual_norm` 1.12 out of a ceiling of 2.0.
  - It lowers still-air RMSE from 0.183 to 0.062 m.
  - Rotor saturation is 0 in every flight.

  Nominal-tracking improvement and wind rejection are separate claims; each has its own evidence (section 4).
- **Learning curve (secondary, validation seeds only):** improvement continues well past 150k decisions.
- **Team SAC vs PID** is in a separate descriptive section (section 8). **No PPO-vs-SAC test of any kind** was
  pre-declared or computed.

## 1. What was run, and verification

**Flights (Confirmed).**
- Who and when: Claire flew all six runs sequentially on 2026-11-15, 12:04:44 to 15:10:53 local, with the frozen
  `evaluation/run_protocol.py fly` at HEAD 5fefb29. Every run logged `RESULT: DONE` and wrote its outputs.
- Committed in **0ec9121** under `results/eval/eval-v1/`. The UTF-16 console logs are excluded and gitignored; their
  facts are in each `results.json` (`meta`, `check0`).

| # | Run | Model | Seeds | Flights | Wall clock |
|---|---|---|---|---|---|
| 1 | `pid_ppo/` (PRIMARY) | `ppo_s0_1p5m` final model, 1,501,184 decisions, sha256 6b9e1993… | eval 300000–300029, `--check0` | 1,080 | 55.3 min |
| 2–5 | `ppo_validation/ppo_150000`, `ppo_400000`, `ppo_800000`, `ppo_final` | checkpoints 150k / 400k / 800k and the final model | validation 400000–400009, no check0 | 360 each | 17.6–18.7 min each |
| 6 | `pid_sac/` (descriptive context) | team SAC `final_model.zip`, sha256 2d9e5cd0… | eval 300000–300029, `--check0` | 1,080 | 55.9 min |

**Blind amendment first (Confirmed).** Prespec amendment A3 (`docs/residual_vs_direct_prespec.md`, the ceiling
fallback) was committed and pushed as **c3198fa** before any of these result files was opened.

**Verification before use: all six runs PASS (Confirmed).** Each run was checked for:
- protocol hash d3f677dc…;
- the 5 code blobs equal to the frozen IDs (`docs/eval_protocol_v1.md`);
- the consistency check over the full grid;
- the realization guard checking every reset (1,080 or 360), with `realizations.json` identical to the reference
  PID run (`pid/` or `pid_validation/`);
- every action finite and of shape 4, with the action count equal to the summed decisions;
- total mass 1.6186144 kg equal to the PID run, HESAI 0 kg;
- check0 PASS for runs 1 and 6 (cell turns / training_range / 300000); runs 2–5 carry no check0, as E6 requires;
- each model's sha256 equal to `results/m6/m6_model_hashes.json` (PPO) or `models/sac_team/SOURCE.json` (SAC).

All runs used Isaac Sim 6.2.0, SB3 2.7.0, gymnasium 1.2.3, numpy 2.3.1, torch 2.14.1+cpu, device cpu.

**Analyses (Confirmed; no Isaac needed):**

```powershell
# Directory: <repo root>
python residual_ppo\evaluation\run_protocol.py report --controller-run residual_ppo/results/eval/eval-v1/pid_ppo `
    --out residual_ppo/results/eval/eval-v1/pid_ppo/report_vs_pid
python residual_ppo\evaluation\run_protocol.py report --controller-run residual_ppo/results/eval/eval-v1/pid_sac `
    --out residual_ppo/results/eval/eval-v1/pid_sac/report_vs_pid
python residual_ppo\evaluation\learning_curve.py `
    --run ppo_150000=residual_ppo/results/eval/eval-v1/ppo_validation/ppo_150000 `
    --run ppo_400000=residual_ppo/results/eval/eval-v1/ppo_validation/ppo_400000 `
    --run ppo_800000=residual_ppo/results/eval/eval-v1/ppo_validation/ppo_800000 `
    --run ppo_final=residual_ppo/results/eval/eval-v1/ppo_validation/ppo_final `
    --out residual_ppo/results/eval/eval-v1/learning_curve
```

- Both reports confirmed identical realizations and passed consistency for both runs.
- The frozen `report` mode prints no `RESULT:` line, only `written: …`.
- `learning_curve.py` ended with `RESULT: DONE`.

**Statistics (`docs/eval_protocol_v1.md`, E5).**
- The paired RMSE difference is per seed: the mean over the 6 scenarios of controller − PID.
- The CI is a 95% percentile bootstrap over seeds, 10,000 resamples, RNG seed 0.
- The paired failure test is McNemar exact; failure-rate CIs are Clopper–Pearson exact 95%.
- There is one primary result per primary profile and no multiplicity correction.

## 2. PRIMARY — PID + PPO residual vs PID (eval seeds, 180 paired flights per profile)

The profile roles come from E5:
- primary: training_range, unseen_gusts, outside_range;
- reported, not primary: still_air;
- secondary (D4): strong_range, beyond_strong. They cannot overturn or add to a primary result.

Endpoints follow **E5 rule 3 order**:
- (a) failure count with exact CI;
- (b) paired McNemar exact;
- (c) median decisions flown;
- (d) position RMSE.

Source: `results/eval/eval-v1/pid_ppo/report_vs_pid.{json,txt}` (Confirmed).

**(a)–(c): failures, paired failures, decisions flown**

| Profile | Role | (a) Failures PID+PPO [95% CP] | (a) Failures PID [95% CP] | (b) Paired: only PPO / only PID / both / neither | (b) McNemar exact p | (c) Median decisions (IQR), PID+PPO; PID |
|---|---|---|---|---|---|---|
| training_range | **primary** | 0/180 [0.000, 0.020] | 0/180 [0.000, 0.020] | 0 / 0 / 0 / 180 | 1 | 154 (84–214); 154 (84–214) |
| unseen_gusts | **primary** | 0/180 [0.000, 0.020] | 0/180 [0.000, 0.020] | 0 / 0 / 0 / 180 | 1 | 154 (84–214); 154 (84–214) |
| outside_range | **primary** | 0/180 [0.000, 0.020] | 110/180 [0.536, 0.683] | 0 / 110 / 0 / 70 | 1.5e−33 | 154 (84–214); 62 (39–95) |
| still_air | reported, not primary | 0/180 [0.000, 0.020] | 0/180 [0.000, 0.020] | 0 / 0 / 0 / 180 | 1 | 154 (84–214); 154 (84–214) |
| strong_range | secondary | 16/180 [0.052, 0.140] | 179/180 [0.969, 1.000] | 0 / 163 / 16 / 1 | 1.7e−49 | 132 (77–207); 23 (21–29) |
| beyond_strong | secondary | 107/180 [0.519, 0.667] | 180/180 [0.980, 1.000] | 0 / 73 / 107 / 0 | 2.1e−22 | 66 (41–120); 18 (17–19) |

**(d): position RMSE**

| Profile | Role | RMSE label | RMSE PID+PPO / PID (m) | Paired diff PID+PPO − PID [95% CI] (m) | Verdict | E5 rule 2 ("RMSE never alone when there are failures") |
|---|---|---|---|---|---|---|
| training_range | **primary** | time-indexed | 0.100 / 0.483 | **−0.3825 [−0.4557, −0.3141]** | improves on PID | does not apply (0 failures) |
| unseen_gusts | **primary** | time-indexed | 0.103 / 0.484 | **−0.3813 [−0.4411, −0.3235]** | improves on PID | does not apply |
| outside_range | **primary** | time-indexed, failed flights included | 0.252 / 1.641 | **−1.3889 [−1.4397, −1.3350]** | improves on PID, read with (a)–(b) | **applies** (PID 110 failures) |
| still_air | reported, not primary | time-indexed | 0.062 / 0.183 | −0.1213 [−0.1285, −0.1142] | not an endpoint | does not apply |
| strong_range | secondary | over flown portion only | 0.690 / 1.651 | −0.9608 [−1.1060, −0.8062] | — | **applies** |
| beyond_strong | secondary | over flown portion only | 1.515 / 1.604 | −0.0890 [−0.2340, +0.0460] | CI includes 0; see section 4 | **applies** |

**Reading.**
- **(Confirmed)** In all three primary profiles the CI lies entirely below 0. That is the pre-declared "improves on
  PID" criterion.
- **(Confirmed)** In outside_range the residual removes PID's failure mode completely: 110 → 0 failures, and every
  discordant pair is "only PID failed".
- **(Confirmed)** In the secondary profiles the order (a)–(c) carries the result: far fewer failures and 5.7× (strong)
  and 3.7× (beyond) longer median flights.
- **(Inference)** The beyond_strong RMSE CI includes 0 because of survivorship (section 4), not because there is no
  gain.
- **(Confirmed)** Identical decision medians in the no-failure profiles are expected, not a bug. A flight that does not
  fail runs to `max_decisions`, and the realization guard makes `max_decisions` identical for both controllers.

## 3. SECONDARY — learning curve (validation seeds 400000–400009, 60 paired flights per profile)

These results are descriptive, with no checkpoint selection. The primary PPO result is the final model on the eval
seeds (section 2). Source: `results/eval/eval-v1/learning_curve.{json,txt,csv}` (Confirmed).

**Paired RMSE difference vs PID, m [95% CI]** (secondary rows are RMSE over the flown portion only; † = E5 rule 2
applies)

| Profile | 150k | 400k | 800k | final (1,501,184) |
|---|---|---|---|---|
| still_air | −0.100 [−0.111, −0.090] | −0.126 [−0.134, −0.117] | −0.116 [−0.125, −0.107] | −0.117 [−0.126, −0.107] |
| training_range | −0.258 [−0.358, −0.173] | −0.322 [−0.441, −0.220] | −0.352 [−0.485, −0.237] | −0.377 [−0.522, −0.251] |
| unseen_gusts | −0.252 [−0.343, −0.181] | −0.309 [−0.413, −0.228] | −0.338 [−0.453, −0.247] | −0.361 [−0.485, −0.261] |
| outside_range † | −1.032 [−1.088, −0.978] | −1.203 [−1.264, −1.135] | −1.315 [−1.380, −1.238] | −1.399 [−1.472, −1.314] |
| strong_range † | −0.380 [−0.627, −0.120] | −0.725 [−0.927, −0.475] | −0.928 [−1.115, −0.683] | −1.030 [−1.226, −0.762] |
| beyond_strong † | **+0.201 [+0.084, +0.315]** | +0.015 [−0.135, +0.160] | −0.140 [−0.364, +0.089] | −0.198 [−0.439, +0.050] |

**Failures out of 60 (PID failures in the first column) and median decisions**

| Profile | PID | 150k | 400k | 800k | final |
|---|---|---|---|---|---|
| outside_range failures | 34 | 0 | 0 | 0 | 0 |
| strong_range failures | 59 | 13 | 4 | 3 | 3 |
| beyond_strong failures | 60 | 42 | 33 | 32 | 32 |
| strong_range median decisions | 24 | 105.5 | 143.5 | 143.5 | 143.5 |
| beyond_strong median decisions | 17 | 57.5 | 77 | 80 | 80 |

McNemar exact p < 1e−5 in every row with failures.

**Reading.**
- **Improvement continues well past 150k (Confirmed).**
  - training_range RMSE falls 0.216 → 0.152 → 0.122 → 0.097 m (PID 0.474), and the paired gain grows from
    −0.258 to −0.377 m.
  - outside_range RMSE falls 0.603 → 0.236 m.
  - Between 800k and the final model, failure counts do not change, but RMSE still improves: training_range
    0.122 → 0.097 m, strong_range 0.733 → 0.632 m.
- **Strong-wind robustness between 150k and 800k (Confirmed arithmetic; how much depends on the measure):**
  - *RMSE:* most of the strong_range gain arrives in this interval. Of the final −1.030 m, −0.380 m (37%) is present
    at 150k, a further −0.548 m (53%) arrives between 150k and 800k, and −0.102 m (10%) after 800k.
  - *Failures that remain at 150k:* most are removed in this interval. strong_range goes 13 → 3 per 60;
    beyond_strong 42 → 32. Median beyond_strong flight length rises 57.5 → 80 decisions.
  - *Failures relative to PID:* most of the reduction is **already present at 150k**. In strong_range, 46 of the
    56 failures removed by the final model (59 → 3) are already gone at 150k; in beyond_strong, 18 of 28.
- **(Documented + Inference) The 150k checkpoint was trained only in the curriculum's `short_slow` stage.**
  - That stage (`path_sac_config.CURRICULUM`) has mean wind 0–2 m/s, no wind in 50% of episodes, and no circle or
    figure-eight routes.
  - Even so, it already removes most of PID's strong-wind (10–14 m/s) failures.
  - No training stage ever exceeded 7 m/s mean wind (`full`: 2–7 m/s). So outside_range, strong_range and
    beyond_strong are all extrapolation for every checkpoint.
- **(Confirmed, weak) still_air is not monotonic.** It is best at 400k (0.051 m), then 0.060–0.061 m, with overlapping
  CIs over 10 seeds. **(Inference)** The later curriculum stages may trade a little nominal precision for strong-wind
  robustness.
- **(Confirmed) Validation and eval agree for the final model.** training_range RMSE is 0.097 m (validation) vs
  0.100 m (eval); still_air is 0.060 vs 0.062 m.

## 4. Methods note — survivorship in "RMSE over flown portion only" (why E5 orders failures before RMSE)

**Worked example: the 150k checkpoint in beyond_strong (validation seeds; Confirmed).**

| | PID | PID + PPO 150k |
|---|---|---|
| Failures | 60/60 | 42/60 (McNemar exact p = 7.6e−6; all 18 discordant pairs are "only PID failed") |
| Median decisions flown (IQR) | 17 (17–18), about 1.7 s at 10 Hz | 57.5 (26–91), about 5.8 s |
| Flew longer than PID in the paired cell | — | 60 of 60 cells |
| Position RMSE, flown portion only | 1.622 m | 1.823 m |
| Paired RMSE difference | — | **+0.201 m [+0.084, +0.315]** ("worse" than PID) |

**What happens (Inference).**
- For a failed flight, position RMSE averages the tracking error only over the decisions actually flown.
- PID fails after about 1.7 s, so its RMSE averages only the start of the flight, before the wind has built up a
  large error.
- The 150k residual survives about 3.4 times longer. Its RMSE therefore also includes the later, more displaced part of
  the flight, which PID never reaches.
- The two RMSEs average over different time windows. The controller that survives longer is penalized for surviving.
- The paired RMSE difference is therefore "worse" by +0.20 m, with a CI that excludes 0. Yet the same controller
  fails 18 fewer flights and flies longer in every single pair.

**Why the protocol already handles this (Confirmed, `docs/eval_protocol_v1.md`, E5 rules 2 and 3).**
- In strong_range and beyond_strong the endpoints are read in a fixed order: (a) failure count, (b) McNemar,
  (c) median decisions, (d) RMSE labelled "over flown portion only".
- Wherever either controller has a failure, RMSE is never interpreted alone.
- Read in that order, the 150k example is unambiguous: fewer failures, significantly, and longer flights. The RMSE
  sign is an artefact of unequal flown windows.
- The final model on the eval seeds shows the same pattern more weakly: beyond_strong RMSE CI [−0.234, +0.046]
  includes 0, while failures are 107 vs 180.

**Not possible with the stored data (Confirmed; Proposed for v2).** An RMSE over a *common* window would remove the
artefact, for example the first `min(decisions_PID, decisions_PPO)` decisions of each pair. But `fly` stores no
per-step traces, only per-flight metrics. Computing it would need a protocol v2 change, and it must be pre-declared
before use.

## 5. The still-air finding: two separate claims

`still_air` is the eval profile with `no_wind_probability` 1.0, i.e. exactly zero wind (`meta.still_air_definition`,
Confirmed).

**Claim 1 — the residual fixes nominal PID tracking (zero wind).** Evidence (Confirmed):
- still_air RMSE falls from **0.183 m (PID) to 0.062 m (PID + PPO)**; the paired difference is −0.1213 m
  [−0.1285, −0.1142]. still_air is reported, not primary.
- Completion rises from 116/180 (64.4%) to 180/180 (section 6).
- With zero wind the policy still applies a large residual: **mean `residual_norm` 1.116**. Its ceiling is 2.0, when
  all four normalized components sit at their limits.
- **Rotor `saturation_rate` is 0 in every flight** of every run: PID, PID + PPO and PID + SAC, all six profiles.

Since there is no disturbance in still_air, this whole gain is about how PID tracks the reference on its own.
**(Inference, untested)** Candidate mechanisms:
- PID's 1.5 m/s² horizontal acceleration cap binds on up to 11.2% of still-air physics steps (M2.4, CLAUDE.md
  analysis note (a));
- PID's along-track lag;
- the residual acting as feedforward thrust and torque.

**Claim 2 — the residual rejects wind.** This needs evidence beyond Claim 1. Two independent pieces:
1. **Failures (Confirmed).** PID+PPO fails 0 vs PID's 110 in outside_range, 16 vs 179 in strong_range and 107 vs 180
   in beyond_strong. Neither controller fails in still air, so these reductions cannot come from better nominal
   tracking.
2. **Wind-induced RMSE increase (descriptive, NOT pre-registered, outside the frozen report).** For each controller,
   per seed, compute mean over scenarios of RMSE(wind profile) − RMSE(still_air). The bootstrap is the same as E5,
   through `protocol_stats.per_seed_mean` / `bootstrap_mean`. Values in m [95% CI]:

   | Profile | Increase PID | Increase PID+PPO | Difference PPO − PID |
   |---|---|---|---|
   | training_range | +0.300 [+0.220, +0.384] | +0.038 [+0.028, +0.049] | −0.261 [−0.335, −0.192] |
   | unseen_gusts | +0.301 [+0.233, +0.372] | +0.041 [+0.032, +0.051] | −0.260 [−0.321, −0.202] |
   | outside_range (PID failures; read with them) | +1.458 [+1.392, +1.517] | +0.190 [+0.172, +0.208] | −1.268 [−1.318, −1.214] |

   - Wind adds about 0.30 m RMSE to PID in training_range but only about 0.04 m to PID + PPO.
   - The training_range primary gain splits exactly into the two claims: **−0.3825 = −0.1213 (nominal, still_air)
     + −0.2612 (less wind-induced error)**. That is about one third nominal tracking and two thirds wind rejection
     (Confirmed arithmetic; the split is descriptive).
   - The unseen_gusts split is −0.3813 = −0.1213 + −0.2600.

**Do not merge the claims.** Saying "the residual rejects wind" from the primary RMSE alone would overstate it by about
a third. Saying "it only fixes PID" would ignore the failure results and the difference-in-differences above.

**Residual magnitude by profile (Confirmed, eval, PID + PPO):**

| Profile | still_air | training_range | unseen_gusts | outside_range | strong_range | beyond_strong |
|---|---|---|---|---|---|---|
| Mean `residual_norm` | 1.116 | 1.137 | 1.144 | 1.257 | 1.464 | 1.660 |

- The per-flight maximum is 1.926 in beyond_strong.
- **(Inference)** The residual approaches its limit L = [0.8 N, 0.05, 0.03, 0.02 N·m] in the strongest wind while the
  motors keep headroom. This is the pattern the residual-vs-direct prespec (P3) describes. That test needs the
  direct-control model and has not been run.

## 6. Completion rates (descriptive, outside the frozen report)

**Definition (Confirmed, `path_sac_policy.CompletionTracker`, `path_sac_config.COMPLETION`).** A flight is
*completed* when, after the reference motion has ended, the drone stays within **0.15 m** of the final point at
**≤ 0.2 m/s** for **0.5 s**. A flight can also end failed, or it can neither fail nor complete and run to
`max_decisions`, marked "ran to time limit" below. Completion is a strict arrival criterion and is not an E5 endpoint.

Computed from `pid/flights.csv` and `pid_ppo/flights.csv` (eval seeds, 180 flights per profile; Clopper–Pearson 95%).
The pid_sac rates are in section 8, kept separate by the no-PPO-vs-SAC rule.

| Profile | PID completed [95% CP] | PID failed / ran to time limit | PID+PPO completed [95% CP] | PID+PPO failed / ran to time limit |
|---|---|---|---|---|
| training_range | 24/180 = 13.3% [0.087, 0.192] | 0 / 156 | 180/180 = 100% [0.980, 1.000] | 0 / 0 |
| unseen_gusts | 16/180 = 8.9% [0.052, 0.140] | 0 / 164 | 179/180 = 99.4% [0.969, 1.000] | 0 / 1 |
| outside_range | 0/180 = 0% [0.000, 0.020] | 110 / 70 | 68/180 = 37.8% [0.307, 0.453] | 0 / 112 |
| still_air | 116/180 = 64.4% [0.570, 0.714] | 0 / 64 | 180/180 = 100% [0.980, 1.000] | 0 / 0 |
| strong_range | 0/180 = 0% [0.000, 0.020] | 179 / 1 | 12/180 = 6.7% [0.035, 0.114] | 16 / 152 |
| beyond_strong | 0/180 = 0% [0.000, 0.020] | 180 / 0 | 0/180 = 0% [0.000, 0.020] | 107 / 73 |

**Reading.**
- **(Confirmed)** PID rarely meets the arrival criterion even when it does not fail: 13% in training_range, 64% in
  still air. PID + PPO meets it in essentially every flight up to unseen_gusts.
- **(Confirmed + Inference)** In outside_range PID + PPO never fails but completes only 37.8%: 112 flights ran to the
  time limit without settling. **(Inference)** Holding within 0.15 m at ≤ 0.2 m/s in 8–10 m/s mean wind with gusts is
  harder than staying in bounds.

## 7. E7b — training cost (fact, not a comparison claim)

| | PID + PPO residual (`ppo_s0_1p5m`) | Team PID + SAC residual |
|---|---|---|
| Environment decisions | 1,501,184 (733 rollouts × 2,048; Confirmed) | 1,500,000 (checkpoint `num_timesteps`; Documented) |
| Gradient updates | **10,305** optimizer steps, minibatch 256 (Confirmed, optimizer hook) | **1,494,999** (`_n_updates`, batch 256, one per step after `learning_starts` 5,000; Documented) |
| SB3 epochs | 1,681 (709 of 733 rollouts early-stopped by target_kl 0.02) | n/a |
| Wall clock, hardware | 8 h 30 m 49 s, CPU torch, environment-bound (Confirmed) | ~30 h, RTX 4070 SUPER (team README §7, Documented) |

- **Stated as a fact:** PID + PPO used 10,305 gradient updates, 0.69% of SAC's 1,494,999. This was on essentially the
  same number of environment decisions (1,501,184 vs 1,500,000).
- **Not claimed: equivalence of the two controllers.** No PPO-vs-SAC test was pre-declared. The prespec
  (`docs/residual_vs_direct_prespec.md`, sections 1 and 9) explicitly excludes any cross-algorithm inferential claim.
- **What establishing equivalence would need:** a separate, pre-registered equivalence test, for example two
  one-sided tests with a stated margin on named endpoints. It would have to be run on flights **not yet seen**. All
  30 eval seeds and all 10 validation seeds have now been looked at, so new seeds would be required.
- Wall clock is reported with its hardware and is not a head-to-head comparison: different devices, trainers and
  library versions.

## 8. DESCRIPTIVE CONTEXT ONLY — team SAC (PID + SAC residual) vs PID

This section is context. It shares no statistic with sections 2–7, and no PPO-vs-SAC difference, test or ranking is
computed or implied.
- The model is the team's checkpoint (HF `pcldsc/Iris_drone_residual_correction_SAC` @ 7dd1f14b,
  `models/sac_team/SOURCE.json`).
- It was flown here in our environment with the same frozen protocol and seeds.
- Source: `results/eval/eval-v1/pid_sac/report_vs_pid.{json,txt}` (Confirmed).

**(a)–(c)**

| Profile | Role | (a) Failures PID+SAC [95% CP] | (a) Failures PID [95% CP] | (b) Paired: only SAC / only PID / both / neither | (b) McNemar exact p | (c) Median decisions (IQR), PID+SAC; PID |
|---|---|---|---|---|---|---|
| training_range | primary | 0/180 [0.000, 0.020] | 0/180 [0.000, 0.020] | 0 / 0 / 0 / 180 | 1 | 154 (84–214); 154 (84–214) |
| unseen_gusts | primary | 0/180 [0.000, 0.020] | 0/180 [0.000, 0.020] | 0 / 0 / 0 / 180 | 1 | 154 (84–214); 154 (84–214) |
| outside_range | primary | 0/180 [0.000, 0.020] | 110/180 [0.536, 0.683] | 0 / 110 / 0 / 70 | 1.5e−33 | 154 (84–214); 62 (39–95) |
| still_air | reported, not primary | 0/180 [0.000, 0.020] | 0/180 [0.000, 0.020] | 0 / 0 / 0 / 180 | 1 | 154 (84–214); 154 (84–214) |
| strong_range | secondary | 16/180 [0.052, 0.140] | 179/180 [0.969, 1.000] | 0 / 163 / 16 / 1 | 1.7e−49 | 132 (79–207); 23 (21–29) |
| beyond_strong | secondary | 110/180 [0.536, 0.683] | 180/180 [0.980, 1.000] | 0 / 70 / 110 / 0 | 1.7e−21 | 68 (39–128); 18 (17–19) |

**(d)**

| Profile | Role | RMSE label | RMSE PID+SAC / PID (m) | Paired diff PID+SAC − PID [95% CI] (m) | Verdict | E5 rule 2 |
|---|---|---|---|---|---|---|
| training_range | primary | time-indexed | 0.081 / 0.483 | −0.4014 [−0.4809, −0.3259] | improves on PID | does not apply |
| unseen_gusts | primary | time-indexed | 0.082 / 0.484 | −0.4021 [−0.4673, −0.3385] | improves on PID | does not apply |
| outside_range | primary | time-indexed, failed flights included | 0.173 / 1.641 | −1.4673 [−1.5207, −1.4100] | improves on PID, read with (a)–(b) | applies |
| still_air | reported, not primary | time-indexed | 0.072 / 0.183 | −0.1114 [−0.1198, −0.1032] | not an endpoint | does not apply |
| strong_range | secondary | over flown portion only | 0.658 / 1.651 | −0.9933 [−1.1398, −0.8444] | — | applies |
| beyond_strong | secondary | over flown portion only | 1.558 / 1.604 | −0.0461 [−0.1743, +0.0690] | CI includes 0 (survivorship, section 4) | applies |

**Completion, PID+SAC vs PID** (from `pid_sac/flights.csv` and `pid/flights.csv`; descriptive, outside the frozen
report)

| Profile | PID+SAC completed [95% CP] | PID completed | Difference PID+SAC − PID | PID+SAC failed / ran to time limit |
|---|---|---|---|---|
| training_range | 180/180 = 100% [0.980, 1.000] | 13.3% | +86.7 pp | 0 / 0 |
| unseen_gusts | 180/180 = 100% [0.980, 1.000] | 8.9% | +91.1 pp | 0 / 0 |
| outside_range | 160/180 = 88.9% [0.834, 0.931] | 0% | +88.9 pp | 0 / 20 |
| still_air | 180/180 = 100% [0.980, 1.000] | 64.4% | +35.6 pp | 0 / 0 |
| strong_range | 44/180 = 24.4% [0.184, 0.314] | 0% | +24.4 pp | 16 / 120 |
| beyond_strong | 2/180 = 1.1% [0.001, 0.040] | 0% | +1.1 pp | 110 / 68 |

- **Residual and saturation (Confirmed):** mean `residual_norm` by profile, PID+SAC: still_air 0.959, training_range
  1.048, unseen_gusts 1.078, outside_range 1.325, strong_range 1.576, beyond_strong 1.796. Rotor saturation is 0 in
  every flight.
- **Setup differences (Documented):** the teammate trained it under Isaac Sim 6.1.0, SB3 2.9.0, gymnasium 1.4.0 and
  torch 2.11.0+cu128 on GPU. Here it was evaluated under Isaac Sim 6.2.0, SB3 2.7.0, gymnasium 1.2.3 and torch
  2.14.1+cpu. The checkpoint loads under SB3 2.7.0 with 0 warnings, and the adapter's spaces and contract checks
  passed (M2.5).
- **Parameter counts (Inference, from `net_arch`):** SAC has 301,066 trainable parameters, or 501,260 including the two
  frozen target critics.

## 9. Limitations

- **One training seed, one run.**
  - `ppo_s0_1p5m` is a single PPO run with seed 0.
  - Every PPO number here describes that one trained policy. Seed-to-seed variation in training is unmeasured.
  - The CIs cover evaluation-seed variation only, not training variation.
- **target_kl throttling.** target_kl 0.02, a provisional setting, early-stopped 709 of 733 rollouts (97%). PPO
  therefore made 10,305 of a possible 58,640 gradient steps (17.6%). A different target_kl could change the result in
  either direction. Not tested.
- **Action clipping.**
  - In the `full` training stage, 24.79% of sampled action components were clipped to [−1, 1]. Earlier stages: 0.72%,
    0.98%, 8.17%.
  - These are stochastic, exploration-noise actions (std about 0.21). Clipping a Gaussian biases the effective action
    (milestone3_plan.md P6).
  - The deterministic eval policy reaches a mean `residual_norm` of 1.66, out of a ceiling of 2.0, in beyond_strong.
    The residual limits L bind in strong wind.
- **SAC was trained by a teammate under different versions** (section 8). It is context, not a controlled comparison.
- **All strong-wind profiles are extrapolation.** Training wind never exceeded 7 m/s mean.
- **Learning curve on 10 validation seeds** (60 flights per profile): its CIs are wide, and it is descriptive only.
- **There is no multiplicity correction** across profiles, as fixed in E5.
- **RMSE over flown portion** is confounded by survival time in strong wind (section 4). No common-window RMSE is
  available from stored data.
- **Analyses outside the frozen report** are descriptive and were not pre-registered: completion rates, the
  wind-induced RMSE increase and its split, and `residual_norm` by profile.

## 10. Files

- **Flights:** `results/eval/eval-v1/pid_ppo/`, `pid_sac/`, `ppo_validation/ppo_{150000,400000,800000,final}/`
  (`results.json`, `realizations.json`, `flights.csv`, `summary.csv`, `summary.txt`). Commit 0ec9121.
- **Reports:** `results/eval/eval-v1/pid_ppo/report_vs_pid.{json,txt}`, `pid_sac/report_vs_pid.{json,txt}`;
  learning curve `results/eval/eval-v1/learning_curve.{json,txt,csv}`. Commit 0ec9121.
- **Pre-specification for the next step:** `docs/residual_vs_direct_prespec.md` (3218b53, amended 5fefb29 and
  c3198fa).
- **Training run record:** `docs/milestone6_main_run.md`, `results/m6/`.
