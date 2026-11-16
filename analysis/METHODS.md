# Statistical Analysis Protocol

**Study:** Paired evaluation of residual SAC flight control

**Analysis date:** 14 November 2026

**Numerical review:** 15 November 2026

## 1. Study design and scope

We conducted a retrospective analysis of the first flight SAC checkpoint's archived Isaac Sim evaluation. Our dataset comprises 420 flight records: seven scenarios, three wind profiles, ten environment seeds per condition, and two controller modes. The checkpoint is `flight1/final_model.zip`, trained for 1,500,000 decisions with training seed 0. Its SHA-256 is `2d9e5cd0df1a3cbabf8ae2eaa75884783c9c16bf6a3ced737398291b0b67023d`.

The scenarios are hover, straight, turns/waypoints, altitude change, circle, figure eight, and Dijkstra navigation. The environment seeds are 300000–300009. Evaluation used a frozen deterministic policy and ground-truth simulator state. Inference concerns environment-seed variation for one policy and PID configuration. The study is exploratory and retrospective.

## 2. Source validation and pairing

We matched each `pid_only` flight with its `pid_sac` counterpart using `(scenario, wind_profile, seed)`. We required exactly one record per controller, a complete condition–seed grid, and agreement in recorded wind seed, mean wind speed when present, and reference-route duration. Wind seeds were preserved as integers to prevent floating-point rounding.

We checked finite and nonnegative metrics, valid binary outcomes, failure-flag/reason consistency, duplicate identifiers, and missing pairs. Flight-level CSV values were cross-checked against `results.json`; source summaries were independently recomputed. Source `max_deviation` and `cross_track_max` summarize maxima across flights, whereas our secondary paired analyses treat each flight's maximum as an observation. These estimands are distinct.

The archived report states that paired routes and starting conditions were shared. Exact reset vectors and reference hashes are not stored, limiting independent verification of those additional properties. We retained original source bytes and checksums for traceability.

## 3. Effects and confidence intervals

For each condition and seed, we defined `d = position_RMSE_PID+SAC − position_RMSE_PID`. Negative differences favor SAC. We report controller means, mean and median paired differences, sample standard deviation of differences, and win/tie counts. Differences within 1e-12 count as ties. Relative mean reduction is `100 × (mean_PID − mean_PID+SAC) / mean_PID`, omitted when the PID mean is zero.

Position RMSE is the evaluator's time-indexed tracking metric. Cross-track error measures geometric distance to the reference path independently of progress along it. We analyze the recorded flight-level values. Overall summaries average flight-level RMSE with equal weight across the specified conditions.

We used pointwise 95% bias-corrected and accelerated (BCa) bootstrap intervals with 50,000 resamples. The master seed was 20261114; group-specific seeds were derived by SHA-256 from the master seed and a stable group label. Paired differences were resampled as units. Constant samples received a degenerate-sample label; intervals were unavailable for groups with fewer than two pairs or non-finite BCa limits. Method definitions follow the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

The same ten seed identifiers occur across conditions. For each overview benchmark, we averaged the selected conditions within each seed and then resampled the ten seed blocks. This preserves dependence associated with seed reuse. Conditions receive equal weight; the five overlapping overviews summarize fixed benchmarks.

Reported intervals provide pointwise coverage. Their precision and coverage remain sensitive to the ten observed seed pairs.

## 4. Hypothesis tests and multiplicity

We used two-sided Wilcoxon signed-rank tests with a null distribution of paired differences symmetric about zero. Location-shift interpretation relies on symmetry, and observations within a condition must support independence across environment seeds. Mean paired effects are estimated separately through the bootstrap intervals.

For rank calculations only, differences were rounded to 12 decimal places to avoid spurious floating-point distinctions. Distinct, nonzero differences used exact signed-rank inference. Tied or zero differences used exhaustive sign permutations for the small samples in this study, dropping zero differences from ranks. All-zero samples received p = 1 with an explicit no-rank-information label. These conventions follow the [SciPy signed-rank documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

We also used a two-sided exact sign test on the number of SAC improvements among non-tied pairs, under a Binomial(n, 0.5) null. It measures directional consistency and serves as a separately adjusted robustness check. Both test results are retained. Exact binomial inference follows [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

We applied Holm's step-down adjustment at α = 0.05 within each declared family. It controls family-wise error under dependence among tests. Unavailable tests retain a family slot conservatively. Raw and adjusted p-values, family identifiers, and family sizes are retained. The procedure is consistent with the [statsmodels multiple-testing documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

The four secondary metrics address separate exploratory endpoints. Each receives a 21-condition Holm adjustment for signed-rank tests and a separate 21-condition sign-test adjustment. A joint 84-test adjustment provides a sensitivity analysis of claims spanning the complete secondary set.

| Analysis family | Tests | Interpretation |
| --- | ---: | --- |
| Primary position-RMSE signed-rank | 21 | Primary condition-specific inference |
| Primary position-RMSE sign | 21 | Directional robustness |
| Secondary signed-rank, per metric | 21 each | Four separate exploratory metric families |
| Secondary sign, per metric | 21 each | Directional robustness within each metric |
| Joint secondary signed-rank sensitivity | 84 | Combined scope across all secondary metrics |
| Joint secondary sign sensitivity | 84 | Combined directional robustness |
| Completion exact McNemar | 21 | Secondary paired outcome |
| Failure exact McNemar | 21 | Secondary paired outcome |
| Reported-collision exact McNemar | 21 | Secondary paired outcome |
| Sensitivity position-RMSE signed-rank | 21 | Survivor-conditioned exploratory analysis |
| Sensitivity position-RMSE sign | 21 | Sensitivity robustness |
| Seed-block overview signed-rank | 5 | Exploratory benchmark summaries |
| Seed-block overview sign | 5 | Overview robustness |

Error control applies within the stated family. Primary conclusions use the position-RMSE family. Main secondary results use metric-wise families, with combined-family results reported in the sensitivity appendix.

For ten nonzero pairs, the minimum two-sided exact signed-rank p-value is `2/2^10 = 0.001953125`. It can pass the initial metric-wise threshold of `0.05/21 ≈ 0.002381`. The initial joint-family threshold is `0.05/84 ≈ 0.000595`, below the attainable minimum. The joint sensitivity analysis is consequently conservative at this sample size.

In `secondary_metrics.csv`, `wilcoxon_p_holm` and `sign_test_p_holm` refer to metric-wise adjustment, with `family_size = 21`. Fields ending in `_holm_joint` and `joint_family_size = 84` record the joint sensitivity analysis. Both scopes use identical effects, raw tests, and bootstrap samples.

## 5. Binary outcomes

Completion, failure, and reported collision were analyzed as separate flags. We report both-true, PID-only-true, SAC-only-true, and both-false counts; controller rates; and SAC-minus-PID rate differences. Positive completion differences and negative failure/collision differences favor SAC. Completion requires final-goal settling under the evaluator's specified tolerances.

For exact McNemar inference, let `b` and `c` denote discordant PID-only-true and SAC-only-true pairs. We used a two-sided Binomial(b+c, 0.5) test on `c`, assigning p = 1 when no discordant pairs were observed. This is the exact formulation described in the [statsmodels McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

Controller-rate intervals are marginal exact Clopper–Pearson binomial intervals. Zero observed failures retain uncertainty about population failure risk. Collision flags reflect the source workflow's geometric checks or proxies.

## 6. Observation length and navigation

Failed flights were retained in primary analyses. Their errors summarize the portions observed. Decision counts differ in 54/210 pairs; common-time-window metrics require per-step trajectories.

The sensitivity subset requires both flights to be nonfailed and to have equal recorded counts. It retains 152 pairs in 20 conditions. Its interpretation is conditional on survival and equal-count selection. Empty groups retain a multiple-testing family slot.

The archived Dijkstra route is 6.98 m. Evaluation documentation describes a departure of approximately 2.8 s with the residual disabled. All 14 failed PID+SAC flights have zero recorded mean residual. These findings are consistent with launch-stage failure and limit interpretation of the engaged policy's airborne behavior in those cases. Both controller modes fail all ten outside-range navigation cases.

## 7. Numerical audit and reproducibility

An independent implementation checked all nine tables against the original archived CSV. It reconstructed signed-rank probabilities by enumerating sign assignments, sign/McNemar probabilities from binomial coefficients, BCa limits from resampling and jackknife acceleration, and marginal rate limits from beta quantiles. It also checked metric-wise and joint corrections, paired-flight values, seed-block aggregates, observation lengths, and descriptive effort. All 12,958 field checks passed, using relative and absolute numerical tolerances of 1e-10.

Exact reproduction additionally requires all nine CSV files to match byte for byte under the recorded software versions and resampling configuration. The original source files, input and code hashes, audit record, and artifact manifest accompany our analysis.

## 8. Interpretation boundaries

Our results concern one checkpoint and the specified PID baseline in Isaac Sim. Independent policy-training variation, alternative PID tunings, state-estimation effects, and physical flight require separate evidence. Time-resolved tracking, event recovery, directional wind effects, and correction–integral alignment require per-step reference, state, wind, correction, integral, and event traces. The available summaries support flight-level comparisons.

A confirmation study should declare endpoints and use evaluation cases held out from tuning and model selection. Additional sample size should follow the desired precision or power for those endpoints.
