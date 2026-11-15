# Statistical Analysis Protocol

**Study:** Paired evaluation of residual SAC flight control

**Analysis date:** 14 November 2026

**Numerical review:** 15 November 2026

## 1. Study design and scope

We conducted a retrospective analysis of the first flight SAC checkpoint's archived Isaac Sim evaluation. Our dataset comprises 420 flight records: seven scenarios, three wind profiles, ten environment seeds per condition, and two controller modes. The checkpoint is `flight1/final_model.zip`, trained for 1,500,000 decisions with training seed 0. Its SHA-256 is `2d9e5cd0df1a3cbabf8ae2eaa75884783c9c16bf6a3ced737398291b0b67023d`.

The scenarios are hover, straight, turns/waypoints, altitude change, circle, figure eight, and Dijkstra navigation. The environment seeds are 300000–300009. The evaluation documentation describes a frozen deterministic policy and ground-truth simulator state. Our inference concerns variation across these environment seeds for one policy and one PID configuration. The analysis is exploratory and retrospective; the recorded protocol is not a preregistration.

## 2. Source validation and pairing

We matched each `pid_only` flight with its `pid_sac` counterpart using `(scenario, wind_profile, seed)`. We required exactly one record per controller, a complete condition–seed grid, and agreement in recorded wind seed, mean wind speed when present, and reference-route duration. Wind seeds were preserved as integers to prevent floating-point rounding.

We checked finite and nonnegative metrics, valid binary outcomes, failure-flag/reason consistency, duplicate identifiers, and missing pairs. Flight-level CSV values were cross-checked against `results.json`; source summaries were independently recomputed. Source `max_deviation` and `cross_track_max` summarize maxima across flights, whereas our secondary paired analyses treat each flight's maximum as an observation. These estimands are distinct.

The archived report states that paired routes and starting conditions were shared. Exact reset vectors and reference hashes are not stored, limiting independent verification of those additional properties. We retained original source bytes and checksums for traceability.

## 3. Effects and confidence intervals

For each condition and seed, we defined `d = position_RMSE_PID+SAC − position_RMSE_PID`. Negative differences favor SAC. We report controller means, mean and median paired differences, sample standard deviation of differences, and win/tie counts. Differences within 1e-12 count as ties. Relative mean reduction is `100 × (mean_PID − mean_PID+SAC) / mean_PID`, omitted when the PID mean is zero.

Position RMSE is the evaluator's time-indexed tracking metric. Cross-track error measures geometric distance to the reference path independently of progress along that path. Our analysis consumes recorded flight-level values; it does not reconstruct trajectories. Overall summaries are means of flight-level RMSE, not RMSE calculated from concatenated time samples.

We used pointwise 95% bias-corrected and accelerated (BCa) bootstrap intervals with 50,000 resamples. The master resampling seed was 20261114; group-specific seeds were derived by SHA-256 from the master seed and a stable group label. Within conditions, paired differences were resampled as units. Constant observed samples received a degenerate-sample label; groups with fewer than two pairs had no interval. Non-finite BCa results remained unavailable instead of triggering an undeclared alternative method. Method definitions follow the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

The same ten seed identifiers occur across conditions. For each overview benchmark, we first averaged its selected fixed conditions within each seed and then resampled the ten resulting seed blocks. This preserves any dependence associated with seed reuse. Conditions receive equal weight within the benchmark. The five overview summaries overlap and are not independent experiments or estimates of an unspecified deployment distribution.

All reported intervals are pointwise. We make no claim of simultaneous 95% coverage across conditions. With ten seed observations, coverage and precision remain sensitive to the observed sample despite the large number of bootstrap resamples.

## 4. Hypothesis tests and multiplicity

We used two-sided Wilcoxon signed-rank tests. The null concerns a paired-difference distribution symmetric about zero; interpreting a rejection as a location shift requires the corresponding symmetry assumption. The test is not a direct test of mean equality. Environment-seed observations within a condition must support the independence assumptions of the analysis.

For rank calculations only, differences were rounded to 12 decimal places to avoid spurious floating-point distinctions. Distinct, nonzero differences used exact signed-rank inference. Tied or zero differences used exhaustive sign permutations for the small samples in this study, dropping zero differences from ranks. All-zero samples received p = 1 with an explicit no-rank-information label. These conventions follow the [SciPy signed-rank documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

We also used a two-sided exact sign test on the number of SAC improvements among non-tied pairs, under a Binomial(n, 0.5) null. It measures directional consistency without using effect magnitude. The separately adjusted sign-test results serve as a robustness check; we do not select a test according to the resulting p-value. Exact binomial inference follows [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

We applied Holm's step-down adjustment at α = 0.05 within each declared family. The adjustment does not require independence among its tests. Unavailable tests conservatively retain a family slot. We retain raw and adjusted p-values, family identifiers, and family sizes. The procedure is consistent with the [statsmodels multiple-testing documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).

| Analysis family | Tests | Interpretation |
| --- | ---: | --- |
| Primary position-RMSE signed-rank | 21 | Primary condition-specific inference |
| Primary position-RMSE sign | 21 | Directional robustness |
| Secondary continuous signed-rank | 84 | Four metrics across 21 conditions |
| Secondary continuous sign | 84 | Directional robustness for secondary metrics |
| Completion exact McNemar | 21 | Secondary paired outcome |
| Failure exact McNemar | 21 | Secondary paired outcome |
| Reported-collision exact McNemar | 21 | Secondary paired outcome |
| Sensitivity position-RMSE signed-rank | 21 | Survivor-conditioned exploratory analysis |
| Sensitivity position-RMSE sign | 21 | Sensitivity robustness |
| Seed-block overview signed-rank | 5 | Exploratory benchmark summaries |
| Seed-block overview sign | 5 | Overview robustness |

These families do not provide one global 5% family-wise error rate over the entire study. Primary conclusions use the primary family; secondary, overview, and sensitivity results retain their separate status.

The minimum two-sided exact signed-rank p-value for ten nonzero pairs is `2/2^10 = 0.001953125`. It exceeds the first Holm threshold for an 84-test family, `0.05/84 ≈ 0.000595`. Therefore no secondary-family rejection is attainable under these exact tests at the current sample size. Non-rejection does not establish equivalence or absence of an effect.

## 5. Binary outcomes

We analyzed completion, failure, and reported collision as separate flags. For each outcome, we report both-true, PID-only-true, SAC-only-true, and both-false counts; controller rates; and the SAC-minus-PID rate difference. Positive completion differences favor SAC, while negative failure/collision differences favor SAC. No recorded failure does not necessarily imply completion.

For exact McNemar inference, let `b` and `c` denote discordant PID-only-true and SAC-only-true pairs. We used a two-sided Binomial(b+c, 0.5) test on `c`, assigning p = 1 when no discordant pairs were observed. This is the exact formulation described in the [statsmodels McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html).

Controller-rate intervals are exact Clopper–Pearson binomial intervals. They are marginal rate intervals, not confidence intervals for paired rate differences. A zero observed failure count does not establish zero population failure risk. Source collision flags reflect geometric/proxy checks and are not independently revalidated native-contact measurements.

## 6. Observation length and navigation

We retained all failed flights in primary analyses. Their errors summarize the observed portions of flight. Decision counts differ in 54/210 pairs, and common-time-window metrics cannot be reconstructed from the saved summaries. Equal counts alone do not verify identical sample times.

The sensitivity subset requires both flights to be nonfailed and to have equal recorded decision counts. It retains 152 pairs in 20 conditions. Selection can favor easier cases; the resulting effects do not replace the all-pair estimates. Empty groups provide no estimate and retain a multiple-testing family slot.

The archived Dijkstra route is 6.98 m. Evaluation documentation describes a departure of approximately 2.8 s with the residual disabled. All 14 failed PID+SAC flights have zero recorded mean residual. These findings are consistent with launch-stage failure and limit interpretation of the engaged policy's airborne behavior in those cases. Both controller modes fail all ten outside-range navigation cases.

## 7. Numerical audit and reproducibility

We independently checked all nine tables against the original archived CSV without importing the production statistical functions. The audit reconstructed signed-rank probabilities by enumerating sign assignments, sign/McNemar probabilities from binomial coefficients, BCa limits from explicit resampling and jackknife acceleration, and marginal rate limits from beta quantiles. It also checked paired-flight values, seed-block aggregates, observation lengths, and descriptive effort. All 12,370 field checks passed; numerical comparisons used relative and absolute tolerances of 1e-10.

Exact reproduction additionally requires all nine CSV files to match byte for byte under the recorded software versions and resampling configuration. The original source files, input and code hashes, audit record, and artifact manifest accompany our analysis.

## 8. Interpretation boundaries

Our results concern one trained checkpoint and the specified PID baseline in Isaac Sim. Independent policy-training variation, alternative PID tunings, state-estimation error, and physical flight require separate evidence. The summaries do not support time-resolved tracking curves, event-recovery estimates, headwind/tailwind/crosswind classification, or correction–wind/integral alignment. These require per-step reference, state, wind, correction, integral, and event traces. Final-goal completion time is not a disturbance-recovery measurement.

A confirmation study should declare endpoints and use evaluation cases held out from tuning and model selection. Additional sample size should follow the desired precision or power for those endpoints.
