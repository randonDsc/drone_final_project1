# Statistical methods and scope

Protocol version: 1. Analysis date: 2026-11-14. This protocol documents a **retrospective exploratory analysis**, written after the archive and original report were available. It is not a preregistration. Package versions, parameters, code hashes, and input hashes are saved with every run.

## Fixed data and model

The source is the supplied `runs.zip`, specifically `runs/flight_sac/flight1/evaluation/final_model/`. Its `flights.csv` contains 420 rows: seven scenarios × three wind profiles × ten environment seeds × two controllers. Scenarios are hover, straight, turns/waypoints, altitude, circle, figure eight, and Dijkstra. Seeds are 300000–300009.

The checkpoint is `flight1/final_model.zip`, SHA-256 `2d9e5cd0df1a3cbabf8ae2eaa75884783c9c16bf6a3ced737398291b0b67023d`. Its metadata reports 1,500,000 decisions, training seed 0, 128 observations, and four residual actions. This checkpoint was selected because it is the team's first model with archived evaluation, rather than through a checkpoint sweep in this pipeline. Weight files are hashed as bytes; serialized model objects are never loaded.

The source report identifies Isaac Sim evaluation on 4 November 2026, using simulator ground truth and a frozen deterministic SAC policy. The saved model is a residual controller: its thrust/torque correction is added to the PID command before mixing. The model's four action magnitudes are distinct from a geographic wind vector. The stored policy contract contains the PID gains, residual limits, sampling rates, observation scaling, and curriculum.

The final training stage uses 2–7 m/s mean wind. The archived report defines the evaluation profiles as training-range wind on new seeds, stronger/faster unseen gusts, and outside-range mean wind of 8–10 m/s. These labels are specific to flight1. Flight2 continues the same training lineage to 2.5 million total decisions, with mean-wind stages up to 14 m/s; its configuration snapshots are preserved as evidence. Flight2 has no evaluation table in the archive and contributes no statistical observations here. Earlier hover/PPO experiments are separate tasks.

## Data integrity and pairing

Pairs are keyed by `(scenario, wind_profile, seed)`. Each key must have exactly one `pid_only` and one `pid_sac` record. The complete scenario/profile/seed grid is required. Within a pair, `wind_seed`, recorded `wind_speed` when present, and `route_seconds` must agree. Wind seeds remain Python integers so values above 2^53 are never rounded through floating point.

The loader rejects duplicate/missing flights, undeclared controllers, missing columns, non-finite/negative metrics, invalid outcome flags, and inconsistent failure flags/reasons. It cross-checks each flight against the source JSON and recomputes source summaries. The source summary's `cross_track_max` and `max_deviation` are **maxima across flights**, unlike its mean-RMSE columns. Source files are copied byte for byte and hashed. ZIP names are inspected and only specific data/configuration files are copied; archived scripts and pickle payloads are not executed.

Recorded pairing fields support matching. The original evaluation report states that routes and initial conditions were shared within a pair. Exact reset vectors and reference hashes are absent, so that stronger claim cannot be independently verified from the summary data. Reproduction from `source_data/` verifies the preserved small-file hashes; it does not rehash the unavailable full ZIP or reload the model. Provenance makes that distinction explicit.

## Estimands and resampling units

For each condition and environment seed, define `d_i = RMSE_SAC,i − RMSE_PID,i`. The primary effect estimate is the arithmetic mean of the ten differences. Negative differences favor SAC. Also report controller means, median difference, sample standard deviation of differences, SAC/PID win counts, and ties. Differences within 1e-12 count as ties. The percentage reduction is `100 × (mean_PID − mean_SAC) / mean_PID`, omitted when the PID mean is zero.

Position RMSE is the source evaluator's time-indexed position tracking metric, in metres. This analysis consumes the recorded flight-level metric rather than reconstructing it from trajectories. The overall benchmark averages **flight-level RMSE values**; it is not the square root of a pooled sum of squared position samples. Secondary maxima are each flight's maximum error and are compared as paired observations.

Pointwise 95% confidence intervals for mean differences use SciPy's bias-corrected and accelerated (BCa) bootstrap with 50,000 resamples. Resampling paired differences preserves controller pairing. Group-specific generator seeds are derived by SHA-256 from master seed 20261114 and a stable group label. Small samples and skewed differences can limit interval coverage; resampling 50,000 times does not create additional independent flight data. Constant samples get a degenerate-observed-sample label, and non-finite BCa results remain unavailable rather than silently switching methods. See the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).

Only ten environment seed identifiers occur across the entire benchmark. For an overview, first average the selected fixed conditions within each seed, then resample the resulting ten whole seed blocks. This preserves any cross-condition dependence induced by reusing seeds. Conditions are equally weighted within the specified benchmark; this does not estimate an unknown deployment distribution. The five overviews overlap and are not five independent experiments.

No confidence interval here is adjusted to provide simultaneous coverage across conditions. A pointwise interval excluding zero does not automatically imply rejection after multiple-testing adjustment.

## Paired tests and multiple comparisons

The primary test is two-sided Wilcoxon signed-rank on paired differences. Its null is a difference distribution symmetric about zero; interpreting it as a location-shift test relies on the usual symmetry assumption. It is **not a direct test of equality of means**. Environment-seed cases within a condition must be independent draws for the inferential interpretation. Existing data cannot establish that these seeds were never used in earlier model selection.

Differences are rounded to 12 decimal places for rank calculation only, following SciPy's warning about subtraction-induced false ranks. For distinct nonzero absolute differences use the exact signed-rank distribution. For zeros or ties use an exhaustive sign-permutation calculation for these small samples, dropping zeros from ranks (`zero_method='wilcox'`). All-zero differences receive p=1 with an explicit no-rank-information label. Larger future tied samples would use a deterministic Monte Carlo permutation calculation. See the [SciPy Wilcoxon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).

A two-sided exact sign test considers the number of negative differences among non-ties under a Binomial(n, 0.5) null. It addresses direction rather than difference magnitude and avoids the signed-rank symmetry interpretation. Its adjusted results are a separate robustness check; they are not an alternative selected after observing p-values. The implementation uses [SciPy binomtest](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

Holm's step-down adjustment controls family-wise error at alpha 0.05 within each declared family. It does not require independence between tests. Raw and adjusted p-values, family names, and family sizes are retained. Unavailable tests retain a family slot conservatively. The families are:

| Family | Number | Purpose |
| --- | ---: | --- |
| Primary position-RMSE signed-rank tests | 21 | Main condition-specific inference |
| Primary position-RMSE sign tests | 21 | Direction-only robustness |
| Secondary continuous signed-rank tests | 84 | Four metrics × 21 conditions |
| Secondary continuous sign tests | 84 | Same metrics, direction-only |
| Completion exact McNemar tests | 21 | Secondary paired outcome |
| Failure exact McNemar tests | 21 | Secondary paired outcome |
| Reported-collision exact McNemar tests | 21 | Secondary paired outcome |
| Survivor-conditioned RMSE signed-rank tests | 21 | Exploratory sensitivity |
| Survivor-conditioned RMSE sign tests | 21 | Exploratory sensitivity robustness |
| Seed-block overview signed-rank tests | 5 | Exploratory fixed-benchmark summaries |
| Seed-block overview sign tests | 5 | Overview robustness |

There is no claim of one global 5% error rate across all families. Primary claims use the primary family; secondary and sensitivity results remain separately labeled. The Holm algorithm agrees with the method described in the [statsmodels multiple-testing documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html), although the runtime implementation needs only NumPy/SciPy.

With ten nonzero pairs the smallest possible two-sided exact signed-rank p-value is `2/2^10 = 0.001953125`. The first Holm threshold in an 84-test family is `0.05/84 ≈ 0.000595`. Therefore that secondary family has very limited rejection power at this sample size. Report effect estimates and uncertainty even when adjusted tests do not reject. Non-rejection is not proof of no effect.

## Binary outcomes

Completion, failure, and collision are analyzed as three recorded binary flags. A flight without failure is not necessarily completed. For each outcome report the 2×2 paired counts, controller rates, and SAC-minus-PID rate difference. For completion positive differences are favorable; for failure/collision negative differences are favorable.

Exact McNemar inference conditions on discordant pairs. If `b` pairs are PID-only true and `c` are SAC-only true, use the two-sided binomial test on `c` of `b+c` at p=0.5. If there are no discordant pairs, assign p=1. This matches the exact-binomial formulation in the [statsmodels McNemar documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html). It avoids a small-sample chi-squared approximation.

Controller-specific 95% rate intervals use exact binomial (Clopper–Pearson) intervals. These marginal intervals do not represent the uncertainty of the paired rate difference. In particular, 0/10 failures does not establish a zero population failure probability. Source collision flags reflect geometric/proxy checks and are not independently revalidated native simulator contacts.

## Failed flights, navigation, and unavailable analyses

All available pairs remain in the primary analysis; failures are never dropped to improve a controller's score. Error metrics average only the observed portion of each flight. Fifty-four pairs have different recorded decision counts, so error summaries can compare different horizons. The saved data cannot produce a common-time-window RMSE or a censoring-aware reconstruction. Decision counts are a duration diagnostic, not precise failure timestamps.

The sensitivity subset requires both flights to be nonfailed and to have equal recorded decision counts. This can select easier cases and introduce survivor bias. It is explicitly conditional, excludes empty groups from inference, and does not replace the all-pair analysis. Equal counts alone do not prove identical actual sample times.

Dijkstra is kept distinct from open-air routes. The archived report describes a 6.98 m path and a departure of roughly 2.8 seconds with the residual disabled. All 14 failed SAC flights have zero recorded mean residual, consistent with launch-stage failure. Under outside-range wind both modes fail all ten cases. These are full-system launch/navigation outcomes; the archive cannot establish the performance of an engaged airborne SAC policy during those failed cases. Current repository navigation settings use a different longer warehouse route, so a new simulation would be a new experiment.

Residual norm/change and motor saturation are descriptive only. A normalized residual magnitude is neither physical energy nor a signed correction direction. The archive does not provide held-out per-step position/reference errors, event markers, wind vectors, signed corrections, or PID integral vectors. Consequently this analysis cannot estimate error-over-time curves, gust/corner recovery time, headwind/tailwind/crosswind classes, or correction–wind/integral alignment. Final-goal completion time and training rollout summaries are not replacements.

These inferences describe environment-seed variation for one policy and the specified PID baseline in Isaac Sim. Independent policy-training seeds, stronger baseline controllers, sensor-estimation error, MuJoCo transfer, and real flights require additional experiments. Fresh held-out seeds are appropriate for later confirmation if existing cases informed tuning; sample size should follow the precision or power needed for the chosen claim.
