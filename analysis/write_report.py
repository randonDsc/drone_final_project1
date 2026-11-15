"""Generate a reviewable interpretation and a report-ready LaTeX subsection."""
from pathlib import Path

import numpy as np

from .archive_io import SCENARIOS, PROFILES


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def interval(row):
    if row["mean_difference_ci_low"] is None:
        return "Unavailable ("+row["ci_status"]+")"
    return f"[{row['mean_difference_ci_low']:.3f}, {row['mean_difference_ci_high']:.3f}]"


def write_report(output, context):
    rows, pairs = context["rows"], context["pairs"]
    summary, config, meta = context["summary"], context["config"], context["metadata"]
    primary, overview = context["primary"], context["overview"]
    n = len(meta["validation"]["evaluation_seeds"])
    open_group = next(r for r in overview if r["benchmark"]=="open_air_all_profiles")
    strong = next(r for r in primary if r["scenario"]=="dijkstra" and r["wind_profile"]=="outside_range")
    sac_zero_failures = sum(r["controller"]=="pid_sac" and r["failed"] and r["residual_norm"]==0 for r in rows)
    secondary_rejections = sum(r["wilcoxon_reject_holm"] for r in context["secondary"])
    lines = [
        "# Paired PID versus PID+SAC statistical analysis",
        "",
        "Analysis date: "+config["analysis_date"]+". **Exploratory retrospective analysis of archived Isaac Sim results.**",
        "",
        f"The archive contains {len(rows)} evaluated flights forming {len(pairs)} matched cases in 21 scenario/wind conditions, with {n} environment seeds per condition. "
        f"SAC has lower recorded position RMSE in {summary['open_air_sac_lower_rmse']}/{summary['open_air_pairs']} open-air paired cases. "
        f"Across all conditions, {summary['conditions_lower_mean_rmse']}/21 mean differences favor SAC; {summary['primary_wilcoxon_holm_rejections']}/21 primary two-sided Wilcoxon tests reject after Holm correction. "
        "These results concern the archived checkpoint and its particular PID baseline. They do not establish real-flight performance, training-seed robustness, or cross-simulator agreement.",
        "",
        "## Model, source data and scope",
        "",
        f"The target is `runs/flight_sac/flight1/final_model.zip`, verified from its JSON metadata at **{meta['selected_model']['timesteps']:,} training decisions**, "
        f"{meta['selected_model']['gradient_updates']:,} gradient updates, training seed {meta['selected_model']['training_seed']}, 128 observation values and four residual actions. "
        "The policy was not loaded or deserialized, and neither training nor simulation was run for this analysis.",
        "",
        "- Model ZIP SHA-256: `"+meta["selected_model"]["model_sha256"]+"`.",
        "- Original archive SHA-256: `"+meta["archive_sha256"]+"`.",
        "- The original CSV, JSON, summary, provided PDF and configuration snapshots are preserved in `source_data/`.",
        "- Duplicate identifiers, missing pairs, mismatched recorded wind seeds/speeds or route durations, non-finite metrics and disagreement with source summaries cause explicit errors.",
        "- The available recorded pairing fields match. Exact initial-state vectors and reference hashes are not stored in the evaluation output, so their equality cannot be independently reconstructed from this CSV alone.",
        "",
        "The target's curriculum gradually reaches 2–7 m/s mean wind, gust scale 0.5–1.0 and wind turn rate up to 0.2 rad/s. "
        "`training_range` evaluates within those final-stage wind ranges on new seeds; `unseen_gusts` uses stronger/faster disturbances; `outside_range` uses 8–10 m/s mean wind. "
        "Training calm-flight percentages are not evaluation weights. A mean-wind limit is not an instantaneous gust-speed limit.",
        "",
        "The archive also contains `flight2`, a continuation to 2.5 million total steps with added stages reaching 14 m/s mean wind, but no saved evaluation for that run. "
        "Flight2 and intermediate checkpoints are not independent training replicates. The first model's outside-range label must not be reused as an out-of-training claim for flight2.",
        "",
        "## Statistical methods",
        "",
        "Pairs use `(scenario, wind_profile, seed)`. The primary effect is **position RMSE(SAC) − position RMSE(PID)**; negative values favor SAC. "
        "Point estimates, medians, paired-difference standard deviations and counts of lower/tied errors are provided. "
        f"Pointwise 95% BCa bootstrap intervals use {config['bootstrap_resamples']:,} resamples with deterministic, group-specific random seeds derived from master seed {config['resampling_seed']}. "
        "Whole paired differences are resampled, not individual physics samples. Intervals are not simultaneous confidence bands.",
        "",
        "Two-sided Wilcoxon signed-rank tests use exact calculations for distinct, nonzero differences and permutation calculations for tied/zero samples. "
        f"Only rank-test differences are rounded to {config['rank_round_decimals']} decimal places to avoid spurious floating-point ranks. "
        "The test's symmetry and independent-seed assumptions remain important. A two-sided exact sign test provides a direction-only robustness check. "
        "Holm corrections are applied separately to the 21 primary Wilcoxon p-values and the 21 sign-test p-values at alpha 0.05. "
        "The two tests address different features of the paired differences; they are reported together, not selected according to which is favorable.",
        "",
        "Secondary continuous metrics are cross-track RMSE, per-flight maximum position error, per-flight maximum cross-track error and saturation fraction. "
        "Their 84 tests form a separate Holm-adjusted family for each test type. Outcome analyses use exact McNemar tests on discordant paired flags, "
        "with a separate 21-test Holm family for each outcome (completion, failure and reported collision). Controller-specific proportion intervals are exact binomial intervals; "
        "they are marginal intervals, not confidence intervals for a paired rate difference. All-zero/constant samples are flagged and do not establish population equivalence.",
        "",
        f"There are **{n} unique evaluation seed identifiers**, reused across conditions. Overall summaries first average the chosen fixed benchmark conditions within each seed, "
        f"then resample these {n} whole seed blocks. They do not pretend the {len(pairs)} cases are independent seed draws. "
        "The weighting is equal across selected benchmark conditions; it is not an estimate of an unspecified real-world or training-mixture distribution.",
        "",
        "## Primary effects by condition",
        "",
        "Each row uses all recorded pairs, including failures. Units are metres. `Wins` counts seeds with lower SAC error. "
        "Adjusted p-values below are the primary Wilcoxon family; confidence intervals remain pointwise.",
        "",
        "| Scenario | Wind profile | PID mean RMSE | SAC mean RMSE | SAC − PID | 95% CI | SAC wins | Holm p |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in primary:
        lines.append(f"| {row['scenario']} | {row['wind_profile']} | {row['pid_mean']:.3f} | {row['sac_mean']:.3f} | {row['mean_difference']:+.3f} | {interval(row)} | {row['sac_lower']}/{row['n_pairs']} | {row['wilcoxon_p_holm']:.5f} |")
    lines += [
        "",
        "![Condition effects](figures/rmse_effects.png)",
        "",
        "![Per-seed differences](figures/seed_differences.png)",
        "",
        "## Overall effects with repeated seeds grouped",
        "",
        "These are differences in **means of flight-level RMSE**, not RMSE of concatenated time samples. Each interval resamples whole seed blocks.",
        "",
        "| Fixed benchmark | Paired cases | Independent seed blocks | PID mean | SAC mean | Difference | Pointwise 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in overview:
        lines.append(f"| {row['benchmark']} | {row['paired_cases']} | {row['independent_resampling_units']} | {row['pid_mean']:.3f} | {row['sac_mean']:.3f} | {row['mean_difference']:+.3f} | {interval(row)} |")
    lines += [
        "",
        "![Seed-cluster overview](figures/seed_cluster_overview.png)",
        "",
        "## Outcomes and observed flight length",
        "",
        f"Across all {len(pairs)} cases, PID reports {summary['outcomes']['pid_only']['completed']} completions, {summary['outcomes']['pid_only']['failed']} failures and {summary['outcomes']['pid_only']['collided']} collisions. "
        f"PID+SAC reports {summary['outcomes']['pid_sac']['completed']} completions, {summary['outcomes']['pid_sac']['failed']} failures and {summary['outcomes']['pid_sac']['collided']} collisions. "
        "These pooled counts are descriptive. No recorded failure is not the same as completion. Collision flags come from the source workflow's geometric checks/proxies, not a newly verified native-contact measurement.",
        "",
        "| Scenario | Wind | PID completed | SAC completed | PID failed | SAC failed | Completion McNemar Holm p |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in SCENARIOS:
        for w in PROFILES:
            comp = next(r for r in context["outcomes"] if r["scenario"]==s and r["wind_profile"]==w and r["outcome"]=="completed")
            fail = next(r for r in context["outcomes"] if r["scenario"]==s and r["wind_profile"]==w and r["outcome"]=="failed")
            lines.append(f"| {s} | {w} | {comp['pid_true']}/{n} | {comp['sac_true']}/{n} | {fail['pid_true']}/{n} | {fail['sac_true']}/{n} | {comp['mcnemar_p_holm']:.5f} |")
    lines += [
        "",
        f"**{meta['validation']['different_recorded_lengths']}/{len(pairs)} pairs have different recorded decision counts.** "
        "The original metrics average what was flown. A short terminated flight and a full surviving flight therefore have different observed horizons. "
        "Decision counts diagnose this issue but do not provide precise termination timestamps. Raw trajectories are missing, so common-time-window RMSE cannot be recomputed.",
        "",
        "A secondary sensitivity table keeps only pairs with no recorded failure in either mode and identical decision counts. "
        "This is explicitly survivor-conditioned, can introduce selection bias, and does not replace the all-pair results. "
        "The relevant outputs are `tables/observed_lengths.csv` and `tables/sensitivity_rmse.csv`.",
        "",
        "![Outcomes](figures/outcomes.png)",
        "",
        "![Observed lengths](figures/observed_lengths.png)",
        "",
        "![Conditional sensitivity](figures/rmse_sensitivity.png)",
        "",
        "## Dijkstra takeoff limitation",
        "",
        f"Under outside-range wind, both modes fail all {n} cases. The mean error difference is {strong['mean_difference']:+.3f} m "
        f"with pointwise interval {interval(strong)}. All SAC failures in the dataset with zero recorded mean residual number {sac_zero_failures}. "
        "The supplied report describes a roughly 2.8 s departure during which the residual is disabled; the zero-residual records support that explanation. "
        "These are full navigation-system/takeoff outcomes, not evidence of how an engaged SAC controller performs airborne in those cases. "
        "The archive does not contain the full per-step logs needed to reconstruct every event independently.",
        "",
        "The archived obstacle route is reported as **6.98 m**. The current repository's longer warehouse destination is a different experiment. "
        "Any additional evaluation must preserve or explicitly version the route, launch rules, model identity and wind configuration rather than append incompatible cases.",
        "",
        "## Secondary measurements and unavailable questions",
        "",
        f"The secondary continuous family has {secondary_rejections}/{len(context['secondary'])} Holm-adjusted Wilcoxon rejections. "
        "With only ten nonzero pairs, the minimum two-sided exact signed-rank p-value is 2/1024; the first Holm threshold for 84 comparisons is 0.05/84. "
        "This makes that secondary family very conservative. Failure to reject there does not establish no effect. "
        "Secondary point estimates and pointwise intervals remain available for exploratory interpretation.",
        "",
        "Residual magnitude/change and saturation are described in `tables/control_effort_descriptive.csv`. "
        "Residual norms are normalized action magnitudes, not physical energy, signed torque direction, or alignment with wind. "
        "Zero saturation describes the recorded actuator clipping metric; it does not prove safety or zero control effort.",
        "",
        "The saved evaluation does **not** support error-over-time curves, sustained post-event recovery, headwind/tailwind/crosswind classification, "
        "or correction alignment with wind and PID integral terms. Those require per-step evaluation traces with reference/state/wind vectors, corrections, integrals and event times. "
        "Final-goal completion time is not event recovery time. Training progress logs and replay buffers are not substitutes for held-out evaluation trajectories.",
        "",
        "## Report-ready conclusion and next work",
        "",
        f"For the archived first checkpoint, adding SAC substantially reduces the recorded open-air tracking error relative to the specified PID. "
        f"The equally weighted open-air benchmark has a mean paired difference of {open_group['mean_difference']:+.3f} m, "
        f"with a seed-block pointwise 95% interval of {interval(open_group)}. "
        "Dijkstra results expose a separate launch limitation and should be presented independently. "
        "Inference concerns variation across environment seeds for this checkpoint; both archived path models share one training lineage. "
        "A stronger PID baseline, independently trained SAC policies and fresh held-out evaluation cases would be separate extensions.",
        "",
        "1. Review these retrospective results and agree on a small set of primary claims.",
        "2. Keep the model hash, baseline gains and condition definitions fixed for a confirmation study.",
        "3. Use fresh seeds if these ten already informed tuning/model selection; add runs based on required precision rather than promising that a fixed seed count guarantees power.",
        "4. Let the teammate responsible for temporal analysis collect the missing time-series/event measurements. No additional simulation is required to reproduce the current statistical tables.",
        "5. Evaluate flight2 separately if desired. Its 8–10 m/s cases are within its expanded mean-wind training range, and it is not an independent training seed.",
        "",
        "## Files and reproducibility",
        "",
        "- `analysis_report.pdf`: overview and scientific figures.",
        "- `report_section.tex`: standalone subsection for inclusion in the team report; adjust figure paths as needed.",
        "- `tables/`: all pairs, condition effects, paired outcomes, observed-length diagnostics, survivor-conditioned sensitivity and seed-cluster summaries.",
        "- `source_data/`: original small archived files and saved metadata, with hashes in provenance.",
        "- `provenance.json`, `validation.json`, `summary.json`, `outputs_manifest.json`: parameters, input/code/version hashes, checks and all output checksums.",
        "- `../../METHODS.md`: method definitions, inference families and statistical references.",
        "",
        "No simulation/controller source, model weights, training settings or Git branches were modified by this pipeline. "
        "Only the new analysis directory and its outputs are used. Running code is published separately from the large ignored input archive.",
        "",
    ]
    (output/"report.md").write_text("\n".join(lines), encoding="utf-8")
    latex = [r"% Generated from the archived flight1 results; see report.md for full methods.",
             r"\subsection{Paired statistical evaluation of PID and residual SAC}",
             f"We analyzed {len(rows)} archived Isaac Sim flights, forming {len(pairs)} matched cases across seven scenarios, three wind profiles, and {n} environment seeds per condition. "
             f"The fixed SAC checkpoint had {meta['selected_model']['timesteps']:,} training decisions. "
             "The primary effect was the paired difference in position RMSE (PID+SAC minus PID), so negative differences favor SAC. "
             f"Pointwise 95\\% BCa intervals used {config['bootstrap_resamples']:,} bootstrap resamples. "
             "Two-sided Wilcoxon signed-rank tests were Holm-adjusted across the 21 primary conditions. "
             "Overall intervals resampled whole environment-seed blocks to preserve dependence across conditions.",
             "",
             f"SAC produced lower recorded RMSE in all {summary['open_air_pairs']} open-air pairs and lower mean error in {summary['conditions_lower_mean_rmse']} of the 21 conditions. "
             f"The primary adjusted signed-rank tests rejected in {summary['primary_wilcoxon_holm_rejections']} conditions. "
             f"For the equally weighted open-air benchmark, the mean paired difference was ${open_group['mean_difference']:.3f}$ m "
             f"(pointwise 95\\% seed-block interval $[{open_group['mean_difference_ci_low']:.3f}, {open_group['mean_difference_ci_high']:.3f}]$ m).",
             "",
             r"\begin{table}[ht]", r"\centering", r"\begin{tabular}{lrrr}", r"\hline",
             r"Open-air wind profile & PID RMSE (m) & PID+SAC RMSE (m) & Difference (m) \\", r"\hline"]
    for profile in PROFILES:
        r = next(r for r in overview if r["benchmark"]=="open_air_"+profile)
        latex.append(f"{profile.replace('_',' ')} & {r['pid_mean']:.3f} & {r['sac_mean']:.3f} & {r['mean_difference']:+.3f} \\\\")
    latex += [r"\hline", r"\end{tabular}",
              r"\caption{Descriptive means across six open-air route types. Uncertainty is calculated using whole seed blocks, not by treating every case as independent.}", r"\end{table}", "",
              f"There were {meta['validation']['different_recorded_lengths']} pairs with different recorded decision counts. "
              "Failed flights remain in the primary summaries, but their errors cover shorter observed horizons. "
              "A survivor-conditioned sensitivity analysis is therefore reported separately. "
              "Dijkstra strong-wind cases are further limited by takeoff with the SAC residual disabled; both controllers failed every case in that condition. "
              "These results characterize one checkpoint and one PID baseline rather than training-seed robustness or real-flight safety. "
              "The archive does not contain the time-series measurements needed for event-recovery or correction--integral analysis.", ""]
    (output/"report_section.tex").write_text("\n".join(latex), encoding="utf-8")
