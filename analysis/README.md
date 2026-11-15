# Archived flight statistics

This folder contains Alan's paired PID versus PID+SAC analysis of the first trained flight model. The work uses the evaluation records already inside `runs.zip`; it runs without Isaac Sim, MuJoCo, PyTorch, or additional training. It is stored on `alan/mujoco` alongside the existing project.

## Start here

The completed analysis is in [`outputs/flight1_2026-11-14/`](outputs/flight1_2026-11-14/):

1. Read [`report.md`](outputs/flight1_2026-11-14/report.md) for the findings, statistical methods, and limitations.
2. Open [`analysis_report.pdf`](outputs/flight1_2026-11-14/analysis_report.pdf) for the eight-page overview and figures.
3. Use [`report_section.tex`](outputs/flight1_2026-11-14/report_section.tex) as a draft subsection for the team report. Review it with the team before including it.
4. Inspect [`tables/primary_rmse.csv`](outputs/flight1_2026-11-14/tables/primary_rmse.csv) for the main statistical results and [`tables/paired_flights.csv`](outputs/flight1_2026-11-14/tables/paired_flights.csv) for individual pairs.

The archive provides **420 flights, 210 PID/SAC pairs, 21 scenario/wind conditions, and 10 environment seeds per condition**. SAC has lower recorded position RMSE in all 180 open-air pairs. Twenty of the 21 primary condition comparisons reject the zero-centered paired-difference null after Holm adjustment. The strong-wind Dijkstra condition does not: both modes fail all ten cases, and the recorded SAC correction is zero during those flights. This needs to be interpreted as a launch limitation.

The equally weighted open-air mean of flight-level RMSE is 0.884 m for PID and 0.115 m for PID+SAC. The paired difference is −0.769 m, with a pointwise 95% seed-block bootstrap interval of [−0.873, −0.685] m. Failed flights are retained, and some end early; these averages describe the observed flight portions. They concern this checkpoint and PID configuration.

## What this covers for your assignment

| Question | Available result |
| --- | --- |
| Does SAC help on the same seed and recorded wind? | Paired flight table, per-seed plot, paired scatter plot |
| How large and consistent is the improvement? | Mean/median differences, win counts, 95% intervals, Wilcoxon and sign tests |
| Are multiple comparisons accounted for? | Explicit Holm-adjusted families; see [METHODS.md](METHODS.md) |
| Does completion/failure differ? | Paired outcome tables and exact McNemar tests |
| Do shortened flights affect interpretation? | Length diagnostics and a separate survivor-conditioned sensitivity analysis |
| Does performance vary across the three wind profiles? | Separate effects within each profile; these are not headwind/tailwind/crosswind classifications |
| Error over time, event recovery, correction/wind/integral alignment? | Unavailable in the saved flight summaries; your teammate needs per-step evaluation traces |

Completion time is arrival/settling time at the final goal, not recovery time after a gust. Training logs or replay buffers cannot replace held-out evaluation traces. `flight2` is continued training with a broader wind curriculum and has no saved evaluation in this archive; it is excluded from the comparisons.

## Layout

```text
analysis/
  inputs/                 Original runs.zip, locally copied and ignored by Git
  outputs/
    flight1_2026-11-14/
      source_data/        Original small CSV/JSON/PDF/configuration files
      tables/             Nine statistical and diagnostic CSV tables
      figures/            Seven figures, each as PNG and vector PDF
      report.md           Full interpretation and methods
      analysis_report.pdf Overview and all figures
      report_section.tex  Draft report subsection
      provenance.json     Model/input/code hashes, versions and parameters
      validation.json     Pairing and data consistency checks
      summary.json        Machine-readable headline results
      outputs_manifest.json  Checksums for generated files
      verification.json   Test and independent reproduction record
  tests/                  Statistical correctness and input-integrity tests
  run_analysis.py         Analysis entry point
  verify_outputs.py       Check hashes and reproduce every numeric CSV
  METHODS.md              Definitions, inference assumptions and references
  requirements.txt        Versions used for the completed run
  .cache/                 Ignored development outputs and plotting cache
```

The local copy of `inputs/runs.zip` is 1,037,617,728 bytes. Its SHA-256 is `0c3b9fcd25ae02491bf9c56b77550b8adbe2037d693438f0f4c5732f4336369c`, matching `/Users/alanlin/Downloads/runs.zip`. The original download is unchanged. Large weights, replay buffers, and unrelated experiments remain inside the ignored archive. The small source bundle and analysis outputs can be shared through Git without uploading the 1 GB ZIP.

## Reproduce the analysis

Run commands from the repository root (`drone_final_project1`). The existing `.venv` already contains the required packages:

```bash
.venv/bin/python -m analysis.run_analysis \
  --source-data analysis/outputs/flight1_2026-11-14/source_data \
  --output analysis/outputs/flight1_reproduced
```

This reproduces the statistics and figures from the small saved source bundle. To re-read the intact ZIP instead:

```bash
.venv/bin/python -m analysis.run_analysis \
  --archive analysis/inputs/runs.zip \
  --output analysis/outputs/flight1_from_archive
```

Use a new output directory each time; existing outputs are never overwritten. `--no-figures` skips plotting. Defaults are 50,000 BCa bootstrap resamples, master random seed 20261114, and two-sided tests with alpha 0.05. Each group gets a deterministic resampling seed. The checkpoint hash and ten evaluation seeds are fixed to this archived benchmark; additional models or datasets need a separate declared protocol.

To create a separate analysis environment on another machine, use Python 3.11 or a compatible version:

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r analysis/requirements.txt
```

Then substitute `.venv-analysis/bin/python` in the commands above. No simulator packages are needed. Plot rendering can vary across platforms; numeric tables are checked byte for byte when using the recorded package versions.

## Verify the work

```bash
.venv/bin/python -m pytest analysis/tests -q
.venv/bin/python -m analysis.verify_outputs \
  analysis/outputs/flight1_2026-11-14
```

The verifier checks every listed output hash and the analysis source hashes, reproduces the tables into a temporary directory from `source_data/`, and requires all nine CSV files to match exactly. It does not need the large ZIP. `verification.json` records the completed verification and is outside the generation manifest so it can describe that manifest without a circular hash.

## Read the CSV columns

- `mean_difference` is **SAC minus PID**. A negative error difference favors SAC. `relative_mean_reduction_percent` compares controller means; it is not the average of per-seed percentages.
- `mean_difference_ci_low/high` are pointwise intervals for the mean difference. They are not simultaneous intervals across all conditions.
- `wilcoxon_p` is raw; `wilcoxon_p_holm` is adjusted within `test_family`. Use the adjusted value for the primary claim. The signed-rank test addresses the paired-difference distribution, not a direct test of the mean.
- `sign_test_p_holm` is a separately adjusted direction-only robustness check. Do not choose whichever test gives a favorable conclusion.
- `n_pairs` is the actual number used. In the overview table it is ten seed blocks, not 180 independent open-air samples.
- `ci_status` flags unavailable or degenerate estimates. A constant recorded sample does not establish population equivalence.
- `paired_outcomes.csv` reports `both_true`, `pid_only_true`, `sac_only_true`, and `both_false`. For completion a positive rate difference favors SAC; for failure/collision a negative difference does. Rate confidence intervals are marginal controller intervals, not intervals for the paired rate difference.
- Source `summary.csv` uses the maximum across flights for its maximum-error columns. Our secondary table instead averages paired **flight-level maxima**. These are distinct summaries, recorded explicitly in the methods.

This is a retrospective analysis of one trained checkpoint, not a preregistered confirmation study or a comparison of independent training seeds. See [METHODS.md](METHODS.md) and the full report before making broader claims.
