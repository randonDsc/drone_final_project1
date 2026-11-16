# Residual SAC Flight-Control Evaluation

We evaluated the first flight SAC checkpoint against its PID baseline using 420 archived Isaac Sim flight records. Our paired analysis covers seven scenarios, three wind profiles, and ten environment seeds per condition. The study's scope, methods, and results are documented in the accompanying manuscripts.

## Manuscripts and results

| Material | Purpose |
| --- | --- |
| [Full report](outputs/flight1_2026-11-14/report.md) | Study design, methods, findings, discussion, and limitations |
| [Publication PDF](outputs/flight1_2026-11-14/analysis_report.pdf) | Typeset manuscript, primary estimates, and scientific figures |
| [Full LaTeX manuscript](outputs/flight1_2026-11-14/report.tex) | Directly maintained source of the PDF |
| [Standalone LaTeX subsection](outputs/flight1_2026-11-14/report_section.tex) | Condensed section for integration into the project manuscript |
| [Statistical protocol](METHODS.md) | Estimands, assumptions, test families, and references |
| [Primary results](outputs/flight1_2026-11-14/tables/primary_rmse.csv) | All 21 primary condition comparisons |
| [Secondary results](outputs/flight1_2026-11-14/tables/secondary_metrics.csv) | Metric-wise tests and joint-adjustment sensitivity results |
| [Paired flight records](outputs/flight1_2026-11-14/tables/paired_flights.csv) | All 210 matched controller cases |

SAC achieved lower RMSE in all 180 open-air pairs, with significant primary differences in 20 of 21 conditions. Mean open-air flight-level RMSE decreased from 0.884 m to 0.115 m, an 87.0% reduction. Exploratory metric-wise tests supported lower cross-track RMSE in 20 conditions, maximum position error in 20, and maximum cross-track error in 18. Motor saturation was zero throughout. Strong-wind Dijkstra cases failed in both modes during a phase consistent with residual-disabled launch. These results characterize one checkpoint and one PID configuration.

## Separation of analysis and manuscript preparation

Python modules process data, calculate statistics, verify results, and export CSV/JSON outputs and scientific plots. The Markdown and LaTeX manuscripts are maintained directly as text. The publication PDF is compiled from `report.tex` using LaTeX.

Analysis runs require new output directories and preserve the maintained manuscripts.

## Repository layout

```text
analysis/
  inputs/                     Local original archive; large inputs ignored by Git
  outputs/flight1_2026-11-14/
    source_data/              Preserved original evaluation and configuration files
    tables/                   Nine analytical and diagnostic CSV files
    figures/                  Eleven plots, each in PNG and vector PDF formats
    figure_export_checks.json Axis-label bounds and export-padding checks
    report.md                 Full manuscript in Markdown
    report.tex                Full manuscript in LaTeX
    report_section.tex        Condensed manuscript subsection
    analysis_report.pdf       LaTeX-compiled publication manuscript
    provenance.json           Input/model/code hashes, versions, and parameters
    validation.json           Data integrity and pairing checks
    numerical_audit.json      Independent numerical audit
    summary.json              Structured analytical summaries
    outputs_manifest.json     Artifact sizes and checksums
    verification.json         Software-test and exact reproduction record
  tests/                      Statistical correctness and data-integrity tests
  archive_io.py               Input validation and source preservation
  stats_tools.py              Paired inference and descriptive statistics
  figures.py                  Scientific plots only
  run_analysis.py             Numerical analysis entry point
  audit_results.py            Independent calculations from original flight data
  verify_outputs.py          Artifact and exact numeric reproduction checks
  METHODS.md                  Statistical protocol
  requirements.txt            Recorded package versions
  .cache/                     Ignored plotting cache and temporary build files
```

## Analytical outputs

| CSV file | Rows | Contents |
| --- | ---: | --- |
| `paired_flights.csv` | 210 | Matched source values, paired differences, and eligibility flags |
| `primary_rmse.csv` | 21 | Primary position-RMSE effects, intervals, and adjusted tests |
| `secondary_metrics.csv` | 84 | Four metric-wise 21-test families, with joint 84-test sensitivity fields |
| `paired_outcomes.csv` | 63 | Completion, failure, and collision counts and exact paired tests |
| `sensitivity_rmse.csv` | 21 | Nonfailed, equal-count conditional comparisons |
| `seed_cluster_overview.csv` | 5 | Overall fixed-benchmark effects using ten seed blocks |
| `seed_cluster_values.csv` | 50 | Within-seed benchmark averages |
| `observed_lengths.csv` | 21 | Decision-count differences and sensitivity eligibility |
| `control_effort_descriptive.csv` | 42 | Residual magnitude/change, saturation, and clearance summaries |

`mean_difference` is PID+SAC minus PID. Negative error differences favor SAC. `wilcoxon_p_holm` and `sign_test_p_holm` apply within the stated `test_family`; confidence intervals remain pointwise. Secondary rows use `family_size = 21`, with joint sensitivity results in fields ending in `_holm_joint` and `joint_family_size = 84`. `n_pairs` gives the actual resampling-unit count, including ten seed blocks in overviews. `ci_status` identifies insufficient or degenerate samples. Binary-outcome intervals describe marginal controller rates.

The source summary's maximum-error columns are maxima across flights. Secondary comparisons analyze each flight's maximum as a paired observation.

## Input provenance

The original archive is stored locally as `inputs/runs.zip`, with size 1,037,617,728 bytes and SHA-256 `0c3b9fcd25ae02491bf9c56b77550b8adbe2037d693438f0f4c5732f4336369c`. It is ignored by Git. We preserve selected small source files byte for byte in `source_data/`, enabling numeric reproduction without distributing large weights or replay buffers. Original archived source documents remain unchanged as provenance records.

## Reproducing the numerical analysis

The following commands run from the repository root. The recorded environment uses Python 3.11 and the versions in `requirements.txt`.

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/python -m pip install -r analysis/requirements.txt
.venv-analysis/bin/python -m analysis.run_analysis \
  --source-data analysis/outputs/flight1_2026-11-14/source_data \
  --output analysis/outputs/flight1_reproduced
```

An existing compatible environment can be used by substituting its Python executable. `--no-figures` restricts the run to numerical outputs. The default protocol uses 50,000 BCa bootstrap resamples and master seed 20261114. The checkpoint identity and ten evaluation seeds are fixed to the archived benchmark.

Re-reading the original archive uses:

```bash
.venv-analysis/bin/python -m analysis.run_analysis \
  --archive analysis/inputs/runs.zip \
  --output analysis/outputs/flight1_from_archive
```

## Numerical verification

```bash
.venv-analysis/bin/python -m pytest analysis/tests -q
.venv-analysis/bin/python -m analysis.audit_results \
  analysis/outputs/flight1_2026-11-14 \
  --archive analysis/inputs/runs.zip
.venv-analysis/bin/python -m analysis.verify_outputs \
  analysis/outputs/flight1_2026-11-14
```

The independent audit compares the tables with the original archived CSV using a separate implementation of exact tests, both correction scopes, and BCa intervals. Its record contains 12,958 successful field checks. The verifier checks artifact and source hashes, runs the tests, and requires all nine tables to reproduce byte for byte. Plot checks cover label bounds and padding; a regression test verifies clear margins in the scatter-plot export. `verification.json` is outside the manifest to avoid circular hashing.

## Typesetting the manuscript

LaTeX compilation operates on the directly maintained manuscript and does not invoke Python. These commands run from the repository root with a TeX distribution installed:

```bash
mkdir -p analysis/.cache/latex-build
pdflatex -interaction=nonstopmode -halt-on-error \
  -jobname=analysis_report -output-directory=analysis/.cache/latex-build \
  analysis/outputs/flight1_2026-11-14/report.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -jobname=analysis_report -output-directory=analysis/.cache/latex-build \
  analysis/outputs/flight1_2026-11-14/report.tex
cp analysis/.cache/latex-build/analysis_report.pdf \
  analysis/outputs/flight1_2026-11-14/analysis_report.pdf
```

Artifact checksums are refreshed after manuscript or compiled-PDF revisions. Analytical reproduction and manuscript typesetting remain separate operations.
