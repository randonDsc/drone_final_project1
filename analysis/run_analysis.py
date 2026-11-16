"""Run from the repository root: python -m analysis.run_analysis --output PATH."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import zipfile

from .archive_io import (load_archive, load_source_data, json_write, write_csv,
                         sha256_file, SCENARIOS, PROFILES, METRICS, OUTCOMES)
from .stats_tools import (PRIMARY, SECONDARY, continuous_table, secondary_table, outcome_table,
                          cluster_overview, diagnostic_tables)


DEFAULTS = dict(analysis_date="2026-11-14", bootstrap_resamples=50000, resampling_seed=20261114,
                expected_evaluation_seeds=list(range(300000, 300010)),
                expected_model_sha256="2d9e5cd0df1a3cbabf8ae2eaa75884783c9c16bf6a3ced737398291b0b67023d",
                expected_model_timesteps=1500000,
                confidence_level=.95, alpha=.05, difference_tolerance=1e-12, rank_round_decimals=12,
                interval_method="BCa", exploratory=True,
                difference_definition="SAC minus PID; negative favors SAC for errors",
                model_selection="flight1 final model; no checkpoint selection from these test results",
                primary_family="21 scenario/profile position-RMSE Wilcoxon tests, Holm correction",
                robustness_test="two-sided exact paired sign test, separately Holm-adjusted",
                secondary_family="21 tests per secondary metric; four separate Holm families; exploratory",
                secondary_joint_sensitivity="84 secondary tests in one Holm family",
                binary_families="21 exact McNemar tests per outcome; three separate secondary families",
                sensitivity_family="21 survivor-conditioned RMSE tests, Holm correction",
                overview_family="five exploratory seed-cluster overview tests, Holm correction",
                ci_scope="pointwise 95% intervals, not simultaneous intervals",
                resampling_unit="environment seed pair; whole seed blocks for cross-condition overviews",
                failed_flights="retained in primary summaries; metrics describe observed portions only")


def versions():
    result = dict(python=platform.python_version(), platform=platform.platform())
    for name in ("numpy", "scipy", "matplotlib"):
        result[name] = importlib.metadata.version(name)
    return result


def git_state(root):
    def query(*args):
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    return dict(branch=query("branch", "--show-current"), commit=query("rev-parse", "HEAD"))


def pair_table(pairs):
    result = []
    fields = METRICS+OUTCOMES+("decisions", "completion_time", "failure_reason", "residual_norm", "residual_change", "clearance_violation_rate")
    for pair in pairs:
        row = {k:v for k,v in pair.items() if k not in ("pid", "sac")}
        for controller in ("pid", "sac"):
            row.update({controller+"_"+k:pair[controller][k] for k in fields})
        for metric in METRICS:
            row[metric+"_difference"] = pair["sac"][metric]-pair["pid"][metric]
        row["same_recorded_decision_count"] = pair["pid"]["decisions"]==pair["sac"]["decisions"]
        row["either_failed"] = pair["pid"]["failed"] or pair["sac"]["failed"]
        row["sensitivity_eligible"] = row["same_recorded_decision_count"] and not row["either_failed"]
        result.append(row)
    return result


def analyze(output, *, archive=None, source_data=None, config=None, make_figures=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(DEFAULTS if config is None else config)
    try:
        if (archive is None) == (source_data is None):
            raise ValueError("Choose exactly one input archive or extracted source-data folder")
        if archive is not None:
            rows, pairs, metadata = load_archive(archive, "flight1", output/"source_data")
        else:
            rows, pairs, metadata = load_source_data(source_data, output/"source_data")
        if metadata["selected_run"] != "flight1":
            raise ValueError("This analysis protocol is scoped to the first model")
        if sorted({p["seed"] for p in pairs}) != config["expected_evaluation_seeds"]:
            raise ValueError("Evaluation seeds differ from the frozen archived benchmark; declare a separate analysis protocol")
        if (metadata["selected_model"]["model_sha256"] != config["expected_model_sha256"] or
                metadata["selected_model"]["timesteps"] != config["expected_model_timesteps"]):
            raise ValueError("Checkpoint differs from the frozen first-model identity; declare a separate analysis protocol")
        tables = output/"tables"
        tables.mkdir()
        primary = continuous_table(pairs, [PRIMARY], config, "primary_position_rmse", SCENARIOS, PROFILES)
        secondary = secondary_table(pairs, config, SCENARIOS, PROFILES)
        sensitivity = continuous_table(pairs, [PRIMARY], config, "survivor_conditioned_sensitivity", SCENARIOS, PROFILES, subset=True)
        outcomes = outcome_table(pairs, config, SCENARIOS, PROFILES)
        overview, seed_overview = cluster_overview(pairs, config)
        lengths, effort = diagnostic_tables(pairs, SCENARIOS, PROFILES)
        all_tables = {"paired_flights.csv":pair_table(pairs), "primary_rmse.csv":primary,
                      "secondary_metrics.csv":secondary, "paired_outcomes.csv":outcomes,
                      "sensitivity_rmse.csv":sensitivity, "seed_cluster_overview.csv":overview,
                      "seed_cluster_values.csv":seed_overview, "observed_lengths.csv":lengths,
                      "control_effort_descriptive.csv":effort}
        for name, values in all_tables.items():
            write_csv(tables/name, values)
        code_root = Path(__file__).parent
        provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), configuration=config,
                          input_mode="archive" if archive is not None else "extracted_source_data",
                          archive_rehashed_this_run=archive is not None,
                          dataset=metadata, versions=versions(), repository=git_state(code_root.parent),
                          analysis_sources_sha256={p.name:sha256_file(p) for p in sorted(code_root.glob("*.py"))},
                          evaluation_not_rerun=True, training_not_rerun=True, model_not_deserialized=True,
                          simulator_imported=False, analysis_does_not_modify_simulation_sources=True,
                          narrative_reports_generated=False)
        json_write(output/"provenance.json", provenance)
        json_write(output/"validation.json", metadata["validation"])
        summary = dict(flights=len(rows), pairs=len(pairs), independent_seed_identifiers=len(metadata["validation"]["evaluation_seeds"]),
                       primary_conditions=len(primary),
                       conditions_lower_mean_rmse=sum(r["mean_difference"]<0 for r in primary),
                       primary_wilcoxon_holm_rejections=sum(r["wilcoxon_reject_holm"] for r in primary),
                       primary_sign_holm_rejections=sum(r["sign_test_reject_holm"] for r in primary),
                       open_air_pairs=sum(p["scenario"]!="dijkstra" for p in pairs),
                       open_air_sac_lower_rmse=sum(p["scenario"]!="dijkstra" and p["sac"][PRIMARY]<p["pid"][PRIMARY] for p in pairs),
                       outcomes={c:{o:sum(r[o] for r in rows if r["controller"]==c) for o in OUTCOMES} for c in ("pid_only", "pid_sac")},
                       secondary_metrics={metric:dict(
                           conditions=sum(r["metric"]==metric for r in secondary),
                           lower_mean_error=sum(r["metric"]==metric and r["mean_difference"]<0 for r in secondary),
                           wilcoxon_holm_rejections=sum(r["metric"]==metric and r["wilcoxon_reject_holm"] for r in secondary),
                           sign_holm_rejections=sum(r["metric"]==metric and r["sign_test_reject_holm"] for r in secondary),
                           joint_holm_rejections=sum(r["metric"]==metric and r["wilcoxon_reject_holm_joint"] for r in secondary))
                           for metric in SECONDARY},
                       overviews=overview,
                       note="Exploratory retrospective analysis; confidence intervals are pointwise and concern this checkpoint over environment seeds.")
        json_write(output/"summary.json", summary)
        context = dict(rows=rows, pairs=pairs, primary=primary, secondary=secondary, sensitivity=sensitivity,
                       outcomes=outcomes, overview=overview, lengths=lengths, effort=effort,
                       metadata=metadata, config=config, summary=summary)
        if make_figures:
            from .figures import create_figures
            create_figures(output, context)
        manifest = {str(p.relative_to(output)):dict(bytes=p.stat().st_size, sha256=sha256_file(p))
                    for p in sorted(output.rglob("*")) if p.is_file()}
        json_write(output/"outputs_manifest.json", manifest)
        return summary
    except Exception as error:
        json_write(output/"execution_error.json", dict(type=type(error).__name__, message=str(error)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--archive", type=Path)
    input_group.add_argument("--source-data", type=Path, help="Small source_data folder from a previous output")
    parser.add_argument("--output", type=Path, required=True, help="New output folder; existing folders are never overwritten")
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULTS["bootstrap_resamples"])
    parser.add_argument("--resampling-seed", type=int, default=DEFAULTS["resampling_seed"])
    parser.add_argument("--no-figures", action="store_true", help="Useful for numerical reproduction/checks")
    args = parser.parse_args(argv)
    if args.bootstrap_resamples<1000 or args.resampling_seed<0:
        parser.error("Use at least 1000 bootstrap resamples and a nonnegative resampling seed")
    config = dict(DEFAULTS, bootstrap_resamples=args.bootstrap_resamples, resampling_seed=args.resampling_seed)
    archive = args.archive or (Path(__file__).parent/"inputs/runs.zip" if args.source_data is None else None)
    try:
        summary = analyze(args.output, archive=archive, source_data=args.source_data, config=config, make_figures=not args.no_figures)
    except (ValueError, FileExistsError, FileNotFoundError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(f"Saved analysis of {summary['flights']} flights / {summary['pairs']} pairs to {args.output}")
    print(f"Primary Holm-adjusted Wilcoxon rejections: {summary['primary_wilcoxon_holm_rejections']}/{summary['primary_conditions']}")


if __name__ == "__main__":
    main()
