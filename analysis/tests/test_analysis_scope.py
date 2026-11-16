"""Maintain separation between numerical analysis and manuscript preparation."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageChops

from analysis.audit_results import exact_binomial_tail, holm, signed_rank
from analysis.archive_io import parse_rows, pair_rows, SCENARIOS, PROFILES
from analysis.run_analysis import DEFAULTS, analyze
from analysis.stats_tools import SECONDARY, continuous_table, secondary_table


def test_independent_exact_tests_match_hand_calculated_probabilities():
    assert exact_binomial_tail(0, 10) == 2/1024
    assert exact_binomial_tail(5, 10) == 1
    assert signed_rank(np.array([-1., -2., -3.])) == (0., .25)
    assert signed_rank(np.array([-1., -1., 0.])) == (0., .5)
    assert signed_rank(np.zeros(4)) == (0., 1.)
    assert holm([.04, .001, .03]) == pytest.approx([.06, .003, .06])


def test_analysis_exports_data_and_plots_without_manuscripts(tmp_path):
    source = Path(__file__).parents[1]/"outputs/flight1_2026-11-14/source_data"
    output = tmp_path/"analysis"
    analyze(output, source_data=source, config=dict(DEFAULTS, bootstrap_resamples=1000))
    assert len(list((output/"tables").glob("*.csv"))) == 9
    assert len(list((output/"figures").glob("*.pdf"))) == 11
    assert len(list((output/"figures").glob("*.png"))) == 11
    assert not list(output.rglob("*.md"))
    assert not list(output.rglob("*.tex"))
    assert not (output/"analysis_report.pdf").exists()
    assert json.loads((output/"provenance.json").read_text())["narrative_reports_generated"] is False
    exports = json.loads((output/"figure_export_checks.json").read_text())
    assert len(exports)==11
    assert all(item["axis_labels_within_export"] for item in exports.values())
    assert all(item["x_axis_labels_separated"] for item in exports.values())
    with Image.open(output/"figures/paired_scatter.png") as raw:
        image = raw.convert("RGB")
        box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).getbbox()
        assert min(box[0], box[1], image.width-box[2], image.height-box[3])>=25


def test_secondary_adjustment_preserves_estimates_and_uses_metric_families():
    source = Path(__file__).parents[1]/"outputs/flight1_2026-11-14/source_data/flights.csv"
    pairs = pair_rows(parse_rows(source.read_bytes()))
    config = dict(DEFAULTS, bootstrap_resamples=1000)
    joint = continuous_table(pairs, SECONDARY, config, "secondary_continuous_84", SCENARIOS, PROFILES)
    metric_wise = secondary_table(pairs, config, SCENARIOS, PROFILES)
    numeric = ("pid_mean", "sac_mean", "mean_difference", "median_difference", "difference_sd",
               "bootstrap_seed", "mean_difference_ci_low", "mean_difference_ci_high", "wilcoxon_p", "sign_test_p")
    for before, after in zip(joint, metric_wise):
        assert all(before[field]==after[field] for field in numeric)
        assert after["family_size"]==21 and after["joint_family_size"]==84
        assert after["test_family"]=="secondary_"+after["metric"]
        assert after["wilcoxon_p_holm_joint"]==before["wilcoxon_p_holm"]
        assert after["sign_test_p_holm_joint"]==before["sign_test_p_holm"]
    assert [sum(r["metric"]==m and r["wilcoxon_reject_holm"] for r in metric_wise)
            for m in SECONDARY]==[20, 20, 18, 0]
    assert not any(row["wilcoxon_reject_holm_joint"] for row in metric_wise)
