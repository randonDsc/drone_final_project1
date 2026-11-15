"""Maintain separation between numerical analysis and manuscript preparation."""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.audit_results import exact_binomial_tail, holm, signed_rank
from analysis.run_analysis import DEFAULTS, analyze


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
    assert len(list((output/"figures").glob("*.pdf"))) == 7
    assert len(list((output/"figures").glob("*.png"))) == 7
    assert not list(output.rglob("*.md"))
    assert not list(output.rglob("*.tex"))
    assert not (output/"analysis_report.pdf").exists()
    assert json.loads((output/"provenance.json").read_text())["narrative_reports_generated"] is False
