"""Reject corrupt/misinterpreted source tables before producing inference."""
import copy
import csv
import hashlib
import io
import json

import pytest

from analysis.archive_io import (NUMERIC, OUTCOMES, load_source_data, parse_rows,
                                 validate_reference_files)


def csv_bytes(rows):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


def fixture_tables():
    flights = []
    for controller, values in (("pid_only", (1., 3.)), ("pid_sac", (.5, 1.5))):
        for seed, value in enumerate(values):
            row = dict.fromkeys(NUMERIC, value)
            row.update(dict.fromkeys(OUTCOMES, False))
            row.update(scenario="hover", wind_profile="training_range", controller=controller,
                       seed=seed, wind_seed=9223372036854775801+seed, decisions=100,
                       failure_reason=None, completion_time=None, wind_speed=3.,
                       clearance_violation_rate=None, route_seconds=10., saturation_rate=0.,
                       residual_norm=0. if controller=="pid_only" else .2,
                       residual_change=0. if controller=="pid_only" else .1)
            flights.append(row)
    summary = []
    for controller, mean, maximum in (("pid_only", 2., 3.), ("pid_sac", 1., 1.5)):
        summary.append(dict(scenario="hover", wind_profile="training_range", controller=controller,
                            flights=2, position_rmse=mean, cross_track_rmse=mean,
                            max_deviation=maximum, cross_track_max=maximum,
                            saturation_rate=0., residual_norm=0. if controller=="pid_only" else .2,
                            residual_change=0. if controller=="pid_only" else .1,
                            completion_rate=0., failure_rate=0., collision_rate=0., completion_time=None))
    return parse_rows(csv_bytes(flights)), dict(flights=flights, summary=summary), csv_bytes(summary)


def test_summary_maximum_is_checked_as_maximum_across_flights():
    rows, reference, summary = fixture_tables()
    validate_reference_files(rows, reference, summary)
    wrong = copy.deepcopy(reference)
    wrong["summary"][0]["max_deviation"] = 2.  # Mean, not source's maximum of 3.
    with pytest.raises(ValueError, match="Summary mismatch"):
        validate_reference_files(rows, wrong, summary)


@pytest.mark.parametrize("field,value", [("position_rmse", 99.), ("wind_seed", 9223372036854775803),
                                       ("failed", True), ("failure_reason", "collision")])
def test_csv_json_disagreement_stops_analysis(field, value):
    rows, reference, summary = fixture_tables()
    wrong = copy.deepcopy(reference)
    wrong["flights"][0][field] = value
    with pytest.raises(ValueError, match="CSV/JSON mismatch"):
        validate_reference_files(rows, wrong, summary)


def test_repeated_summary_rows_do_not_silently_replace_each_other():
    rows, reference, summary = fixture_tables()
    reference["summary"].append(reference["summary"][0])
    with pytest.raises(ValueError, match="Summary group identifiers"):
        validate_reference_files(rows, reference, summary)


def test_modified_small_bundle_fails_hash_validation(tmp_path):
    source = tmp_path/"source"
    source.mkdir()
    expected = b"original input bytes"
    (source/"flights.csv").write_bytes(b"modified input bytes")
    metadata = dict(source_files={"flights.csv":dict(sha256=hashlib.sha256(expected).hexdigest())})
    (source/"dataset_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Source file hash mismatch"):
        load_source_data(source, tmp_path/"output")
