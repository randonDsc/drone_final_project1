import pytest

from mujoco_drone.comparison import compare
from mujoco_drone.experiments import write_csv


def test_paired_comparison_checks_protocol_and_includes_failures(tmp_path):
    row = dict(protocol_hash="verified-protocol", seed=1, simulator="mujoco", mode="pid",
               controller_family="original_adapted", metric_version=1, position_rmse=.1,
               cross_track_rmse=.1, max_deviation=.2, altitude_rmse=.01,
               saturation_rate=0, completed=True, failed=False, crashed=False)
    left, right = tmp_path/"left.csv", tmp_path/"right.csv"
    write_csv(left, [row])
    write_csv(right, [dict(row, mode="pid_sac", position_rmse=.2, completed=False, failed=True)])
    result = compare(left, right, tmp_path/"comparison")
    assert result["position_rmse_right_minus_left_mean"] == pytest.approx(.1)
    assert result["failed_rate_right"] == 1
    write_csv(right, [dict(row, protocol_hash="different-task")])
    with pytest.raises(ValueError, match="Protocols/seeds differ"):
        compare(left, right, tmp_path/"invalid")
