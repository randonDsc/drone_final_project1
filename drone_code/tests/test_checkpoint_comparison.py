"""Pairing and clustered-summary checks; per-flight validation has separate tests."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_wind_checkpoints import compare


class CheckpointComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for route in ('hover', 'straight', 'waypoints'):
            for label in ('original', 'adapted'):
                folder = self.root / f'{route}_{label}'
                folder.mkdir()
                (folder / "source").mkdir()
                (folder / "source/evaluation.py").write_text("same source")
                manifest = dict(seeds=[1, 2], scenarios=['control'], duration_s=20, wind_profile='training_range', mean_wind_scenarios={}, route=route, contract={}, sampling_hz=10, git_commit='same', model_sha256=label)
                (folder / 'manifest.json').write_text(json.dumps(manifest))
                flights = []
                for seed in (1, 2):
                    (folder / f'reference_{seed}.json').write_text('{}')
                    for mode in ('pid_only', 'pid_sac'):
                        rmse = .2 if mode == 'pid_only' or label == 'original' else .1
                        flights.append(dict(scenario='control', seed=seed, controller=mode, wind_seed=seed, heading_rad=0, event_s=5, route=route, reference_speed_at_event_m_s=0, route_duration_s=20, position_rmse_m=rmse, post_event_peak_m=rmse, failed=False, recovered=True, recovery_time_s=1))
                        with (folder / f'control_{seed}_{mode}.csv').open('w', newline='') as handle:
                            writer = csv.writer(handle)
                            writer.writerow(['time_s', 'target_x_m', 'target_y_m', 'target_z_m', 'mean_wind_x_m_s', 'mean_wind_y_m_s', 'mean_wind_z_m_s'])
                            writer.writerow([.1, 0, 0, 1, 4, 0, 0])
                (folder / 'flights.json').write_text(json.dumps(flights))

    def run_comparison(self):
        with patch('compare_wind_checkpoints.analyze'), contextlib.redirect_stdout(io.StringIO()):
            compare(self.root)

    def test_known_improvement_counts_seeds_not_conditions(self):
        self.run_comparison()
        result = json.loads((self.root / 'checkpoint_comparison.json').read_text())['groups'][0]
        self.assertEqual(result['pairs'], 6)
        self.assertEqual(result['seed_count'], 2)
        self.assertAlmostEqual(result['mean_delta_m'], -.1)
        self.assertEqual(result['adapted_lower_count'], 6)

    def test_rejects_unmatched_reference(self):
        (self.root / 'hover_adapted/reference_1.json').write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'Reference trajectories differ'):
            self.run_comparison()

    def test_rejects_changed_pid_trace(self):
        path = self.root / 'hover_adapted/control_1_pid_only.csv'
        path.write_text(path.read_text().replace('0.1,', '0.2,'))
        with self.assertRaisesRegex(ValueError, 'Repeated PID baseline differs'):
            self.run_comparison()

    def test_rejects_same_checkpoint(self):
        for folder in self.root.glob('*_adapted'):
            path = folder / 'manifest.json'
            manifest = json.loads(path.read_text())
            manifest['model_sha256'] = 'original'
            path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'same checkpoint'):
            self.run_comparison()


if __name__ == '__main__':
    unittest.main()
