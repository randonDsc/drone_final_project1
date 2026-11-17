import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_evaluation_suite as pipeline


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_reject_duplicate_seeds(self):
        config = dict(routes=['hover'], scenarios=['control'], seeds=[1, 1], duration_s=20)
        path = self.root / 'suite.json'
        pipeline.write(path, config)
        with self.assertRaisesRegex(ValueError, 'Seeds'):
            pipeline.suite_config(path)

    def test_reject_unknown_event(self):
        path = self.root / 'suite.json'
        pipeline.write(path, dict(routes=['hover'], scenarios=['typo'], seeds=[1], duration_s=20))
        with self.assertRaisesRegex(ValueError, 'scenarios'):
            pipeline.suite_config(path)

    def test_reject_unsafe_model_name(self):
        with self.assertRaisesRegex(ValueError, 'NAME=PATH'):
            pipeline.model_entries(['../escape=model.zip'])

    def test_reject_batch_checkpoint_mismatch_before_analysis(self):
        pipeline.write(self.root / 'manifest.json', dict(route='hover', seeds=[1], scenarios=['control'], duration_s=20, model_sha256='wrong'))
        plan = dict(suite=dict(seeds=[1], scenarios=['control'], duration_s=20))
        with self.assertRaisesRegex(ValueError, 'model_sha256'):
            pipeline.check_batch(self.root, plan, dict(sha256='expected'), 'hover')

    def test_dry_run_does_not_create_output(self):
        output = self.root / 'not_created'
        model = dict(name='test', sha256='abc')
        with patch.object(sys, 'argv', ['suite', '--model', 'test=unused', '--output', str(output), '--dry-run']), patch.object(pipeline, 'model_entries', return_value=[model]), contextlib.redirect_stdout(io.StringIO()):
            pipeline.main()
        self.assertFalse(output.exists())

    def test_resume_refuses_changed_source(self):
        pipeline.write(self.root / 'pipeline.json', dict(sources={'old': 'hash'}))
        with patch.object(sys, 'argv', ['suite', '--resume', str(self.root)]):
            with self.assertRaisesRegex(ValueError, 'Code changed'):
                pipeline.main()


if __name__ == '__main__':
    unittest.main()
