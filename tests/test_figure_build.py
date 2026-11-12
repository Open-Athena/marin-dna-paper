"""Contracts for offline data and publication-output protection."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import local_data
import build_figure

class FigureBuildTests(unittest.TestCase):
    def test_remote_missing_is_explicit_and_offline(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(local_data, 'ROOT', Path(directory)), patch.dict('os.environ', {}, clear=True):
            with patch('urllib.request.urlopen', side_effect=AssertionError('network')):
                with self.assertRaisesRegex(local_data.MissingData, 'Original source: s3://bucket/metrics.parquet'):
                    local_data.local_path('s3://bucket/metrics.parquet')

    def test_cached_parquet_is_read_locally(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(local_data, 'ROOT', Path(directory)):
            path = Path(directory) / 'remote/s3/bucket/metrics.parquet'
            path.parent.mkdir(parents=True)
            pd.DataFrame({'value': [0.25], 'se': [0.01]}).to_parquet(path)
            self.assertEqual(local_data.read_polars('s3://bucket/metrics.parquet')['value'][0], 0.25)

    def test_wandb_snapshot_roundtrip(self):
        value = {'series': pd.Series([0.2, 0.3], index=[100, 200]), 'nested': {'loss': pd.DataFrame({'_step': [100], 'loss': [0.4]})}, 'array': (np.array([1., 2.]), np.array([0.1, 0.2]))}
        restored = local_data._decode(json.loads(json.dumps(local_data._encode(value))))
        pd.testing.assert_series_equal(value['series'], restored['series'])
        pd.testing.assert_frame_equal(value['nested']['loss'], restored['nested']['loss'])
        np.testing.assert_array_equal(value['array'][0], restored['array'][0])

    def test_missing_input_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'figures').mkdir()
            output = root / 'figures/existing.pdf'
            output.write_bytes(b'original publication figure')
            entry = {'inputs': ['absent.csv'], 'outputs': {'existing': 'existing'}}
            with patch.object(build_figure, 'ROOT', root), patch.dict(build_figure.MANIFEST, {'missing_test': entry}):
                with self.assertRaises(local_data.MissingData):
                    build_figure.build('missing_test')
            self.assertEqual(output.read_bytes(), b'original publication figure')

    def test_manifest_has_unique_outputs_and_existing_inputs(self):
        seen = set()
        for entry in build_figure.MANIFEST.values():
            for output in entry['outputs'].values():
                self.assertNotIn(output, seen)
                seen.add(output)
            for source in entry['inputs']:
                self.assertTrue((build_figure.ROOT / 'plot_data' / source).is_file())

if __name__ == '__main__': unittest.main()
