"""Scientific regression checks captured from the original evaluation code."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import per_chrom_ap_table, per_chrom_weighted_ap


class MetricTests(unittest.TestCase):
    def test_original_paired_readout_and_joint_bootstrap_results(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame(
            {
                "subset": np.repeat(["missense", "promoter"], 24),
                "chrom": np.tile(np.repeat(["1", "2", "3"], 8), 2),
                "label": np.tile([0, 1], 24),
                "probe": rng.random(48),
                "llr": rng.random(48),
            }
        )
        frame.loc[0, "probe"] = np.nan
        result = per_chrom_ap_table(
            frame, ["probe", "llr"], n_bootstrap=100, rng=0, n_min=1
        )
        # Captured before consolidation, including the joint chromosome bootstrap
        # for the two macro-average rows. This is not a reimplementation oracle.
        expected = np.array(
            [
                [0.65, 0.108100290890635],
                [0.524603174603175, 0.132999876536910],
                [0.703174603174603, 0.131847798701226],
                [0.700793650793651, 0.086426619136434],
                [0.676587301587302, 0.090490970133022],
                [0.612698412698413, 0.041880659530419],
            ]
        )
        np.testing.assert_allclose(result[["value", "se"]], expected, rtol=1e-12)
        self.assertEqual(
            result["subset"].tolist(),
            [
                "missense",
                "missense",
                "promoter",
                "promoter",
                "_macro_avg_",
                "_macro_avg_",
            ],
        )
        reverse = per_chrom_ap_table(
            frame, ["llr", "probe"], n_bootstrap=100, rng=0, n_min=1
        )
        keys = ["score_type", "subset"]
        pd.testing.assert_frame_equal(
            result.sort_values(keys).reset_index(drop=True),
            reverse.sort_values(keys).reset_index(drop=True),
        )

    def test_nonfinite_and_single_class_chromosomes_are_excluded(self):
        value = per_chrom_weighted_ap(
            np.array([0, 1, 0, 1, 1, 1]),
            np.array([0.1, 0.9, np.nan, np.inf, 0.2, 0.4]),
            np.array(["1", "1", "2", "2", "3", "3"]),
        )
        self.assertEqual(value, 1.0)
        self.assertTrue(
            np.isnan(
                per_chrom_weighted_ap(
                    np.array([1, 1]), np.array([0.1, 0.2]), np.array(["1", "1"])
                )
            )
        )

    def test_single_chromosome_has_no_bootstrap_uncertainty(self):
        frame = pd.DataFrame(
            {
                "subset": ["a", "a"],
                "chrom": ["1", "1"],
                "label": [0, 1],
                "score": [0.1, 0.9],
            }
        )
        table = per_chrom_ap_table(frame, ["score"], n_min=1)
        self.assertTrue(table["se"].isna().all())
        self.assertTrue((table["value"] == 1).all())


if __name__ == "__main__":
    unittest.main()
