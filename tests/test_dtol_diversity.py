"""Check taxon accounting and dated group coverage for the DToL snapshot."""
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from plot_dtol_diversity import ROOT, summarize


class DtolTests(unittest.TestCase):
    def setUp(self):
        p = ROOT / 'data/dtol_provenance'
        self.organisms = pd.read_csv(p / 'organisms.csv', keep_default_na=False)
        self.taxonomy = pd.read_csv(p / 'taxonomy.csv', keep_default_na=False)

    def test_snapshot_totals(self):
        broad, insects, taxa = summarize(self.organisms, self.taxonomy)
        self.assertEqual(len(taxa), 1826)
        self.assertEqual(broad.taxa.sum(), 1826)
        self.assertEqual(insects.taxa.sum(), 1219)
        self.assertEqual(insects.set_index('order').loc['Lepidoptera', 'taxa'], 600)

    def test_repeated_records_and_synonyms_do_not_inflate_counts(self):
        repeated = pd.concat([self.organisms, self.organisms.iloc[[0]]])
        alias = self.taxonomy.iloc[[0]].copy()
        alias['organism'] = 'Test synonym'
        row = self.organisms.iloc[[0]].copy()
        row['Organism'] = 'Test synonym'
        _, _, taxa = summarize(pd.concat([repeated, row]), pd.concat([self.taxonomy, alias]))
        self.assertEqual(len(taxa), 1826)

    def test_missing_taxonomy_cannot_silently_drop_a_record(self):
        with self.assertRaises(AssertionError):
            summarize(self.organisms, self.taxonomy.iloc[1:])

    def test_unknown_taxonomy_cannot_be_guessed(self):
        taxonomy = self.taxonomy.copy()
        taxonomy.loc[0, 'kingdom'] = 'Unknown'
        with self.assertRaises(ValueError):
            summarize(self.organisms, taxonomy)


if __name__ == '__main__':
    unittest.main()
