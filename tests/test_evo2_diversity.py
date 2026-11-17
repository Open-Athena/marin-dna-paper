"""Check metadata counting without downloading OpenGenome2 sequences."""
import sys
from pathlib import Path
import unittest
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from plot_evo2_diversity import summarize


def row(species,domain='Eukaryote',phylum='Chordata',kingdom='Eukaryota'):
    return {'species':species,'domain':domain,'phylum':phylum,'kingdom':kingdom,
            'class':'','order':'','family':'','source_dataset':'fixture'}


class EvoMetadataTests(unittest.TestCase):
    def test_duplicates_and_source_overlap_do_not_inflate_unique_counts(self):
        rows=[row('species A'),row('species A'),row('species A','Organelle'),row('species B','Organelle')]
        s,_,stats=summarize(pd.DataFrame(rows))
        a=s.set_index('group').loc['Animals']
        self.assertEqual((a.genome_rna_taxa,a.organelle_host_taxa,a.unique_taxa_union,a.overlap),(1,2,2,1))
        self.assertEqual(stats['metadata_records'],4)
        self.assertEqual(stats['unique_taxon_labels'],2)

    def test_taxonomy_fill_uses_same_label_and_keeps_unplaced_taxa(self):
        rows=[row('species A'),row('species A','Organelle',''),row('species B',phylum='')]
        s,_,_=summarize(pd.DataFrame(rows))
        s=s.set_index('group')
        self.assertEqual(s.loc['Animals','organelle_host_taxa'],1)
        self.assertEqual(s.loc['Unplaced eukaryotes','unique_taxa_union'],1)

    def test_conflicting_taxonomy_is_rejected(self):
        with self.assertRaises(AssertionError):
            summarize(pd.DataFrame([row('species A'),row('species A',phylum='Ascomycota')]))

    def test_gtdb_phylum_suffixes_are_not_merged(self):
        rows=[row('species A','Prokaryote','Bacillota','Bacteria'),
              row('species B','Prokaryote','Bacillota_A','Bacteria')]
        s,m,_=summarize(pd.DataFrame(rows))
        self.assertEqual(set(m.phylum),{'Bacillota','Bacillota_A'})
        self.assertEqual(s.iloc[0].unique_taxa_union,2)


if __name__=='__main__':
    unittest.main()
