"""Compare species and higher-rank coverage using harmonised NCBI taxonomy.

Run with --resolve to refresh Marin name resolution from the cached taxdump;
otherwise use the bundled resolved taxonomy. DToL uses its saved resolution.
"""
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from resolve_dtol_taxonomy import ROOT, resolve

OUT = ROOT / 'plot_data'
ALIASES = {'Ateles geoffroyi a': 'Ateles geoffroyi'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resolve', action='store_true')
    args = parser.parse_args()
    ap = ROOT/'data/dataset_provenance/annotation_genomes.tsv'
    pp = ROOT/'data/dataset_provenance/projection_species.tsv'
    dp = ROOT/'data/dtol_provenance/taxonomy.csv'
    mp = OUT/'marin_comparison_taxonomy.csv'
    a = pd.read_csv(ap, sep='\t', keep_default_na=False)
    p = pd.read_csv(pp, sep='\t', keep_default_na=False)
    pnames = p.species.str.replace('_', ' ').replace(ALIASES)
    if args.resolve:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'names.csv'
            pd.DataFrame({'Organism': sorted(set(a.species) | set(pnames))}).to_csv(path, index=False)
            m = resolve(path, taxid_overrides=dict(zip(a.species, a['Organism Taxonomic ID'])))
            m.to_csv(mp, index=False)
    m = pd.read_csv(mp, keep_default_na=False)
    d = pd.read_csv(dp, keep_default_na=False)
    assert set(m.organism) == set(a.species) | set(pnames)
    assert (d.ncbi_name == d.species).all(), 'DToL IDs must be at species rank'
    d['species_taxid'] = d.taxid
    lookup = m.set_index('organism').species_taxid
    ai, pi = set(lookup.loc[a.species]), set(lookup.loc[pnames])
    m = m.drop_duplicates('species_taxid')
    d = d.drop_duplicates('species_taxid')
    mi, di = set(m.species_taxid), set(d.species_taxid)
    assert ai | pi == mi
    # Every shared species must carry identical ranked taxonomy in both snapshots.
    ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    shared = m.set_index('species_taxid').loc[sorted(mi & di), ranks]
    pd.testing.assert_frame_equal(shared, d.set_index('species_taxid').loc[sorted(mi & di), ranks])
    datasets = {'Marin combined': m, 'DToL all': d, 'DToL animals': d[d.kingdom == 'Metazoa'],
                'Marin annotation': m[m.species_taxid.isin(ai)], 'Marin projection': m[m.species_taxid.isin(pi)]}
    rows = []
    for rank in ranks:
        row = {'rank': rank}
        for label, data in datasets.items():
            row[label] = data[rank].replace('', pd.NA).nunique()
        ms, ds = set(m[rank])-{''}, set(d[rank])-{''}
        row.update(shared=len(ms & ds), marin_only=len(ms-ds), dtol_only=len(ds-ms))
        rows.append(row)
    richness = pd.DataFrame(rows)
    richness.to_csv(OUT/'marin_dtol_rank_richness.csv', index=False)
    overlaps = m[m.species_taxid.isin(di)][['species_taxid', *ranks]]
    overlaps.to_csv(OUT/'marin_dtol_shared_species.csv', index=False)
    compositions = []
    for label, data in datasets.items():
        for rank in ['kingdom', 'phylum', 'class', 'order']:
            for group, count in data[rank].value_counts().items():
                compositions.append({'dataset': label, 'rank': rank, 'taxon': group,
                                     'species': int(count), 'percent': 100*count/len(data)})
    pd.DataFrame(compositions).to_csv(OUT/'marin_dtol_composition.csv', index=False)
    report = {'marin_annotation_species': len(ai), 'marin_projection_species': len(pi),
              'within_marin_overlap': len(ai & pi), 'marin_combined_species': len(mi),
              'projection_additional_species': len(pi-ai), 'dtol_species': len(di),
              'shared_species': len(mi & di), 'marin_only_species': len(mi-di),
              'dtol_only_species': len(di-mi), 'jaccard_species': len(mi & di)/len(mi | di),
              'marin_only_animal_phyla': sorted(set(m.phylum)-set(d.phylum)),
              'dtol_only_animal_phyla': sorted(set(d[d.kingdom=='Metazoa'].phylum)-set(m.phylum)),
              'missing_rank_records': {label: {rank: int(data[rank].eq('').sum()) for rank in ranks} for label, data in datasets.items()},
              'input_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ap, pp, dp, mp]},
              'taxonomy_source': json.loads((ROOT/'data/dtol_provenance/sources.json').read_text()),
              'label_normalization': ALIASES,
              'alias_evidence': 'https://karlssonlab.org/zoonomia/ (GCA_004024785.1 is Ateles geoffroyi)',
              'weighting': 'Each distinct species taxid counts once; these are taxonomic counts, not training-token proportions.'}
    (OUT/'marin_dtol_comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(richness.to_string(index=False))
    print(json.dumps({k: v for k, v in report.items() if k not in ['input_sha256', 'taxonomy_source']}, indent=2))
    for label, data in [('Marin', m), ('DToL', d)]:
        print(label, 'classes:', data['class'].value_counts().head(8).to_dict())
        print(label, 'mammal orders / families:', data[data['class']=='Mammalia']['order'].nunique(), data[data['class']=='Mammalia'].family.nunique())


if __name__ == '__main__':
    main()
