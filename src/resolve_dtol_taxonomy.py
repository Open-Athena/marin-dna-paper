"""Resolve the supplied DToL names against a cached NCBI taxonomy dump."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DUMP = ROOT / 'plot_data/remote/https/ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz'
URL = 'https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz'
DEST = ROOT / 'data/dtol_provenance'


def resolve(source: Path, dump: Path = DUMP, taxid_overrides: dict | None = None) -> pd.DataFrame:
    d = pd.read_csv(source, dtype=str, keep_default_na=False)
    assert d.Organism.str.strip().ne('').all()
    wanted = set(d.Organism.str.strip())
    matches = {name: [] for name in wanted}
    scientific = {}
    nodes = {}
    merged = {}
    with tarfile.open(dump, 'r:gz') as archive:
        for line in archive.extractfile('merged.dmp'):
            old, new = [v.strip() for v in line.decode().split('|')[:2]]
            merged[int(old)] = int(new)
        for line in archive.extractfile('names.dmp'):
            taxid, name, _, kind = [v.strip() for v in line.decode().split('|')[:4]]
            if kind == 'scientific name':
                scientific[int(taxid)] = name
            if name in wanted:
                matches[name].append((int(taxid), kind))
        for line in archive.extractfile('nodes.dmp'):
            taxid, parent, rank = [v.strip() for v in line.decode().split('|')[:3]]
            nodes[int(taxid)] = (int(parent), rank)
    rows = []
    for name in sorted(wanted):
        hits = matches[name]
        exact = [h for h in hits if h[1] == 'scientific name']
        choices = exact or hits
        if len({h[0] for h in choices}) != 1 and taxid_overrides and name in taxid_overrides:
            taxid = int(taxid_overrides[name])
            while taxid in merged:
                taxid = merged[taxid]
            choices = [(taxid, 'manifest taxid fallback')]
        assert len({h[0] for h in choices}) == 1, (name, choices)
        taxid, kind = choices[0]
        row = {'organism': name, 'taxid': taxid, 'ncbi_name': scientific[taxid], 'match_type': kind}
        lineage = []
        current = taxid
        while current != 1:
            parent, rank = nodes[current]
            lineage.append(scientific[current])
            if rank in ['superkingdom', 'domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
                row[rank] = scientific[current]
            if rank == 'species':
                row['species_taxid'] = current
            current = parent
        row['lineage'] = ';'.join(reversed(lineage))
        rows.append(row)
    return pd.DataFrame(rows).fillna('')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    args = parser.parse_args()
    resolved = resolve(args.input)
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / 'organisms.csv').write_bytes(args.input.read_bytes())
    resolved.to_csv(DEST / 'taxonomy.csv', index=False)
    report = {'input_path': str(args.input.resolve()),
              'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
              'taxonomy_url': URL, 'taxonomy_sha256': hashlib.sha256(DUMP.read_bytes()).hexdigest(),
              'taxonomy_cache_date_utc': datetime.fromtimestamp(DUMP.stat().st_mtime, timezone.utc).date().isoformat(),
              'resolution': 'Exact scientific name preferred; otherwise an unambiguous exact NCBI name or synonym. No fuzzy or genus-only matches.',
              'matched_names': len(resolved), 'match_types': resolved.match_type.value_counts().to_dict()}
    (DEST / 'sources.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
