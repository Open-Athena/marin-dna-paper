"""Plot the supplied Darwin Tree of Life snapshot on dated reference trees.

uv run --locked python src/plot_dtol_diversity.py [--fetch]
Counts use unique NCBI taxa resolved from the supplied organism names.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from plot_dataset_timetree import ROOT, CACHE, ANIMAL_GROUPS, collapse, leaves, normalize, parse_newick
from plot_dataset_phylogeny import BLUE, ORANGE

GREEN = '#547e65'
COLORS = {'Metazoa': BLUE, 'Viridiplantae': GREEN, 'Fungi': ORANGE}
LABELS = {
    'Magnoliopsida': 'Flowering plants', 'mosses': 'Mosses', 'liverworts': 'Liverworts',
    'Chlorophyta': 'Green algae (Chlorophyta)', 'Ascomycota': 'Ascomycota · sac fungi',
    'Basidiomycota': 'Basidiomycota', 'Mortierellomycota': 'Mortierellomycota',
    'Mucoromycota': 'Mucoromycota', 'Arthropoda': 'Arthropods (incl. insects)',
    'Annelida': 'Annelids', 'Bryozoa': 'Bryozoans', 'Cnidaria': 'Cnidarians',
    'Echinodermata': 'Echinoderms', 'Mollusca': 'Molluscs', 'Nemertea': 'Ribbon worms',
    'Actinopteri': 'Ray-finned fishes', 'Chondrichthyes': 'Cartilaginous fishes',
    'Amphibia': 'Amphibians', 'Aves': 'Birds', 'Mammalia': 'Mammals',
    'Lepidosauria': 'Lepidosaurs',
    'Coleoptera': 'Coleoptera · beetles', 'Diptera': 'Diptera · flies',
    'Lepidoptera': 'Lepidoptera · moths & butterflies', 'Hymenoptera': 'Hymenoptera · wasps, bees & ants',
    'Hemiptera': 'Hemiptera · true bugs', 'Trichoptera': 'Trichoptera · caddisflies',
    'Ephemeroptera': 'Ephemeroptera · mayflies', 'Odonata': 'Odonata · dragonflies & damselflies',
    'Dermaptera': 'Dermaptera · earwigs', 'Plecoptera': 'Plecoptera · stoneflies',
    'Psocodea': 'Psocodea · barklice & relatives', 'Mecoptera': 'Mecoptera · scorpionflies',
    'Megaloptera': 'Megaloptera · alderflies & relatives',
    'Neuroptera': 'Neuroptera · lacewings & relatives', 'Raphidioptera': 'Raphidioptera · snakeflies',
}
EXTRA_SELECTIONS = {
    'mosses': {'Andreaeopsida', 'Bryopsida', 'Polytrichopsida', 'Sphagnopsida'},
    'liverworts': {'Jungermanniopsida', 'Marchantiopsida'},
    'Chlorophyta': {'Chlorophyceae'},
    'Ascomycota': {'Dothideomycetes', 'Eurotiomycetes', 'Lecanoromycetes',
                   'Leotiomycetes', 'Lichinomycetes', 'Sordariomycetes'},
    'Basidiomycota': {'Agaricomycetes', 'Tremellomycetes'},
    'Mortierellomycota': {'Mortierellomycetes'}, 'Mucoromycota': {'Mucoromycetes'},
}
SOURCES = {
    'overview': ('eukaryota_class.nwk', '23103', 'class'),
    'insects': ('metazoa_order.nwk', '211973', 'order'),
}


def group_for(row):
    if row.kingdom == 'Metazoa':
        return row['class'] if row.phylum == 'Chordata' else row.phylum
    if row.kingdom == 'Fungi':
        return row.phylum
    if row.kingdom == 'Viridiplantae':
        if row.phylum == 'Chlorophyta':
            return 'Chlorophyta'
        if row['class'] == 'Magnoliopsida':
            return 'Magnoliopsida'
        for group in ['mosses', 'liverworts']:
            if row['class'] in EXTRA_SELECTIONS[group]:
                return group
    raise ValueError(f'Unmapped organism: {row.organism}')


def summarize(organisms, taxonomy):
    assert taxonomy.organism.is_unique, 'Duplicate taxonomy matches'
    names = set(organisms.Organism.str.strip())
    assert names == set(taxonomy.organism), 'Input and resolved taxonomy differ'
    d = taxonomy.copy()
    d['group'] = d.apply(group_for, axis=1)
    assert d.groupby('taxid').group.nunique().le(1).all(), 'Conflicting groups'
    d = d.drop_duplicates('taxid')
    broad = d.groupby(['group', 'kingdom']).size().reset_index(name='taxa')
    insects = d[d['class'] == 'Insecta'].groupby('order').size().reset_index(name='taxa')
    assert broad.taxa.sum() == len(d)
    return broad, insects, d


def panel(ax, tree, values, colors, title, maximum, step, denominator, highlight=None):
    ts = leaves(tree)
    for i, n in enumerate(ts):
        n.y = len(ts) - 1 - i
    def locate(n):
        if n.children:
            for c in n.children:
                locate(c)
                assert n.age >= c.age
            n.y = (n.children[0].y + n.children[-1].y) / 2
    locate(tree)
    x = lambda age: 4.5 * (1 - age / maximum)
    ax.set_xlim(-.15, 12.9)
    ax.set_ylim(-2.4, len(ts) + 2)
    ax.axis('off')
    for tick in range(0, maximum + 1, step):
        ax.plot([x(tick)] * 2, [-.45, len(ts)-.55], color='#dde3e7', lw=.65, zorder=0)
        ax.text(x(tick), -.95, str(tick), ha='center', va='top', fontsize=9, color='#52616b')
    ax.text(2.25, -1.9, 'Millions of years ago', ha='center', fontsize=10, color='#24323b')
    for i, n in enumerate(ts):
        color = colors[n.key]
        if i % 2 == 0:
            ax.axhspan(n.y-.45, n.y+.45, color='#f2f5f6', zorder=-2)
        if n.key == highlight:
            ax.axhspan(n.y-.45, n.y+.45, color='#e4f0f7', zorder=-1)
        ax.text(4.72, n.y, n.name, va='center', fontsize=10, color=color)
        ax.barh(n.y, .95*values[n.key]/max(values.values()), left=10.2, height=.24, color=color)
        ax.text(11.8, n.y, f'{values[n.key]:,}', ha='right', va='center', fontsize=10.5, weight='bold', color=color)
        ax.text(12.85, n.y, f'{100*values[n.key]/denominator:.1f}%', ha='right', va='center', fontsize=9, color='#52616b')
    def branches(n):
        if n.children:
            ax.plot([x(n.age)]*2, [n.children[0].y, n.children[-1].y], color='#687984', lw=1.1)
            for c in n.children:
                ax.plot([x(n.age), x(c.age)], [c.y, c.y], color=colors[c.key] if not c.children else '#687984', lw=1.1)
                branches(c)
    branches(tree)
    ax.scatter([x(tree.age)], [tree.y], s=14, color='#24323b', zorder=5)
    ax.annotate(f'{tree.age:.0f} Ma', (x(tree.age), tree.y), xytext=(4, 7), textcoords='offset points',
                fontsize=8, weight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1))
    ax.text(0, len(ts)+1.35, title, fontsize=14, weight='bold', color='#24323b')
    ax.text(0, len(ts)+.5, f'Time axis: {maximum}–0 Ma', fontsize=10, color='#52616b')
    ax.text(11.8, len(ts)+.5, 'Taxa', fontsize=10, ha='right', weight='bold')
    ax.text(12.85, len(ts)+.5, '%', fontsize=10, ha='right', weight='bold')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    provenance = ROOT / 'data/dtol_provenance'
    organisms = pd.read_csv(provenance/'organisms.csv', dtype=str, keep_default_na=False)
    taxonomy = pd.read_csv(provenance/'taxonomy.csv', keep_default_na=False)
    broad, insects, unique = summarize(organisms, taxonomy)
    selections = {k: set(ANIMAL_GROUPS.get(k, EXTRA_SELECTIONS.get(k, [k]))) for k in broad.group}
    insect_selections = {k: {'Psocoptera', 'Phthiraptera'} if k == 'Psocodea' else {k} for k in insects.order}
    trees, records = {}, []
    for key, (filename, taxid, rank) in SOURCES.items():
        path = CACHE / filename
        url = f'https://www.timetree.org/ajax/direct_download?direct-download-format=newick&direct-download-id={taxid}&direct-download-rank={rank}'
        if args.fetch:
            with urllib.request.urlopen(url, timeout=60) as r:
                raw = r.read()
            parse_newick(raw.decode())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        if not path.exists():
            raise SystemExit(f'Missing {path}; run with --fetch')
        tree = parse_newick(path.read_text())
        for n in leaves(tree):
            n.name = normalize(n.name)
        selection = selections if key == 'overview' else insect_selections
        trees[key] = collapse(tree, selection, LABELS)
        records.append({'panel': key, 'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'cache_date_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat(),
                        'root_age_ma': trees[key].age, 'mapping': {k: sorted(v) for k, v in selection.items()}})
    total, ni = len(unique), int(insects.taxa.sum())
    kingdoms = unique.kingdom.value_counts()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(22, 13))
    # Equal panel widths make the 1600-to-400 Ma change a physical 4x zoom.
    left = fig.add_axes([.025, .19, .46, .625])
    right = fig.add_axes([.525, .19, .46, .625])
    panel(left, trees['overview'], broad.set_index('group').taxa.to_dict(),
          {r.group: COLORS[r.kingdom] for r in broad.itertuples()}, 'A   Whole-file coverage', 1600, 400, total, 'Arthropoda')
    panel(right, trees['insects'], insects.set_index('order').taxa.to_dict(),
          {k: BLUE for k in insects.order}, 'B   Insect orders · 4× time-axis zoom', 400, 100, ni)
    fig.text(.025, .955, 'Darwin Tree of Life: taxonomic coverage', fontsize=25, weight='bold', color='#24323b')
    fig.text(.025, .915, f'{total:,} distinct taxa in the supplied dtol.csv snapshot', fontsize=16, color='#52616b')
    statuses = organisms['Current Status'].value_counts()
    status_text = f'All {len(organisms):,} records: {statuses.index[0]}' if len(statuses) == 1 else 'Status counts are recorded in the source manifest'
    fig.text(.025, .88, status_text, fontsize=12, color='#52616b')
    for xpos, kingdom, label in [(.025, 'Metazoa', 'Animals'), (.205, 'Viridiplantae', 'Green plants'), (.4, 'Fungi', 'Fungi')]:
        fig.text(xpos, .843, f'{label}  {kingdoms.get(kingdom, 0):,}', fontsize=12, weight='bold', color=COLORS[kingdom])
    fig.text(.53, .843, f'Insects: {ni:,} taxa · {100*ni/total:.1f}% of this file', fontsize=12, weight='bold', color=BLUE)
    fig.text(.025, .145, 'Panel A percentages use all taxa; panel B percentages use insects only. Insects are also included within Arthropoda in panel A.', fontsize=11, color='#43525c')
    fig.text(.025, .112, 'Counts are distinct resolved NCBI taxa, not specimens, assemblies or sequence volume. Multiple ToL IDs on a row are counted once.', fontsize=11, color='#43525c')
    fig.text(.025, .079, 'Branch lengths are TimeTree point estimates (Ma); uncertainty is not shown. Group tips are placed at the present, with within-group diversification collapsed.', fontsize=10.5, color='#43525c')
    fig.text(.025, .046, 'Scope: the supplied annotation-complete snapshot, not the entire DToL project. Colours indicate major lineages; there is one cohort in this file.', fontsize=10.5, color='#43525c')
    date = max(r['cache_date_utc'] for r in records)
    fig.text(.025, .018, f'Sources: supplied dtol.csv; NCBI Taxonomy name resolution; TimeTree higher-taxon exports (accessed {date}). See the companion source manifest for mappings and hashes.', fontsize=9, color='#65727a')
    for ext in ['pdf', 'svg', 'png']:
        fig.savefig(ROOT/f'figures/dtol_taxonomic_diversity.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    broad.to_csv(ROOT/'plot_data/dtol_taxonomic_groups.csv', index=False)
    insects.to_csv(ROOT/'plot_data/dtol_insect_orders.csv', index=False)
    report = {'input_sha256': hashlib.sha256((provenance/'organisms.csv').read_bytes()).hexdigest(),
              'taxonomy_sha256': hashlib.sha256((provenance/'taxonomy.csv').read_bytes()).hexdigest(),
              'taxonomy_provenance': 'data/dtol_provenance/sources.json',
              'input_records': len(organisms), 'unique_taxa': total, 'insect_taxa': ni,
              'kingdom_counts': kingdoms.to_dict(), 'status_counts': statuses.to_dict(), 'timetree_sources': records,
              'count_unit': 'Distinct resolved NCBI taxid; multiple specimens and synonym labels do not multiply counts.',
              'scope': 'User-supplied snapshot only; no live project records added.'}
    (ROOT/'plot_data/dtol_diversity_sources.json').write_text(json.dumps(report, indent=2)+'\n')
    print(broad.to_string(index=False))
    print(insects.to_string(index=False))


if __name__ == '__main__':
    main()
