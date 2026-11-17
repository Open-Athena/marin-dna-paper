"""OpenGenome2 metadata coverage: dated eukaryotes and microbial taxon counts.

uv run --locked python src/plot_evo2_diversity.py [--fetch]
This summarizes the released metadata, not a census of all training sequences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from plot_dataset_timetree import CACHE, ROOT, collapse, draw, leaves, normalize, parse_newick
from plot_dataset_phylogeny import BLUE, ORANGE

REVISION = '84d2a7e690c8bb395d1f4868822dfa031ea561c6'
META = ROOT/'plot_data/remote/https/huggingface.co/arcinstitute/opengenome2/species_metadata.csv'
META_URL = f'https://huggingface.co/datasets/arcinstitute/opengenome2/resolve/{REVISION}/species_metadata.csv'
TREE = CACHE/'taxon_23103_phylum.nwk'
TREE_URL = 'https://www.timetree.org/ajax/direct_download?' + urllib.parse.urlencode({
    'direct-download-format':'newick','direct-download-id':'23103','direct-download-rank':'phylum'})
ANIMALS = set('Acanthocephala Annelida Arthropoda Brachiopoda Bryozoa Chaetognatha Chordata Cnidaria Ctenophora Dicyemida Echinodermata Entoprocta Gastrotricha Gnathostomulida Hemichordata Kinorhyncha Mollusca Nematoda Nematomorpha Nemertea Onychophora Orthonectida Phoronida Placozoa Platyhelminthes Porifera Priapulida Rotifera Tardigrada Xenacoelomorpha'.split())
FUNGI = set('Ascomycota Basidiomycota Blastocladiomycota Chytridiomycota Cryptomycota Microsporidia Mucoromycota Olpidiomycota Sanchytriomycota Zoopagomycota'.split())
GROUP_PHYLA = {
    'Animals': ANIMALS, 'Fungi': FUNGI,
    'Green plants': {'Streptophyta','Chlorophyta','Prasinodermophyta'},
    'Red algae': {'Rhodophyta'}, 'Amoebozoans': {'Discosea','Evosea','Tubulinea'},
    'Alveolates': {'Apicomplexa','Ciliophora','Perkinsozoa'},
    'Stramenopiles': {'Bacillariophyta','Oomycota'},
    'Rhizarians': {'Cercozoa','Endomyxa','Foraminifera'},
    'Discobans': {'Euglenozoa','Heterolobosea'},
    'Metamonads': {'Fornicata','Parabasalia','Preaxostyla'},
    'Haptophytes': {'Haptophyta'},
}
CLASS_GROUP = {
    'Dinophyceae':'Alveolates','Cryptophyceae':'Cryptophytes','Glaucocystophyceae':'Glaucophytes',
    'Ichthyosporea':'Ichthyosporeans','Filasterea':'Filastereans',
    'Choanoflagellata':'Choanoflagellates','Aphelidea':'Aphelids',
    **{c:'Stramenopiles' for c in ['Phaeophyceae','Bigyra','Eustigmatophyceae','Xanthophyceae',
        'Synurophyceae','Bolidophyceae','Raphidophyceae','Dictyochophyceae','Pelagophyceae',
        'Chrysophyceae','Hyphochytriomycetes','Olisthodiscophyceae','Pinguiophyceae']},
}
SOURCE_GROUPS = {
    'Animals': ANIMALS, 'Fungi': FUNGI,
    'Green plants': {'Streptophyta','Chlorophyta','Prasinodermophyta'},
    'Red algae': {'Rhodophyta'}, 'Amoebozoans': {'Discosea','Evosea','Tubulinea'},
    'Alveolates': {'Ciliophora','Dinophyceae (no phylum)','Perkinsozoa','Apicomplexa','Vitrellaceae (no phylum)'},
    'Stramenopiles': {'Bigyra (no phylum)','Ochrophyta','Chrysomerophyceae (no phylum)','Oomycota'},
    'Rhizarians': {'Cercozoa','Endomyxa','Foraminifera','Polycystinea (no phylum)'},
    'Discobans': {'Euglenozoa','Jakobida (no phylum)','Heterolobosea'},
    'Metamonads': {'Parabasalia','Fornicata','Preaxostyla'},
    'Haptophytes': {'Haptophyta'}, 'Cryptophytes': {'Cryptophyceae (no phylum)'},
    'Glaucophytes': {'Glaucocystophyceae (no phylum)'},
    'Choanoflagellates': {'Choanoflagellata (no phylum)'},
    'Filastereans': {'Filasterea (no phylum)'}, 'Ichthyosporeans': {'Ichthyosporea (no phylum)'},
    'Aphelids': {'Aphelidea (no phylum)'}, 'Rotosphaerids': {'Rotosphaerida (no phylum)'},
}


def classify(row: pd.Series) -> str:
    if row.kingdom in ['Bacteria','Archaea']:
        return row.kingdom
    for group, phyla in GROUP_PHYLA.items():
        if row.phylum in phyla:
            return group
    if row['class'] in CLASS_GROUP:
        return CLASS_GROUP[row['class']]
    if row.order == 'Rotosphaerida':
        return 'Rotosphaerids'
    if row.order == 'Jakobida':
        return 'Discobans'
    if row.family in ['Chromeraceae','Vitrellaceae']:
        return 'Alveolates'
    return 'Unplaced eukaryotes'


def summarize(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    required = {'domain','source_dataset','kingdom','phylum','class','order','family','species'}
    assert required <= set(d.columns)
    assert (d.species != '').all()
    d = d.copy()
    d['group'] = d.apply(classify,axis=1)
    # A species can occur in multiple source components. Fill missing taxonomy
    # only from another row with the *same* species label, never from a relative.
    assigned = d[d.group != 'Unplaced eukaryotes'].groupby('species').group.agg(set)
    assert assigned.map(len).le(1).all(), 'Conflicting taxonomic classifications'
    lookup = assigned.map(lambda x: next(iter(x))).to_dict()
    d['group'] = d.species.map(lookup).fillna(d.group)
    out = []
    for group, rows in d.groupby('group'):
        g = set(rows.loc[rows.domain != 'Organelle','species'])
        o = set(rows.loc[rows.domain == 'Organelle','species'])
        out.append({'group':group,'genome_rna_taxa':len(g),'organelle_host_taxa':len(o),
                    'unique_taxa_union':len(g|o),'overlap':len(g&o)})
    summary = pd.DataFrame(out)
    assert summary.unique_taxa_union.sum() == d.species.nunique()
    microbes = (d[d.kingdom.isin(['Bacteria','Archaea'])]
                .groupby(['kingdom','phylum']).species.nunique().reset_index(name='taxon_labels'))
    stats = {'metadata_records':len(d),'unique_taxon_labels':int(d.species.nunique()),
             'unique_eukaryotic_labels':int(d.loc[d.kingdom=='Eukaryota','species'].nunique()),
             'source_rows':d.groupby('source_dataset').size().to_dict(),
             'source_unique_labels':d.groupby('source_dataset').species.nunique().to_dict()}
    return summary,microbes,stats


def microbial_panel(ax, microbes: pd.DataFrame) -> None:
    ax.set_xlim(0,1)
    ax.set_ylim(0,20)
    ax.axis('off')
    ax.text(0,19.4,'B   Bacterial and archaeal coverage',fontsize=14,weight='bold',color='#24323b')
    ax.text(0,18.7,'GTDB labels in the released metadata · bars are counts, not time',fontsize=10,color='#52616b')
    for kingdom,top,color in [('Bacteria',17.4,BLUE),('Archaea',9.0,'#547e65')]:
        rows=microbes[microbes.kingdom==kingdom].sort_values('taxon_labels',ascending=False)
        total=int(rows.taxon_labels.sum())
        ax.text(0,top,f'{kingdom}: {total:,} taxon labels / {len(rows)} phylum labels',fontsize=12,weight='bold',color=color)
        values=list(rows.head(6)[['phylum','taxon_labels']].itertuples(index=False,name=None))
        values.append((f'Other {len(rows)-6} phylum labels',int(rows.iloc[6:].taxon_labels.sum())))
        maximum=max(v for _,v in values)
        for i,(label,value) in enumerate(values):
            y=top-1-i*.87
            ax.text(0,y,label,fontsize=10,va='center',color='#24323b')
            ax.barh(y,.32*value/maximum,left=.5,height=.32,color=color)
            ax.text(.96,y,f'{value:,}',ha='right',va='center',fontsize=10,color=color,weight='bold')
    ax.text(0,.55,'GTDB phylum labels are retained verbatim, including suffixes.',fontsize=9,color='#65727a')


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch',action='store_true')
    args=parser.parse_args()
    for path,url in [(META,META_URL),(TREE,TREE_URL)]:
        if args.fetch:
            path.parent.mkdir(parents=True,exist_ok=True)
            with urllib.request.urlopen(url,timeout=60) as response:
                raw=response.read()
            if path.suffix=='.nwk':
                parse_newick(raw.decode())
            path.write_bytes(raw)
        if not path.exists():
            raise SystemExit(f'Missing {path}; run with --fetch.')
    d=pd.read_csv(META,dtype=str,keep_default_na=False)
    summary,microbes,stats=summarize(d)
    tree=parse_newick(TREE.read_text())
    for n in leaves(tree):
        n.name=normalize(n.name)
    available={n.name for n in leaves(tree)}
    groups=set(summary.group)-{'Bacteria','Archaea','Unplaced eukaryotes'}
    selection={k:SOURCE_GROUPS[k]&available for k in groups}
    assert all(selection.values())
    reduced=collapse(tree,selection,{k:k for k in groups})
    s=summary.set_index('group')
    values={k:(int(s.loc[k,'genome_rna_taxa']),int(s.loc[k,'organelle_host_taxa'])) for k in groups}
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig=plt.figure(figsize=(20,12))
    ax=fig.add_axes([.025,.205,.62,.61])
    # Reuse the dated topology renderer, but omit its genome-scale bars and
    # short cohort headers: the two columns now count taxon labels by source.
    draw(ax,reduced,{k:(0,0) for k in groups},'A   Eukaryotic coverage',1600,400)
    for patch in list(ax.patches):
        if patch.get_zorder()==3:
            patch.remove()
    for t in list(ax.texts):
        if t.get_text() in ['Ann.','Proj.','–']:
            t.remove()
    ts=leaves(reduced)
    for n in ts:
        g,o=values[n.key]
        ax.text(13.75,n.y,f'{g:,}' if g else '–',ha='right',va='center',fontsize=9.5,color=BLUE,weight='bold')
        ax.text(14.55,n.y,f'{o:,}' if o else '–',ha='right',va='center',fontsize=9.5,color=ORANGE,weight='bold')
    ax.text(13.75,len(ts)+.5,'G/R',ha='right',fontsize=9,color=BLUE,weight='bold')
    ax.text(14.55,len(ts)+.5,'Org.',ha='right',fontsize=9,color=ORANGE,weight='bold')
    right=fig.add_axes([.69,.205,.285,.61])
    microbial_panel(right,microbes)
    fig.text(.025,.96,'Evo 2 / OpenGenome2: taxonomic coverage',fontsize=24,weight='bold',color='#24323b')
    fig.text(.025,.92,'A view of the released species metadata — not a complete census of the training corpus',fontsize=14,color='#52616b')
    fig.text(.025,.875,f'{stats["unique_taxon_labels"]:,} distinct species/taxon labels across {stats["metadata_records"]:,} metadata records',fontsize=13,color='#24323b')
    fig.text(.025,.838,'G/R: genomes, transcripts or ncRNA',fontsize=11,color=BLUE,weight='bold')
    fig.text(.285,.838,'Org.: organelle host taxa (may overlap G/R)',fontsize=11,color=ORANGE,weight='bold')
    missing=s.loc['Unplaced eukaryotes']
    fig.text(.025,.155,f'Unplaced eukaryotes: {int(missing.genome_rna_taxa):,} G/R and {int(missing.organelle_host_taxa):,} organelle-host labels; retained in totals, omitted from the dated tree.',fontsize=10.5,color='#52616b')
    fig.text(.025,.12,'Additional training components include metagenomes, viral sequences and plasmids; these have no corresponding source rows in this metadata file.',fontsize=10.5,color='#43525c')
    fig.text(.025,.088,'Counts are unique labels within source groups, not training weights or genome counts. Organelles are assigned to their hosts; viruses are not branches of this cellular tree.',fontsize=10,color='#43525c')
    fig.text(.025,.057,'Eukaryotic branches use TimeTree point estimates (Ma); within-group diversification is collapsed. Microbial bars have no evolutionary-time scale.',fontsize=10,color='#43525c')
    fig.text(.025,.025,f'Sources: arcinstitute/opengenome2 species_metadata.csv @ {REVISION[:12]}; TimeTree web export, 2026-11-17. Metadata taxonomy: GTDB and NCBI.',fontsize=9,color='#65727a')
    for ext in ['pdf','svg','png']:
        fig.savefig(ROOT/f'figures/evo2_taxonomic_diversity.{ext}',dpi=180,facecolor='white')
    plt.close(fig)
    summary.to_csv(ROOT/'plot_data/evo2_taxonomic_groups.csv',index=False)
    microbes.to_csv(ROOT/'plot_data/evo2_microbial_phyla.csv',index=False)
    report={'metadata_url':META_URL,'metadata_revision':REVISION,
            'metadata_sha256':hashlib.sha256(META.read_bytes()).hexdigest(),
            'timetree_url':TREE_URL,'timetree_sha256':hashlib.sha256(TREE.read_bytes()).hexdigest(),
            'scope':'Released metadata only; not a complete census of all OpenGenome2 training components.',
            'count_unit':'Distinct species string; placeholders retained; no cross-taxonomy synonym resolution.',
            'taxonomy_fill':'Missing classification filled only from another record with exactly the same species string.',
            'eukaryote_root_age_ma':reduced.age,'source_group_mapping':{k:sorted(v) for k,v in selection.items()},**stats}
    (ROOT/'plot_data/evo2_diversity_sources.json').write_text(json.dumps(report,indent=2)+'\n')
    print(summary.to_string(index=False));print(json.dumps(stats,indent=2))

if __name__=='__main__':
    main()
