"""Plot the cohort counts on dated higher-taxon trees exported by TimeTree.

uv run --locked python src/plot_dataset_timetree.py [--fetch]

Raw exports are cached under plot_data/remote (gitignored). The default is
fully offline; --fetch refreshes the public web exports. This is a group-level
view, not a species tree inferred from the training data. Terminal lineages
include diversification hidden inside each collapsed taxonomic group.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = '#2878a5', '#d77b32'

# Display names only; topology and dates come from the TimeTree exports.
ANIMAL_LABELS = {'Porifera': 'Sponges',
 'Ctenophora': 'Comb jellies',
 'Placozoa': 'Placozoans',
 'Cnidaria': 'Corals, anemones & jellyfish',
 'Xenacoelomorpha': 'Xenacoelomorph worms',
 'Mollusca': 'Molluscs',
 'Annelida': 'Segmented worms',
 'Nemertea': 'Ribbon worms',
 'Platyhelminthes': 'Flatworms',
 'Brachiopoda': 'Lamp shells',
 'Bryozoa': 'Bryozoans',
 'Priapulida': 'Priapulid worms',
 'Nematoda': 'Roundworms',
 'Tardigrada': 'Water bears',
 'Arthropoda': 'Arthropods (including insects)',
 'Echinodermata': 'Sea stars, urchins & relatives',
 'Hemichordata': 'Acorn worms & relatives',
 'Leptocardii': 'Lancelets',
 'Ascidiacea': 'Tunicates',
 'Myxini': 'Hagfish',
 'Hyperoartia': 'Lampreys',
 'Chondrichthyes': 'Sharks, rays & chimaeras',
 'ray_finned': 'Ray-finned fishes',
 'Coelacanthiformes': 'Coelacanths',
 'Amphibia': 'Amphibians',
 'Mammalia': 'Mammals → panel B',
 'Lepidosauria': 'Lizards, snakes & tuatara',
 'Testudines': 'Turtles',
 'Crocodylia': 'Crocodilians',
 'Aves': 'Birds'}

MAMMAL_LABELS = {'Monotremata': 'Platypus & echidnas',
 'Didelphimorphia': 'Opossums',
 'Dasyuromorphia': 'Quolls, devils & relatives',
 'Diprotodontia': 'Kangaroos, koalas & relatives',
 'Microbiotheria': 'Monito del monte',
 'Peramelemorphia': 'Bandicoots & bilbies',
 'Afrosoricida': 'Tenrecs & golden moles',
 'Macroscelidea': 'Elephant shrews',
 'Tubulidentata': 'Aardvarks',
 'Hyracoidea': 'Hyraxes',
 'Proboscidea': 'Elephants',
 'Sirenia': 'Manatees & dugongs',
 'Cingulata': 'Armadillos',
 'Pilosa': 'Sloths & anteaters',
 'Rodentia': 'Rodents',
 'Lagomorpha': 'Rabbits, hares & pikas',
 'Scandentia': 'Tree shrews',
 'Dermoptera': 'Colugos',
 'Primates': 'Primates',
 'Eulipotyphla': 'Shrews, moles & hedgehogs',
 'Chiroptera': 'Bats',
 'Carnivora': 'Carnivorans',
 'Pholidota': 'Pangolins',
 'Perissodactyla': 'Horses, rhinos & tapirs',
 'Artiodactyla': 'Even-toed ungulates & whales'}

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'plot_data/remote/https/www.timetree.org/cohort_figure'

@dataclass(eq=False)
class DatedNode:
    name: str = ''
    children: list[DatedNode] = field(default_factory=list)
    length: float = 0.0
    age: float = 0.0
    key: str = ''
    y: float = 0.0


def leaves(node: DatedNode) -> list[DatedNode]:
    return [node] if not node.children else [t for c in node.children for t in leaves(c)]


def parse_newick(text: str) -> DatedNode:
    """Parse quoted/unquoted labels and lengths; validate an ultrametric tree."""
    tokens = re.findall(r"'(?:[^']|'')*'|\[[^\]]*\]|[^\s(),:;]+|[(),:;]", text)
    tokens = [t for t in tokens if not t.startswith('[')]
    i = 0
    def read() -> DatedNode:
        nonlocal i
        n = DatedNode()
        if tokens[i] == '(':
            i += 1
            while True:
                n.children.append(read())
                if tokens[i] == ')':
                    i += 1
                    break
                assert tokens[i] == ','
                i += 1
        if tokens[i] not in ',():;':
            n.name = tokens[i].strip("'").replace("''", "'")
            i += 1
        if tokens[i] == ':':
            i += 1
            n.length = float(tokens[i])
            assert n.length >= -1e-7, 'Negative source branch length'
            i += 1
        if n.children:
            ages = [c.age + c.length for c in n.children]
            assert max(ages) - min(ages) < .001, ('Non-ultrametric source', ages)
            n.age = max(ages)  # tolerate export rounding without negative zero-length edges
        return n
    tree = read()
    assert tokens[i:] == [';']
    names = [n.name for n in leaves(tree)]
    assert len(names) == len(set(names)), 'Duplicate source tip labels'
    return tree


def mrca(tree: DatedNode, names: set[str]) -> DatedNode:
    assert names, 'Empty clade selection'
    def find(n: DatedNode) -> tuple[int, DatedNode | None]:
        if not n.children:
            return int(n.name in names), n if len(names) == 1 and n.name in names else None
        hits = 0
        for c in n.children:
            count, result = find(c)
            if result is not None:
                return len(names), result
            hits += count
        return hits, n if hits == len(names) else None
    count, result = find(tree)
    assert count == len(names) and result is not None, ('Unmatched source taxa', names)
    return result


def collapse(tree: DatedNode, selections: dict[str, set[str]], labels: dict[str, str]) -> DatedNode:
    chosen = {mrca(tree, names): key for key, names in selections.items()}
    assert len(chosen) == len(selections), 'Overlapping clade roots'
    # Clades must not nest; a collapsed group must never swallow another group.
    def keys_below(n: DatedNode) -> set[str]:
        keys = {chosen[n]} if n in chosen else set()
        for c in n.children:
            keys |= keys_below(c)
        if n in chosen:
            assert len(keys) == 1, ('Non-disjoint clade collapse', keys)
        return keys
    keys_below(tree)
    def prune(n: DatedNode) -> DatedNode | None:
        if n in chosen:
            k = chosen[n]
            return DatedNode(name=labels[k], key=k)  # group represented at present
        cs = [p for c in n.children if (p := prune(c)) is not None]
        if not cs:
            return None
        if len(cs) == 1:
            return cs[0]  # retain absolute ages, not the old edge lengths
        return DatedNode(name=n.name, children=cs, age=n.age)
    result = prune(tree)
    assert result is not None
    assert {n.key for n in leaves(result)} == set(selections)
    return result


# These are source-tree taxon labels, not substitute species. All displayed
# counts still come from the complete cohort manifests.
ANIMAL_GROUPS = {
    'Porifera': ['Calcarea', 'Homoscleromorpha', 'Demospongiae', 'Hexactinellida'],
    'Ctenophora': ['Tentaculata'], 'Placozoa': ['Uniplacotomia'],
    'Cnidaria': ['Anthozoa', 'Scyphozoa', 'Cubozoa', 'Hydrozoa', 'Myxozoa'],
    'Xenacoelomorpha': ['Xenoturbellidae (no class)', 'Acoela (no class)'],
    'Mollusca': ['Cephalopoda', 'Monoplacophora', 'Scaphopoda', 'Gastropoda',
                'Bivalvia', 'Caudofoveata', 'Solenogastres', 'Polyplacophora'],
    'Annelida': ['Hrabeiella (no class)', 'Sipuncula', 'Polychaeta', 'Clitellata'],
    'Nemertea': ['Palaeonemertea', 'Enopla', 'Pilidiophora'],
    'Platyhelminthes': ['Monogenea', 'Trematoda', 'Cestoda', 'Rhabdocoela', 'Rhabditophora'],
    'Brachiopoda': ['Rhynchonellata', 'Lingulata', 'Craniata'],
    'Bryozoa': ['Phylactolaemata', 'Gymnolaemata', 'Stenolaemata'],
    'Priapulida': ['Priapulimorpha'], 'Nematoda': ['Enoplea', 'Chromadorea'],
    'Tardigrada': ['Eutardigrada', 'Heterotardigrada'],
    'Arthropoda': ['Pycnogonida', 'Arachnida', 'Merostomata', 'Chilopoda', 'Diplopoda',
                  'Symphyla', 'Pauropoda', 'Ostracoda', 'Mystacocarida', 'Ichthyostraca',
                  'Branchiopoda', 'Remipedia', 'Cephalocarida', 'Hexanauplia',
                  'Malacostraca', 'Thecostraca', 'Diplura (no class)', 'Collembola', 'Protura', 'Insecta'],
    'Echinodermata': ['Asteroidea', 'Ophiuroidea', 'Echinoidea', 'Holothuroidea', 'Crinoidea'],
    'Hemichordata': ['Enteropneusta'], 'Leptocardii': ['Leptocardii'],
    'Ascidiacea': ['Ascidiacea'], 'Myxini': ['Myxini'], 'Hyperoartia': ['Hyperoartia'],
    'Chondrichthyes': ['Chondrichthyes'], 'ray_finned': ['Actinopteri', 'Cladistia'],
    'Coelacanthiformes': ['Coelacanthiformes (no class)'], 'Amphibia': ['Amphibia'],
    'Mammalia': ['Mammalia'], 'Lepidosauria': ['Lepidosauria'],
    'Testudines': ['Testudinata (no class)'], 'Crocodylia': ['Crocodylia (no class)'], 'Aves': ['Aves'],
}


def normalize(name: str) -> str:
    return name.replace('_', ' ').strip()


def counts() -> tuple[dict, dict]:
    p = ROOT / 'data/dataset_provenance'
    a = pd.read_csv(p / 'annotation_genomes.tsv', sep='\t')
    b = pd.read_csv(p / 'projection_species.tsv', sep='\t')
    def key(r) -> str:
        if r.phylum != 'Chordata':
            return r.phylum
        if r['class'] in ['Actinopteri', 'Cladistia']:
            return 'ray_finned'
        return r['class'] if pd.notna(r['class']) else r.order
    animal = a.apply(key, axis=1).value_counts().to_dict()
    ac = {k: (v, len(b) if k == 'Mammalia' else 0) for k, v in animal.items()}
    ma = a[a['class'] == 'Mammalia'].groupby('order').size().to_dict()
    mb = b.groupby('order').size().to_dict()
    mc = {k: (ma.get(k, 0), mb.get(k, 0)) for k in ma.keys() | mb.keys()}
    assert tuple(map(sum, zip(*ac.values()))) == (499, 108)
    assert tuple(map(sum, zip(*mc.values()))) == (81, 108)
    return ac, mc


def draw(ax, tree: DatedNode, data: dict, title: str, maximum: int, step: int) -> None:
    ts = leaves(tree)
    for i, n in enumerate(ts):
        n.y = len(ts) - 1 - i
    def locate(n: DatedNode) -> None:
        if n.children:
            for c in n.children:
                locate(c)
                assert n.age >= c.age, 'Non-monotonic displayed ages'
            n.y = (n.children[0].y + n.children[-1].y) / 2
    locate(tree)
    # Reserve nearly half the width for the dated tree; keep count columns fixed.
    x = lambda age: 6.2 * (1 - age / maximum)
    ax.set_xlim(-.1, 14.7)
    ax.set_ylim(-2.4, len(ts) + 2)
    ax.axis('off')
    for tick in range(0, maximum + 1, step):
        ax.plot([x(tick), x(tick)], [-.45, len(ts)-.55], color='#dde3e7', lw=.65, zorder=0)
        ax.text(x(tick), -.95, str(tick), ha='center', va='top', fontsize=8.5, color='#4e5963')
    ax.text(3.1, -1.85, 'Millions of years ago', ha='center', fontsize=10, color='#24323b')
    for i, n in enumerate(ts):
        a, b = data[n.key]
        if i % 2 == 0:
            ax.axhspan(n.y-.47, n.y+.47, color='#f2f5f6', zorder=-2)
        if n.key == 'Mammalia':
            ax.axhspan(n.y-.47, n.y+.47, color='#fff0e2', zorder=-1)
        ax.text(6.45, n.y, n.name, fontsize=8.7, va='center', color='#24323b',
                weight='bold' if n.key == 'Mammalia' else 'normal')
        ax.barh(n.y+.13, a/125*1.3, left=11.7, height=.19, color=BLUE, zorder=3)
        ax.barh(n.y-.13, b/125*1.3, left=11.7, height=.19, color=ORANGE, zorder=3)
        ax.text(13.75, n.y, str(a) if a else '–', color=BLUE, ha='right', va='center', fontsize=9.5, weight='bold')
        ax.text(14.55, n.y, str(b) if b else '–', color=ORANGE, ha='right', va='center', fontsize=9.5, weight='bold')
    def branch(n: DatedNode) -> None:
        if not n.children:
            return
        ax.plot([x(n.age)]*2, [n.children[0].y,n.children[-1].y], color='#687984', lw=1.05)
        for c in n.children:
            ax.plot([x(n.age),x(c.age)], [c.y,c.y], color='#687984', lw=1.05)
            branch(c)
    branch(tree)
    ax.scatter([x(tree.age)], [tree.y], s=14, color='#24323b', zorder=5)
    ax.annotate(f'{tree.age:.0f} Ma', (x(tree.age),tree.y), xytext=(4,7),
                textcoords='offset points', fontsize=8, weight='bold', color='#24323b',
                bbox=dict(facecolor='white',edgecolor='none',pad=1))
    ax.text(0,len(ts)+1.35,title,fontsize=13,weight='bold',color='#24323b')
    ax.text(0,len(ts)+.55,f'Time axis: {maximum}–0 Ma',fontsize=10,color='#53636e')
    ax.text(13.75,len(ts)+.4,'Ann.',ha='right',fontsize=9,color=BLUE,weight='bold')
    ax.text(14.55,len(ts)+.4,'Proj.',ha='right',fontsize=9,color=ORANGE,weight='bold')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fetch', action='store_true', help='Refresh TimeTree exports explicitly')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    metadata = []
    source_trees = {}
    for rank in ['class','order']:
        params = {'direct-download-format':'newick','direct-download-id':'211973',
                  'direct-download-rank':rank}
        url = 'https://www.timetree.org/ajax/direct_download?' + urllib.parse.urlencode(params)
        path = CACHE / f'metazoa_{rank}.nwk'
        if args.fetch:
            with urllib.request.urlopen(url,timeout=60) as response:
                content = response.read()
            parse_newick(content.decode())  # reject an error page before replacing cache
            path.write_bytes(content)
        if not path.exists():
            raise SystemExit(f'Missing {path}; run with --fetch to retrieve the source.')
        content = path.read_bytes()
        tree = parse_newick(content.decode())
        for n in leaves(tree):
            n.name = normalize(n.name)
        assert tree.age > 0, 'Source does not contain positive divergence times'
        source_trees[rank] = tree
        metadata.append({'rank':rank,'url':url,'sha256':hashlib.sha256(content).hexdigest(),
                         'tips':len(leaves(tree)), 'root_age_ma':tree.age,
                         'cache_date_utc':datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat()})
    accessed = max(item['cache_date_utc'] for item in metadata)
    alabels = ANIMAL_LABELS
    mlabels = MAMMAL_LABELS
    available = {n.name for n in leaves(source_trees['class'])}
    selection = {k:set(v)&available for k,v in ANIMAL_GROUPS.items()}
    assert all(selection.values()), 'Source lacks an entire displayed group'
    broad = collapse(source_trees['class'], selection, alabels)
    mselection = {k: {k} for k in mlabels}
    mselection['Afrosoricida'] = {'Tenrecidae (no order)', 'Chrysochloridae (no order)'}
    mammal = collapse(source_trees['order'], mselection, mlabels)
    ac, mc = counts()
    assert {n.key for n in leaves(broad)} == set(ac)
    assert {n.key for n in leaves(mammal)} == set(mc)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig, axes = plt.subplots(1,2,figsize=(20,11.5))
    fig.subplots_adjust(left=.025,right=.985,top=.88,bottom=.04,wspace=.045)
    draw(axes[0],broad,ac,'A   Across animals',800,200)
    draw(axes[1],mammal,mc,'B   Mammals — 4× time-axis zoom',200,50)
    fig.text(.025,.965,'499 annotation-derived genomes · 18 animal phyla',color=BLUE,fontsize=14,weight='bold')
    fig.text(.525,.965,'108 alignment-projected genomes · placental mammals',color=ORANGE,fontsize=14,weight='bold')
    fig.text(.025,.925,'CDS · upstream · downstream',fontsize=11,color='#52616b')
    fig.text(.525,.925,'ncRNA · enhancer regions',fontsize=11,color='#52616b')
    out = ROOT/'figures/dataset_timetree'
    for ext in ['pdf','svg','png']:
        fig.savefig(out.with_suffix('.'+ext),dpi=180,facecolor='white')
    plt.close(fig)
    report = {'source':'TimeTree live higher-taxon web export','retrieved':accessed,
              'citation':'https://doi.org/10.1093/molbev/msac174', 'sources':metadata,
              'interpretation':'TimeTree reference topology and dates; group-level tips at present. Terminal branches collapse within-group diversification. No species substitutions used.',
              'source_cache':'plot_data/remote/https/www.timetree.org/cohort_figure (not redistributed)',
              'panels':{'animals':{'root_age_ma':broad.age,'axis_max_ma':800,'annotation':499,'projection':108},
                        'mammals':{'root_age_ma':mammal.age,'axis_max_ma':200,'annotation':81,'projection':108}},
              'class_group_mapping':{k:sorted(v) for k,v in selection.items()},
              'order_group_mapping':{k:sorted(v) for k,v in mselection.items()}}
    (ROOT/'plot_data/dataset_timetree_sources.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['panels'],indent=2))

if __name__ == '__main__':
    main()
