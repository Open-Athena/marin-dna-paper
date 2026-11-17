"""Draw cohort coverage on a simplified cladogram, not an inferred/time-scaled tree.

Run: uv run --locked python src/plot_dataset_phylogeny.py
Counts come only from the manuscript's saved species manifests. Internal
relationships are a deliberately coarse scaffold; multifurcations collapse
relationships rather than assert simultaneous divergence. The animal root,
xenacoelomorph placement and placental root are left unresolved.

Reference scaffold: NCBI Taxonomy; Asher & Helgen (2010),
https://pmc.ncbi.nlm.nih.gov/articles/PMC2865478/ .
No branch lengths or within-order species relationships are inferred.
"""
from pathlib import Path
from dataclasses import dataclass, field

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = '#2878a5', '#d77b32'

@dataclass
class Node:
    name: str
    children: list['Node'] = field(default_factory=list)
    key: str = ''
    x: float = 0
    y: float = 0


def leaf(key: str, name: str) -> Node:
    return Node(name, key=key)


def group(name: str, *children: Node) -> Node:
    return Node(name, list(children))


def animal_tree() -> Node:
    archosaurs = group('', leaf('Crocodylia', 'Crocodilians'), leaf('Aves', 'Birds'))
    sauropsids = group('', leaf('Lepidosauria', 'Lizards, snakes & tuatara'),
        group('', leaf('Testudines', 'Turtles'), archosaurs))
    tetrapods = group('Tetrapods', leaf('Amphibia', 'Amphibians'),
        group('Amniotes', leaf('Mammalia', 'Mammals → panel B'), sauropsids))
    bony = group('Bony vertebrates', leaf('ray_finned', 'Ray-finned fishes'),
        group('', leaf('Coelacanthiformes', 'Coelacanths'), tetrapods))
    vertebrates = group('Vertebrates', group('', leaf('Myxini', 'Hagfish'),
        leaf('Hyperoartia', 'Lampreys')),
        group('', leaf('Chondrichthyes', 'Sharks, rays & chimaeras'), bony))
    chordates = group('Chordates', leaf('Leptocardii', 'Lancelets'),
        group('', leaf('Ascidiacea', 'Tunicates'), vertebrates))
    deuterostomes = group('Deuterostomes', group('',
        leaf('Echinodermata', 'Sea stars, urchins & relatives'),
        leaf('Hemichordata', 'Acorn worms & relatives')), chordates)
    ecdysozoa = group('Ecdysozoa', leaf('Priapulida', 'Priapulid worms'),
        leaf('Nematoda', 'Roundworms'), group('', leaf('Tardigrada', 'Water bears'),
        leaf('Arthropoda', 'Arthropods (including insects)')))
    spiralia = group('Spiralia', *[leaf(k, n) for k, n in [
        ('Mollusca', 'Molluscs'), ('Annelida', 'Segmented worms'),
        ('Nemertea', 'Ribbon worms'), ('Platyhelminthes', 'Flatworms'),
        ('Brachiopoda', 'Lamp shells'), ('Bryozoa', 'Bryozoans')]])
    bilateria = group('Bilateria', leaf('Xenacoelomorpha', 'Xenacoelomorph worms'),
        group('Protostomes', spiralia, ecdysozoa), deuterostomes)
    return group('Animals', leaf('Porifera', 'Sponges'), leaf('Ctenophora', 'Comb jellies'),
        leaf('Placozoa', 'Placozoans'), leaf('Cnidaria', 'Corals, anemones & jellyfish'), bilateria)


def mammal_tree() -> Node:
    names = {
        'Monotremata': 'Platypus & echidnas', 'Didelphimorphia': 'Opossums',
        'Dasyuromorphia': 'Quolls, devils & relatives', 'Diprotodontia': 'Kangaroos, koalas & relatives',
        'Microbiotheria': 'Monito del monte', 'Peramelemorphia': 'Bandicoots & bilbies',
        'Afrosoricida': 'Tenrecs & golden moles', 'Macroscelidea': 'Elephant shrews',
        'Tubulidentata': 'Aardvarks', 'Hyracoidea': 'Hyraxes',
        'Proboscidea': 'Elephants', 'Sirenia': 'Manatees & dugongs',
        'Cingulata': 'Armadillos', 'Pilosa': 'Sloths & anteaters',
        'Rodentia': 'Rodents', 'Lagomorpha': 'Rabbits, hares & pikas',
        'Primates': 'Primates', 'Dermoptera': 'Colugos', 'Scandentia': 'Tree shrews',
        'Eulipotyphla': 'Shrews, moles & hedgehogs', 'Chiroptera': 'Bats',
        'Carnivora': 'Carnivorans', 'Pholidota': 'Pangolins',
        'Perissodactyla': 'Horses, rhinos & tapirs', 'Artiodactyla': 'Even-toed ungulates & whales'}
    def order(k: str) -> Node:
        return leaf(k, names[k])
    return group('Mammals', order('Monotremata'),
        group('Theria', group('Marsupials', *[order(k) for k in
            ['Didelphimorphia', 'Dasyuromorphia', 'Diprotodontia', 'Microbiotheria', 'Peramelemorphia']]),
            group('Placentals',
                group('Afrotheria', *[order(k) for k in ['Afrosoricida', 'Macroscelidea',
                    'Tubulidentata', 'Hyracoidea', 'Proboscidea', 'Sirenia']]),
                group('Xenarthra', order('Cingulata'), order('Pilosa')),
                group('Boreoeutheria',
                    group('Euarchontoglires', group('Glires', order('Rodentia'), order('Lagomorpha')),
                        order('Scandentia'), group('', order('Dermoptera'), order('Primates'))),
                    group('Laurasiatheria', *[order(k) for k in ['Eulipotyphla', 'Chiroptera',
                        'Carnivora', 'Pholidota', 'Perissodactyla', 'Artiodactyla']])))))


def tips(n: Node) -> list[Node]:
    return [n] if not n.children else [t for c in n.children for t in tips(c)]


def layout(n: Node, max_depth: int) -> None:
    leaves = tips(n)
    for i, t in enumerate(leaves):
        t.y = len(leaves) - 1 - i
    def place(node: Node, depth: int) -> None:
        node.x = depth / max_depth * 4.7
        if node.children:
            for c in node.children:
                place(c, depth + 1)
            node.y = (node.children[0].y + node.children[-1].y) / 2
        else:
            node.x = 5.0
    place(n, 0)


def depth(n: Node) -> int:
    return 0 if not n.children else 1 + max(depth(c) for c in n.children)


def draw(ax, tree: Node, counts: dict[str, tuple[int, int]], title: str) -> None:
    layout(tree, depth(tree))
    leaves = tips(tree)
    ax.set_xlim(-.2, 13.5)
    ax.set_ylim(-.8, len(leaves) + 1.8)
    ax.axis('off')
    for i, tip in enumerate(leaves):
        a, b = counts[tip.key]
        if i % 2 == 0:
            ax.axhspan(tip.y-.48, tip.y+.48, color='#f3f5f5', zorder=0)
        if tip.key == 'Mammalia':
            ax.axhspan(tip.y-.48, tip.y+.48, color='#fff0e2', zorder=0)
        ax.text(5.16, tip.y, tip.name, fontsize=9, va='center', color='#24323b',
                weight='bold' if tip.key == 'Mammalia' else 'normal')
        # Fixed-width bars use the same count scale in both panels.
        ax.barh(tip.y+.14, a/125*1.8, left=10.0, height=.20, color=BLUE, zorder=3)
        ax.barh(tip.y-.14, b/125*1.8, left=10.0, height=.20, color=ORANGE, zorder=3)
        ax.text(12.25, tip.y, str(a) if a else '–', color=BLUE, ha='right', va='center', fontsize=10, weight='bold')
        ax.text(13.15, tip.y, str(b) if b else '–', color=ORANGE, ha='right', va='center', fontsize=10, weight='bold')
    # Label only selected clades to keep the branch scaffold legible.
    label_nodes = {'Bilateria', 'Protostomes', 'Spiralia', 'Ecdysozoa', 'Chordates',
        'Tetrapods', 'Marsupials', 'Placentals', 'Afrotheria', 'Xenarthra',
        'Euarchontoglires', 'Laurasiatheria'}
    def branches(n: Node) -> None:
        if not n.children:
            return
        ax.plot([n.x,n.x], [n.children[0].y,n.children[-1].y], color='#89949b', lw=1.05)
        for c in n.children:
            ax.plot([n.x,c.x], [c.y,c.y], color='#89949b', lw=1.05)
            branches(c)
        if n.name in label_nodes:
            ax.text(n.x+.05,n.y+.25,n.name,fontsize=7.2,color='#4a565e',
                    bbox=dict(facecolor='white',edgecolor='none',pad=.6), zorder=4)
    branches(tree)
    ax.text(0,len(leaves)+1.25,title,fontsize=13,weight='bold',color='#24323b')
    ax.text(12.25,len(leaves)+.2,'Ann.',ha='right',fontsize=9,color=BLUE,weight='bold')
    ax.text(13.15,len(leaves)+.2,'Proj.',ha='right',fontsize=9,color=ORANGE,weight='bold')


def main() -> None:
    p = ROOT/'data/dataset_provenance'
    a = pd.read_csv(p/'annotation_genomes.tsv',sep='\t')
    b = pd.read_csv(p/'projection_species.tsv',sep='\t')
    def animal_key(row) -> str:
        if row.phylum != 'Chordata':
            return row.phylum
        if row['class'] in ['Actinopteri','Cladistia']:
            return 'ray_finned'
        return row['class'] if pd.notna(row['class']) else row.order
    a['plot_group'] = a.apply(animal_key,axis=1)
    tree, mammals = animal_tree(), mammal_tree()
    ac = a.groupby('plot_group').size().to_dict()
    assert set(ac)=={t.key for t in tips(tree)}, 'Unrepresented animal groups'
    animal_counts = {k:(v,len(b) if k=='Mammalia' else 0) for k,v in ac.items()}
    ma = a[a['class']=='Mammalia'].groupby('order').size().to_dict()
    mb = b.groupby('order').size().to_dict()
    assert set(ma)|set(mb)=={t.key for t in tips(mammals)}, 'Unrepresented mammalian orders'
    mammal_counts = {k:(ma.get(k,0),mb.get(k,0)) for k in set(ma)|set(mb)}
    assert sum(x[0] for x in animal_counts.values())==499
    assert sum(x[1] for x in animal_counts.values())==108
    assert sum(x[0] for x in mammal_counts.values())==81
    assert sum(x[1] for x in mammal_counts.values())==108
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(19,13.5))
    fig.subplots_adjust(left=.025,right=.985,top=.84,bottom=.115,wspace=.06)
    draw(ax1,tree,animal_counts,'A   Breadth across animals')
    draw(ax2,mammals,mammal_counts,'B   Mammals, expanded by order')
    fig.text(.025,.965,'Two cohorts at very different evolutionary scales',fontsize=23,weight='bold',color='#24323b')
    fig.text(.025,.925,'Annotation-derived: 499 genomes across 18 animal phyla',color=BLUE,fontsize=14,weight='bold')
    fig.text(.525,.925,'Alignment-projected: 108 placental mammals',color=ORANGE,fontsize=14,weight='bold')
    fig.text(.025,.891,'CDS · upstream · downstream',fontsize=11,color='#52616b')
    fig.text(.525,.891,'ncRNA · enhancer regions',fontsize=11,color='#52616b')
    fig.text(.025,.069,'Counts are selected genomes, not sequence counts or sampling weights. Bars share a linear scale across panels.',fontsize=11,color='#43525c')
    fig.text(.025,.044,'Simplified cladogram: branch lengths are arbitrary; multifurcations collapse relationships. Only represented groups are shown.',fontsize=10,color='#65727a')
    fig.text(.025,.023,'Sources: saved cohort manifests; NCBI Taxonomy; Asher & Helgen (2010). Mammal labels follow the manifests (Artiodactyla includes whales).',fontsize=9,color='#65727a')
    out=ROOT/'figures/dataset_phylogeny'
    for ext in ['pdf','svg','png']:
        fig.savefig(out.with_suffix('.'+ext),dpi=180,facecolor='white')
    rows=[]
    for panel,t,counts in [('animals',tree,animal_counts),('mammals',mammals,mammal_counts)]:
        for n in tips(t):
            x,y=counts[n.key]
            rows.append({'panel':panel,'taxon':n.key,'label':n.name,'annotation_genomes':x,'projection_genomes':y})
    pd.DataFrame(rows).to_csv(ROOT/'plot_data/dataset_phylogeny_counts.csv',index=False)
    print('Wrote PDF, SVG, PNG and count table. All cohort totals verified.')

if __name__=='__main__':
    main()
