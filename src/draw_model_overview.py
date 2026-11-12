# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["matplotlib==3.10.8"]
# ///
"""Generate Figure 1A/B: uv run src/draw_model_overview.py.

Recipe sources: paper.tex, Architecture and training / training datasets;
marin-dna docs/research/experiments/485-m5-1-inference-context.md.
The diagrams show the later 255-base recipe, not mixture schedules.
SVG text remains editable; PDF fonts are embedded.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
for font in Path('/usr/share/fonts/truetype/lato').glob('*.ttf'):
    fontManager.addfont(font)
plt.rcParams.update({
    'font.family': 'Lato', 'font.size': 14, 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'text.color': '#252522', 'svg.hashsalt': 'marin-overview',
})
INK = '#465e70'
GRAY = '#817866'
COLORS = ['#465e70', '#a54e2a', '#2b766b', '#8a7090', '#a17d2f']


def canvas(height):
    fig, ax = plt.subplots(figsize=(7, height / 100))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 700), ylim=(height, 0))
    ax.axis('off')
    return fig, ax


def text(ax, x, y, value, size=14, color='#252522', align='left', bold=False):
    return ax.text(x, y, value, fontsize=size, color=color, ha=align,
                   va='center', weight='bold' if bold else 'normal')


def arrow(ax, start, end, color=GRAY):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=10, lw=1, color=color))


def box(ax, x, y, w, h, edge=INK, fill='white'):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle='round,pad=0,rounding_size=5',
        edgecolor=edge, facecolor=fill, linewidth=0.8))


def save(fig, stem):
    for extension in ('pdf', 'svg'):
        metadata = {'Creator': 'draw_model_overview.py'}
        if extension == 'pdf':
            metadata.update(CreationDate=None, ModDate=None)
        else:
            metadata.update(Date=None)
        fig.savefig(ROOT / 'figures' / f'{stem}.{extension}', metadata=metadata)
    plt.close(fig)


def training_sequences():
    fig, ax = canvas(245)
    text(ax, 15, 22, 'Multiple species', bold=True)
    text(ax, 245, 22, 'Select regions', bold=True)
    text(ax, 510, 22, 'Sample sequences', bold=True)
    for i, seq in enumerate(('ACGTAC…', 'TGCACT…', 'GACTGA…')):
        y = 67 + i * 50
        text(ax, 15, y, f'Species {i + 1}', size=12, color=GRAY)
        text(ax, 15, y + 20, seq, size=17, color=INK)
    for i, name in enumerate(('CDS', 'Upstream', 'Downstream', 'ncRNA',
                              'Enhancer regions')):
        y = 54 + i * 31
        box(ax, 245, y, 178, 26, edge=COLORS[i])
        text(ax, 256, y + 13, name, color=COLORS[i])
    arrow(ax, (170, 128), (231, 128))
    arrow(ax, (435, 128), (493, 128))
    for i, (seq, color) in enumerate(zip(('ACGTAC…', 'GACTGA…', 'TGCACT…'),
                                        (COLORS[0], COLORS[3], COLORS[4]))):
        y = 63 + i * 49
        box(ax, 510, y, 164, 33, edge=color)
        text(ax, 592, y + 16.5, seq, size=17, color=color, align='center')
    text(ax, 592, 225, '255 bases per example', size=12, align='center')
    save(fig, 'multispecies_training_sequences')


def next_base_prediction():
    fig, ax = canvas(285)
    text(ax, 350, 20, 'Predict each base from preceding bases',
         size=15, align='center', bold=True)
    xs = [105, 160, 215, 270, 325]
    text(ax, 12, 70, 'Target', size=12, color=GRAY)
    text(ax, 12, 204, 'Input', size=12, color=GRAY)
    for x, inp, target in zip(xs, ('BOS', 'A', 'C', 'G', '…'),
                             ('A', 'C', 'G', 'T', '…')):
        text(ax, x, 70, target, size=17, align='center', color='#a54e2a')
        text(ax, x, 204, inp, size=15, align='center', color=INK)
        arrow(ax, (x, 187), (x, 160), color=INK)
        arrow(ax, (x, 115), (x, 88), color='#a54e2a')
    box(ax, 68, 115, 297, 45, fill='#f3f6f8')
    text(ax, 216.5, 137.5, 'Qwen3 causal Transformer',
         size=14, bold=True, align='center', color=INK)
    ax.plot([365, 410, 410], [137.5, 137.5, 100], color=GRAY, lw=1)
    ax.plot([410, 410], [137.5, 177], color=GRAY, lw=1)
    arrow(ax, (410, 100), (450, 100))
    arrow(ax, (410, 177), (450, 177))
    text(ax, 463, 100, 'Sequence\nprobabilities', color='#a54e2a')
    text(ax, 463, 177, 'Learned\nrepresentations', color='#2b766b')
    text(ax, 216, 232, '255 DNA bases + BOS = 256 tokens',
         size=12, color=GRAY, align='center')
    text(ax, 350, 265, 'DNA training, without variant-effect labels',
         size=13, align='center')
    save(fig, 'next_base_prediction')


if __name__ == '__main__':
    training_sequences()
    next_base_prediction()
