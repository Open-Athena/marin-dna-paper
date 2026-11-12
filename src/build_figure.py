"""Build one manifest recipe atomically, using local inputs by default."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'plot_data/figures.json').read_text())['figures']

def build(name: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    from local_data import local_path
    entry = MANIFEST[name]
    for source in entry['inputs']:
        local_path(ROOT / 'plot_data' / source)
    with tempfile.TemporaryDirectory(prefix='paper-figure-') as temporary:
        out = Path(temporary)
        if 'artwork' in entry:
            import cairosvg
            source = ROOT / entry['artwork']
            stem = next(iter(entry['outputs']))
            shutil.copyfile(source, out / (stem + '.svg'))
            cairosvg.svg2pdf(url=str(source), write_to=str(out / (stem + '.pdf')))
        else:
            from figures import data
            data.FIGURES_DIR = out
            module = importlib.import_module(entry['module'])
            for attribute in ['FIGURES_DIR', 'OUTPUT_DIR', 'OUT_DIR']:
                if hasattr(module, attribute): setattr(module, attribute, out)
            if hasattr(module, 'KAPLAN_FIT_REPORT_PATH'):
                module.KAPLAN_FIT_REPORT_PATH = out / 'sequence_loss_scaling_fit.txt'
            if entry['module'].startswith('live.'):
                from live import _leaderboard
                _leaderboard.OUTPUT_DIR = out
            call = entry['call']
            if call == 'transfer':
                from utils.figure_style import palette
                df = data.load_transfer(); params = sorted({int(p) for p in df['params'].dropna().unique()})
                module.build(df, palette(params), params)
            elif call == 'scaling':
                from utils.figure_style import palette
                df = data.load_scaling_results(); params = sorted({int(p) for p in df['params'].dropna().unique()})
                module.build(data.load_scaling_history(), df, palette(params))
            elif call == 'mixture': module.build(data.load_mixture())
            elif call == 'history': module.build(data.load_mixture(), data.load_mixture_history())
            elif call == 'leaderboard': module.build(data.load_leaderboard())
            elif call == 'cost': module.build(data.load_leaderboard(), data.load_inference_costs())
            elif call in {'llr', 'probe'}: module.build(call, data.load_mixture())
            elif call == 'overview':
                module.ROOT = out
                (out / 'figures').mkdir()
                module.training_sequences(); module.next_base_prediction()
                for path in (out / 'figures').glob('*'): shutil.move(path, out / path.name)
            else: getattr(module, call)()
        # Check all promised outputs before replacing any existing publication assets.
        copies = []
        for old, new in entry['outputs'].items():
            for extension in entry.get('extensions', ['pdf', 'svg']):
                source = out / f'{old}.{extension}'
                if not source.is_file(): raise RuntimeError(f'{name} did not produce {source.name}')
                copies.append((source, ROOT / 'figures' / f'{new}.{extension}'))
        for source, dest in copies:
            dest.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, dest)
        for source in out.glob('*_fit.txt'):
            if source.is_file(): shutil.copyfile(source, ROOT / 'figures' / source.name)
        print(f'Built {name}')

def main() -> int:
    from local_data import MissingData
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', choices=[*MANIFEST, 'all', 'status'])
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    if args.name == 'status':
        for name, entry in MANIFEST.items():
            state = 'external snapshots required (see recipe)' if entry['remote'] else 'local inputs'
            if 'artwork' in entry: state = 'authored SVG conversion'
            print(f'{name}: {state}; recipe={entry.get("module", entry.get("artwork"))}')
        return 0
    names = MANIFEST if args.name == 'all' else [args.name]
    failed = False
    for name in names:
        try: build(name)
        except MissingData as exc:
            print(f'BLOCKED {name}: {exc}', file=sys.stderr)
            failed = True
            if args.name != 'all': return 2
    return 2 if failed and args.strict else 0

if __name__ == '__main__':
    raise SystemExit(main())
