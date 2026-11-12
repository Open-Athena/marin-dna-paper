"""Emit Make rules from the tracked figure manifest (standard library only)."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
entries = json.loads((root / 'plot_data/figures.json').read_text())['figures']
local = []
all_outputs = []
print('FIGURE_CODE := $(shell find src -name "*.py") pyproject.toml uv.lock plot_data/figures.json')
for name, entry in entries.items():
    outputs = [f'figures/{new}.{ext}' for new in entry['outputs'].values() for ext in entry.get('extensions', ['pdf', 'svg'])]
    all_outputs.extend(outputs)
    if not entry['remote']: local.extend(outputs)
    deps = ' '.join('plot_data/' + p for p in entry['inputs'])
    # Preserve checked-in blocked figures in ordinary paper builds.
    if entry['remote']:
        print(' '.join(outputs) + ' &:')
        print(f'\t$(FIGURE_PYTHON) src/build_figure.py {name}')
        print(f'figure-{name}:\n\t$(FIGURE_PYTHON) src/build_figure.py {name}')
    else:
        print((' '.join(outputs) + ' &: $(FIGURE_CODE) ' + deps).rstrip())
        print(f'\t$(FIGURE_PYTHON) src/build_figure.py {name}')
        print(f'figure-{name}: ' + ' '.join(outputs))
    print(f'fetch-figure-{name}:\n\tFIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py {name}')
    print(f'.PHONY: figure-{name} fetch-figure-{name}')
print('LOCAL_FIGURES := ' + ' '.join(local))
print('REPRODUCIBLE_FIGURES := ' + ' '.join(p for n,e in entries.items() if not e['remote'] for p in [f'figures/{new}.{ext}' for new in e['outputs'].values() for ext in e.get('extensions',['pdf','svg'])]))
