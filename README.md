# Marin DNA paper

Work in progress.

## Sources

- [Marin DNA blog](https://openathena.ai/blog/marin-dna/): original source
- [Marin DNA](https://github.com/Open-Athena/marin-dna): code and experiments.
- [VCZ publication](https://github.com/sgkit-dev/vcf-zarr-publication):
  template, repository layout, and build conventions.


## Figure 1 schematics

Regenerate the vector PDF and editable SVG artwork for panels A and B with:

```sh
uv run src/draw_model_overview.py
make
```

The script declares its Matplotlib dependency inline and uses the Lato font
when installed. The SVGs retain editable text; PDFs embed their fonts.

## Rebuilding blog figures

The blog plotting recipes are ported into `src/`, with their input snapshots in `plot_data/`.
Install the locked plotting environment with `uv sync --locked`.
Normal rendering uses local data only; it does not require a checkout of `marin-dna`.

```sh
make figures                       # All figures backed by bundled inputs
make                               # Rebuild those figures and compile the paper
make figure-learning_rate_transfer # One recipe
make figures/learning_rate_transfer.pdf
make figure-status                 # Inventory and external-data requirements
make figures-all                   # Strict rebuild; fails if any data is missing
```

Each recipe writes PDF and editable SVG outputs under `figures/`.
The cooldown appendix is a CSV/Markdown analysis rather than a plot.
`plot_data/figures.json` maps original recipes to manuscript filenames and declares local prerequisites.
`figures.mk` is generated from that manifest by `src/make_rules.py`.
Blocked figures retain their existing checked-in PDFs during ordinary paper builds.
Explicit `figure-<name>` targets always attempt external-data recipes, so missing inputs cannot silently pass as a successful regeneration.

Remote metric inputs are cached under `plot_data/remote/<scheme>/<host>/<path>`.
A missing-data error prints the exact expected filename and original URL.
To retrieve snapshots and render a particular recipe, use `make fetch-figure-<name>` with the appropriate AWS/W&B credentials.
W&B appendix snapshots are stored as JSON in `plot_data/wandb/`; query definitions remain in each appendix recipe.
`make fetch-sweeps` explicitly refreshes the five sweep/history CSVs from W&B.
Fetches are never part of the normal build, and downloaded caches are gitignored.
After fetching a blocked recipe, rebuild it explicitly when its inputs change.

The seven authored blog schematics are preserved in `plot_data/artwork/` and converted to PDF with CairoSVG.
These targets reproduce vector artwork, not plots from numeric measurements.
The existing Figure 1A/B generator is also included in the build as `figure-model_overview`.

`make clean` removes only LaTeX intermediates.
`make figure-clean` removes outputs reproducible from bundled inputs, preserving blocked figures.
Run the build-contract checks with `uv run --locked python -m unittest discover -s tests`.
