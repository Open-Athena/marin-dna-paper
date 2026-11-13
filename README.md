# Marin DNA paper

Work in progress.

Sources: [Marin DNA blog](https://openathena.ai/blog/marin-dna/),
[Marin DNA experiments](https://github.com/Open-Athena/marin-dna), and
[VCZ publication template](https://github.com/sgkit-dev/vcf-zarr-publication).

## Build the paper and figures

Install the plotting environment with `uv sync --locked`. The paper also needs
GNU Make 4.3 or later, `pdflatex`, and BibTeX.

```sh
make                                      # Bundled figures, then paper.pdf
make figures                              # The 13 figures with bundled inputs
make figures/learning_rate_transfer.pdf    # Incremental build of one figure
make figure-learning_rate_transfer        # Force regeneration of one figure
uv run --locked python src/plot.py --help  # All 22 paper figures
uv run --locked python src/plot.py --output-dir /tmp/figures learning_rate_transfer
```

Every figure has a Click command matching its filename without the extension.
Commands write PDF and SVG; `sequence_loss_scaling` also writes
`sequence_loss_scaling_fit.txt`. Authored SVGs are converted to vector PDF.
The overview diagrams use Lato when available. SVG text stays editable.

The Makefile explicitly lists the paper's figures and their bundled inputs.
All plotting and saving is in `src/plot.py`; `src/local_data.py` contains shared
input loaders, source URLs, model metadata, and lineage definitions.
`src/metrics.py` contains the chromosome-weighted AUPRC and bootstrap calculations.
There is no figure manifest or generated Makefile.

## External snapshots

Nine figures require original predictions or metric Parquets that are not bundled.
Ordinary paper builds use their checked-in PDFs. Explicit regeneration fails with
the expected local path and original URL when an input is missing; it leaves
existing publication files intact. An absent PDF also triggers this check.
Changing plotting code does not automatically regenerate these external figures.

```sh
make fetch-figure-vep_by_model_size         # Fetch missing inputs, then render
uv run --locked python src/plot.py --fetch vep_by_model_size
make figure-vep_by_model_size              # Regenerate from downloaded inputs
make figures-all                          # Force all 22; fails if inputs are missing
make fetch-sweeps                          # Refresh four W&B sweep/history CSVs
```

Fetching is explicit and may require AWS or W&B credentials. Remote inputs are
cached under `plot_data/remote/<scheme>/<host>/<path>` and are gitignored.
`FIGURE_FETCH=1` is also supported for explicit retrieval. Normal builds are offline.
See [plot_data/README.md](plot_data/README.md) for the input inventory and provenance.

`make clean` removes LaTeX intermediates; `make mrproper` also removes `paper.pdf`.
`make figure-clean` removes only outputs reproducible from bundled inputs, retaining
the external figures' checked-in PDFs.

Run the offline build and scientific-calculation checks with:

```sh
uv run --locked python -m unittest discover -s tests
```
