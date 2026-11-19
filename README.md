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
make figures                              # The figures with bundled inputs
make figures/learning_rate_transfer.pdf    # Incremental build of one figure
make figure-learning_rate_transfer        # Force regeneration of one figure
uv run --locked python src/plot.py --help  # 20 figures; time tree uses a separate script
uv run --locked python src/plot.py --output-dir /tmp/figures learning_rate_transfer
```

The 20 figures in `src/plot.py` have Click commands matching their filenames without the extension.
The supplementary cohort time tree uses `src/plot_dataset_timetree.py` and has matching Make targets.
Plots write PDF and SVG; `sequence_loss_scaling` also writes
`sequence_loss_scaling_fit.txt`. Existing SVG diagrams in `figures/` are converted
to vector PDF. `plot_data/` holds the CSVs, model registry, and downloaded inputs.
The overview diagrams use Lato when available. SVG text stays editable.

The Makefile explicitly lists the paper's figures and their bundled inputs.
The main figure recipes are in `src/plot.py`; `src/local_data.py` contains shared
input loaders, source URLs, model metadata, and lineage definitions.
`src/metrics.py` contains the chromosome-weighted AUPRC and bootstrap calculations.
The plotting code’s original license is retained in `src/LICENSE`.

## External snapshots

Some figures require original predictions or metric Parquets that are not bundled.
Ordinary paper builds use their checked-in PDFs. Explicit regeneration fails with
the expected local path and original URL when an input is missing; it leaves
existing publication files intact. An absent PDF also triggers this check.
Changing plotting code does not automatically regenerate these external figures.

```sh
make fetch-figure-vep_by_model_size         # Fetch missing inputs, then render
uv run --locked python src/plot.py --fetch vep_by_model_size
make figure-vep_by_model_size              # Regenerate from downloaded inputs
make figures-all                          # Force all 21; fails if inputs are missing
make fetch-sweeps                          # Refresh four W&B sweep/history CSVs
```

Fetching is explicit and may require AWS or W&B credentials. Remote inputs are
cached under `plot_data/remote/<scheme>/<host>/<path>` and are gitignored.
`FIGURE_FETCH=1` is also supported for explicit retrieval. Normal builds are offline.

`make clean` removes LaTeX intermediates; `make mrproper` also removes `paper.pdf`.
`make figure-clean` removes only outputs reproducible from bundled inputs, retaining
the source SVGs and external figures' checked-in PDFs.

Run the scientific-calculation and cohort tree-data checks with:

```sh
uv run --locked python -m unittest discover -s tests
```

## Cohort time tree

`uv run --locked python src/plot_dataset_timetree.py` renders the dated cohort overview as PDF, SVG, and PNG in `figures/dataset_timetree.*`.
Add `--fetch` to retrieve or refresh TimeTree's higher-taxon exports; the raw exports are cached under `plot_data/remote/` and are not bundled.
The offline renderer preserves source node ages while collapsing taxonomic groups and obtains counts from `data/dataset_provenance/`.
`plot_data/dataset_timetree_sources.json` records source URLs, hashes, and the mapping between source taxa and figure groups.
The two panels have explicitly different time axes; terminal branches include diversification hidden inside each group, and age uncertainty is not drawn.

### Figure 3 data

`src/prepare_figure3_data.py` extracts the measured AWS checkpoint data into two
bundled CSVs. Normal figure builds read only these CSVs and need no AWS access:

- `plot_data/figure3_endpoints.csv`: 12 rows covering four conditions at step
  22,000, their promoter and missense AUPRC, and their unweighted two-class mean.
- `plot_data/figure3_trajectories.csv`: 216 checkpoint rows covering three
  lineages and nine plots, including the stored eight-class macro-average and
  standard errors, cumulative tokens, and the transition position.

AUPRC values and standard errors are stored as fractions; tokens are absolute
counts. Rows include original S3 URIs, metric names, and checkpoint identifiers
for provenance. The export preserves the former S5 checkpoint selection and
horizon, including its deliberate pre-transition omission; no measurements or
uncertainty estimates are recomputed. Only Panel A's two-class means are derived.

```sh
# Fetch missing Parquets and export both CSVs using the configured AWS profile.
AWS_PROFILE=marin-dna make fetch-figure3-data
# Or export using already cached Parquets, without AWS access.
make figure3-data
# Render entirely from the bundled CSVs.
make figure-upstream_coding_mixture_balance
# Fetch, export, and render together.
AWS_PROFILE=marin-dna make fetch-figure-upstream_coding_mixture_balance
```

The script also accepts `--output-dir` to write exports to a different directory.
Raw Parquets remain in the gitignored `plot_data/remote/` cache. CSVs are updated
only by the explicit preparation targets, not by ordinary figure builds.
