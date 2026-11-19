# Marin DNA paper

Work in progress.

Sources: [Marin DNA blog](https://openathena.ai/blog/marin-dna/),
[Marin DNA experiments](https://github.com/Open-Athena/marin-dna), and
[VCZ publication template](https://github.com/sgkit-dev/vcf-zarr-publication).

## Build the paper and figures

Install the plotting environment with `uv sync --locked`. The paper also needs
GNU Make 4.3 or later, `pdflatex`, and BibTeX.

Every data-driven figure renders from CSVs checked into `plot_data/`. No AWS
credentials, raw Parquets, network access, or untracked caches are needed to
render. Schematics use local Python code or source SVG artwork rather than
measurement tables.

```sh
make                                      # Figures and paper.pdf
make figures                              # All figures from local inputs
make figures-all                          # Force regeneration of every figure
make figure-vep_by_model_size              # One figure
uv run --locked python src/plot.py --help
uv run --locked python src/plot.py --output-dir /tmp/figures vep_by_model_size
```

Plot commands, CSV basenames, and figure filenames share descriptive labels,
not manuscript figure numbers. A figure with multiple tables adds a descriptive
suffix, such as `upstream_coding_mixture_balance_endpoints.csv` and
`upstream_coding_mixture_balance_trajectories.csv`. The standalone zero-shot
mixture trajectory figure remains available even though its content is now in
the combined upstream/CDS figure.

The Makefile tracks CSV dependencies, so editing an export rebuilds its figure.
Plots write PDF and SVG; `sequence_loss_scaling` also writes its fitted-parameter
report. `dataset_timetree` additionally writes PNG. Source SVG diagrams remain
editable and are converted to PDF. The plotting code's original license is
retained in `src/LICENSE`.

## Prepare plotting data

`src/process_data.py` contains the S3 metric/prediction processing and exposes
Click commands named after each data-driven figure, plus `all`. Processing is
separate from rendering and only runs when explicitly requested:

```sh
uv run --locked python src/process_data.py --help
make process-data                         # Re-export all using cached sources
make data-vep_by_model_size                # One export using cached sources
AWS_PROFILE=marin-dna make fetch-data      # Fetch missing sources and export all
AWS_PROFILE=marin-dna make fetch-data-vep_by_model_size
AWS_PROFILE=marin-dna make fetch-figure-vep_by_model_size  # Export, then render
uv run --locked python src/process_data.py --output-dir /tmp/plot-data vep_by_model_size
```

The script preserves the original evaluation methods: scaling readouts use the
same finite-probe variants and chromosome-weighted AUPRC with 1,000 bootstrap
replicates and seed 0; leaderboard and lineage exports retain stored metrics
and standard errors. Mixture trajectories preserve the former supplementary
checkpoint selection and horizon, including its deliberate pre-transition
omission. The upstream/CDS endpoints use step 22,000, with an unweighted mean of
the two classes; the later macro-average is the stored eight-class result.
Undefined benchmark cells remain missing rather than being imputed.

The CSVs retain the values consumed by the plots, with source URIs and model or
checkpoint identifiers on externally derived metric tables. AUPRC values and
SEs are fractions except for `region_specialist_performance.csv`, whose
explicit `_pct` columns use percentages. Token coordinates are unrounded.
Readers preserve floating-point precision through the CSV round trip.

Original S3 and pinned public benchmark inputs are cached under the gitignored
`plot_data/remote/` directory. `--fetch` (or `FIGURE_FETCH=1`) retrieves missing
inputs; it does not refresh already cached snapshots. A preparation failure
leaves that figure's existing exports intact. Source metric values are not
estimated from figure images.

The transfer, loss-scaling, and throughput commands prepare figure-specific
CSVs from the existing frozen source CSVs in `plot_data/`. Those snapshots are
retained as provenance and inputs to processing. `make fetch-sweeps` remains
the separate, explicit W&B source-refresh command; after refreshing sources,
run `process_data.py` to update plotting exports.

## Cohort time tree

`process_data.py dataset_timetree` exports dated tree nodes, parent relationships,
labels, source hashes, and cohort counts to `plot_data/dataset_timetree.csv`.
Preparation uses cached TimeTree higher-taxon Newick exports and the manifests
in `data/dataset_provenance/`; `--fetch` retrieves missing public exports.
`plot_dataset_timetree.py` renders only the bundled node CSV. Collapsed nodes
retain their original ages, and the two panels retain distinct time axes.
Terminal branches include diversification hidden inside each taxonomic group;
this is not a species tree inferred from the training data.

## Cleanup and scientific checks

`make figure-clean` removes generated artwork while keeping all source CSVs and
SVG diagrams, so `make figures` can rebuild offline. `make clean` removes LaTeX
intermediates; `make mrproper` also removes `paper.pdf`.

Run the scientific-calculation and cohort tree-data checks with:

```sh
uv run --locked python -m unittest discover -s tests
```
