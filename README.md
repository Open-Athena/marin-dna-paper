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
Plots write PDF and SVG; `sequence_loss_scaling` also writes
`sequence_loss_scaling_fit.txt`. Existing SVG diagrams in `figures/` are converted
to vector PDF. `plot_data/` holds the CSVs, model registry, and downloaded inputs.
The overview diagrams use Lato when available. SVG text stays editable.

The Makefile explicitly lists the paper's figures and their bundled inputs.
All plotting and saving is in `src/plot.py`; `src/local_data.py` contains shared
input loaders, source URLs, model metadata, and lineage definitions.
`src/metrics.py` contains the chromosome-weighted AUPRC and bootstrap calculations.
The plotting code’s original license is retained in `src/LICENSE`.

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

`make clean` removes LaTeX intermediates; `make mrproper` also removes `paper.pdf`.
`make figure-clean` removes only outputs reproducible from bundled inputs, retaining
the source SVGs and external figures' checked-in PDFs.

Run the offline build and scientific-calculation checks with:

```sh
uv run --locked python -m unittest discover -s tests
```

## Cohort time tree

`uv run --locked python src/plot_dataset_timetree.py` renders the dated cohort overview as PDF, SVG, and PNG in `figures/dataset_timetree.*`.
Add `--fetch` to retrieve or refresh TimeTree's higher-taxon exports; the raw exports are cached under `plot_data/remote/` and are not bundled.
The offline renderer preserves source node ages while collapsing taxonomic groups and obtains counts from `data/dataset_provenance/`.
`plot_data/dataset_timetree_sources.json` records source URLs, hashes, and the mapping between source taxa and figure groups.
The two panels have explicitly different time axes; terminal branches include diversification hidden inside each group, and age uncertainty is not drawn.

`uv run --locked python src/plot_evo2_diversity.py` renders the matching OpenGenome2 metadata overview in `figures/evo2_taxonomic_diversity.*`.
Its `--fetch` option downloads the pinned species metadata and TimeTree eukaryote export into the same ignored cache area.
The figure combines a dated eukaryotic tree with bacterial and archaeal phylum-count bars; it explicitly distinguishes the released metadata from a complete training-corpus census.
Aggregated counts and source hashes are saved in `plot_data/evo2_*`.

`uv run --locked python src/plot_dtol_diversity.py` renders the matching Darwin Tree of Life snapshot in `figures/dtol_taxonomic_diversity.*`.
The supplied CSV and exact-name NCBI taxonomy resolution are bundled in `data/dtol_provenance/`; counts use distinct resolved taxon IDs, not ToL specimen IDs.
This snapshot contains 1,826 taxa, all marked Annotation Complete. The overview colours animals, green plants and fungi, and a second panel expands the 1,219 insects on a 4× time-axis zoom.
The default render is offline; `--fetch` retrieves the two required TimeTree exports. Counts and dated-tree mappings are written to `plot_data/dtol_*`.
To resolve a replacement CSV, first cache `https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz` at `plot_data/remote/https/ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz`, then run `uv run --locked python src/resolve_dtol_taxonomy.py /path/to/dtol.csv`.
Resolution requires an unambiguous exact scientific name or NCBI synonym; unmapped names or plotting groups fail explicitly. Source hashes and dates are recorded alongside the resolved taxonomy.

`uv run --locked python src/compare_marin_dtol.py` compares taxonomic richness, species overlap and composition between the combined Marin cohorts and the supplied DToL snapshot.
Both use the same NCBI taxonomy snapshot; Marin's two cohorts are deduplicated by species taxon ID. Tables and provenance are saved in `plot_data/marin_dtol_*`.
Add `--resolve` to regenerate `plot_data/marin_comparison_taxonomy.csv` from the cached taxonomy dump. The assembly label `Ateles_geoffroyi_a` is explicitly normalised, and an unresolved historical annotation name can fall back to its manifest taxon ID; these resolutions are recorded in the output.
