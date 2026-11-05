# Marin DNA paper

Placeholder-only LaTeX scaffold for the Marin DNA publication.

## Build

Install GNU Make and a TeX Live distribution including pdfLaTeX, BibTeX,
Merriweather, Euler, and the packages required by `oup-contemporary.cls`
(a full TeX Live installation includes these).

Run `make` to produce `paper.pdf`. Run `make clean` to remove auxiliary
files, or `make mrproper` to also remove the built PDF. BibTeX runs once
citations and a bibliography command are present; uncomment the bibliography
command in `paper.tex` when adding references.

## Layout

- `paper.tex`: manuscript headings, metadata TODOs, and figure/table slots.
- `paper.bib`: bibliography placeholder.
- `figures/`: future figure assets; current slots render directly in LaTeX.
- `plot_data/`: future plot inputs.
- `src/`: future analysis and plotting code.
- `oup-contemporary.cls`, `vancouver-authoryear.bst`: template files reused
  unchanged from the VCZ publication repository, with existing notices retained.

The A4 numeric-reference OUP/GigaScience configuration follows VCZ.
Journal metadata is provisional and does not establish a submission venue.

## Sources

- [Marin DNA blog](https://openathena.ai/blog/marin-dna/): outline inspiration.
- [Marin DNA](https://github.com/Open-Athena/marin-dna): code and experiments.
- [VCZ publication](https://github.com/sgkit-dev/vcf-zarr-publication):
  template, repository layout, and build conventions.

No manuscript prose, results, references, datasets, or blog images are included.
