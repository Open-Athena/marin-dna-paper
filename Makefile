.DEFAULT_GOAL := all
FIGURE_PYTHON ?= uv run --locked python
include figures.mk

.PHONY: all clean mrproper figures figures-all figure-status figure-clean fetch-sweeps
all: paper.pdf

figures.mk: src/make_rules.py plot_data/figures.json
	python3 src/make_rules.py > $@

figures: $(LOCAL_FIGURES)
	$(FIGURE_PYTHON) src/build_figure.py status

figures-all:
	$(FIGURE_PYTHON) src/build_figure.py all --strict

figure-status:
	$(FIGURE_PYTHON) src/build_figure.py status

fetch-sweeps:
	$(FIGURE_PYTHON) src/fetch_sweeps.py

figure-clean:
	rm -f $(REPRODUCIBLE_FIGURES)

paper.pdf: $(LOCAL_FIGURES) paper.tex paper.bib oup-contemporary.cls vancouver-authoryear.bst $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	@if grep -q '\\citation{' paper.aux && grep -q '\\bibdata{' paper.aux; then bibtex paper; fi
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

clean:
	rm -f paper.aux paper.log paper.out paper.bbl paper.blg paper.fls paper.fdb_latexmk paper.synctex.gz

mrproper: clean
	rm -f paper.pdf
