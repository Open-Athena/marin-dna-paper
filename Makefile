.PHONY: all clean mrproper
all: paper.pdf

paper.pdf: paper.tex paper.bib oup-contemporary.cls vancouver-authoryear.bst $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	@if grep -q '\\citation{' paper.aux && grep -q '\\bibdata{' paper.aux; then bibtex paper; fi
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

clean:
	rm -f paper.aux paper.log paper.out paper.bbl paper.blg paper.fls paper.fdb_latexmk paper.synctex.gz

mrproper: clean
	rm -f paper.pdf
