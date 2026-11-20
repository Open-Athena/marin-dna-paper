.DEFAULT_GOAL := all
FIGURE_PYTHON ?= uv run --locked python
PLOT := $(FIGURE_PYTHON) src/plot.py
PROCESS_DATA := $(FIGURE_PYTHON) src/process_data.py
FIGURE_CODE := src/plot.py src/local_data.py src/metrics.py pyproject.toml uv.lock

DATA_FIGURES := upstream_coding_mixture_balance vep_by_model_size vep_by_sequence_likelihood missense_readout_scaling mixture_zero_shot_trajectories mixture_probe_trajectories mendelian_zero_shot_comparison mendelian_probe_comparison region_specialist_performance hyperparameter_transfer regional_hyperparameter_transfer sequence_loss_scaling vep_performance_vs_throughput dataset_timetree
SCHEMATIC_FIGURES := multispecies_training_sequences next_base_prediction
ARTWORK_FIGURES := variant_scoring_workflow training_dataset_provenance evaluation_benchmarks
ALL_FIGURES := $(DATA_FIGURES) $(SCHEMATIC_FIGURES) $(ARTWORK_FIGURES)
GENERATED_SVGS := $(addprefix figures/,$(addsuffix .svg,$(DATA_FIGURES) $(SCHEMATIC_FIGURES)))

CSV_upstream_coding_mixture_balance := plot_data/upstream_coding_mixture_balance_endpoints.csv plot_data/upstream_coding_mixture_balance_trajectories.csv
CSV_vep_by_model_size := plot_data/vep_by_model_size.csv
CSV_vep_by_sequence_likelihood := plot_data/vep_by_sequence_likelihood.csv
CSV_missense_readout_scaling := plot_data/missense_readout_scaling.csv
CSV_mixture_zero_shot_trajectories := plot_data/mixture_zero_shot_trajectories.csv
CSV_mixture_probe_trajectories := plot_data/mixture_probe_trajectories.csv
CSV_mendelian_zero_shot_comparison := plot_data/mendelian_zero_shot_comparison.csv
CSV_mendelian_probe_comparison := plot_data/mendelian_probe_comparison.csv
CSV_region_specialist_performance := plot_data/region_specialist_performance.csv
CSV_hyperparameter_transfer := plot_data/hyperparameter_transfer.csv
CSV_regional_hyperparameter_transfer := plot_data/regional_hyperparameter_transfer.csv
CSV_sequence_loss_scaling := plot_data/sequence_loss_scaling_results.csv plot_data/sequence_loss_scaling_history.csv
CSV_vep_performance_vs_throughput := plot_data/vep_performance_vs_throughput_leaderboard.csv plot_data/vep_performance_vs_throughput_inference_costs.csv
CSV_dataset_timetree := plot_data/dataset_timetree.csv

.PHONY: all figures figures-all process-data fetch-data fetch-sweeps figure-clean clean mrproper
all: paper.pdf
# All rendering is offline. Missing exports fail rather than silently fetching.
figures: $(addprefix figures/,$(addsuffix .pdf,$(ALL_FIGURES))) $(GENERATED_SVGS) figures/sequence_loss_scaling_fit.txt
figures-all: $(addprefix figure-,$(ALL_FIGURES))
process-data:
	$(PROCESS_DATA) all
fetch-data:
	$(PROCESS_DATA) --fetch all
fetch-sweeps:
	$(FIGURE_PYTHON) src/fetch_sweeps.py

# Preparation is explicit and never a prerequisite of normal rendering.
define data_targets
.PHONY: data-$(1) fetch-data-$(1) fetch-figure-$(1)
data-$(1):
	$$(PROCESS_DATA) $(1)
fetch-data-$(1):
	$$(PROCESS_DATA) --fetch $(1)
fetch-figure-$(1): fetch-data-$(1)
	$$(MAKE) figure-$(1)
endef
$(foreach name,$(DATA_FIGURES),$(eval $(call data_targets,$(name))))

define plot_targets
figures/$(1).pdf figures/$(1).svg &: $$(FIGURE_CODE) $$(CSV_$(1))
	$$(PLOT) $(1)
.PHONY: figure-$(1)
figure-$(1):
	$$(PLOT) $(1)
endef
$(foreach name,$(filter-out dataset_timetree sequence_loss_scaling,$(DATA_FIGURES)) $(SCHEMATIC_FIGURES),$(eval $(call plot_targets,$(name))))

figures/sequence_loss_scaling.pdf figures/sequence_loss_scaling.svg figures/sequence_loss_scaling_fit.txt &: $(FIGURE_CODE) $(CSV_sequence_loss_scaling)
	$(PLOT) sequence_loss_scaling
.PHONY: figure-sequence_loss_scaling
figure-sequence_loss_scaling:
	$(PLOT) sequence_loss_scaling

figures/dataset_timetree.pdf figures/dataset_timetree.svg figures/dataset_timetree.png &: src/plot_dataset_timetree.py pyproject.toml uv.lock $(CSV_dataset_timetree)
	$(FIGURE_PYTHON) src/plot_dataset_timetree.py
.PHONY: figure-dataset_timetree
figure-dataset_timetree:
	$(FIGURE_PYTHON) src/plot_dataset_timetree.py

# Source SVG artwork and code-native schematics have no external measurements.
define artwork_targets
figures/$(1).pdf: $$(FIGURE_CODE) figures/$(1).svg
	$$(PLOT) $(1)
.PHONY: figure-$(1)
figure-$(1):
	$$(PLOT) $(1)
endef
$(foreach name,$(ARTWORK_FIGURES),$(eval $(call artwork_targets,$(name))))
$(foreach name,$(SCHEMATIC_FIGURES) $(ARTWORK_FIGURES),$(eval .PHONY: fetch-figure-$(name)))
$(foreach name,$(SCHEMATIC_FIGURES) $(ARTWORK_FIGURES),$(eval fetch-figure-$(name): figure-$(name)))

# Figures are built explicitly with `make figures` or `make figure-<name>`.
paper.pdf: paper.tex paper.bib oup-contemporary.cls vancouver-authoryear.bst
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	@if grep -q '\\citation{' paper.aux && grep -q '\\bibdata{' paper.aux; then bibtex paper; fi
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

figure-clean:
	rm -f $(addprefix figures/,$(addsuffix .pdf,$(ALL_FIGURES))) $(GENERATED_SVGS) figures/dataset_timetree.png figures/sequence_loss_scaling_fit.txt
clean:
	rm -f paper.aux paper.log paper.out paper.bbl paper.blg paper.fls paper.fdb_latexmk paper.synctex.gz
mrproper: clean
	rm -f paper.pdf
