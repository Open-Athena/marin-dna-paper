.DEFAULT_GOAL := all
FIGURE_PYTHON ?= uv run --locked python
PLOT := $(FIGURE_PYTHON) src/plot.py
FIGURE_CODE := src/plot.py src/local_data.py src/metrics.py pyproject.toml uv.lock

# Every PDF referenced by paper.tex, in manuscript order.
PAPER_FIGURES := \
	figures/multispecies_training_sequences.pdf \
	figures/next_base_prediction.pdf \
	figures/variant_scoring_workflow.pdf \
	figures/vep_performance_vs_throughput.pdf \
	figures/upstream_coding_mixture_balance.pdf \
	figures/missense_readout_scaling.pdf \
	figures/training_dataset_provenance.pdf \
	figures/annotation_derived_training_pool.pdf \
	figures/training_mixture_histories.pdf \
	figures/hyperparameter_transfer_workflow.pdf \
	figures/learning_rate_transfer.pdf \
	figures/sequence_loss_scaling.pdf \
	figures/evaluation_benchmarks.pdf \
	figures/mendelian_zero_shot_comparison.pdf \
	figures/mendelian_probe_comparison.pdf \
	figures/region_specialist_performance.pdf \
	figures/mixture_zero_shot_trajectories.pdf \
	figures/mixture_probe_trajectories.pdf \
	figures/vep_by_model_size.pdf \
	figures/vep_by_sequence_likelihood.pdf \
	figures/optimizer_parameter_transfer.pdf \
	figures/regional_hyperparameter_transfer.pdf

# These figures can be regenerated entirely from bundled inputs.
LOCAL_FIGURES := \
	figures/multispecies_training_sequences.pdf \
	figures/next_base_prediction.pdf \
	figures/variant_scoring_workflow.pdf \
	figures/vep_performance_vs_throughput.pdf \
	figures/training_dataset_provenance.pdf \
	figures/annotation_derived_training_pool.pdf \
	figures/training_mixture_histories.pdf \
	figures/hyperparameter_transfer_workflow.pdf \
	figures/learning_rate_transfer.pdf \
	figures/sequence_loss_scaling.pdf \
	figures/evaluation_benchmarks.pdf \
	figures/optimizer_parameter_transfer.pdf \
	figures/regional_hyperparameter_transfer.pdf

.PHONY: all figures figures-all fetch-sweeps figure-clean clean mrproper
all: paper.pdf

figures: $(LOCAL_FIGURES) $(LOCAL_FIGURES:.pdf=.svg) figures/sequence_loss_scaling_fit.txt

# Explicit regeneration includes figures whose source snapshots are external.
figures-all: \
	figure-multispecies_training_sequences \
	figure-next_base_prediction \
	figure-variant_scoring_workflow \
	figure-vep_performance_vs_throughput \
	figure-upstream_coding_mixture_balance \
	figure-missense_readout_scaling \
	figure-training_dataset_provenance \
	figure-annotation_derived_training_pool \
	figure-training_mixture_histories \
	figure-hyperparameter_transfer_workflow \
	figure-learning_rate_transfer \
	figure-sequence_loss_scaling \
	figure-evaluation_benchmarks \
	figure-mendelian_zero_shot_comparison \
	figure-mendelian_probe_comparison \
	figure-region_specialist_performance \
	figure-mixture_zero_shot_trajectories \
	figure-mixture_probe_trajectories \
	figure-vep_by_model_size \
	figure-vep_by_sequence_likelihood \
	figure-optimizer_parameter_transfer \
	figure-regional_hyperparameter_transfer

fetch-sweeps:
	$(FIGURE_PYTHON) src/fetch_sweeps.py

figures/multispecies_training_sequences.pdf figures/multispecies_training_sequences.svg &: $(FIGURE_CODE)
	$(PLOT) multispecies_training_sequences

.PHONY: figure-multispecies_training_sequences fetch-figure-multispecies_training_sequences
figure-multispecies_training_sequences:
	$(PLOT) multispecies_training_sequences
fetch-figure-multispecies_training_sequences:
	$(PLOT) --fetch multispecies_training_sequences

figures/next_base_prediction.pdf figures/next_base_prediction.svg &: $(FIGURE_CODE)
	$(PLOT) next_base_prediction

.PHONY: figure-next_base_prediction fetch-figure-next_base_prediction
figure-next_base_prediction:
	$(PLOT) next_base_prediction
fetch-figure-next_base_prediction:
	$(PLOT) --fetch next_base_prediction

figures/variant_scoring_workflow.pdf figures/variant_scoring_workflow.svg &: $(FIGURE_CODE) plot_data/artwork/variant_scoring_workflow.svg
	$(PLOT) variant_scoring_workflow

.PHONY: figure-variant_scoring_workflow fetch-figure-variant_scoring_workflow
figure-variant_scoring_workflow:
	$(PLOT) variant_scoring_workflow
fetch-figure-variant_scoring_workflow:
	$(PLOT) --fetch variant_scoring_workflow

figures/vep_performance_vs_throughput.pdf figures/vep_performance_vs_throughput.svg &: $(FIGURE_CODE) plot_data/model_leaderboard.csv plot_data/inference_costs.csv
	$(PLOT) vep_performance_vs_throughput

.PHONY: figure-vep_performance_vs_throughput fetch-figure-vep_performance_vs_throughput
figure-vep_performance_vs_throughput:
	$(PLOT) vep_performance_vs_throughput
fetch-figure-vep_performance_vs_throughput:
	$(PLOT) --fetch vep_performance_vs_throughput

# Keep the checked-in PDF usable without external snapshots.
figures/upstream_coding_mixture_balance.pdf figures/upstream_coding_mixture_balance.svg &:
	$(PLOT) upstream_coding_mixture_balance

.PHONY: figure-upstream_coding_mixture_balance fetch-figure-upstream_coding_mixture_balance
figure-upstream_coding_mixture_balance:
	$(PLOT) upstream_coding_mixture_balance
fetch-figure-upstream_coding_mixture_balance:
	$(PLOT) --fetch upstream_coding_mixture_balance

# Keep the checked-in PDF usable without external snapshots.
figures/missense_readout_scaling.pdf figures/missense_readout_scaling.svg &:
	$(PLOT) missense_readout_scaling

.PHONY: figure-missense_readout_scaling fetch-figure-missense_readout_scaling
figure-missense_readout_scaling:
	$(PLOT) missense_readout_scaling
fetch-figure-missense_readout_scaling:
	$(PLOT) --fetch missense_readout_scaling

figures/training_dataset_provenance.pdf figures/training_dataset_provenance.svg &: $(FIGURE_CODE) plot_data/artwork/training_dataset_provenance.svg
	$(PLOT) training_dataset_provenance

.PHONY: figure-training_dataset_provenance fetch-figure-training_dataset_provenance
figure-training_dataset_provenance:
	$(PLOT) training_dataset_provenance
fetch-figure-training_dataset_provenance:
	$(PLOT) --fetch training_dataset_provenance

figures/annotation_derived_training_pool.pdf figures/annotation_derived_training_pool.svg &: $(FIGURE_CODE) plot_data/artwork/annotation_derived_training_pool.svg
	$(PLOT) annotation_derived_training_pool

.PHONY: figure-annotation_derived_training_pool fetch-figure-annotation_derived_training_pool
figure-annotation_derived_training_pool:
	$(PLOT) annotation_derived_training_pool
fetch-figure-annotation_derived_training_pool:
	$(PLOT) --fetch annotation_derived_training_pool

figures/training_mixture_histories.pdf figures/training_mixture_histories.svg &: $(FIGURE_CODE) plot_data/artwork/training_mixture_histories.svg
	$(PLOT) training_mixture_histories

.PHONY: figure-training_mixture_histories fetch-figure-training_mixture_histories
figure-training_mixture_histories:
	$(PLOT) training_mixture_histories
fetch-figure-training_mixture_histories:
	$(PLOT) --fetch training_mixture_histories

figures/hyperparameter_transfer_workflow.pdf figures/hyperparameter_transfer_workflow.svg &: $(FIGURE_CODE) plot_data/artwork/hyperparameter_transfer_workflow.svg
	$(PLOT) hyperparameter_transfer_workflow

.PHONY: figure-hyperparameter_transfer_workflow fetch-figure-hyperparameter_transfer_workflow
figure-hyperparameter_transfer_workflow:
	$(PLOT) hyperparameter_transfer_workflow
fetch-figure-hyperparameter_transfer_workflow:
	$(PLOT) --fetch hyperparameter_transfer_workflow

figures/learning_rate_transfer.pdf figures/learning_rate_transfer.svg &: $(FIGURE_CODE) plot_data/transfer_validation_results.csv
	$(PLOT) learning_rate_transfer

.PHONY: figure-learning_rate_transfer fetch-figure-learning_rate_transfer
figure-learning_rate_transfer:
	$(PLOT) learning_rate_transfer
fetch-figure-learning_rate_transfer:
	$(PLOT) --fetch learning_rate_transfer

figures/sequence_loss_scaling.pdf figures/sequence_loss_scaling.svg figures/sequence_loss_scaling_fit.txt &: $(FIGURE_CODE) plot_data/parameter_scaling_results.csv plot_data/parameter_scaling_history.csv
	$(PLOT) sequence_loss_scaling

.PHONY: figure-sequence_loss_scaling fetch-figure-sequence_loss_scaling
figure-sequence_loss_scaling:
	$(PLOT) sequence_loss_scaling
fetch-figure-sequence_loss_scaling:
	$(PLOT) --fetch sequence_loss_scaling

figures/evaluation_benchmarks.pdf figures/evaluation_benchmarks.svg &: $(FIGURE_CODE) plot_data/artwork/evaluation_benchmarks.svg
	$(PLOT) evaluation_benchmarks

.PHONY: figure-evaluation_benchmarks fetch-figure-evaluation_benchmarks
figure-evaluation_benchmarks:
	$(PLOT) evaluation_benchmarks
fetch-figure-evaluation_benchmarks:
	$(PLOT) --fetch evaluation_benchmarks

# Keep the checked-in PDF usable without external snapshots.
figures/mendelian_zero_shot_comparison.pdf figures/mendelian_zero_shot_comparison.svg &:
	$(PLOT) mendelian_zero_shot_comparison

.PHONY: figure-mendelian_zero_shot_comparison fetch-figure-mendelian_zero_shot_comparison
figure-mendelian_zero_shot_comparison:
	$(PLOT) mendelian_zero_shot_comparison
fetch-figure-mendelian_zero_shot_comparison:
	$(PLOT) --fetch mendelian_zero_shot_comparison

# Keep the checked-in PDF usable without external snapshots.
figures/mendelian_probe_comparison.pdf figures/mendelian_probe_comparison.svg &:
	$(PLOT) mendelian_probe_comparison

.PHONY: figure-mendelian_probe_comparison fetch-figure-mendelian_probe_comparison
figure-mendelian_probe_comparison:
	$(PLOT) mendelian_probe_comparison
fetch-figure-mendelian_probe_comparison:
	$(PLOT) --fetch mendelian_probe_comparison

# Keep the checked-in PDF usable without external snapshots.
figures/region_specialist_performance.pdf figures/region_specialist_performance.svg &:
	$(PLOT) region_specialist_performance

.PHONY: figure-region_specialist_performance fetch-figure-region_specialist_performance
figure-region_specialist_performance:
	$(PLOT) region_specialist_performance
fetch-figure-region_specialist_performance:
	$(PLOT) --fetch region_specialist_performance

# Keep the checked-in PDF usable without external snapshots.
figures/mixture_zero_shot_trajectories.pdf figures/mixture_zero_shot_trajectories.svg &:
	$(PLOT) mixture_zero_shot_trajectories

.PHONY: figure-mixture_zero_shot_trajectories fetch-figure-mixture_zero_shot_trajectories
figure-mixture_zero_shot_trajectories:
	$(PLOT) mixture_zero_shot_trajectories
fetch-figure-mixture_zero_shot_trajectories:
	$(PLOT) --fetch mixture_zero_shot_trajectories

# Keep the checked-in PDF usable without external snapshots.
figures/mixture_probe_trajectories.pdf figures/mixture_probe_trajectories.svg &:
	$(PLOT) mixture_probe_trajectories

.PHONY: figure-mixture_probe_trajectories fetch-figure-mixture_probe_trajectories
figure-mixture_probe_trajectories:
	$(PLOT) mixture_probe_trajectories
fetch-figure-mixture_probe_trajectories:
	$(PLOT) --fetch mixture_probe_trajectories

# Keep the checked-in PDF usable without external snapshots.
figures/vep_by_model_size.pdf figures/vep_by_model_size.svg &:
	$(PLOT) vep_by_model_size

.PHONY: figure-vep_by_model_size fetch-figure-vep_by_model_size
figure-vep_by_model_size:
	$(PLOT) vep_by_model_size
fetch-figure-vep_by_model_size:
	$(PLOT) --fetch vep_by_model_size

# Keep the checked-in PDF usable without external snapshots.
figures/vep_by_sequence_likelihood.pdf figures/vep_by_sequence_likelihood.svg &:
	$(PLOT) vep_by_sequence_likelihood

.PHONY: figure-vep_by_sequence_likelihood fetch-figure-vep_by_sequence_likelihood
figure-vep_by_sequence_likelihood:
	$(PLOT) vep_by_sequence_likelihood
fetch-figure-vep_by_sequence_likelihood:
	$(PLOT) --fetch vep_by_sequence_likelihood

figures/optimizer_parameter_transfer.pdf figures/optimizer_parameter_transfer.svg &: $(FIGURE_CODE) plot_data/transfer_validation_results.csv
	$(PLOT) optimizer_parameter_transfer

.PHONY: figure-optimizer_parameter_transfer fetch-figure-optimizer_parameter_transfer
figure-optimizer_parameter_transfer:
	$(PLOT) optimizer_parameter_transfer
fetch-figure-optimizer_parameter_transfer:
	$(PLOT) --fetch optimizer_parameter_transfer

figures/regional_hyperparameter_transfer.pdf figures/regional_hyperparameter_transfer.svg &: $(FIGURE_CODE) plot_data/transfer_validation_results.csv
	$(PLOT) regional_hyperparameter_transfer

.PHONY: figure-regional_hyperparameter_transfer fetch-figure-regional_hyperparameter_transfer
figure-regional_hyperparameter_transfer:
	$(PLOT) regional_hyperparameter_transfer
fetch-figure-regional_hyperparameter_transfer:
	$(PLOT) --fetch regional_hyperparameter_transfer

paper.pdf: $(PAPER_FIGURES) paper.tex paper.bib oup-contemporary.cls vancouver-authoryear.bst
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	@if grep -q '\\citation{' paper.aux && grep -q '\\bibdata{' paper.aux; then bibtex paper; fi
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex
	pdflatex -interaction=nonstopmode -halt-on-error paper.tex

figure-clean:
	rm -f $(LOCAL_FIGURES) $(LOCAL_FIGURES:.pdf=.svg) figures/sequence_loss_scaling_fit.txt

clean:
	rm -f paper.aux paper.log paper.out paper.bbl paper.blg paper.fls paper.fdb_latexmk paper.synctex.gz

mrproper: clean
	rm -f paper.pdf
