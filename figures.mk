FIGURE_CODE := $(shell find src -name "*.py") pyproject.toml uv.lock plot_data/figures.json
figures/learning_rate_transfer.pdf figures/learning_rate_transfer.svg &: $(FIGURE_CODE) plot_data/transfer_validation_results.csv
	$(FIGURE_PYTHON) src/build_figure.py learning_rate_transfer
figure-learning_rate_transfer: figures/learning_rate_transfer.pdf figures/learning_rate_transfer.svg
fetch-figure-learning_rate_transfer:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py learning_rate_transfer
.PHONY: figure-learning_rate_transfer fetch-figure-learning_rate_transfer
figures/optimizer_parameter_transfer.pdf figures/optimizer_parameter_transfer.svg &: $(FIGURE_CODE) plot_data/transfer_validation_results.csv
	$(FIGURE_PYTHON) src/build_figure.py optimizer_parameter_transfer
figure-optimizer_parameter_transfer: figures/optimizer_parameter_transfer.pdf figures/optimizer_parameter_transfer.svg
fetch-figure-optimizer_parameter_transfer:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py optimizer_parameter_transfer
.PHONY: figure-optimizer_parameter_transfer fetch-figure-optimizer_parameter_transfer
figures/regional_hyperparameter_transfer.pdf figures/regional_hyperparameter_transfer.svg &: $(FIGURE_CODE) plot_data/transfer_validation_results.csv
	$(FIGURE_PYTHON) src/build_figure.py regional_hyperparameter_transfer
figure-regional_hyperparameter_transfer: figures/regional_hyperparameter_transfer.pdf figures/regional_hyperparameter_transfer.svg
fetch-figure-regional_hyperparameter_transfer:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py regional_hyperparameter_transfer
.PHONY: figure-regional_hyperparameter_transfer fetch-figure-regional_hyperparameter_transfer
figures/sequence_loss_scaling.pdf figures/sequence_loss_scaling.svg &: $(FIGURE_CODE) plot_data/parameter_scaling_results.csv plot_data/parameter_scaling_history.csv
	$(FIGURE_PYTHON) src/build_figure.py sequence_loss_scaling
figure-sequence_loss_scaling: figures/sequence_loss_scaling.pdf figures/sequence_loss_scaling.svg
fetch-figure-sequence_loss_scaling:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py sequence_loss_scaling
.PHONY: figure-sequence_loss_scaling fetch-figure-sequence_loss_scaling
figures/vep_by_model_size.pdf figures/vep_by_model_size.svg &:
	$(FIGURE_PYTHON) src/build_figure.py vep_by_model_size
figure-vep_by_model_size:
	$(FIGURE_PYTHON) src/build_figure.py vep_by_model_size
fetch-figure-vep_by_model_size:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py vep_by_model_size
.PHONY: figure-vep_by_model_size fetch-figure-vep_by_model_size
figures/vep_by_sequence_likelihood.pdf figures/vep_by_sequence_likelihood.svg &:
	$(FIGURE_PYTHON) src/build_figure.py vep_by_sequence_likelihood
figure-vep_by_sequence_likelihood:
	$(FIGURE_PYTHON) src/build_figure.py vep_by_sequence_likelihood
fetch-figure-vep_by_sequence_likelihood:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py vep_by_sequence_likelihood
.PHONY: figure-vep_by_sequence_likelihood fetch-figure-vep_by_sequence_likelihood
figures/missense_readout_scaling.pdf figures/missense_readout_scaling.svg &:
	$(FIGURE_PYTHON) src/build_figure.py missense_readout_scaling
figure-missense_readout_scaling:
	$(FIGURE_PYTHON) src/build_figure.py missense_readout_scaling
fetch-figure-missense_readout_scaling:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py missense_readout_scaling
.PHONY: figure-missense_readout_scaling fetch-figure-missense_readout_scaling
figures/upstream_reweighting.pdf figures/upstream_reweighting.svg &: $(FIGURE_CODE) plot_data/data_mixture_results.csv plot_data/data_mixture_history.csv
	$(FIGURE_PYTHON) src/build_figure.py upstream_reweighting
figure-upstream_reweighting: figures/upstream_reweighting.pdf figures/upstream_reweighting.svg
fetch-figure-upstream_reweighting:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py upstream_reweighting
.PHONY: figure-upstream_reweighting fetch-figure-upstream_reweighting
figures/historical_lineage_trajectories.pdf figures/historical_lineage_trajectories.svg &: $(FIGURE_CODE) plot_data/data_mixture_results.csv plot_data/data_mixture_history.csv
	$(FIGURE_PYTHON) src/build_figure.py historical_lineage_trajectories
figure-historical_lineage_trajectories: figures/historical_lineage_trajectories.pdf figures/historical_lineage_trajectories.svg
fetch-figure-historical_lineage_trajectories:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py historical_lineage_trajectories
.PHONY: figure-historical_lineage_trajectories fetch-figure-historical_lineage_trajectories
figures/historical_leaderboard.pdf figures/historical_leaderboard.svg &: $(FIGURE_CODE) plot_data/model_leaderboard.csv
	$(FIGURE_PYTHON) src/build_figure.py historical_leaderboard
figure-historical_leaderboard: figures/historical_leaderboard.pdf figures/historical_leaderboard.svg
fetch-figure-historical_leaderboard:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py historical_leaderboard
.PHONY: figure-historical_leaderboard fetch-figure-historical_leaderboard
figures/vep_performance_vs_throughput.pdf figures/vep_performance_vs_throughput.svg &: $(FIGURE_CODE) plot_data/model_leaderboard.csv plot_data/inference_costs.csv
	$(FIGURE_PYTHON) src/build_figure.py vep_performance_vs_throughput
figure-vep_performance_vs_throughput: figures/vep_performance_vs_throughput.pdf figures/vep_performance_vs_throughput.svg
fetch-figure-vep_performance_vs_throughput:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py vep_performance_vs_throughput
.PHONY: figure-vep_performance_vs_throughput fetch-figure-vep_performance_vs_throughput
figures/mixture_zero_shot_trajectories.pdf figures/mixture_zero_shot_trajectories.svg &:
	$(FIGURE_PYTHON) src/build_figure.py mixture_zero_shot_trajectories
figure-mixture_zero_shot_trajectories:
	$(FIGURE_PYTHON) src/build_figure.py mixture_zero_shot_trajectories
fetch-figure-mixture_zero_shot_trajectories:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py mixture_zero_shot_trajectories
.PHONY: figure-mixture_zero_shot_trajectories fetch-figure-mixture_zero_shot_trajectories
figures/mendelian_zero_shot_comparison.pdf figures/mendelian_zero_shot_comparison.svg &:
	$(FIGURE_PYTHON) src/build_figure.py mendelian_zero_shot_comparison
figure-mendelian_zero_shot_comparison:
	$(FIGURE_PYTHON) src/build_figure.py mendelian_zero_shot_comparison
fetch-figure-mendelian_zero_shot_comparison:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py mendelian_zero_shot_comparison
.PHONY: figure-mendelian_zero_shot_comparison fetch-figure-mendelian_zero_shot_comparison
figures/mixture_probe_trajectories.pdf figures/mixture_probe_trajectories.svg &:
	$(FIGURE_PYTHON) src/build_figure.py mixture_probe_trajectories
figure-mixture_probe_trajectories:
	$(FIGURE_PYTHON) src/build_figure.py mixture_probe_trajectories
fetch-figure-mixture_probe_trajectories:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py mixture_probe_trajectories
.PHONY: figure-mixture_probe_trajectories fetch-figure-mixture_probe_trajectories
figures/mendelian_probe_comparison.pdf figures/mendelian_probe_comparison.svg &:
	$(FIGURE_PYTHON) src/build_figure.py mendelian_probe_comparison
figure-mendelian_probe_comparison:
	$(FIGURE_PYTHON) src/build_figure.py mendelian_probe_comparison
fetch-figure-mendelian_probe_comparison:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py mendelian_probe_comparison
.PHONY: figure-mendelian_probe_comparison fetch-figure-mendelian_probe_comparison
figures/region_specialist_performance.pdf figures/region_specialist_performance.svg &:
	$(FIGURE_PYTHON) src/build_figure.py region_specialist_performance
figure-region_specialist_performance:
	$(FIGURE_PYTHON) src/build_figure.py region_specialist_performance
fetch-figure-region_specialist_performance:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py region_specialist_performance
.PHONY: figure-region_specialist_performance fetch-figure-region_specialist_performance
figures/upstream_coding_mixture_balance.pdf figures/upstream_coding_mixture_balance.svg &:
	$(FIGURE_PYTHON) src/build_figure.py upstream_coding_mixture_balance
figure-upstream_coding_mixture_balance:
	$(FIGURE_PYTHON) src/build_figure.py upstream_coding_mixture_balance
fetch-figure-upstream_coding_mixture_balance:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py upstream_coding_mixture_balance
.PHONY: figure-upstream_coding_mixture_balance fetch-figure-upstream_coding_mixture_balance
figures/mixture_tree.pdf figures/mixture_tree.svg &: $(FIGURE_CODE) plot_data/data_mixture_results.csv plot_data/data_mixture_history.csv
	$(FIGURE_PYTHON) src/build_figure.py mixture_tree
figure-mixture_tree: figures/mixture_tree.pdf figures/mixture_tree.svg
fetch-figure-mixture_tree:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py mixture_tree
.PHONY: figure-mixture_tree fetch-figure-mixture_tree
figures/cooldown_effects.csv figures/cooldown_effects.md &:
	$(FIGURE_PYTHON) src/build_figure.py cooldown_effects
figure-cooldown_effects:
	$(FIGURE_PYTHON) src/build_figure.py cooldown_effects
fetch-figure-cooldown_effects:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py cooldown_effects
.PHONY: figure-cooldown_effects fetch-figure-cooldown_effects
figures/pooled_vs_unpooled_raw.pdf figures/pooled_vs_unpooled_raw.svg figures/pooled_vs_unpooled_matched.pdf figures/pooled_vs_unpooled_matched.svg &:
	$(FIGURE_PYTHON) src/build_figure.py pooled_vs_unpooled
figure-pooled_vs_unpooled:
	$(FIGURE_PYTHON) src/build_figure.py pooled_vs_unpooled
fetch-figure-pooled_vs_unpooled:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py pooled_vs_unpooled
.PHONY: figure-pooled_vs_unpooled fetch-figure-pooled_vs_unpooled
figures/region_loss_scaling.pdf figures/region_loss_scaling.svg &:
	$(FIGURE_PYTHON) src/build_figure.py region_loss_scaling
figure-region_loss_scaling:
	$(FIGURE_PYTHON) src/build_figure.py region_loss_scaling
fetch-figure-region_loss_scaling:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py region_loss_scaling
.PHONY: figure-region_loss_scaling fetch-figure-region_loss_scaling
figures/training_dataset_provenance.pdf figures/training_dataset_provenance.svg &: $(FIGURE_CODE) plot_data/artwork/data_provenance_training_datasets.svg
	$(FIGURE_PYTHON) src/build_figure.py training_dataset_provenance
figure-training_dataset_provenance: figures/training_dataset_provenance.pdf figures/training_dataset_provenance.svg
fetch-figure-training_dataset_provenance:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py training_dataset_provenance
.PHONY: figure-training_dataset_provenance fetch-figure-training_dataset_provenance
figures/evaluation_benchmarks.pdf figures/evaluation_benchmarks.svg &: $(FIGURE_CODE) plot_data/artwork/eval_datasets.svg
	$(FIGURE_PYTHON) src/build_figure.py evaluation_benchmarks
figure-evaluation_benchmarks: figures/evaluation_benchmarks.pdf figures/evaluation_benchmarks.svg
fetch-figure-evaluation_benchmarks:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py evaluation_benchmarks
.PHONY: figure-evaluation_benchmarks fetch-figure-evaluation_benchmarks
figures/variant_scoring_workflow.pdf figures/variant_scoring_workflow.svg &: $(FIGURE_CODE) plot_data/artwork/eval_apparatus.svg
	$(FIGURE_PYTHON) src/build_figure.py variant_scoring_workflow
figure-variant_scoring_workflow: figures/variant_scoring_workflow.pdf figures/variant_scoring_workflow.svg
fetch-figure-variant_scoring_workflow:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py variant_scoring_workflow
.PHONY: figure-variant_scoring_workflow fetch-figure-variant_scoring_workflow
figures/annotation_derived_training_pool.pdf figures/annotation_derived_training_pool.svg &: $(FIGURE_CODE) plot_data/artwork/annotation_derived_training_pool.svg
	$(FIGURE_PYTHON) src/build_figure.py annotation_derived_training_pool
figure-annotation_derived_training_pool: figures/annotation_derived_training_pool.pdf figures/annotation_derived_training_pool.svg
fetch-figure-annotation_derived_training_pool:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py annotation_derived_training_pool
.PHONY: figure-annotation_derived_training_pool fetch-figure-annotation_derived_training_pool
figures/hyperparameter_transfer_workflow.pdf figures/hyperparameter_transfer_workflow.svg &: $(FIGURE_CODE) plot_data/artwork/parameter_transfer_methodology_v1.svg
	$(FIGURE_PYTHON) src/build_figure.py hyperparameter_transfer_workflow
figure-hyperparameter_transfer_workflow: figures/hyperparameter_transfer_workflow.pdf figures/hyperparameter_transfer_workflow.svg
fetch-figure-hyperparameter_transfer_workflow:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py hyperparameter_transfer_workflow
.PHONY: figure-hyperparameter_transfer_workflow fetch-figure-hyperparameter_transfer_workflow
figures/training_mixture_histories.pdf figures/training_mixture_histories.svg &: $(FIGURE_CODE) plot_data/artwork/continued_training_data_exposures.svg
	$(FIGURE_PYTHON) src/build_figure.py training_mixture_histories
figure-training_mixture_histories: figures/training_mixture_histories.pdf figures/training_mixture_histories.svg
fetch-figure-training_mixture_histories:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py training_mixture_histories
.PHONY: figure-training_mixture_histories fetch-figure-training_mixture_histories
figures/multispecies_training_sequences.pdf figures/multispecies_training_sequences.svg figures/next_base_prediction.pdf figures/next_base_prediction.svg &: $(FIGURE_CODE)
	$(FIGURE_PYTHON) src/build_figure.py model_overview
figure-model_overview: figures/multispecies_training_sequences.pdf figures/multispecies_training_sequences.svg figures/next_base_prediction.pdf figures/next_base_prediction.svg
fetch-figure-model_overview:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py model_overview
.PHONY: figure-model_overview fetch-figure-model_overview
figures/mixture_continuation_design.pdf figures/mixture_continuation_design.svg &: $(FIGURE_CODE) plot_data/artwork/mini_fig9_mixture.svg
	$(FIGURE_PYTHON) src/build_figure.py mixture_continuation_design
figure-mixture_continuation_design: figures/mixture_continuation_design.pdf figures/mixture_continuation_design.svg
fetch-figure-mixture_continuation_design:
	FIGURE_FETCH=1 $(FIGURE_PYTHON) src/build_figure.py mixture_continuation_design
.PHONY: figure-mixture_continuation_design fetch-figure-mixture_continuation_design
LOCAL_FIGURES := figures/learning_rate_transfer.pdf figures/learning_rate_transfer.svg figures/optimizer_parameter_transfer.pdf figures/optimizer_parameter_transfer.svg figures/regional_hyperparameter_transfer.pdf figures/regional_hyperparameter_transfer.svg figures/sequence_loss_scaling.pdf figures/sequence_loss_scaling.svg figures/upstream_reweighting.pdf figures/upstream_reweighting.svg figures/historical_lineage_trajectories.pdf figures/historical_lineage_trajectories.svg figures/historical_leaderboard.pdf figures/historical_leaderboard.svg figures/vep_performance_vs_throughput.pdf figures/vep_performance_vs_throughput.svg figures/mixture_tree.pdf figures/mixture_tree.svg figures/training_dataset_provenance.pdf figures/training_dataset_provenance.svg figures/evaluation_benchmarks.pdf figures/evaluation_benchmarks.svg figures/variant_scoring_workflow.pdf figures/variant_scoring_workflow.svg figures/annotation_derived_training_pool.pdf figures/annotation_derived_training_pool.svg figures/hyperparameter_transfer_workflow.pdf figures/hyperparameter_transfer_workflow.svg figures/training_mixture_histories.pdf figures/training_mixture_histories.svg figures/multispecies_training_sequences.pdf figures/multispecies_training_sequences.svg figures/next_base_prediction.pdf figures/next_base_prediction.svg figures/mixture_continuation_design.pdf figures/mixture_continuation_design.svg
REPRODUCIBLE_FIGURES := figures/learning_rate_transfer.pdf figures/learning_rate_transfer.svg figures/optimizer_parameter_transfer.pdf figures/optimizer_parameter_transfer.svg figures/regional_hyperparameter_transfer.pdf figures/regional_hyperparameter_transfer.svg figures/sequence_loss_scaling.pdf figures/sequence_loss_scaling.svg figures/upstream_reweighting.pdf figures/upstream_reweighting.svg figures/historical_lineage_trajectories.pdf figures/historical_lineage_trajectories.svg figures/historical_leaderboard.pdf figures/historical_leaderboard.svg figures/vep_performance_vs_throughput.pdf figures/vep_performance_vs_throughput.svg figures/mixture_tree.pdf figures/mixture_tree.svg figures/training_dataset_provenance.pdf figures/training_dataset_provenance.svg figures/evaluation_benchmarks.pdf figures/evaluation_benchmarks.svg figures/variant_scoring_workflow.pdf figures/variant_scoring_workflow.svg figures/annotation_derived_training_pool.pdf figures/annotation_derived_training_pool.svg figures/hyperparameter_transfer_workflow.pdf figures/hyperparameter_transfer_workflow.svg figures/training_mixture_histories.pdf figures/training_mixture_histories.svg figures/multispecies_training_sequences.pdf figures/multispecies_training_sequences.svg figures/next_base_prediction.pdf figures/next_base_prediction.svg figures/mixture_continuation_design.pdf figures/mixture_continuation_design.svg
