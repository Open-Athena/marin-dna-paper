# Paper figure data

The six CSV snapshots and model registry come from Open-Athena/marin-dna commit
`d8c4803cbbbffafb24890cd0c75134d78368d55c` (preserved blog branch). The original
plotting project is eric-czech/marin-dna-post-202606 commit
`2abef91b37a16fde9c9cdf1cfa0046942442b97f`.

[UPSTREAM.md](UPSTREAM.md) preserves the historical inventory and scientific
caveats; its paths describe the upstream project, not the current paper build.
[UPSTREAM_LICENSE](UPSTREAM_LICENSE) retains the source license. The historical
recipes and data used only by figures outside this paper have been removed.

## Bundled inputs

| Input | Paper figure commands |
|---|---|
| `transfer_validation_results.csv` | `learning_rate_transfer`, `optimizer_parameter_transfer`, `regional_hyperparameter_transfer` |
| `parameter_scaling_results.csv`, `parameter_scaling_history.csv` | `sequence_loss_scaling` |
| `model_leaderboard.csv`, `inference_costs.csv` | `vep_performance_vs_throughput` |
| `artwork/<figure>.svg` | `training_dataset_provenance`, `evaluation_benchmarks`, `variant_scoring_workflow`, `annotation_derived_training_pool`, `hyperparameter_transfer_workflow`, `training_mixture_histories` |
| No numeric inputs (schematic definitions in `src/plot.py`) | `multispecies_training_sequences`, `next_base_prediction` |

The six authored SVG sources use the same filenames as their paper outputs.
Their contents are preserved from the blog snapshot. The dated leaderboard CSV is
used only for the throughput comparison; it is never substituted for offline
prediction or metric inputs.

## Figures requiring external snapshots

| Figure commands | Inputs |
|---|---|
| `vep_by_model_size`, `vep_by_sequence_likelihood` | `parameter_scaling_results.csv` and paired endpoint prediction Parquets for eight models, each with Mendelian and SGE data |
| `missense_readout_scaling` | The same endpoint predictions plus commit-pinned Evo 2 probe metrics |
| `mixture_zero_shot_trajectories`, `mixture_probe_trajectories` | `data_mixture_results.csv`, lineage definitions, and checkpoint metric/probe-metric Parquets |
| `mendelian_zero_shot_comparison`, `mendelian_probe_comparison` | `models.yaml` and model-specific Parquets, including pinned Evo 2 metrics |
| `region_specialist_performance` | `models.yaml` and specialist, GPN-Star, and Evo 2 metric Parquets |
| `upstream_coding_mixture_balance` | Upstream/CDS experiment checkpoint metric Parquets |

Source URLs and checkpoint selections live beside their loaders in `src/plot.py`
and `src/local_data.py`. Remote paths map predictably to
`remote/<scheme>/<host>/<path>`; a missing-input error shows both paths.
Use `make fetch-figure-<name>` or `python src/plot.py --fetch <name>` in the locked
environment to retrieve missing inputs explicitly. Downloads are gitignored.
Source S3 keys are not immutable versions: preserve retrieval provenance alongside
any snapshots subsequently committed for publication.

`make fetch-sweeps` explicitly refreshes the four W&B-derived CSVs: transfer
results, parameter-scaling results/history, and mixture results. It does not
refresh the dated leaderboard or inference-cost snapshots.

The refactor preserves paired-readout filtering, chromosome weighting, bootstrap
uncertainty, and checkpoint omissions, including the omitted final pre-shift
m5.1 checkpoint. No numeric data is inferred from rendered charts.
