# Figure data provenance

The seven CSV snapshots, model registry, plotting recipes, shared typography, and evaluation helpers were copied from Open-Athena/marin-dna commit `d8c4803cbbbffafb24890cd0c75134d78368d55c` (preserved blog branch).
The plotting project originally came from eric-czech/marin-dna-post-202606 commit `2abef91b37a16fde9c9cdf1cfa0046942442b97f`.
`UPSTREAM.md` preserves its original figure inventory and scientific caveats; its execution paths describe the upstream project, not this port.
The source license is retained in `UPSTREAM_LICENSE`.

`figures.json` records recipe-to-output mappings and bundled dependencies.
The dated `model_leaderboard.csv` supplies the historical leaderboard and throughput plot only; it is not substituted for current offline evaluation metrics.
Paired scaling readouts, mixture trajectories, specialists, upstream/CDS balance, and current leaderboards require their original Parquet inputs.
The three W&B appendix analyses require explicit snapshot retrieval.
The port preserves original filters, uncertainty calculations, and checkpoint omissions, including the omitted final pre-shift m5.1 checkpoint.
No numeric data is inferred from rendered charts.

`artwork/` contains authored SVG sources copied from this paper's blog snapshot, with the matching upstream filenames.
Downloaded Parquets preserve their remote hierarchy under `remote/`.
W&B function-result snapshots use JSON under `wandb/` and retain Series/DataFrame indexes and array structure.
Keep retrieved snapshot provenance alongside any snapshots subsequently committed for publication; remote S3 keys themselves are not immutable versions.
