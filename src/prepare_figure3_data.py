"""Export measured Figure 3 plotting data from AWS metric Parquets.

Run with --fetch to download missing inputs, or without it to use cached inputs.
The existing lineage selection is shared with plot.py so the original S5
checkpoint selection, branch cutoffs, and stored macro-average SEs are preserved.
Values and SEs are fractions (not percentages); token counts are absolute.
"""

from pathlib import Path
import tempfile

import click
import numpy as np
import polars as pl

import local_data
import plot


def prepare_endpoints() -> pl.DataFrame:
    """Four step-22,000 conditions, two measured classes, and their mean."""
    frames = []
    sources = {}
    for arm, config in plot.ARMS.items():
        uri = f"{plot.S3_BASE}/{config.prefix}-step-22000/mendelian_traits.parquet"
        sources[arm] = uri
        frame = local_data.read_polars(uri).filter(
            (pl.col("score_type") == plot.SCORE_TYPE)
            & pl.col("subset").is_in(list(plot.BALANCE_SUBSETS.values()))
        )
        frames.append(frame.with_columns(
            pl.lit(arm).alias("arm"), pl.lit(22000).alias("step")
        ))
    endpoints = plot.balance_endpoint_data(pl.concat(frames))
    rows = []
    for row in endpoints.iter_rows(named=True):
        rows.append({
            **row, "step": 22000, "score_type": plot.SCORE_TYPE,
            "aggregation": ("unweighted_two_class_mean"
                            if row["subset"] == "two_class_mean" else "measured"),
            "source_uri": sources[row["arm"]],
        })
    return pl.DataFrame(rows).sort("arm", "subset")


def prepare_trajectories() -> pl.DataFrame:
    """Exactly the checkpoints drawn in Panel B, with stored uncertainty."""
    results = local_data.load_mixture().set_index("mix")
    own_tokens = {mix: float(results.loc[mix, "tokens"]) for mix in results.index}
    staged = "exp135-zoonomia-m5.1"
    token_cutoff = max(row[0] for row in plot._composed_checkpoints(staged, results, own_tokens))
    transition_tokens = sum(local_data.inherited_components(staged, own_tokens).values())
    rows = []
    for leaf, _label, _color, _marker in plot.LINEAGES:
        checkpoints = plot._composed_checkpoints(leaf, results, own_tokens)
        keep = plot._lineage_keep(np.array([row[0] for row in checkpoints]), leaf, token_cutoff)
        for (tokens, mix, step), selected in zip(checkpoints, keep, strict=True):
            if not selected:
                continue
            model_id = plot._model_id(mix, step)
            metrics = plot._read_checkpoint("llr", mix, step)
            for subset in plot.ALL_METRIC_SUBSETS:
                value, se = metrics[subset]
                rows.append({
                    "lineage": leaf, "subset": subset,
                    "model_id": model_id, "step": step, "tokens": tokens,
                    "value": value, "se": se,
                    "transition_tokens": transition_tokens,
                    "score_type": plot.SCORE_TYPE, "split": "train",
                    "source_uri": f"{plot.RESULTS_ROOT}/metrics/{model_id}/mendelian_traits.parquet",
                })
    frame = pl.DataFrame(rows).sort("lineage", "subset", "tokens")
    assert frame.height == 216
    assert frame.select("lineage", "subset", "tokens").unique().height == frame.height
    return frame


@click.command()
@click.option("--fetch", is_flag=True, help="Download missing AWS snapshots with configured credentials.")
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False),
              default=local_data.ROOT, show_default=True)
def main(fetch: bool, output_dir: Path) -> None:
    """Write figure3_endpoints.csv and figure3_trajectories.csv."""
    local_data.FETCH = fetch
    try:
        endpoints = prepare_endpoints()
        trajectories = prepare_trajectories()
    except local_data.MissingData as exc:
        raise click.ClickException(str(exc)) from exc
    # Finish loading and validating both panels before touching existing CSVs.
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=".figure3-") as directory:
        staging = Path(directory)
        for name, frame in [("figure3_endpoints.csv", endpoints),
                            ("figure3_trajectories.csv", trajectories)]:
            frame.write_csv(staging / name)
        for path in sorted(staging.iterdir()):
            path.replace(output_dir / path.name)
    click.echo(f"Wrote {endpoints.height} endpoints/means and {trajectories.height} trajectory rows to {output_dir}")


if __name__ == "__main__":
    main()
