"""Prepare checked-in plotting CSVs from original measurements.

Only this entry point processes S3 metric/prediction data. Plotting commands
consume the exported CSVs and never fetch remote snapshots. Use --fetch to
retrieve missing source files with the active AWS profile; default is offline.
"""
from __future__ import annotations

import functools
from functools import lru_cache
import sys
import tempfile
from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import polars as pl

import local_data
import metrics as _metrics
import plot
from local_data import Model, ALL_DATASETS, models_for_dataset
from plot import (
    ALL_METRIC_SUBSETS,
    ARMS,
    BALANCE_SUBSETS,
    BASELINE_IDS,
    BY_MIX,
    EVO2,
    EVO2_MODELS,
    FAMILIES,
    FINAL_STEP,
    LOSS_COLUMNS,
    MARIN,
    MATCHED_LOSS_COLUMN,
    MISSENSE_DATASET,
    PANEL_ORDER,
    PANEL_REGIONS,
    READOUTS,
    ROLE_ORDER,
    SCALING_SUBSETS,
    SCORE_TYPE,
    SHORT_NAMES,
    SPECIALIST_DATASET,
    SPECIALIST_IDS,
    SPECIALIST_SUBSET_DISPLAY,
    SUBSET,
    TOKENS_PER_STEP,
    WORLD_CONFIG
)

PROBE_RESULTS = (
    "s3://oa-bolinas/snakemake/analysis/evals_v2/results/probe/"
    "{model}/{dataset}.parquet"
)


RESULTS_ROOT = "s3://oa-bolinas/snakemake/analysis/evals_v2/results"


S3_BASE = "s3://oa-bolinas/snakemake/analysis/evals_v2/results/metrics"


S3 = "s3://oa-bolinas"

SPLIT = "train"

EVO2_METRICS_GIST_OWNER = "gonzalobenegas"

EVO2_METRICS_GIST_ID = "3649e68fb63ca1f3443e4486078eb4d8"

EVO2_METRICS_GIST_COMMIT = "1bce02fe0d831382d24ecbac305d401f153c65fc"

EVO2_METRICS_GIST_BASE = (
    f"https://gist.githubusercontent.com/{EVO2_METRICS_GIST_OWNER}/"
    f"{EVO2_METRICS_GIST_ID}/raw/{EVO2_METRICS_GIST_COMMIT}"
)

EVO2_PROBE_METRICS_GIST_COMMIT = "3ac60cd5e37148fd135c4d55bed12386dc065dc6"

EVO2_PROBE_METRICS_GIST_BASE = (
    f"https://gist.githubusercontent.com/{EVO2_METRICS_GIST_OWNER}/"
    f"{EVO2_METRICS_GIST_ID}/raw/{EVO2_PROBE_METRICS_GIST_COMMIT}"
)

EVO2_DATASET_SHORT: dict[str, str] = {
    "mendelian_traits": "mendelian",
}

PROTOCOLS: dict[str, dict[str, dict[str, str]]] = {
    "marin_dna": {
        # Default LLR / JSD pick the FWD+RC `_avg` columns. The per-strand
        # `_fwd` variants are surfaced on the dashboard's Protocols page
        # for AVG-vs-FWD exploration; they're not exposed in the
        # leaderboards' protocol toggle (see `PROTOCOL_OPTIONS` in
        # `dashboard/src/components/controls.js`). The `_rc` columns are
        # still in the parquet for diagnostics but aren't surfaced.
        "LLR": {
            "mendelian_traits": "minus_llr_avg",
            "complex_traits": "abs_llr_avg",
        },
        "JSD": {
            "mendelian_traits": "jsd_avg",
            "complex_traits": "jsd_avg",
        },
        "LLR-FWD": {
            "mendelian_traits": "minus_llr_fwd",
            "complex_traits": "abs_llr_fwd",
        },
        "JSD-FWD": {
            "mendelian_traits": "jsd_fwd",
            "complex_traits": "jsd_fwd",
        },
    },
    "conservation": {
        "score": {d: "score" for d in ALL_DATASETS},
    },
    "alphagenome": {
        "L2": {d: "alphagenome_max_l2" for d in ALL_DATASETS},
    },
    "gpn_star": {
        "cLLR": {
            "mendelian_traits": "minus_llr_calibrated",
            "complex_traits": "abs_llr_calibrated",
        },
        "LLR": {
            "mendelian_traits": "minus_llr",
            "complex_traits": "abs_llr",
        },
    },
    # evo2 mirrors marin_dna; mendelian_traits only (extend when complex lands).
    "evo2": {
        "LLR": {
            "mendelian_traits": "minus_llr_avg",
        },
        "JSD": {
            "mendelian_traits": "jsd_avg",
        },
        "LLR-FWD": {
            "mendelian_traits": "minus_llr_fwd",
        },
        "JSD-FWD": {
            "mendelian_traits": "jsd_fwd",
        },
    },
}

DEFAULT_PROTOCOL: dict[str, str] = {
    "marin_dna": "LLR",
    "conservation": "score",
    "alphagenome": "L2",
    "gpn_star": "cLLR",
    "evo2": "LLR",
}


@functools.lru_cache(maxsize=None)
def _read_parquet(path: str) -> pl.DataFrame:
    """Cached S3/gist read so families that share a per-dataset parquet
    (conservation, alphagenome, gpn_star) only fetch once per process."""
    return local_data.read_polars(path)


def _parquet_path(method: Model, dataset: str) -> str:
    match method.family:
        case "marin_dna":
            return (
                f"{S3}/snakemake/analysis/evals_v2/results/metrics/"
                f"{method.id}/{dataset}.parquet"
            )
        case "conservation":
            return (
                f"{S3}/snakemake/conservation_eval/results/"
                f"{dataset}/metrics_{SPLIT}.parquet"
            )
        case "alphagenome":
            return f"{S3}/snakemake/alphagenome_eval/results/metrics/{dataset}.parquet"
        case "gpn_star":
            return f"{S3}/snakemake/gpn_star_eval/results/metrics/{dataset}.parquet"
        case "evo2":
            short = EVO2_DATASET_SHORT[dataset]
            return f"{EVO2_METRICS_GIST_BASE}/{short}_{method.id}_train_metrics.parquet"
        case _:
            raise ValueError(f"unknown family {method.family!r}")


def fetch_method_metrics(
    method: Model, dataset: str, protocol: str | None = None
) -> pl.DataFrame:
    """Return rows ``[subset, value, se, n, n_positives]`` for one
    ``(method, dataset, protocol)``.

    Matched-pair datasets (mendelian_traits, complex_traits) emit one row per
    consequence ``subset`` plus the ``_global_`` / ``_macro_avg_`` aggregates;
    ``subset`` carries the consequence name and ``n`` is total variants (or K
    on the macro row). See ``normalized_rows`` for the column semantics.

    When ``protocol`` is ``None``, defaults to ``DEFAULT_PROTOCOL[family]``.
    """
    assert dataset in method.datasets, (
        f"{method.id!r} is not registered for dataset {dataset!r}"
    )
    protocol = protocol or DEFAULT_PROTOCOL[method.family]
    assert protocol in PROTOCOLS[method.family], (
        f"unknown protocol {protocol!r} for family {method.family!r}; "
        f"options: {list(PROTOCOLS[method.family])}"
    )
    score_type = PROTOCOLS[method.family][protocol][dataset]
    path = _parquet_path(method, dataset)
    df = _read_parquet(path)
    match method.family:
        case "marin_dna" | "alphagenome" | "evo2":
            df = df.filter(pl.col("score_type") == score_type).filter(
                pl.col("split") == SPLIT
            )
        case "conservation":
            df = df.filter(pl.col("score_name") == method.id)
        case "gpn_star":
            df = df.filter(pl.col("score_type") == score_type).filter(
                pl.col("model") == method.id
            )
        case _:
            raise ValueError(f"unknown family {method.family!r}")
    if df.height == 0:
        raise LookupError(
            f"no metrics rows for {method.id!r} on {dataset!r} with protocol "
            f"{protocol!r} (score_type={score_type!r}) in {path}. The pipeline "
            f"may need to be re-run with this protocol included."
        )
    df = df.with_columns(
        pl.when(pl.col("subset") == _metrics.MACRO_AVG_SUBSET)
        .then(pl.col("n_groups"))
        .otherwise(pl.col("n_rows"))
        .alias("n"),
        pl.col("n_groups").alias("n_positives"),
    )
    return df.select(["subset", "value", "se", "n", "n_positives"])


def normalized_rows(
    dataset: str, displays: frozenset[str] | None = None
) -> pl.DataFrame:
    """Long-form table of all metrics for one dataset.

    Emits one row per ``(method, protocol, subset)`` — so each method
    contributes one row block per protocol registered for its family.

    Protocols whose metrics aren't in the parquet yet (e.g. the JSD path
    before ``compute_metrics`` has been re-run with the new score column)
    log a warning and are skipped rather than failing the build.

    Columns:
      - ``method_id``       — ``Model.id`` (primary key for a row's method)
      - ``method_display``  — ``Model.display``
      - ``family``          — ``Model.family``
      - ``protocol``        — protocol name (e.g. ``LLR``, ``JSD``, ``cLLR``)
      - ``subset``          — consequence subset OR ``_global_`` /
        ``_macro_avg_`` (matched-pair).
      - ``value``           — AUPRC.
      - ``se``              — cluster-bootstrap SE.
      - ``n``               — total variants in the subset (positives +
        matched negatives), except on the ``_macro_avg_`` row where it
        carries K = number of qualifying subsets.
      - ``n_positives``     — positives in the subset (= ``n_groups`` from
        the source AUPRC parquet); K on the macro row. Drives the dashboard's
        ≥30-positives display threshold; never rendered.
    """
    # Soft-fail surface, intentionally narrow: only the two legitimate
    # "no data for this protocol yet" exception types.
    #   * `LookupError` — `fetch_method_metrics` raises this when the
    #     parquet exists but has no rows for the requested protocol's
    #     `score_type` (e.g. marin_dna JSD before `metrics.smk` was rerun
    #     with the new score column).
    #   * `pl.exceptions.ComputeError` — `pl.read_parquet` raises this
    #     when the parquet file isn't on S3 yet (e.g. a freshly added
    #     external-family entry that the eval pipeline hasn't produced
    #     output for). `FileNotFoundError` covers the local-path case in
    #     tests.
    # Everything else (`AssertionError`, `KeyError` from a malformed
    # registry, `ValueError`, etc.) propagates so config bugs fail loud
    # instead of yielding a silently-empty dashboard.
    soft_fail = (LookupError, pl.exceptions.ComputeError, FileNotFoundError)
    rows: list[dict] = []
    for method in models_for_dataset(dataset):
        if displays is not None and method.display not in displays:
            continue
        for protocol in PROTOCOLS[method.family]:
            try:
                df = fetch_method_metrics(method, dataset, protocol)
            except soft_fail as exc:
                print(
                    f"  ! {method.family}/{protocol} skip for {method.id} "
                    f"({dataset}): {exc}",
                    file=sys.stderr,
                )
                continue
            for row in df.iter_rows(named=True):
                rows.append(
                    {
                        "method_id": method.id,
                        "method_display": method.display,
                        "family": method.family,
                        "protocol": protocol,
                        "subset": row["subset"],
                        "value": float(row["value"]),
                        "se": float(row["se"]),
                        "n": int(row["n"]),
                        "n_positives": int(row["n_positives"]),
                    }
                )
    return pl.DataFrame(rows)


_PROBE_ROW_SCHEMA: dict[str, pl.DataType] = {
    "method_id": pl.String,
    "method_display": pl.String,
    "family": pl.String,
    "protocol": pl.String,
    "subset": pl.String,
    "value": pl.Float64,
    "se": pl.Float64,
    "n": pl.Int64,
    "n_positives": pl.Int64,
}

PROBE_FAMILIES = ("marin_dna", "evo2")


def _probe_parquet_path(method: Model, dataset: str) -> str:
    """Path to one model's frozen-embedding probe-metrics parquet, dispatched by family.

    ``marin_dna`` probes are the ``evals_v2`` pipeline's ``probe_metrics`` output on S3; ``evo2``
    probes are computed off-pipeline and published to the pinned Evo 2 gist, mirroring
    ``_parquet_path``'s zero-shot ``case "evo2"``.
    """
    match method.family:
        case "marin_dna":
            return (
                f"{S3}/snakemake/analysis/evals_v2/results/probe_metrics/"
                f"{method.id}/{dataset}.parquet"
            )
        case "evo2":
            short = EVO2_DATASET_SHORT[dataset]
            return (
                f"{EVO2_PROBE_METRICS_GIST_BASE}/"
                f"{short}_{method.id}_train_probe_metrics.parquet"
            )
        case _:
            raise ValueError(f"family {method.family!r} has no probe-metrics source")


def probe_normalized_rows(
    dataset: str, displays: frozenset[str] | None = None
) -> pl.DataFrame:
    """Long-form linear-probe (supervised) rows for the dashboard's Supervised mode.

    Reads the frozen-embedding probe metrics (per-subset per-chromosome-weighted AUPRC +
    chromosome-cluster bootstrap SE + a ``_macro_avg_`` row, #347) for every probe-capable model
    (``PROBE_FAMILIES``) registered for ``dataset``, keeps the ``probe_score`` rows on the
    ``train`` split, and maps them onto the same schema as ``normalized_rows`` so the leaderboard
    heatmap renders them unchanged. Source is per family (``_probe_parquet_path``): ``marin_dna``
    from the ``evals_v2`` pipeline on S3, ``evo2`` from the pinned Evo 2 gist.

    Only families with per-allele ``emb_ref``/``emb_alt`` embeddings have probes; zero-shot-only
    families contribute nothing here. Each family's metrics already carry the ``_macro_avg_``
    aggregate (**no** ``_global_`` — probe scores come from a separate per-subset classifier),
    so nothing is synthesized here. Models whose probe parquet isn't published yet are skipped
    with a warning (same soft-fail posture as ``normalized_rows``).

    Emits ``[method_id, method_display, family, protocol, subset, value, se, n, n_positives]``
    with ``protocol == "probe"`` (the Supervised view has a single protocol). On the
    ``_macro_avg_`` row ``n`` / ``n_positives`` are overloaded to K = the number of subsets
    entering the macro (the heatmap's "K subsets" header), mirroring ``fetch_method_metrics``'s
    matched-pair macro; per-subset ``n`` / ``n_positives`` are the probe parquet's total
    variants / positives.
    """
    # OSError covers the 404 case (a probe-capable model registered for the dataset whose probe
    # run hasn't landed — S3 for marin_dna, gist for evo2); the others mirror `normalized_rows`'
    # soft-fail set.
    soft_fail = (LookupError, pl.exceptions.ComputeError, FileNotFoundError, OSError)
    rows: list[dict] = []
    for method in models_for_dataset(dataset):
        if displays is not None and method.display not in displays:
            continue
        if method.family not in PROBE_FAMILIES:
            continue
        try:
            # Inside the try so a per-model path-construction LookupError (e.g. a probe family
            # registered for a dataset absent from `EVO2_DATASET_SHORT`) soft-fails that one
            # model — matching the zero-shot path — rather than aborting the whole build. A
            # non-probe family reaching `_probe_parquet_path` still raises `ValueError` (not in
            # `soft_fail`) so a `PROBE_FAMILIES`/dispatch mismatch fails loud.
            path = _probe_parquet_path(method, dataset)
            df = _read_parquet(path).filter(
                (pl.col("score_type") == "probe_score") & (pl.col("split") == SPLIT)
            )
        except soft_fail as exc:
            print(f"  ! probe skip for {method.id} ({dataset}): {exc}", file=sys.stderr)
            continue
        if df.height == 0:
            print(
                f"  ! probe no probe_score rows for {method.id} ({dataset}) in {path}",
                file=sys.stderr,
            )
            continue
        # Require the #347 schema — an `se` column and the pipeline-emitted `_macro_avg_`
        # row. A pre-#347 parquet (per-subset only, no SE, no aggregate — e.g. a #341
        # scaling-ladder run) is skipped rather than shown with blank error bars and no
        # Macro Avg column; re-run its `compute_probe_metrics` to surface it.
        if (
            "se" not in df.columns
            or not (df["subset"] == _metrics.MACRO_AVG_SUBSET).any()
        ):
            print(
                f"  ! probe stale-schema for {method.id} ({dataset}): missing se/_macro_avg_ "
                f"(pre-#347) — re-run compute_probe_metrics to surface it",
                file=sys.stderr,
            )
            continue
        # K = subsets entering the macro = per-subset rows with a finite value. A finite probe
        # value already implies the subset cleared `min_variants=300` (⇒ ≥30 positives under
        # 1:9), i.e. exactly the set the pipeline's macro `n_min` averaged and the heatmap
        # renders as columns — so no separate positives threshold is needed here. Overloaded
        # onto the macro row's n / n_positives (the "K subsets" header), matching
        # `fetch_method_metrics`'s matched-pair macro.
        k = df.filter(
            (pl.col("subset") != _metrics.MACRO_AVG_SUBSET)
            & pl.col("value").is_not_null()
        ).height
        for row in df.iter_rows(named=True):
            is_macro = row["subset"] == _metrics.MACRO_AVG_SUBSET
            rows.append(
                {
                    "method_id": method.id,
                    "method_display": method.display,
                    "family": method.family,
                    "protocol": "probe",
                    "subset": row["subset"],
                    "value": _nan_float(row["value"]),
                    "se": _nan_float(row["se"]),
                    "n": k if is_macro else int(row["n"]),
                    "n_positives": k if is_macro else int(row["n_pos"]),
                }
            )
    return pl.DataFrame(rows, schema=_PROBE_ROW_SCHEMA)


def _nan_float(value: object) -> float:
    """Preserve undefined metric values when Parquet nulls become None."""
    return float("nan") if value is None else float(value)

def _model_rows() -> pd.DataFrame:
    """Return the eight endpoint model IDs and their parameter counts."""
    metadata = pd.read_csv(local_data.SCALING_RESULTS_PATH)[
        ["run_name", "params"]
    ].copy()
    assert len(metadata) == 8, f"expected 8 scaling models, got {len(metadata)}"
    assert metadata["run_name"].is_unique
    assert metadata["params"].is_unique
    metadata["model"] = (
        metadata["run_name"].str.removeprefix("dna-bolinas-") + f"-step-{FINAL_STEP}"
    )
    return metadata.sort_values("params").reset_index(drop=True)


def _read_paired_metrics(model: str, dataset: str) -> pd.DataFrame:
    """Compute both readouts' AUPRC on identical finite probe-score rows."""
    path = PROBE_RESULTS.format(model=model, dataset=dataset)
    print(f"Reading {path}")
    data = local_data.read_pandas(path)
    required = {
        "label",
        "subset",
        "chrom",
        "probe_score",
        "llr_fwd",
        "llr_rc",
    }
    assert required.issubset(data.columns), (
        f"{path} missing columns {sorted(required - set(data.columns))}"
    )

    # A probe can be absent for a subset that failed its training-data gate.
    # Restrict both readouts to the exact rows on which the probe is defined.
    finite_probe = np.isfinite(data["probe_score"].to_numpy(dtype=float))
    data = data.loc[finite_probe].copy()
    assert len(data) > 0, f"{path} has no finite probe predictions"
    data["minus_llr_avg"] = (
        -(data["llr_fwd"].to_numpy(dtype=float) + data["llr_rc"].to_numpy(dtype=float))
        / 2
    )
    assert np.isfinite(data["minus_llr_avg"]).all()

    metrics = _metrics.per_chrom_ap_table(
        data,
        ["minus_llr_avg", "probe_score"],
        n_bootstrap=1000,
        rng=0,
    )
    wanted = {subset for subset, _ in SCALING_SUBSETS[dataset]}
    metrics = metrics[metrics["subset"].isin(wanted)].copy()
    expected = {(score, subset) for score, *_ in READOUTS for subset in wanted}
    observed = set(zip(metrics["score_type"], metrics["subset"], strict=True))
    assert observed == expected, (
        f"{model}/{dataset}: expected metric cells {sorted(expected)}, "
        f"got {sorted(observed)}"
    )
    assert metrics["value"].between(0, 1).all()
    assert metrics["se"].notna().all()
    return metrics


@lru_cache(maxsize=1)
def load_parameter_scaling_metrics() -> pd.DataFrame:
    """Load all eight endpoints for both datasets and both paired readouts."""
    frames: list[pd.DataFrame] = []
    for row in _model_rows().itertuples(index=False):
        for dataset in SCALING_SUBSETS:
            metrics = _read_paired_metrics(row.model, dataset)
            metrics["dataset"] = dataset
            metrics["params"] = int(row.params)
            metrics["model"] = row.model
            frames.append(metrics)
    result = pd.concat(frames, ignore_index=True)
    expected_rows = (
        8 * sum(len(subsets) for subsets in SCALING_SUBSETS.values()) * len(READOUTS)
    )
    assert len(result) == expected_rows, (
        f"expected {expected_rows} endpoint metric rows, got {len(result)}"
    )
    return result


def load_loss_vs_auprc(
    results: pd.DataFrame | None = None,
    metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join paired AUPRCs to each consequence's matched validation LL."""
    if results is None:
        results = pd.read_csv(local_data.SCALING_RESULTS_PATH)
    if metrics is None:
        metrics = load_parameter_scaling_metrics()

    required_results = {"params", *LOSS_COLUMNS}
    assert required_results.issubset(results.columns), (
        f"scaling results missing {sorted(required_results - set(results.columns))}"
    )
    metadata = results[["params", *LOSS_COLUMNS]].copy()
    assert len(metadata) == 8
    assert metadata["params"].is_unique
    assert np.isfinite(metadata[list(LOSS_COLUMNS)].to_numpy(dtype=float)).all()

    data = metrics.merge(metadata, on="params", how="left", validate="many_to_one")
    assert len(data) == len(metrics)
    data["ll"] = np.nan
    for subset, loss_column in MATCHED_LOSS_COLUMN.items():
        mask = data["subset"] == subset
        data.loc[mask, "ll"] = -data.loc[mask, loss_column]
    assert data["ll"].notna().all()
    assert np.isfinite(data["ll"].to_numpy(dtype=float)).all()
    return data


def _evo2_path(model: str) -> str:
    short = EVO2_DATASET_SHORT[MISSENSE_DATASET]
    return f"{EVO2_PROBE_METRICS_GIST_BASE}/{short}_{model}_train_probe_metrics.parquet"


def load_missense_comparison(
    marin_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load paired missense AUPRCs for both model families."""
    if marin_metrics is None:
        marin_metrics = load_parameter_scaling_metrics()
    required = {"dataset", "subset", "score_type", "value", "se", "n", "params"}
    assert required.issubset(marin_metrics.columns), (
        f"MarinDNA metrics missing {sorted(required - set(marin_metrics.columns))}"
    )
    marin = marin_metrics[
        (marin_metrics["dataset"] == MISSENSE_DATASET)
        & (marin_metrics["subset"] == SUBSET)
    ][["score_type", "value", "se", "n", "params"]].copy()
    assert len(marin) == 8 * len(READOUTS)
    marin["family"] = MARIN

    evo_frames: list[pd.DataFrame] = []
    for model, params in EVO2_MODELS:
        path = _evo2_path(model)
        print(f"Reading {path}")
        frame = local_data.read_polars(path).filter(
            (pl.col("subset") == SUBSET)
            & pl.col("score_type").is_in([score for score, *_ in READOUTS])
            & (pl.col("split") == "train")
        )
        assert frame.height == len(READOUTS), (
            f"{model}: expected {len(READOUTS)} missense score rows, got {frame.height}"
        )
        evo = frame.select("score_type", "value", "se", "n").to_pandas()
        evo["params"] = params
        evo["family"] = EVO2
        evo_frames.append(evo)

    result = pd.concat([marin, *evo_frames], ignore_index=True)
    assert len(result) == (8 + len(EVO2_MODELS)) * len(READOUTS)
    assert result["value"].between(0, 1).all()
    assert result["se"].notna().all()
    assert (result["se"] >= 0).all()
    assert result["n"].nunique() == 1, (
        f"MarinDNA and Evo 2 missense rows differ: {sorted(result['n'].unique())}"
    )
    expected_cells = {
        (family, score) for family in FAMILIES for score, *_rest in READOUTS
    }
    assert (
        set(zip(result["family"], result["score_type"], strict=True)) == expected_cells
    )
    return result


def _model_id(mix: str, step: int) -> str:
    run = BY_MIX[mix]
    short = SHORT_NAMES.get(mix, mix)
    return f"mix-v0.9-p1B-i{run.index}-{short}-step-{step}"


@lru_cache(maxsize=None)
def _read_checkpoint(kind: str, mix: str, step: int) -> dict[str, tuple[float, float]]:
    """Return stored ``(AUPRC, SE)`` values for the eight subsets and their macro."""
    result_kind, score_type, _metric_label, _method_label = WORLD_CONFIG[kind]
    model_id = _model_id(mix, step)
    path = f"{RESULTS_ROOT}/{result_kind}/{model_id}/mendelian_traits.parquet"
    data = (
        local_data.read_polars(path)
        .filter(
            (pl.col("score_type") == score_type)
            & (pl.col("split") == "train")
            & pl.col("subset").is_in(ALL_METRIC_SUBSETS)
        )
        .select("subset", "value", "se")
    )
    assert data.height == len(ALL_METRIC_SUBSETS), (
        f"{model_id}/{kind}: expected {len(ALL_METRIC_SUBSETS)} metric rows, "
        f"got {data.height}"
    )
    assert data["subset"].n_unique() == len(ALL_METRIC_SUBSETS)
    assert data["value"].is_not_null().all()
    assert data["se"].is_not_null().all()
    assert data["value"].is_between(0, 1, closed="both").all()
    assert data["se"].is_between(0, 1, closed="both").all()
    return {subset: (float(value), float(se)) for subset, value, se in data.iter_rows()}


def _chain(leaf: str) -> list[str]:
    """Return the root-to-leaf stage sequence for one lineage."""
    chain: list[str] = []
    mix: str | None = leaf
    while mix is not None:
        chain.append(mix)
        mix = local_data.BY_MIX[mix].parent
    return chain[::-1]


def _phase_start_step(mix: str, results: pd.DataFrame) -> float:
    row = results.loc[mix]
    own_steps = round(float(row["tokens"]) / TOKENS_PER_STEP)
    return float(row["num_train_steps"]) - own_steps


def _composed_checkpoints(
    leaf: str, results: pd.DataFrame, own_tokens: dict[str, float]
) -> list[tuple[float, str, int]]:
    """Select inherited checkpoints and their cumulative-token coordinates."""
    chain = _chain(leaf)
    offsets = [
        sum(local_data.inherited_components(mix, own_tokens).values()) for mix in chain
    ]
    checkpoints = []
    for phase_index, mix in enumerate(chain):
        start = _phase_start_step(mix, results)
        cutoff = (
            offsets[phase_index + 1] + TOKENS_PER_STEP
            if phase_index + 1 < len(chain)
            else np.inf
        )
        for step in BY_MIX[mix].steps:
            # Preserve S5's omission of the point 0.9B tokens before the first
            # post-transition checkpoint, as requested for Figure 3.
            if (leaf == "exp135-zoonomia-m5.1"
                    and mix == "uniform_to_uniform_1" and step == 29579):
                continue
            tokens = offsets[phase_index] + (step - start) * TOKENS_PER_STEP
            if tokens <= cutoff:
                checkpoints.append((tokens, mix, step))
    checkpoints.sort()
    expected = 8 if leaf == "exp135-zoonomia-m5.1" else 9
    assert len(checkpoints) == expected
    assert np.all(np.diff([row[0] for row in checkpoints]) > 0)
    return checkpoints


def _lineage_keep(tokens: np.ndarray, leaf: str, token_cutoff: float) -> np.ndarray:
    """Retain the original S5 horizon, including the first control point beyond it."""
    keep = np.ones(len(tokens), dtype=bool)
    if leaf != "exp135-zoonomia-m5.1":
        keep = tokens <= token_cutoff
        if keep.any() and not keep.all():
            keep[np.argmax(~keep)] = True
    return keep


def load_plot_data() -> pd.DataFrame:
    """Load both specialists and both comparators for all five variant classes."""
    required_ids = {
        *SPECIALIST_IDS.values(),
        *BASELINE_IDS.values(),
    }
    models = {
        model.id: model
        for model in local_data.models_for_dataset(SPECIALIST_DATASET)
        if model.id in required_ids
    }
    assert set(models) == required_ids, (
        f"missing registered models: {sorted(required_ids - set(models))}"
    )

    records: list[dict[str, Any]] = []
    for subset in PANEL_ORDER:
        role_to_model_id = {
            **SPECIALIST_IDS,
            **BASELINE_IDS,
        }
        for role, model_id in role_to_model_id.items():
            model = models[model_id]
            metrics = fetch_method_metrics(model, SPECIALIST_DATASET)
            row = metrics.filter(metrics["subset"] == subset)
            assert row.height == 1, (
                f"expected one {subset!r} row for {model_id!r}, got {row.height}"
            )
            values = row.row(0, named=True)
            value = float(values["value"])
            se = float(values["se"])
            n = int(values["n"])
            n_positives = int(values["n_positives"])
            assert 0.0 <= value <= 1.0
            assert 0.0 <= se <= 1.0
            assert 0 < n_positives <= n
            records.append(
                {
                    "subset": subset,
                    "consequence": SPECIALIST_SUBSET_DISPLAY[subset],
                    "region": PANEL_REGIONS[subset],
                    "role": role,
                    "model_id": model_id,
                    "protocol": DEFAULT_PROTOCOL[model.family],
                    "auprc_pct": 100.0 * value,
                    "se_pct": 100.0 * se,
                    "n": n,
                    "n_positives": n_positives,
                }
            )

    frame = pd.DataFrame.from_records(records)
    expected_rows = len(PANEL_ORDER) * len(ROLE_ORDER)
    assert len(frame) == expected_rows
    assert not frame.isna().any().any()
    return frame


def balance_endpoint_data(data: pl.DataFrame) -> pl.DataFrame:
    """Select common-checkpoint values and SEs, including the two-class mean."""
    endpoint = data.filter(pl.col("step") == 22000)
    assert endpoint.height == 8
    assert endpoint.select("arm", "subset").unique().height == 8
    assert set(endpoint["arm"]) == set(ARMS)
    assert set(endpoint["subset"]) == set(BALANCE_SUBSETS.values())
    assert endpoint["value"].is_not_null().all()
    assert endpoint["value"].is_between(0, 1).all()
    assert endpoint["se"].is_not_null().all()
    assert endpoint["se"].is_finite().all() and (endpoint["se"] >= 0).all()
    # Match compute_auprc_metrics: the two subsets have disjoint match_group
    # bootstrap units, so the SE of their unweighted mean is sqrt(sum(SE²))/2.
    means = endpoint.group_by("arm").agg(
        pl.col("value").mean(),
        (pl.col("se").pow(2).sum().sqrt() / 2).alias("se"),
    ).with_columns(
        pl.lit("two_class_mean").alias("subset")
    )
    columns = ["arm", "subset", "value", "se"]
    return pl.concat([endpoint.select(columns), means.select(columns)])


def prepare_endpoints() -> pl.DataFrame:
    """Four step-22,000 conditions, two measured classes, and their mean."""
    frames = []
    sources = {}
    for arm, config in plot.ARMS.items():
        uri = f"{S3_BASE}/{config.prefix}-step-22000/mendelian_traits.parquet"
        sources[arm] = uri
        frame = local_data.read_polars(uri).filter(
            (pl.col("score_type") == plot.SCORE_TYPE)
            & pl.col("subset").is_in(list(plot.BALANCE_SUBSETS.values()))
        )
        frames.append(frame.with_columns(
            pl.lit(arm).alias("arm"), pl.lit(22000).alias("step")
        ))
    endpoints = balance_endpoint_data(pl.concat(frames))
    rows = []
    for row in endpoints.iter_rows(named=True):
        rows.append({
            **row, "step": 22000, "score_type": plot.SCORE_TYPE,
            "aggregation": ("unweighted_two_class_mean"
                            if row["subset"] == "two_class_mean" else "measured"),
            "source_uri": sources[row["arm"]],
        })
    return pl.DataFrame(rows).sort("arm", "subset")


def prepare_trajectories(kind: str = "llr") -> pl.DataFrame:
    """Exactly the checkpoints drawn in Panel B, with stored uncertainty."""
    results = local_data.load_mixture().set_index("mix")
    own_tokens = {mix: float(results.loc[mix, "tokens"]) for mix in results.index}
    staged = "exp135-zoonomia-m5.1"
    token_cutoff = max(row[0] for row in _composed_checkpoints(staged, results, own_tokens))
    transition_tokens = sum(local_data.inherited_components(staged, own_tokens).values())
    rows = []
    for leaf, _label, _color, _marker in plot.LINEAGES:
        checkpoints = _composed_checkpoints(leaf, results, own_tokens)
        keep = _lineage_keep(np.array([row[0] for row in checkpoints]), leaf, token_cutoff)
        for (tokens, mix, step), selected in zip(checkpoints, keep, strict=True):
            if not selected:
                continue
            model_id = _model_id(mix, step)
            metrics = _read_checkpoint(kind, mix, step)
            for subset in plot.ALL_METRIC_SUBSETS:
                value, se = metrics[subset]
                rows.append({
                    "lineage": leaf, "subset": subset,
                    "model_id": model_id, "step": step, "tokens": tokens,
                    "value": value, "se": se,
                    "transition_tokens": transition_tokens,
                    "score_type": plot.WORLD_CONFIG[kind][1], "split": "train",
                    "source_uri": f"{RESULTS_ROOT}/{plot.WORLD_CONFIG[kind][0]}/{model_id}/mendelian_traits.parquet",
                })
    frame = pl.DataFrame(rows).sort("lineage", "subset", "tokens")
    assert frame.height == 216
    assert frame.select("lineage", "subset", "tokens").unique().height == frame.height
    return frame


def prepare_dataset_timetree() -> pl.DataFrame:
    """Export collapsed dated trees and cohort counts as ordered CSV node rows."""
    import hashlib
    import urllib.parse
    import urllib.request
    import plot_dataset_timetree as tt

    trees = {}
    provenance = {}
    for rank in ("class", "order"):
        params = {"direct-download-format": "newick", "direct-download-id": "211973",
                  "direct-download-rank": rank}
        url = "https://www.timetree.org/ajax/direct_download?" + urllib.parse.urlencode(params)
        path = tt.CACHE / f"metazoa_{rank}.nwk"
        if not path.exists():
            if not local_data.FETCH:
                raise local_data.MissingData(f"Missing {path}; use process_data.py --fetch dataset_timetree")
            with urllib.request.urlopen(url, timeout=60) as response:
                content = response.read()
            tt.parse_newick(content.decode())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        content = path.read_bytes()
        tree = tt.parse_newick(content.decode())
        for node in tt.leaves(tree):
            node.name = tt.normalize(node.name)
        assert tree.age > 0
        trees[rank] = tree
        provenance[rank] = {"source_uri": url, "source_sha256": hashlib.sha256(content).hexdigest()}
    available = {node.name for node in tt.leaves(trees["class"])}
    selection = {key: set(names) & available for key, names in tt.ANIMAL_GROUPS.items()}
    assert all(selection.values())
    broad = tt.collapse(trees["class"], selection, tt.ANIMAL_LABELS)
    mammal_selection = {key: {key} for key in tt.MAMMAL_LABELS}
    mammal_selection["Afrosoricida"] = {"Tenrecidae (no order)", "Chrysochloridae (no order)"}
    mammal = tt.collapse(trees["order"], mammal_selection, tt.MAMMAL_LABELS)
    animal_counts, mammal_counts = tt.counts()
    rows = []
    for panel, rank, tree, counts in [("animals", "class", broad, animal_counts),
                                       ("mammals", "order", mammal, mammal_counts)]:
        assert {node.key for node in tt.leaves(tree)} == set(counts)
        def visit(node, parent_id):
            node_id = len(rows)
            annotation, projection = counts.get(node.key, (0, 0))
            rows.append({"panel": panel, "node_id": node_id, "parent_id": parent_id,
                         "name": node.name, "key": node.key, "age_ma": node.age,
                         "annotation": annotation, "projection": projection, **provenance[rank]})
            for child in node.children:
                visit(child, node_id)
        visit(tree, -1)
    return pl.DataFrame(rows)


def _with_metric_sources(frame: pl.DataFrame, *, probe=False) -> pl.DataFrame:
    models = {model.id: model for model in models_for_dataset("mendelian_traits")}
    path_fn = _probe_parquet_path if probe else _parquet_path
    return frame.with_columns(pl.col("method_id").replace_strict(
        {model_id: path_fn(models[model_id], "mendelian_traits")
         for model_id in frame["method_id"].unique()}
    ).alias("source_uri"))


def prepare_zero_shot_comparison() -> pl.DataFrame:
    rows = normalized_rows("mendelian_traits", plot.ZERO_SHOT_DISPLAYS).with_columns(
        pl.col("family").replace_strict(DEFAULT_PROTOCOL, default=None).alias("_default")
    ).filter(pl.col("protocol") == pl.col("_default")).drop("_default")
    plot._table_for_displays(rows, plot.ZERO_SHOT_DISPLAYS)
    return _with_metric_sources(rows)


def prepare_probe_comparison() -> pl.DataFrame:
    rows = probe_normalized_rows("mendelian_traits", plot.PROBE_DISPLAYS)
    plot._table_for_displays(rows, plot.PROBE_DISPLAYS)
    return _with_metric_sources(rows, probe=True)


def prepare_specialists() -> pd.DataFrame:
    frame = load_plot_data()
    models = {model.id: model for model in models_for_dataset(SPECIALIST_DATASET)}
    frame["source_uri"] = frame["model_id"].map(
        lambda model_id: _parquet_path(models[model_id], SPECIALIST_DATASET))
    return frame


def prepare_scaling() -> pd.DataFrame:
    frame = load_parameter_scaling_metrics().copy()
    frame["source_uri"] = [PROBE_RESULTS.format(model=row.model, dataset=row.dataset)
                           for row in frame.itertuples()]
    return frame


def prepare_missense() -> pd.DataFrame:
    frame = load_missense_comparison()
    model_ids = dict(zip(_model_rows()["params"], _model_rows()["model"], strict=True))
    evo_ids = {params: model for model, params in EVO2_MODELS}
    frame["source_uri"] = [
        PROBE_RESULTS.format(model=model_ids[row.params], dataset=MISSENSE_DATASET)
        if row.family == MARIN else _evo2_path(evo_ids[row.params])
        for row in frame.itertuples()
    ]
    return frame


# Keys match plot commands and output basenames. Shared frozen source snapshots
# remain inputs; each exported table is named for the figure that consumes it.
RECIPES = {
    "upstream_coding_mixture_balance": lambda: {
        "upstream_coding_mixture_balance_endpoints": prepare_endpoints(),
        "upstream_coding_mixture_balance_trajectories": prepare_trajectories(),
    },
    "vep_by_model_size": lambda: {"vep_by_model_size": prepare_scaling()},
    "vep_by_sequence_likelihood": lambda: {
        "vep_by_sequence_likelihood": load_loss_vs_auprc(metrics=prepare_scaling())},
    "missense_readout_scaling": lambda: {"missense_readout_scaling": prepare_missense()},
    "mixture_zero_shot_trajectories": lambda: {"mixture_zero_shot_trajectories": prepare_trajectories("llr")},
    "mixture_probe_trajectories": lambda: {"mixture_probe_trajectories": prepare_trajectories("probe")},
    "mendelian_zero_shot_comparison": lambda: {"mendelian_zero_shot_comparison": prepare_zero_shot_comparison()},
    "mendelian_probe_comparison": lambda: {"mendelian_probe_comparison": prepare_probe_comparison()},
    "region_specialist_performance": lambda: {"region_specialist_performance": prepare_specialists()},
    "hyperparameter_transfer": lambda: {"hyperparameter_transfer": local_data.load_transfer()},
    "regional_hyperparameter_transfer": lambda: {"regional_hyperparameter_transfer": local_data.load_transfer()},
    "sequence_loss_scaling": lambda: {
        "sequence_loss_scaling_results": local_data.load_scaling_results(),
        "sequence_loss_scaling_history": local_data.load_scaling_history(),
    },
    "vep_performance_vs_throughput": lambda: {
        "vep_performance_vs_throughput_leaderboard": local_data.load_leaderboard(),
        "vep_performance_vs_throughput_inference_costs": local_data.load_inference_costs(),
    },
    "dataset_timetree": lambda: {"dataset_timetree": prepare_dataset_timetree()},
}


def export(name: str, output_dir: Path) -> None:
    """Validate a complete figure's data before replacing any of its exports."""
    try:
        frames = RECIPES[name]()
    except local_data.MissingData as exc:
        raise click.ClickException(str(exc)) from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir, prefix=f".{name}-") as directory:
        staging = Path(directory)
        for stem, frame in frames.items():
            assert len(frame) > 0, f"Empty export: {stem}"
            if isinstance(frame, pl.DataFrame):
                frame.write_csv(staging / f"{stem}.csv")
            else:
                frame.to_csv(staging / f"{stem}.csv", index=False)
        for path in sorted(staging.iterdir()):
            path.replace(output_dir / path.name)
    for stem, frame in frames.items():
        click.echo(f"Wrote {output_dir / (stem + '.csv')} ({len(frame)} rows)")


@click.group()
@click.option("--fetch", is_flag=True, help="Download missing original snapshots with configured credentials.")
@click.option("--output-dir", type=click.Path(path_type=Path, file_okay=False),
              default=local_data.ROOT, show_default=True)
@click.pass_context
def cli(ctx: click.Context, fetch: bool, output_dir: Path) -> None:
    """Prepare plotting CSVs. Commands use the same descriptive labels as plots."""
    previous = local_data.FETCH
    local_data.FETCH = fetch
    ctx.call_on_close(lambda: setattr(local_data, "FETCH", previous))
    ctx.obj = output_dir


def _register(name):
    @cli.command(name, help=f"Prepare the {name} plotting CSVs.")
    @click.pass_obj
    def command(output_dir):
        export(name, output_dir)


for _name in RECIPES:
    _register(_name)


@cli.command("all")
@click.pass_obj
def all_data(output_dir: Path) -> None:
    """Prepare every data-driven figure, reusing shared source calculations."""
    for name in RECIPES:
        export(name, output_dir)


if __name__ == "__main__":
    cli()
