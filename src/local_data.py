"""Paper figure inputs, frozen source URLs, and explicit snapshot retrieval."""

from __future__ import annotations

import functools
import os
import sys
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from urllib.request import urlopen

import boto3
import pandas as pd
import polars as pl
import yaml

import metrics as _metrics

# Local

ROOT = Path(__file__).resolve().parents[1] / "plot_data"

FETCH = False


class MissingData(RuntimeError):
    pass


def local_path(source: str | Path) -> Path:
    source = str(source)
    parsed = urlparse(source)
    if parsed.scheme not in {"s3", "https", "http"}:
        path = Path(source)
    else:
        path = ROOT / "remote" / parsed.scheme / parsed.netloc / parsed.path.lstrip("/")
    if path.is_file():
        return path
    if not (FETCH or os.getenv("FIGURE_FETCH") == "1") or parsed.scheme not in {
        "s3",
        "https",
        "http",
    }:
        raise MissingData(
            f"Missing input: {path}\nOriginal source: {source}\nUse make fetch-figure-<name> with appropriate credentials, or place the snapshot at this path."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        if parsed.scheme == "s3":
            boto3.client("s3", region_name="us-east-2").download_file(
                parsed.netloc, parsed.path.lstrip("/"), str(temporary)
            )
        else:
            with urlopen(source) as response:
                temporary.write_bytes(response.read())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_polars(source, **kwargs):
    return pl.read_parquet(local_path(source), **kwargs)


def read_pandas(source, **kwargs):
    return pd.read_parquet(local_path(source), **kwargs)


# Csv

DATA_PATH = ROOT / "transfer_validation_results.csv"

SCALING_RESULTS_PATH = ROOT / "parameter_scaling_results.csv"

SCALING_HISTORY_PATH = ROOT / "parameter_scaling_history.csv"

MIXTURE_RESULTS_PATH = ROOT / "data_mixture_results.csv"

MODEL_LEADERBOARD_PATH = ROOT / "model_leaderboard.csv"

INFERENCE_COSTS_PATH = ROOT / "inference_costs.csv"


def _round_sig(x: float, sig: int = 6) -> float:
    """Round `x` to `sig` significant figures (NaN-safe). Snaps float-precision-jittered values."""
    if x is None or pd.isna(x) or x == 0:
        return x
    return float(f"{x:.{sig}g}")


def load_transfer() -> pd.DataFrame:
    df = pd.read_csv(local_path(DATA_PATH))
    # Different runs can store the same nominal sweep value with slightly different
    # float bit patterns (e.g. 2e-9 vs 1.999999999999999e-9), which then show up as
    # duplicate x-axis ticks. Snap sweep-axis fields to 6 sig figs to fix this.
    for col in ("learning_rate", "beta2", "epsilon"):
        df[col] = df[col].apply(_round_sig)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    return df


def load_scaling_results() -> pd.DataFrame:
    df = pd.read_csv(local_path(SCALING_RESULTS_PATH))
    print(f"Loaded {len(df)} rows from {SCALING_RESULTS_PATH}")
    return df


def load_scaling_history() -> pd.DataFrame:
    df = pd.read_csv(local_path(SCALING_HISTORY_PATH))
    print(f"Loaded {len(df)} rows from {SCALING_HISTORY_PATH}")
    return df


def load_mixture() -> pd.DataFrame:
    df = pd.read_csv(local_path(MIXTURE_RESULTS_PATH))
    print(f"Loaded {len(df)} rows from {MIXTURE_RESULTS_PATH}")
    return df


def load_leaderboard() -> pd.DataFrame:
    # First line is a provenance comment; the header is on line 2.
    df = pd.read_csv(local_path(MODEL_LEADERBOARD_PATH), skiprows=1)
    print(f"Loaded {len(df)} rows from {MODEL_LEADERBOARD_PATH}")
    return df


def load_inference_costs() -> pd.DataFrame:
    df = pd.read_csv(local_path(INFERENCE_COSTS_PATH), comment="#")
    print(f"Loaded {len(df)} rows from {INFERENCE_COSTS_PATH}")
    return df


# Models

Family = Literal["marin_dna", "conservation", "alphagenome", "gpn_star", "evo2"]

ALL_FAMILIES: tuple[Family, ...] = (
    "marin_dna",
    "conservation",
    "alphagenome",
    "gpn_star",
    "evo2",
)

ALL_DATASETS: tuple[str, ...] = (
    "mendelian_traits",
    "complex_traits",
    "sge",
)


@dataclass(frozen=True)
class Model:
    """Registry fields needed to locate and label the paper's metric inputs."""

    id: str
    display: str
    family: Family
    datasets: tuple[str, ...]


MODELS_YAML = ROOT / "models.yaml"


def _coerce(raw: dict) -> Model:
    family = raw["family"]
    assert family in ALL_FAMILIES, f"unknown model family: {family!r}"
    datasets = tuple(raw["datasets"])
    assert set(datasets).issubset(ALL_DATASETS), f"unknown datasets for {raw['id']!r}"
    return Model(raw["id"], raw["display"], family, datasets)


@cache
def load_models(path: Path = MODELS_YAML) -> tuple[Model, ...]:
    """Load and validate ``plot_data/models.yaml``. Cached per (process, path)."""
    raw_list = yaml.safe_load(path.read_text())
    assert isinstance(raw_list, list), (
        f"models.yaml must be a top-level list, got {type(raw_list).__name__}"
    )
    methods = tuple(_coerce(r) for r in raw_list)
    seen: set[str] = set()
    for m in methods:
        assert m.id not in seen, f"duplicate method id {m.id!r}"
        seen.add(m.id)
    return methods


def models_for_dataset(dataset: str) -> tuple[Model, ...]:
    """Methods registered for ``dataset``, in registry order."""
    assert dataset in ALL_DATASETS, f"unknown dataset {dataset!r}"
    return tuple(m for m in load_models() if dataset in m.datasets)


# Leaderboard data

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
    return read_polars(path)


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


# Mixture

COOLDOWN_KEEP_FRACTION = 0.8

_THIRD = 1 / 3

_FIFTH = 1 / 5


@dataclass(frozen=True)
class Run:
    name: str  # tree-path id + size tier (display label)
    mix: str  # W&B `mix=` tag (join key to the CSV)
    weights: dict[str, float]  # training mixture weights over REGION_ORDER
    parent: str | None  # parent's `mix`, or None for roots
    branch: str  # "root" | "pre_cooldown" | "final"
    keep_fraction: float | None = None


LINEAGE: tuple[Run, ...] = (
    # Tree 1 — uniform (CUD ⅓) base, ~52B own.
    Run(
        "1·L",
        "uniform",
        {"cds": _THIRD, "upstream": _THIRD, "downstream": _THIRD},
        None,
        "root",
    ),
    Run(
        "1.7·L",
        "uniform_to_uniform_1",
        {"upstream": _THIRD, "cds": _THIRD, "downstream": _THIRD},
        "uniform",
        "pre_cooldown",
    ),
    Run(
        "1.7.2·L",
        "exp135-zoonomia-m5.1",
        {
            "cds": _FIFTH,
            "upstream": _FIFTH,
            "downstream": _FIFTH,
            "ncrna_exon": _FIFTH,
            "ccre_non_promoter": _FIFTH,
        },
        "uniform_to_uniform_1",
        "final",
    ),
    # Tree 5 — zoonomia uniform (CUDNE ⅕) base, then a pre_cooldown continuation
    # chain (m1 -> m1.1 -> m1.2 -> m1.3). The forks step progressively earlier into
    # each parent's own tokens: m1.1 at 0.8 of m1; m1.2 at 0.60 of m1.1's own (step
    # 41412 of [23664, 53244]); m1.3 at 0.40 of m1.2's own (step 53244 of
    # [41412, 70992]) — all before the parent's own cooldown but short of 0.8,
    # hence the explicit keep_fraction overrides. m1.3 is the terminal leg.
    Run(
        "5·L",
        "exp135-zoonomia-m1",
        {
            "cds": _FIFTH,
            "upstream": _FIFTH,
            "downstream": _FIFTH,
            "ncrna_exon": _FIFTH,
            "ccre_non_promoter": _FIFTH,
        },
        None,
        "root",
    ),
    Run(
        "5.1·L",
        "exp135-zoonomia-m1.1",
        {
            "cds": _FIFTH,
            "upstream": _FIFTH,
            "downstream": _FIFTH,
            "ncrna_exon": _FIFTH,
            "ccre_non_promoter": _FIFTH,
        },
        "exp135-zoonomia-m1",
        "pre_cooldown",
    ),
    Run(
        "5.1.1·L",
        "exp135-zoonomia-m1.2",
        {
            "cds": _FIFTH,
            "upstream": _FIFTH,
            "downstream": _FIFTH,
            "ncrna_exon": _FIFTH,
            "ccre_non_promoter": _FIFTH,
        },
        "exp135-zoonomia-m1.1",
        "pre_cooldown",
        keep_fraction=0.60,
    ),
    Run(
        "5.1.1.1·L",
        "exp135-zoonomia-m1.3",
        {
            "cds": _FIFTH,
            "upstream": _FIFTH,
            "downstream": _FIFTH,
            "ncrna_exon": _FIFTH,
            "ccre_non_promoter": _FIFTH,
        },
        "exp135-zoonomia-m1.2",
        "pre_cooldown",
        keep_fraction=0.40,
    ),
    # Tree 6 — zoonomia upstream-tilted (U25, CDNE 0.1875) base, then a pre_cooldown
    # continuation chain (m3 -> m3.1 -> m3.2 -> m3.3), mirroring tree 5. m3.2 forks at
    # 0.60 of m3.1's own (step 41405 of [23660, 53239]); m3.3 at 0.40 of m3.2's own
    # (step 53235 of [41405, 70984]). m3.3 is the terminal leg.
    Run(
        "6·L",
        "exp135-zoonomia-m3",
        {
            "cds": 0.1875,
            "upstream": 0.25,
            "downstream": 0.1875,
            "ncrna_exon": 0.1875,
            "ccre_non_promoter": 0.1875,
        },
        None,
        "root",
    ),
    Run(
        "6.1·L",
        "exp135-zoonomia-m3.1",
        {
            "cds": 0.1875,
            "upstream": 0.25,
            "downstream": 0.1875,
            "ncrna_exon": 0.1875,
            "ccre_non_promoter": 0.1875,
        },
        "exp135-zoonomia-m3",
        "pre_cooldown",
    ),
    Run(
        "6.1.1·L",
        "exp135-zoonomia-m3.2",
        {
            "cds": 0.1875,
            "upstream": 0.25,
            "downstream": 0.1875,
            "ncrna_exon": 0.1875,
            "ccre_non_promoter": 0.1875,
        },
        "exp135-zoonomia-m3.1",
        "pre_cooldown",
        keep_fraction=0.60,
    ),
    Run(
        "6.1.1.1·L",
        "exp135-zoonomia-m3.3",
        {
            "cds": 0.1875,
            "upstream": 0.25,
            "downstream": 0.1875,
            "ncrna_exon": 0.1875,
            "ccre_non_promoter": 0.1875,
        },
        "exp135-zoonomia-m3.2",
        "pre_cooldown",
        keep_fraction=0.40,
    ),
)

BY_MIX: dict[str, Run] = {r.mix: r for r in LINEAGE}


def keep_fraction(run: Run) -> float:
    """Fraction of the parent's OWN tokens inherited at `run`'s branch point.

    A `final` branch inherits the parent's entire run (1.0). A `pre_cooldown`
    branch inherits up to the fork: the per-run `keep_fraction` override when set
    (the chained continuations fork at ~0.60), else the flat COOLDOWN_KEEP_FRACTION
    (0.8) that the bulk of pre_cooldown branches sit at. Roots inherit nothing,
    but are never asked (they have no parent); we return 1.0 defensively.
    """
    if run.keep_fraction is not None:
        return run.keep_fraction
    return COOLDOWN_KEEP_FRACTION if run.branch == "pre_cooldown" else 1.0


def inherited_components(mix: str, own_tokens: dict[str, float]) -> dict[str, float]:
    """Per-component tokens `mix` inherited at warm-start (everything its parent
    accumulated up to the branch point). `own_tokens` maps mix → its own token tag.
    """
    run = BY_MIX[mix]
    if run.parent is None:
        return {}
    parent = BY_MIX[run.parent]
    comp = dict(inherited_components(run.parent, own_tokens))  # grandparent+ at 100%
    f = keep_fraction(run)
    parent_own = own_tokens[run.parent]
    for region, w in parent.weights.items():
        if w > 0:
            comp[region] = comp.get(region, 0.0) + f * parent_own * w
    return comp


def _nan_float(value: object) -> float:
    """Preserve undefined metric values when Parquet nulls become None."""
    return float("nan") if value is None else float(value)
