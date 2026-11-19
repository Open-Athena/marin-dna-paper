"""Paper figure inputs, frozen source URLs, and explicit snapshot retrieval."""

from __future__ import annotations

import os
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


def load_plot_csv(name: str) -> pd.DataFrame:
    """Read an exported plotting table without altering stored binary floats."""
    return pd.read_csv(local_path(ROOT / f"{name}.csv"), float_precision="round_trip")


def load_plot_polars(name: str) -> pl.DataFrame:
    return pl.from_pandas(load_plot_csv(name))


def load_upstream_coding_mixture_balance() -> tuple[pl.DataFrame, pl.DataFrame]:
    return tuple(load_plot_polars(f"upstream_coding_mixture_balance_{part}")
                 for part in ("endpoints", "trajectories"))


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
