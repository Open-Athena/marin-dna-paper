"""Chromosome-weighted AUPRC and bootstrap uncertainty used by the paper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# Metrics

MACRO_AVG_SUBSET = "_macro_avg_"


def _per_chrom_components(
    y: np.ndarray,
    score: np.ndarray,
    chrom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-chromosome AUPRC and finite-score weight for the *contributing* chromosomes.

    The shared primitive behind ``per_chrom_weighted_ap`` (whose value is the
    weight-average of ``ap``) and ``per_chrom_ap_table``'s chromosome-cluster bootstrap
    SE. Returns ``(chroms, ap, weight)`` over only the chromosomes that carry both
    classes among their finite-score variants (``0 < positives < n``); single-class or
    all-non-finite chromosomes are dropped (AUPRC is undefined there). ``ap[i]`` is
    chromosome ``chroms[i]``'s AUPRC and ``weight[i]`` its finite-score variant count.

    Because ``average_precision_score`` is invariant to exact row duplication, a
    chromosome-cluster bootstrap can resample these ``(ap, weight)`` pairs with
    replacement rather than re-scoring: a chromosome's ``ap`` is fixed, only its weight
    scales with the resample multiplicity.

    Non-finite scores (NaN / ±inf) are dropped per chromosome — the ``iter3``
    finite-score guard — so an unaligned or uncomputable locus neither counts as a class
    member nor enters the size weight.
    """
    # Labels must be 0/1 (or bool). Cast like the module's other metrics
    # (``auprc_with_bootstrap_se``) so a 0/1-valued object/float column still
    # works, then fail loud on a non-binary encoding: an un-cast object/str label
    # would make ``y[keep].sum()`` string-concatenate and silently skip every
    # chromosome (a wrong ``nan``) instead of raising.
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    chrom = np.asarray(chrom)
    assert len(y) == len(score) == len(chrom), (
        f"length mismatch: y={len(y)} score={len(score)} chrom={len(chrom)}"
    )
    assert np.isin(y, (0, 1)).all(), (
        f"labels must be 0/1 (or bool), got values={np.unique(y)[:5]}"
    )
    chroms: list = []
    aps: list[float] = []
    weights: list[float] = []
    for c in np.unique(chrom):
        keep = (chrom == c) & np.isfinite(score)
        n = int(keep.sum())
        # Keep a chromosome only if it carries at least one of each class among its
        # finite-score variants — AUPRC is undefined on a single class.
        if 0 < int(y[keep].sum()) < n:
            chroms.append(c)
            aps.append(float(average_precision_score(y[keep], score[keep])))
            weights.append(float(n))
    return (
        np.asarray(chroms),
        np.asarray(aps, dtype=float),
        np.asarray(weights, dtype=float),
    )


def _weighted_ap(
    ap: np.ndarray, weight: np.ndarray, mult: np.ndarray | None = None
) -> float:
    """Size-weighted mean of per-chromosome AUPRCs, optionally re-weighted by a
    bootstrap multiplicity ``mult`` (per-chromosome resample counts). ``nan`` when the
    total weight is zero — no contributing chromosome, or a resample that drew none of a
    subset's chromosomes."""
    w = weight if mult is None else weight * mult
    total = float(w.sum())
    return float((ap * w).sum() / total) if total > 0 else float("nan")


def per_chrom_weighted_ap(
    y: np.ndarray,
    score: np.ndarray,
    chrom: np.ndarray,
) -> float:
    """Per-chromosome-weighted AUPRC — the #314/#319 linear-probe headline metric.

    For each unique chromosome, computes ``average_precision_score`` over that
    chromosome's finite-score variants, then returns the mean of those
    per-chromosome AUPRCs weighted by each chromosome's (finite-score) variant
    count.

    Why per-chromosome rather than a single global pooled AUPRC: pooling
    variants across chromosomes lets a cross-fold score-calibration offset
    masquerade as discrimination, systematically *understating* a probe that is
    well-calibrated *within* each chromosome (#314 saw e.g. exp166-4B 5'UTR
    global 0.134 vs per-chrom 0.322). Scoring within each chromosome and then
    size-weighting removes that artifact. This is TraitGym's metric and the
    single headline number for the protocol.

    Chromosomes lacking both classes among their finite-score variants are
    skipped (AUPRC is undefined on a single class). Non-finite scores (NaN /
    ±inf) are dropped per chromosome — the ``iter3`` finite-score guard — so an
    unaligned or uncomputable locus neither counts as a class member nor enters
    the size weight.

    Args:
        y: 0/1 (or bool) label per variant.
        score: numeric score per variant; non-finite entries are dropped.
        chrom: chromosome id per variant (any comparable dtype — string names
            like ``"chr1"`` or ints).

    Returns:
        Sample-size-weighted mean of the per-chromosome AUPRCs, or ``nan`` when
        no chromosome has both classes among its finite-score variants.
    """
    _, ap, weight = _per_chrom_components(y, score, chrom)
    return _weighted_ap(ap, weight)


def _per_chrom_cluster_bootstrap_se(
    ap: np.ndarray,
    weight: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator | int | None,
) -> float:
    """Chromosome-cluster bootstrap SE of a subset's per-chromosome-weighted AUPRC.

    Resamples the contributing chromosomes with replacement (``average_precision_score``
    is duplication-invariant, so a repeated chromosome just re-weights that cluster) and
    returns the ``ddof=1`` std of the resampled weighted means. ``nan`` when
    ``n_bootstrap <= 0`` or fewer than two chromosomes contribute — a cluster-bootstrap
    SE needs at least two clusters.

    A fresh ``default_rng(rng)`` is constructed here rather than threaded in from the
    caller, so this cell's SE is a pure function of its own data + seed — independent of
    how many other cells were bootstrapped before it (mirrors ``auprc_with_bootstrap_se``,
    which ``compute_auprc_metrics`` calls once per cell). With an ``int`` seed the shared
    value only correlates the per-iteration noise across cells; it doesn't couple their
    data or make one cell's SE depend on the ``score_columns`` order.
    """
    m = len(ap)
    if n_bootstrap <= 0 or m < 2:
        return float("nan")
    gen = np.random.default_rng(rng)
    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = gen.integers(0, m, size=m)
        boot[b] = _weighted_ap(ap[idx], weight[idx])
    return float(np.nanstd(boot, ddof=1))


def _macro_joint_chrom_bootstrap_se(
    components: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_bootstrap: int,
    rng: np.random.Generator | int | None,
) -> float:
    """Joint chromosome-cluster bootstrap SE of the macro-average AUPRC.

    Chromosomes are *shared* across consequence subsets, so the macro-average's SE can't
    use the disjoint-cluster ``sqrt(sum(se**2)) / K`` formula (that's the matched-pair
    ``compute_auprc_metrics``, whose subsets partition their ``match_group``s). Instead
    resample the UNION of contributing chromosomes ONCE per iteration and recompute every
    subset's per-chromosome-weighted AUPRC under that single shared resample, then average
    across subsets — a chromosome that is 'lucky' across several subsets moves them
    together, and the SE reflects that coupling. ``nan`` when ``n_bootstrap <= 0`` or the
    union has fewer than two chromosomes.

    Every retained iteration averages over ALL K subsets (like the point macro), so the
    bootstrap stays centered on the point estimate. A draw that assigns multiplicity 0 to
    every chromosome some subset owns leaves that subset undefined; that whole draw is
    dropped (``nan``) rather than averaging the survivors — averaging survivors would bias
    the center toward a smaller-K macro and let subset-dropout masquerade as sampling
    noise. Under the heavy chromosome overlap of genome-wide consequence subsets this is
    vanishingly rare (only near-disjoint chromosome sets trigger it).

    ``components`` is the list of ``(chroms, ap, weight)`` from ``_per_chrom_components``
    for the qualifying subsets. A fresh ``default_rng(rng)`` is constructed here so the
    macro SE is order-independent (see ``_per_chrom_cluster_bootstrap_se``).
    """
    if n_bootstrap <= 0:
        return float("nan")
    union = sorted({c for chroms, _, _ in components for c in chroms.tolist()})
    n_union = len(union)
    if n_union < 2:
        return float("nan")
    pos = {c: i for i, c in enumerate(union)}
    # Align each subset's (ap, weight) onto the shared union index; a chromosome a subset
    # lacks gets weight 0, so it drops out of that subset's weighted mean under any
    # resample multiplicity.
    aligned: list[tuple[np.ndarray, np.ndarray]] = []
    for chroms, ap, weight in components:
        ap_u = np.zeros(n_union, dtype=float)
        w_u = np.zeros(n_union, dtype=float)
        for c, a, wi in zip(chroms.tolist(), ap, weight):
            ap_u[pos[c]] = a
            w_u[pos[c]] = wi
        aligned.append((ap_u, w_u))
    gen = np.random.default_rng(rng)
    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        mult = np.bincount(
            gen.integers(0, n_union, size=n_union), minlength=n_union
        ).astype(float)
        vals = np.array([_weighted_ap(ap_u, w_u, mult) for ap_u, w_u in aligned])
        # Keep the draw only if every subset is defined under it, so each retained macro is
        # a fixed-K mean centered on the point estimate; a dropped-out subset voids the draw.
        boot[b] = float(np.mean(vals)) if np.isfinite(vals).all() else np.nan
    return float(np.nanstd(boot, ddof=1))


def per_chrom_ap_table(
    df: pd.DataFrame,
    score_columns: list[str],
    *,
    label_col: str = "label",
    subset_col: str = "subset",
    chrom_col: str = "chrom",
    n_bootstrap: int = 1000,
    rng: np.random.Generator | int | None = 0,
    n_min: int = 30,
) -> pd.DataFrame:
    """Per-subset per-chromosome-weighted AUPRC + chromosome-cluster bootstrap SE, with a
    ``_macro_avg_`` aggregate row per score column (#341 scaling ladder; SE + macro #347).

    For each ``(subset, score_col)`` computes ``per_chrom_weighted_ap`` and its
    chromosome-cluster bootstrap SE. Pass the ``compute_probe`` predictions frame with
    ``score_columns=["probe_score", "minus_llr_avg"]`` to score the probe and its matched
    zero-shot LLR baseline on **identical rows** under the identical metric.

    Aggregate: one ``_macro_avg_`` row per score column — the unweighted mean over subsets
    with ``n_pos >= n_min`` and a finite value. Its SE is a JOINT chromosome bootstrap
    (``_macro_joint_chrom_bootstrap_se``): chromosomes are shared across subsets, so the
    disjoint-cluster SE-of-mean formula (``compute_auprc_metrics``) would misstate it.
    **No ``_global_`` row** — probe scores come from a separate per-subset classifier, so a
    pooled global ranking is undefined (unlike the matched-pair leaderboard). The macro row
    keeps honest ``n`` / ``n_pos`` sums and an all-chromosomes ``n_chrom`` (the #348 dashboard
    read layer will overload the displayed ``n`` to K, as ``fetch_method_metrics`` already
    does for the matched-pair macro).

    ``n`` / ``n_pos`` / ``n_chrom`` are subset-level counts (independent of the score
    column); on the ``_macro_avg_`` row ``n`` / ``n_pos`` are sums over the qualifying
    subsets and ``n_chrom`` is the count of distinct chromosomes across them (same
    all-chromosomes sense as the per-subset ``n_chrom``, deduped). ``value`` / ``se`` are
    per ``(subset, score_col)`` and ``NaN`` when no
    chromosome carries both classes among that score's finite values (e.g. a skipped
    probe's all-``NaN`` ``probe_score`` beside a finite baseline row). ``se`` is also
    ``NaN`` when ``n_bootstrap <= 0`` or fewer than two chromosomes contribute.

    Args:
        df: variant frame with ``label`` / ``subset`` / ``chrom`` columns plus every name
            in ``score_columns`` (row-aligned).
        score_columns: score column names to score (e.g. ``probe_score``).
        label_col: 0/1 label column name.
        subset_col: consequence-subset column name (the grouping key).
        chrom_col: chromosome column name (per-chrom weighting + bootstrap-cluster key).
        n_bootstrap: chromosome-cluster bootstrap iterations; ``<= 0`` skips the resample
            (``se = NaN``, point estimates only) — the macro row is still emitted.
        rng: seed int (recommended), ``numpy.random.Generator``, or ``None``. Each SE
            builds its own ``default_rng(rng)``, so with an ``int`` seed (``0`` default →
            bit-stable pipeline output) every cell's SE is reproducible AND independent of
            the ``score_columns`` order; ``None`` → fresh randomness. Passing a live
            ``Generator`` re-couples cells to draw order (as in ``auprc_with_bootstrap_se``).
        n_min: minimum ``n_pos`` for a subset to enter the macro average. Default 30
            (project-wide leaderboard convention; a 300-variant 1:9-matched subset ⇒ ~30
            positives, matching the probe's ``min_variants`` gate).

    Returns:
        DataFrame ``[score_type, subset, value, se, n, n_pos, n_chrom]`` — one row per
        ``(subset, score_col)`` in first-appearance subset order, then one ``_macro_avg_``
        row per score column.
    """
    for col in (label_col, subset_col, chrom_col):
        assert col in df.columns, f"df missing required column {col!r}"
    missing = [c for c in score_columns if c not in df.columns]
    assert not missing, f"df missing score columns {missing}"
    # Fail loud on nulls in the grouping / weighting keys: a null ``subset`` would be
    # silently dropped by ``groupby`` (pandas ``dropna=True`` default) and a null
    # ``chrom``/``label`` would mis-weight or mis-count the per-chromosome AUPRC — a
    # silent-corruption risk (mirrors ``compute_sge_metrics``'s ``subset`` notna guard).
    for col in (subset_col, chrom_col, label_col):
        assert df[col].notna().all(), (
            f"{col!r} contains nulls; per_chrom_ap_table would silently drop / "
            f"mis-weight those rows — fail loud instead"
        )

    rows: list[dict] = []
    # Per score column, gather the qualifying subsets' components (for the joint macro
    # bootstrap), point values (for the macro mean), counts, and the union of their (all)
    # chromosomes (for the macro's n_chrom) — in first-appearance order so the macro is
    # deterministic.
    qual_components: dict[str, list[tuple]] = {sc: [] for sc in score_columns}
    qual_values: dict[str, list[float]] = {sc: [] for sc in score_columns}
    qual_counts: dict[str, list[tuple[int, int]]] = {sc: [] for sc in score_columns}
    qual_chroms: dict[str, set] = {sc: set() for sc in score_columns}
    for subset_name, sub in df.groupby(subset_col, sort=False):
        n = int(len(sub))
        n_pos = int(np.asarray(sub[label_col]).astype(int).sum())
        sub_chroms = pd.unique(sub[chrom_col])
        n_chrom = int(sub_chroms.size)
        for score_col in score_columns:
            chroms, ap, weight = _per_chrom_components(
                sub[label_col].to_numpy(),
                sub[score_col].to_numpy(),
                sub[chrom_col].to_numpy(),
            )
            value = _weighted_ap(ap, weight)
            # A fresh default_rng(rng) is built inside the helper per call, so this cell's
            # SE doesn't depend on the score_columns order (see _per_chrom_cluster_bootstrap_se).
            se = _per_chrom_cluster_bootstrap_se(ap, weight, n_bootstrap, rng)
            rows.append(
                {
                    "score_type": score_col,
                    "subset": str(subset_name),
                    "value": value,
                    "se": se,
                    "n": n,
                    "n_pos": n_pos,
                    "n_chrom": n_chrom,
                }
            )
            if n_pos >= n_min and np.isfinite(value):
                qual_components[score_col].append((chroms, ap, weight))
                qual_values[score_col].append(value)
                qual_counts[score_col].append((n, n_pos))
                qual_chroms[score_col].update(sub_chroms.tolist())

    for score_col in score_columns:
        comps = qual_components[score_col]
        if comps:
            macro = {
                "score_type": score_col,
                "subset": MACRO_AVG_SUBSET,
                "value": float(np.mean(qual_values[score_col])),
                "se": _macro_joint_chrom_bootstrap_se(comps, n_bootstrap, rng),
                "n": sum(n for n, _ in qual_counts[score_col]),
                "n_pos": sum(n_pos for _, n_pos in qual_counts[score_col]),
                # All distinct chromosomes across the qualifying subsets (deduped) — same
                # all-chromosomes sense as the per-subset n_chrom, not the contributing-only
                # union the joint bootstrap uses internally.
                "n_chrom": len(qual_chroms[score_col]),
            }
        else:
            # No subset cleared n_min with a finite value — still emit the macro row (NaN)
            # so the aggregate is always present (the dashboard loader keys on it).
            macro = {
                "score_type": score_col,
                "subset": MACRO_AVG_SUBSET,
                "value": float("nan"),
                "se": float("nan"),
                "n": 0,
                "n_pos": 0,
                "n_chrom": 0,
            }
        rows.append(macro)

    return pd.DataFrame(
        rows,
        columns=["score_type", "subset", "value", "se", "n", "n_pos", "n_chrom"],
    )
