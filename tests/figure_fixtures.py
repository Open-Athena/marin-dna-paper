"""Synthetic Parquet-shaped inputs; these are never used for publication figures."""

import hashlib

import numpy as np
import pandas as pd
import polars as pl

SUBSETS = (
    "missense_variant",
    "splicing",
    "synonymous_variant",
    "tss_proximal",
    "5_prime_UTR_variant",
    "3_prime_UTR_variant",
    "distal",
    "non_coding_transcript_exon_variant",
)
EXTERNAL_FIGURES = (
    "vep_by_model_size",
    "vep_by_sequence_likelihood",
    "missense_readout_scaling",
    "mixture_zero_shot_trajectories",
    "mixture_probe_trajectories",
    "mendelian_zero_shot_comparison",
    "mendelian_probe_comparison",
    "region_specialist_performance",
    "upstream_coding_mixture_balance",
)


def prediction_frame(source, **kwargs):
    seed = int(hashlib.sha256(str(source).encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "subset": np.repeat(SUBSETS, 120),
            "chrom": np.tile(np.repeat(["1", "2", "3"], 40), len(SUBSETS)),
            "label": np.tile([0, 1], 60 * len(SUBSETS)),
            "probe_score": rng.random(120 * len(SUBSETS)),
            "llr_fwd": rng.random(120 * len(SUBSETS)),
            "llr_rc": rng.random(120 * len(SUBSETS)),
        }
    )


def metric_frame(source, **kwargs):
    seed = int(hashlib.sha256(str(source).encode()).hexdigest()[:8], 16)
    return pl.DataFrame(
        [
            {
                "subset": subset,
                "score_type": score,
                "split": "train",
                "model": "GPN-Star-M",
                "value": 0.2 + (seed % 100) / 1000 + i * 0.015,
                "se": 0.01,
                "n": 120,
                "n_pos": 60,
                "n_rows": 120,
                "n_groups": 60,
            }
            for i, subset in enumerate((*SUBSETS, "_macro_avg_"))
            for score in (
                "minus_llr_avg",
                "probe_score",
                "jsd_avg",
                "minus_llr_fwd",
                "jsd_fwd",
                "minus_llr_calibrated",
                "minus_llr",
                "alphagenome_max_l2",
            )
        ]
    )
