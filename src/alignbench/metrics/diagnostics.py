"""Secondary diagnostics: batch silhouette and warp magnitude (from warpalign.metrics)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ..axis import FISH_OIL_AXIS, TimeAxis


def study_silhouette(chromas_a: np.ndarray, chromas_b: np.ndarray) -> float:
    """Study-label silhouette of per-m/z max projections in PCA space (≤ 50 components).
    Lower = less batch separation. The max projection is largely RT-invariant, so this is a
    secondary diagnostic, not a measure of alignment."""
    feats = np.concatenate([np.asarray(chromas_a).max(axis=1), np.asarray(chromas_b).max(axis=1)])
    labels = np.array([0] * len(chromas_a) + [1] * len(chromas_b))
    n_comp = min(50, feats.shape[0] - 1, feats.shape[1])
    return float(silhouette_score(PCA(n_components=n_comp).fit_transform(feats), labels))


def warp_magnitude(src_bins: np.ndarray, axis: TimeAxis = FISH_OIL_AXIS) -> float:
    """Mean |w(t) − t| in minutes for source positions src_bins [..., T] (bins)."""
    src = np.asarray(src_bins, dtype=np.float64)
    return float(np.abs(src - np.arange(src.shape[-1])).mean()) * axis.bin_min
