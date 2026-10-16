"""Summary statistics and tests across seeds or pairs (from warpalign.metrics)."""

from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon


def holm(pvals) -> np.ndarray:
    """Holm–Bonferroni adjusted p-values (monotone, capped at 1), in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p, kind='stable')
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def mean_sd(x) -> tuple[float, float]:
    """Mean and sample s.d. (ddof 1; NaN for a single value)."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float('nan')


def paired_wilcoxon(x, y) -> tuple[float, float]:
    """Two-sided paired Wilcoxon signed-rank test of x vs y: (statistic, p). Identical inputs
    (every difference zero) give (0, 1) instead of SciPy's NaN."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.all(x == y):
        return 0.0, 1.0
    res = wilcoxon(x, y, alternative='two-sided')
    return float(res.statistic), float(res.pvalue)
