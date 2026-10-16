"""TIC correlation: Pearson r of normalised and raw TICs, smoothed and masked variants.

From warpalign.metrics and warpalign.evaluation. Every function that used the module-level
evaluation window takes `bins` (default: the fish-oil axis's eval_bins) or `axis`.

Smoothed TIC r: raw TICs of apex-scan bins are dominated by one-bin-wide peaks, so resampling at
fractional positions (any warp) lowers raw TIC r even without misalignment. Every method is
therefore also scored after smoothing both batches' TICs with the same Gaussian (SMOOTH_SIGMAS).

Empty bins: some runs record scans with no ions, which bin to exact zeros; 'r non-empty' scores
only bins non-empty in both samples.

Coverage: an output bin of a warped sample is *outside* the acquired range if its source position
lies before that sample's acquisition start (it reads zeros) or past the last bin (it reads the
repeated edge value); 'r in range' masks those out.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..axis import FISH_OIL_AXIS, TimeAxis

EVAL_BINS = FISH_OIL_AXIS.eval_bins
SMOOTH_SIGMAS = (1.0, 2.0)


def pearson_rows(a: np.ndarray, b: np.ndarray, bins: slice | None = EVAL_BINS) -> np.ndarray:
    """Pearson r for every (a_i, b_j) row pair of a [n_a, T] and b [n_b, T], i-major flat
    (index i · n_b + j). Computed on the time bins `bins` (default: the fish-oil evaluation
    window; None = all bins). A row that is constant over those bins gives NaN."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    if bins is not None:
        a, b = a[:, bins], b[:, bins]
    az = a - a.mean(axis=1, keepdims=True)
    bz = b - b.mean(axis=1, keepdims=True)
    na = np.linalg.norm(az, axis=1, keepdims=True)
    nb = np.linalg.norm(bz, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        r = (az / na) @ (bz / nb).T
    return np.clip(r, -1.0, 1.0).ravel()


def smooth_rows(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing along time (the last axis) with σ = sigma bins; sigma 0 = unchanged.
    Edges are padded with the end value, so a zero-filled run start stays zero."""
    x = np.asarray(x, dtype=np.float64)
    return gaussian_filter1d(x, sigma, axis=-1, mode='nearest') if sigma > 0 else x


def pearson_rows_smoothed(
    a: np.ndarray, b: np.ndarray, sigmas: tuple[float, ...], bins: slice | None = EVAL_BINS
) -> dict[float, np.ndarray]:
    """pearson_rows after smoothing both a and b with the same Gaussian, for each σ in sigmas."""
    return {s: pearson_rows(smooth_rows(a, s), smooth_rows(b, s), bins) for s in sigmas}


def all_pairs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows of a [n_a, ...] and b [n_b, ...] expanded to every (a_i, b_j), i-major: row
    i · n_b + j of the results is (a_i, b_j)."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return np.repeat(a, len(b), axis=0), np.tile(b, (len(a),) + (1,) * (b.ndim - 1))


def pearson_paired(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None, bins: slice | None = EVAL_BINS
) -> np.ndarray:
    """Pearson r of each row pair (a_k, b_k) of a, b [K, T], on the time bins `bins` and, if
    given, only where mask [K, T] is True. NaN for fewer than 3 bins or a constant row."""
    a = np.atleast_2d(np.asarray(a, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    m = np.ones(a.shape, dtype=bool) if mask is None else np.atleast_2d(np.asarray(mask, bool))
    if bins is not None:
        a, b, m = a[:, bins], b[:, bins], m[:, bins]
    w = m.astype(np.float64)
    n = w.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        az = (a - (a * w).sum(axis=1, keepdims=True) / n) * w
        bz = (b - (b * w).sum(axis=1, keepdims=True) / n) * w
        r = (az * bz).sum(axis=1) / np.sqrt((az**2).sum(axis=1) * (bz**2).sum(axis=1))
    r = np.where(n[:, 0] >= 3, r, np.nan)
    return np.clip(r, -1.0, 1.0)


def pearson_rows_masked(
    a: np.ndarray, b: np.ndarray, valid_a: np.ndarray, valid_b: np.ndarray, bins: slice = EVAL_BINS
) -> np.ndarray:
    """Like pearson_rows, but each (a_i, b_j) uses only the bins in `bins` where both
    valid_a[i] and valid_b[j] are True (NaN if fewer than 3 such bins or a constant row)."""
    xa, xb = all_pairs(np.asarray(a), np.asarray(b))
    va, vb = all_pairs(np.asarray(valid_a), np.asarray(valid_b))
    return pearson_paired(xa, xb, va & vb, bins)


def pearson(a: np.ndarray, b: np.ndarray, bins: slice | None = EVAL_BINS) -> float:
    """Pearson r of two 1-D signals on `bins`."""
    return float(pearson_rows(a, b, bins)[0])


def tic_r_pairs(chromas_a: np.ndarray, chromas_b: np.ndarray, bins: slice | None = EVAL_BINS) -> np.ndarray:
    """Normalised-TIC Pearson r for every (A_i, B_j), i-major, on the evaluation window."""
    return pearson_rows(np.asarray(chromas_a).sum(-1), np.asarray(chromas_b).sum(-1), bins)


def within_batch_r(x: np.ndarray, bins: slice | None = EVAL_BINS) -> tuple[float, float, int]:
    """Mean and s.d. of Pearson r over all pairs of distinct rows of x [n, T], and the number of
    pairs."""
    n = len(x)
    if n < 2:
        raise ValueError('within_batch_r needs at least two rows')
    r = pearson_rows(x, x, bins).reshape(n, n)
    v = np.array([r[i, j] for i, j in combinations(range(n), 2)])
    return float(v.mean()), float(v.std()), len(v)


def per_a_mean(r: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    """Per batch-A sample mean of an i-major per-pair array."""
    return np.asarray(r).reshape(n_a, n_b).mean(axis=1)


# ── The per-pair metric set ──────────────────────────────────────────────────


def metric_suffixes(sigmas: tuple[float, ...] = SMOOTH_SIGMAS) -> dict[str, str]:
    """Report label → saved-array suffix of every per-pair TIC metric: an array saved as <key>
    holds normalised TIC r, <key><suffix> the others (e.g. 'pchip_pooled_raw_s1')."""
    return {
        'TIC r': '',
        **{f'TIC r σ{s:g}': f'_s{s:g}' for s in sigmas},
        'raw r': '_raw',
        **{f'raw r σ{s:g}': f'_raw_s{s:g}' for s in sigmas},
        'r in range': '_valid',
        'r non-empty': '_nonempty',
        'raw r non-empty': '_raw_nonempty',
    }


METRICS = metric_suffixes()


def in_range(src: np.ndarray, start: np.ndarray, axis: TimeAxis = FISH_OIL_AXIS) -> np.ndarray:
    """[N, T] True where source positions src [N, T] (or [T], shared) lie inside each sample's
    acquired range [start_i, T − 1]."""
    src = np.broadcast_to(np.asarray(src, dtype=np.float64), (len(start), axis.n_bins))
    return (src >= np.asarray(start)[:, None]) & (src <= axis.n_bins - 1)


def tic_metrics(
    tic_a: np.ndarray,
    tic_b: np.ndarray,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    valid_a: np.ndarray,
    valid_b: np.ndarray,
    paired: bool = False,
    axis: TimeAxis = FISH_OIL_AXIS,
    sigmas: tuple[float, ...] = SMOOTH_SIGMAS,
) -> tuple[dict[str, np.ndarray], float]:
    """(per-pair metrics keyed by metric_suffixes(sigmas), outside fraction) of warped TICs.

    paired=False: every (A_i, B_j) of tic_a [n_a, T] and tic_b [n_b, T], i-major. paired=True:
    row k of tic_a against row k of tic_b (each pair warped separately, as COW-TIC).
    valid_*: in-range masks (in_range). Non-empty bins: raw TIC > 0 in both warped samples."""
    bins = axis.eval_bins
    xs = [np.asarray(x, dtype=np.float64) for x in (tic_a, tic_b, raw_a, raw_b)]
    va, vb = np.asarray(valid_a, bool), np.asarray(valid_b, bool)
    outside = float(1.0 - np.concatenate([va, vb])[:, bins].mean())
    if not paired:
        (ta, tb), (ra, rb), (va, vb) = all_pairs(*xs[:2]), all_pairs(*xs[2:]), all_pairs(va, vb)
    else:
        ta, tb, ra, rb = xs
    nonempty = (ra > 0) & (rb > 0)
    out = {'': pearson_paired(ta, tb, bins=bins), '_raw': pearson_paired(ra, rb, bins=bins)}
    for s in sigmas:
        out[f'_s{s:g}'] = pearson_paired(smooth_rows(ta, s), smooth_rows(tb, s), bins=bins)
        out[f'_raw_s{s:g}'] = pearson_paired(smooth_rows(ra, s), smooth_rows(rb, s), bins=bins)
    out['_valid'] = pearson_paired(ta, tb, va & vb, bins)
    out['_nonempty'] = pearson_paired(ta, tb, nonempty, bins)
    out['_raw_nonempty'] = pearson_paired(ra, rb, nonempty, bins)
    return out, outside
