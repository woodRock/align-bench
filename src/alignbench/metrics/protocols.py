"""Scoring protocols: every method's correspondence map scored the same way on a study.

joint-align scripts/32_fair_benchmark.py, generalised to any study and method. For an evaluation
pair (A, B) and every sample pair (A_i, B_j), the map m(t) = g(B, A, t; i=j, j=i) takes RT bins in
B_j to RT bins in A_i; it is clipped to [−T, 2T] and made non-decreasing. Then

  tic_bframe    A_i read at m(t), B_j raw:  r( A_i(m(t)), B_j(t) )
  tic_midpoint  both read in the frame halfway between them: h(t) = (t + m(t)) / 2 (+1e-9·t, so
                strictly increasing), s = h⁻¹(τ) on the grid, r( A_i(m(s)), B_j(s) )

B-frame is the headline protocol for real data (Jesse, 2026-10-17); midpoint is symmetric and
smooths both batches alike, so it is reported alongside.

Metrics per sample pair (TIC r on the axis's evaluation window):
  tic_r            normalised TIC (the sum of the sqrt-L2 chromatogram): 32_fair_benchmark's number
  tic_r_s1/_s2     both read signals smoothed with the same Gaussian, σ = 1 / 2 bins
  raw_r, _s1, _s2  the same on the raw TIC
  tic_r_valid      only bins whose read positions lie inside both samples' acquired range
  tic_r_nonempty   only bins where both read raw TICs are > 0 (scans with no ions bin to zeros)
  raw_r_nonempty
  outside          fraction of evaluated (bin, sample) reads outside the acquired range

Unmasked r is warpalign metrics.pearson (pearson_rows, as script 32), 0 when a signal is constant
on the window (as script 32's corr). Masked r is metrics.tic.pearson_paired, NaN when fewer than 3
bins remain. Note warpalign's evaluation.PairEval used pearson_paired for every metric: the two
agree to rounding, not bit for bit.

Each protocol returns tidy rows: one dict per (pair, sample pair, metric).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from ..axis import TimeAxis
from ..data import Study
from ..methods.base import Fit
from ..warps import warp_signal
from .tic import SMOOTH_SIGMAS, pearson, pearson_paired, smooth_rows

Pair = tuple[int, int]


def monotone_map(m: np.ndarray, n_bins: int) -> np.ndarray:
    """script 32's guard: clip to [−T, 2T], then non-decreasing."""
    return np.maximum.accumulate(np.clip(m, -n_bins, 2 * n_bins))


def bframe_positions(m: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(positions read in A, positions read in B) for the B-frame protocol."""
    return m, grid


def midpoint_positions(m: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(positions read in A, positions read in B) in the frame halfway between A and B."""
    h = (grid + m) / 2 + 1e-9 * grid
    s = np.interp(grid, h, grid)
    return np.interp(s, grid, m), s


FRAMES: dict[str, Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]] = {
    'tic_bframe': bframe_positions,
    'tic_midpoint': midpoint_positions,
}
HEADLINE = 'tic_bframe'


def _corr(x: np.ndarray, y: np.ndarray, bins: slice) -> float:
    r = pearson(x, y, bins)
    return 0.0 if np.isnan(r) else r


def score_reads(tic_a, tic_b, raw_a, raw_b, pa, pb, start_a: int, start_b: int, axis: TimeAxis,
                sigmas: Sequence[float] = SMOOTH_SIGMAS) -> dict[str, float]:
    """Every metric for one sample pair, A read at positions pa and B at pb."""
    bins, last = axis.eval_bins, axis.n_bins - 1
    read = lambda x, p: warp_signal(np.asarray(x, dtype=float), p)
    xa, xb, ra, rb = read(tic_a, pa), read(tic_b, pb), read(raw_a, pa), read(raw_b, pb)
    out = {'tic_r': _corr(xa, xb, bins), 'raw_r': _corr(ra, rb, bins)}
    for s in sigmas:
        out[f'tic_r_s{s:g}'] = _corr(smooth_rows(xa, s), smooth_rows(xb, s), bins)
        out[f'raw_r_s{s:g}'] = _corr(smooth_rows(ra, s), smooth_rows(rb, s), bins)
    va = (pa >= start_a) & (pa <= last)
    vb = (pb >= start_b) & (pb <= last)
    nonempty = (ra > 0) & (rb > 0)
    out['tic_r_valid'] = float(pearson_paired(xa, xb, va & vb, bins)[0])
    out['tic_r_nonempty'] = float(pearson_paired(xa, xb, nonempty, bins)[0])
    out['raw_r_nonempty'] = float(pearson_paired(ra, rb, nonempty, bins)[0])
    out['outside'] = float(1.0 - np.concatenate([va[bins], vb[bins]]).mean())
    return out


def eval_pairs(study: Study) -> list[Pair]:
    """The study's evaluation pairs (A, B) as batch indices: study.meta['eval_pairs'] (batch names)
    where every batch is present, else every pair (k, l), k < l."""
    names = study.meta.get('eval_pairs')
    if names:
        have = set(study.batch_names)
        return [(study.index(a), study.index(b)) for a, b in names if a in have and b in have]
    K = len(study)
    return [(a, b) for a in range(K) for b in range(a + 1, K)]


def tic_protocol(study: Study, fit: Fit, protocol: str = HEADLINE, pairs: Sequence[Pair] | None = None,
                 sigmas: Sequence[float] = SMOOTH_SIGMAS) -> list[dict]:
    """Score fit on study under one TIC protocol ('tic_bframe' or 'tic_midpoint')."""
    if protocol not in FRAMES:
        raise ValueError(f'unknown TIC protocol {protocol!r}; one of {sorted(FRAMES)}')
    frame, axis = FRAMES[protocol], study.axis
    grid = np.arange(axis.n_bins, dtype=float)
    rows = []
    for a, b in (eval_pairs(study) if pairs is None else pairs):
        A, B = study[a], study[b]
        tic_a, tic_b, start_a, start_b = A.tic, B.tic, A.acquisition_start, B.acquisition_start
        for i in range(len(A)):
            for j in range(len(B)):
                m = monotone_map(np.asarray(fit.map(b, a, grid, i=j, j=i), dtype=float), axis.n_bins)
                pa, pb = frame(m, grid)
                met = score_reads(tic_a[i], tic_b[j], A.tic_raw[i], B.tic_raw[j], pa, pb,
                                  int(start_a[i]), int(start_b[j]), axis, sigmas)
                for metric, value in met.items():
                    rows.append({'protocol': protocol, 'method': fit.method, 'batch_a': A.name,
                                 'batch_b': B.name, 'sample_a': i, 'sample_b': j, 'metric': metric,
                                 'value': value})
    return rows


def tic_bframe(study: Study, fit: Fit, **kw) -> list[dict]:
    return tic_protocol(study, fit, 'tic_bframe', **kw)


def tic_midpoint(study: Study, fit: Fit, **kw) -> list[dict]:
    return tic_protocol(study, fit, 'tic_midpoint', **kw)


def as_array(rows: list[dict], batch_a: str, batch_b: str, metric: str = 'tic_r',
             protocol: str | None = None, method: str | None = None) -> np.ndarray:
    """(n_a, n_b) array of one metric for one pair from tidy rows (script 32's layout)."""
    sel = [r for r in rows if r['batch_a'] == batch_a and r['batch_b'] == batch_b and r['metric'] == metric
           and (protocol is None or r['protocol'] == protocol) and (method is None or r['method'] == method)]
    if not sel:
        raise KeyError(f'no rows for {batch_a} × {batch_b}, {metric}, {protocol}, {method}')
    n_a, n_b = max(r['sample_a'] for r in sel) + 1, max(r['sample_b'] for r in sel) + 1
    out = np.full((n_a, n_b), np.nan)
    for r in sel:
        out[r['sample_a'], r['sample_b']] = r['value']
    return out


# ── The other protocols, as tidy rows ────────────────────────────────────────


def _row(protocol: str, fit: Fit, batch_a: str, batch_b: str, metric: str, value: float, **extra) -> dict:
    return {'protocol': protocol, 'method': fit.method, 'batch_a': batch_a, 'batch_b': batch_b,
            'sample_a': -1, 'sample_b': -1, 'metric': metric, 'value': float(value), **extra}


def _direct_edges(study: Study, fit: Fit) -> set[Pair]:
    edges = fit.params.get('edges')
    K = len(study)
    return {tuple(e) for e in edges} if edges is not None else {(a, b) for a in range(K) for b in range(a + 1, K)}


def map_error(study: Study, fit: Fit, **_) -> list[dict]:
    """Error against the truth (synthetic studies): per ordered batch pair, the median and 95th
    percentile of |ĝ − g*| (minutes) inside the mask, and the summary over all / direct / indirect
    pairs (direct = batch pairs whose anchors the fit used: fit.params['edges'], default all)."""
    from .truth import map_errors, summarise

    errs = map_errors(study, fit.map)
    names = study.batch_names
    rows = []
    for (k, l), e in errs.items():
        med, p95 = (float(np.median(e)), float(np.percentile(e, 95))) if len(e) else (np.nan, np.nan)
        rows.append(_row('map_error', fit, names[l], names[k], 'median_min', med, n=len(e)))
        rows.append(_row('map_error', fit, names[l], names[k], 'p95_min', p95, n=len(e)))
    for metric, value in summarise(errs, _direct_edges(study, fit)).items():
        rows.append(_row('map_error', fit, '*', '*', f'{metric}_min', value))
    return rows


def cycle_inconsistency(study: Study, fit: Fit, **_) -> list[dict]:
    """Median and max over ordered batch triplets of the mean |g_m→l ∘ g_k→m − g_k→l| (minutes),
    inside the truth mask (synthetic studies)."""
    from .truth import cycle_inconsistency as cyc

    med, mx = cyc(study, fit.map)
    return [_row('cycle_inconsistency', fit, '*', '*', 'median_min', med),
            _row('cycle_inconsistency', fit, '*', '*', 'max_min', mx)]


def heldout_anchor_error(study: Study, fit: Fit, *, anchors=None, refit=None, **_) -> list[dict]:
    """Median |held-out anchor error| over the RT-blocked folds (metrics.anchors), per batch pair and
    overall, in bins and minutes. Needs the anchors and refit(anchors) → Fit (the runner passes
    both): the method is refitted on each fold's training anchors."""
    from .anchors import rt_folds

    if anchors is None or refit is None:
        raise ValueError('heldout_anchor_error needs anchors= and refit= (the method refitted on a subset)')
    from .anchors import anchor_errors

    err, test_rows = [], []
    for tr, te in rt_folds(anchors):
        sub = anchors.subset(te)
        err.append(anchor_errors(refit(anchors.subset(tr)).map, sub))
        test_rows.append(np.flatnonzero(te))
    e, idx = np.abs(np.concatenate(err)), np.concatenate(test_rows)
    names, bmin = study.batch_names, study.axis.bin_min
    rows = [_row('heldout_anchor_error', fit, '*', '*', 'median_bins', np.median(e), n=len(e)),
            _row('heldout_anchor_error', fit, '*', '*', 'median_min', np.median(e) * bmin, n=len(e))]
    ba, bb = anchors.ba[idx], anchors.bb[idx]
    for a, b in sorted(set(zip(ba.tolist(), bb.tolist()))):
        m = (ba == a) & (bb == b)
        rows.append(_row('heldout_anchor_error', fit, names[a], names[b], 'median_min',
                         np.median(e[m]) * bmin, n=int(m.sum())))
    return rows


def warp_magnitude(study: Study, fit: Fit, pairs: Sequence[Pair] | None = None, **_) -> list[dict]:
    """Mean |m(t) − t| (minutes) of the A ← B map per evaluation pair, over every sample pair."""
    grid = np.arange(study.axis.n_bins, dtype=float)
    rows = []
    for a, b in (eval_pairs(study) if pairs is None else pairs):
        A, B = study[a], study[b]
        mags = [np.abs(np.asarray(fit.map(b, a, grid, i=j, j=i)) - grid).mean()
                for i in range(len(A)) for j in range(len(B))]
        rows.append(_row('warp_magnitude', fit, A.name, B.name, 'mean_min', np.mean(mags) * study.axis.bin_min))
    return rows


def silhouette(study: Study, fit: Fit, pairs: Sequence[Pair] | None = None, **_) -> list[dict]:
    """Batch-label silhouette after warping A into B's frame (diagnostics.study_silhouette; lower =
    less batch separation; largely RT-invariant, so secondary). Batch-level maps only: NaN for a
    per-sample fit, which has no single warp per A sample."""
    from ..warps import warp_rows
    from .diagnostics import study_silhouette

    grid = np.arange(study.axis.n_bins, dtype=float)
    rows = []
    for a, b in (eval_pairs(study) if pairs is None else pairs):
        A, B = study[a], study[b]
        if fit.per_sample:
            value = np.nan
        else:
            src = np.asarray(fit.map(b, a, grid), dtype=float)
            value = study_silhouette(warp_rows(A.chromas, src), B.chromas)
        rows.append(_row('silhouette', fit, A.name, B.name, 'silhouette', value))
    return rows


PROTOCOLS: dict[str, Callable[..., list[dict]]] = {
    'tic_bframe': tic_bframe,
    'tic_midpoint': tic_midpoint,
    'map_error': map_error,
    'cycle_inconsistency': cycle_inconsistency,
    'heldout_anchor_error': heldout_anchor_error,
    'warp_magnitude': warp_magnitude,
    'silhouette': silhouette,
}
NEEDS_TRUTH = frozenset({'map_error', 'cycle_inconsistency'})
NEEDS_REFIT = frozenset({'heldout_anchor_error'})


def available() -> list[str]:
    return sorted(PROTOCOLS)


def run(name: str, study: Study, fit: Fit, **ctx) -> list[dict]:
    """Run protocol `name`; ctx: anchors=, refit= (heldout_anchor_error), pairs= (pairwise ones)."""
    if name not in PROTOCOLS:
        raise KeyError(f'unknown protocol {name!r}; available: {", ".join(available())}')
    if name in NEEDS_TRUTH and study.truth is None:
        raise ValueError(f'protocol {name!r} needs a study with truth (synthetic); {study.name!r} has none')
    return PROTOCOLS[name](study, fit, **ctx)
