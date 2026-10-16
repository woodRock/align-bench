"""Error against the truth of a synthetic study (trustalign.metrics): map error, cycle
inconsistency, rejection of corrupted anchors.

trustalign's functions took a World; these take a Study whose `truth` is set (a synthetic study),
and any correspondence map g(k, l, t) in RT bins.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import permutations

import numpy as np

from ..data import Study, Truth

MapFn = Callable[[int, int, np.ndarray], np.ndarray]
Edge = tuple[int, int]


def _truth(study: Study) -> Truth:
    if study.truth is None:
        raise ValueError(f'study {study.name!r} has no truth (only synthetic studies do)')
    return study.truth


def inside_mask(study: Study, k: int, l: int, t_min: np.ndarray) -> np.ndarray:
    """Times in batch k where compounds exist in both batches: t inside batch k's observed compound
    span and its true image inside batch l's."""
    truth = _truth(study)
    lo_k, hi_k = truth.observed_span(k)
    lo_l, hi_l = truth.observed_span(l)
    true = truth.true_map(k, l, t_min)
    return (t_min >= lo_k) & (t_min <= hi_k) & (true >= lo_l) & (true <= hi_l)


def map_errors(study: Study, g: MapFn, pairs=None) -> dict[tuple[int, int], np.ndarray]:
    """|ĝ_k→l(t) − g*_k→l(t)| in minutes, at every RT bin inside the mask, for every ordered pair
    (or only the ordered pairs given)."""
    truth, times, bin_min = _truth(study), study.axis.times, study.axis.bin_min
    out = {}
    for k, l in (permutations(range(len(study)), 2) if pairs is None else pairs):
        t = times[inside_mask(study, k, l, times)]
        out[(k, l)] = np.abs(g(k, l, t / bin_min) * bin_min - truth.true_map(k, l, t))
    return out


def summarise(errs: dict[tuple[int, int], np.ndarray], direct: set[Edge]) -> dict[str, float]:
    """Median and 95th percentile over t and pairs: all pairs, pairs with direct anchors, without."""

    def stats(sel) -> tuple[float, float]:
        v = np.concatenate([errs[p] for p in sel]) if sel else np.empty(0)
        return (float(np.median(v)), float(np.percentile(v, 95))) if len(v) else (np.nan, np.nan)

    with_d = [p for p in errs if (min(p), max(p)) in direct]
    without = [p for p in errs if (min(p), max(p)) not in direct]
    out = {}
    for name, sel in (('all', list(errs)), ('direct', with_d), ('indirect', without)):
        out[f'median_{name}'], out[f'p95_{name}'] = stats(sel)
    return out


def cycle_inconsistency(study: Study, g: MapFn) -> tuple[float, float]:
    """Median and max over ordered triplets (k, m, l) of the mean |g_m→l(g_k→m(t)) − g_k→l(t)|
    (minutes) over t inside the k → l mask. Zero by construction for a joint-type model."""
    times, bin_min = study.axis.times, study.axis.bin_min
    vals = []
    for k, m, l in permutations(range(len(study)), 3):
        t = times[inside_mask(study, k, l, times)] / bin_min
        if len(t):
            vals.append(float(np.mean(np.abs(g(m, l, g(k, m, t)) - g(k, l, t)))) * bin_min)
    return (float(np.median(vals)), float(np.max(vals))) if vals else (np.nan, np.nan)


def rejection(weights: np.ndarray, bad: np.ndarray) -> tuple[float, float]:
    """Share of corrupted anchors given weight 0 (recall) and of clean anchors given weight 0."""
    recall = float(np.mean(weights[bad] == 0)) if bad.any() else np.nan
    false = float(np.mean(weights[~bad] == 0)) if (~bad).any() else np.nan
    return recall, false
