"""The built-in methods, each reduced to its correspondence map.

  unaligned       the identity
  constant-shift  batch A shifted by the median pooled-anchor drift  ┐ pooled anchors between the
  linear          least-squares line through the pooled anchors      │ two batches (match_pooled,
  pooled-pchip    PCHIP through the pooled anchors                   ┘ A = target, B = source)
  composed-pchip  pairwise PCHIP on harvested anchors, composed along shortest anchor paths
  cow-tic         COW of each target sample's TIC onto each source sample's TIC (per sample pair)
  icoshift        icoshift, likewise

Direction: g(k, l, t) maps RT in batch k to RT in batch l. The pooled methods are pairwise and
asymmetric: g(k, l) uses match_pooled(study[l], study[k]) (batch l plays warpalign's A, matched
in its first n_ref samples against B[0] = batch k's first sample), so g(b, a) for an evaluation
pair (A = a, B = b) is exactly warpalign's / joint-align's A → B-frame warp. Maps are built
lazily per batch (or sample) pair and cached.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
from typing import Any, ClassVar

import numpy as np

from ..anchors import DEFAULT_MATCH, AnchorResult, Anchors, MatchConfig, match_pooled
from ..data import Study
from ..maps import composed_pchip, identity_map
from ..warps import WarpFn, constant_shift_warp, linear_warp, pchip_warp
from .base import Fit
from .tic import COW_N_SEGS, COW_SLACK, ICO_MAX_SHIFT, ICO_N_INTERVALS, cow_tic_align, icoshift_align


def _src_at(fn: WarpFn | None, t: np.ndarray, bin_min: float) -> np.ndarray:
    """A minutes → minutes warp read at bins t (warps.src_from_fn at arbitrary bins: identical on
    the grid, including the near-integer snapping)."""
    t = np.asarray(t, dtype=float)
    if fn is None:
        return t.copy()
    src = np.asarray(fn(t * bin_min), dtype=np.float64) / bin_min
    near = np.round(src)
    return np.where(np.abs(src - near) < 1e-9, near, src)


class Unaligned:
    name = 'unaligned'
    needs_anchors = False

    def fit(self, study: Study, anchors: Anchors | None = None, **params: Any) -> Fit:
        return Fit(identity_map(), self.name)


class _PooledWarp:
    """A warp through pooled anchors between each batch pair (lazily matched, cached)."""

    name = ''
    needs_anchors = False
    warp = staticmethod(pchip_warp)

    def fit(self, study: Study, anchors: Anchors | None = None, **params: Any) -> Fit:
        cfg = replace(DEFAULT_MATCH, **params) if params else DEFAULT_MATCH
        pooled: dict[tuple[int, int], AnchorResult] = {}
        warps: dict[tuple[int, int], WarpFn | None] = {}
        axis = study.axis

        def g(k: int, l: int, t: np.ndarray, *, i: int | None = None, j: int | None = None) -> np.ndarray:
            if k == l:
                return np.asarray(t, dtype=float)
            if (l, k) not in pooled:
                pooled[(l, k)] = match_pooled(study[l], study[k], cfg)
                warps[(l, k)] = self.warp(pooled[(l, k)].anchors, axis)
            return _src_at(warps[(l, k)], t, axis.bin_min)

        return Fit(g, self.name, dict(params), extras={'pooled_anchors': pooled, 'match_config': cfg})


class PooledPCHIP(_PooledWarp):
    name = 'pooled-pchip'
    warp = staticmethod(pchip_warp)


class Linear(_PooledWarp):
    name = 'linear'
    warp = staticmethod(linear_warp)


class ConstantShift(_PooledWarp):
    name = 'constant-shift'
    warp = staticmethod(constant_shift_warp)


class ComposedPCHIP:
    """Pairwise PCHIP per batch pair with anchors, composed along shortest anchor paths
    (trustalign baselines.composed_pchip). edges: the batch pairs whose anchors are used
    (default: every pair with at least one anchor)."""

    name = 'composed-pchip'
    needs_anchors = True

    def fit(self, study: Study, anchors: Anchors | None, edges: set[tuple[int, int]] | None = None,
            **params: Any) -> Fit:
        if anchors is None:
            raise ValueError('composed-pchip needs anchors (anchors.harvest(study))')
        K = len(study)
        if edges is None:
            edges = {(a, b) for a, b in combinations(range(K), 2) if np.any((anchors.ba == a) & (anchors.bb == b))}
        g, paths = composed_pchip(anchors, K, edges, study.axis)
        return Fit(g, self.name, {'edges': sorted(edges)}, extras={'paths': paths})


class _PerSampleTic:
    """A TIC-only warp per (target sample, source sample) pair: g(k, l, t; i, j) aligns batch l
    sample j (query) onto batch k sample i (reference) and reads the warp at t."""

    name = ''
    needs_anchors = False
    defaults: ClassVar[dict[str, Any]] = {}

    def align(self, query: np.ndarray, ref: np.ndarray, **kw: Any):
        raise NotImplementedError

    def fit(self, study: Study, anchors: Anchors | None = None, signal: str = 'tic', **params: Any) -> Fit:
        if signal not in ('tic', 'tic_raw'):
            raise ValueError(f"signal must be 'tic' (normalised) or 'tic_raw', not {signal!r}")
        kw = {**self.defaults, **params}
        tics = [getattr(b, signal) for b in study.batches]
        grid = np.arange(study.axis.n_bins, dtype=float)
        cache: dict[tuple[int, int, int, int], np.ndarray] = {}

        def g(k: int, l: int, t: np.ndarray, *, i: int | None = None, j: int | None = None) -> np.ndarray:
            if k == l and i == j:
                return np.asarray(t, dtype=float)
            if i is None or j is None:
                raise ValueError(f'{self.name} is per sample pair: pass i (in batch {k}) and j (in batch {l})')
            key = (k, l, i, j)
            if key not in cache:
                cache[key] = self.align(tics[l][j], tics[k][i], axis=study.axis, **kw).src
            src = cache[key]
            t = np.asarray(t, dtype=float)
            return src.copy() if t.shape == grid.shape and np.array_equal(t, grid) else np.interp(t, grid, src)

        return Fit(g, self.name, {'signal': signal, **kw}, per_sample=True)


class CowTic(_PerSampleTic):
    name = 'cow-tic'
    defaults: ClassVar[dict[str, Any]] = {'n_segs': COW_N_SEGS, 'slack': COW_SLACK}
    align = staticmethod(cow_tic_align)  # type: ignore[assignment]


class Icoshift(_PerSampleTic):
    name = 'icoshift'
    defaults: ClassVar[dict[str, Any]] = {'n_intervals': ICO_N_INTERVALS, 'max_shift': ICO_MAX_SHIFT}
    align = staticmethod(icoshift_align)  # type: ignore[assignment]


BUILTINS = (Unaligned, ConstantShift, Linear, PooledPCHIP, ComposedPCHIP, CowTic, Icoshift)

__all__ = ['BUILTINS', 'ComposedPCHIP', 'ConstantShift', 'CowTic', 'Icoshift', 'Linear', 'MatchConfig',
           'PooledPCHIP', 'Unaligned']
