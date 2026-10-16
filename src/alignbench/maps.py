"""Correspondence maps: what every alignment method estimates, and how methods are compared.

A correspondence map g(k, l, t) takes RT bins t in batch k to the RT bins in batch l where the same
compounds elute. Batch-level methods ignore the sample indices i (in k) and j (in l); per-sample
methods such as COW-TIC use them. Every scoring protocol consumes this one interface.

Adapters build maps from what methods produce:
  from_latent     a model with one warp per batch into a shared frame (latent(k, t), inverse(k, τ);
                  e.g. jointalign.JointWarp): g = inverse_l ∘ latent_k
  from_pair_fns   per-pair functions t_k → t_l (both directions), e.g. pairwise PCHIP
  compose_paths   per-edge maps chained along shortest anchor paths (trustalign.baselines)
  pchip_pairwise  the pairwise PCHIP baseline through pooled sub-bin anchors (jointalign)
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise
from typing import Protocol

import numpy as np
from scipy.interpolate import PchipInterpolator

from .anchors import Anchors
from .axis import FISH_OIL_AXIS, TimeAxis
from .graphs import Edge, adjacency

PairFn = Callable[[np.ndarray], np.ndarray]  # RT bins in one batch → RT bins in another


class CorrespondenceMap(Protocol):
    def __call__(
        self, k: int, l: int, t: np.ndarray, *, i: int | None = None, j: int | None = None
    ) -> np.ndarray:
        """RT bins in batch k (sample i) → RT bins in batch l (sample j)."""
        ...


class LatentModel(Protocol):
    def latent(self, k: int, t: np.ndarray) -> np.ndarray: ...
    def inverse(self, k: int, tau: np.ndarray) -> np.ndarray: ...


class _Map:
    """A batch-level map from a plain g(k, l, t), accepting (and ignoring) sample indices."""

    def __init__(self, g: Callable[[int, int, np.ndarray], np.ndarray], name: str = '') -> None:
        self._g, self.name = g, name

    def __call__(self, k: int, l: int, t: np.ndarray, *, i: int | None = None, j: int | None = None) -> np.ndarray:
        return self._g(k, l, t)

    def __repr__(self) -> str:
        return f'CorrespondenceMap({self.name or self._g!r})'


def batch_map(g: Callable[[int, int, np.ndarray], np.ndarray], name: str = '') -> CorrespondenceMap:
    """A batch-level map from g(k, l, t) (trustalign's MapFn)."""
    return _Map(g, name)


def identity_map() -> CorrespondenceMap:
    return _Map(lambda k, l, t: np.asarray(t, dtype=float), 'identity')


def from_latent(model: LatentModel) -> CorrespondenceMap:
    """g(k, l, t) = model.inverse(l, model.latent(k, t)), as jointalign.JointWarp.map."""
    return _Map(lambda k, l, t: model.inverse(l, model.latent(k, t)), type(model).__name__)


def from_pair_fns(fns: dict[tuple[int, int], PairFn]) -> CorrespondenceMap:
    """g from per-ordered-pair functions; k == l is the identity, a missing pair an error."""

    def g(k: int, l: int, t: np.ndarray) -> np.ndarray:
        if k == l:
            return np.asarray(t, dtype=float)
        if (k, l) not in fns:
            raise ValueError(f'no map from batch {k} to batch {l}')
        return fns[(k, l)](t)

    return _Map(g, 'pairwise')


# ── Shortest anchor paths ────────────────────────────────────────────────────


def edge_counts(anchors: Anchors, edges: set[Edge]) -> dict[Edge, int]:
    """Number of anchors on each edge (a, b), a < b."""
    return {(a, b): int(np.sum((anchors.ba == a) & (anchors.bb == b))) for a, b in edges}


def shortest_paths(K: int, edges: set[Edge], n_pair: dict[Edge, int]) -> dict[tuple[int, int], list[int]]:
    """For every ordered (k, l) connected in the graph: the fewest-hop path, ties broken by the
    largest total number of anchors on its edges, then by the lowest batch indices."""
    adj = adjacency(K, edges)
    out: dict[tuple[int, int], list[int]] = {}
    for src in range(K):
        hops, best, prev = {src: 0}, {src: 0}, {src: -1}
        frontier = [src]
        while frontier:
            nxt: list[int] = []
            for u in frontier:
                for v in adj[u]:
                    score = best[u] + n_pair[(min(u, v), max(u, v))]
                    if v not in hops:
                        hops[v], best[v], prev[v] = hops[u] + 1, score, u
                        nxt.append(v)
                    elif hops[v] == hops[u] + 1 and score > best[v]:
                        best[v], prev[v] = score, u
            frontier = sorted(nxt)
        for dst in hops:
            if dst != src:
                path, v = [dst], dst
                while prev[v] != -1:
                    v = prev[v]
                    path.append(v)
                out[(src, dst)] = path[::-1]
    return out


def compose_paths(
    edge_maps: dict[tuple[int, int], PairFn], paths: dict[tuple[int, int], list[int]]
) -> CorrespondenceMap:
    """g(k, l, ·): the edge maps chained along paths[(k, l)]; identity for k == l; an error for
    batches the graph does not connect."""

    def g(k: int, l: int, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        if k == l:
            return t
        if (k, l) not in paths:
            raise ValueError(f'batches {k} and {l} are not connected')
        for u, v in pairwise(paths[(k, l)]):
            t = edge_maps[(u, v)](t)
        return t

    return _Map(g, 'composed')


# ── Pairwise PCHIP through pooled sub-bin anchors ───────────────────────────


def pchip_pairwise(anchors: Anchors, a: int, b: int, axis: TimeAxis = FISH_OIL_AXIS) -> PairFn:
    """Pairwise baseline (jointalign.multibatch.pchip_pairwise): median-aggregate the pooled anchors
    per integer bin of batch a, keep a monotone subsequence, PCHIP with end points pinned.
    Returns t_a → t_b (identity with fewer than two anchors)."""
    pa = anchors.pair(a, b)
    if len(pa) < 2:
        return lambda t: np.asarray(t, dtype=float)
    last = axis.n_bins - 1
    ra = np.round(pa.ta).astype(int)
    groups = np.unique(ra)
    mx = np.array([np.median(pa.ta[ra == r]) for r in groups])
    my = np.array([np.median(pa.tb[ra == r]) for r in groups])
    mono = [0]
    for i in range(1, len(mx)):
        if my[i] > my[mono[-1]]:
            mono.append(i)
    xs, ys = mx[mono], my[mono]
    keep = (xs > 0) & (xs < last) & (ys > 0) & (ys < last)
    f = PchipInterpolator(np.concatenate([[0], xs[keep], [last]]), np.concatenate([[0], ys[keep], [last]]))
    return lambda t: f(np.asarray(t, dtype=float))


def composed_pchip(
    anchors: Anchors, K: int, edges: set[Edge], axis: TimeAxis = FISH_OIL_AXIS
) -> tuple[CorrespondenceMap, dict[tuple[int, int], list[int]]]:
    """Pairwise PCHIP per kept edge, composed along shortest anchor paths (trustalign
    baselines.composed_pchip). Returns (map, paths)."""
    maps: dict[tuple[int, int], PairFn] = {}
    for a, b in edges:
        maps[(a, b)] = pchip_pairwise(anchors, a, b, axis)
        maps[(b, a)] = pchip_pairwise(anchors, b, a, axis)
    paths = shortest_paths(K, edges, edge_counts(anchors, edges))
    return compose_paths(maps, paths), paths
