"""Anchor-graph topologies and anchor perturbations, for robustness benchmarks (trustalign.graphs).

Anchors are harvested once per study on every cross-batch sample pair (anchors.harvest); a
condition is then derived by keeping the anchors of some batch pairs (topology, thin), subsampling
each kept pair (budget), corrupting a fraction of what is kept (corrupt) and adding timing noise
(jitter).
"""

from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np

from .anchors import Anchors

Edge = tuple[int, int]  # (a, b) with a < b

TOPOLOGIES = ('complete', 'star', 'weak_star', 'chain', 'ring', 'er0.2', 'er0.4',
              'chain+1', 'chain+2', 'chain+3', 'density0.6', 'density0.4', 'density0.2')


def weak_hub(sizes: list[int]) -> int:
    """The batch with the fewest samples (lowest index on ties)."""
    return int(np.argmin(sizes))


def hub_of(topology: str, sizes: list[int]) -> int:
    """The hub a star fit uses: the weak hub for 'weak_star', else batch 0 (the reference)."""
    return weak_hub(sizes) if topology == 'weak_star' else 0


def _connected(K: int, edges: set[Edge]) -> bool:
    return len(reachable(0, K, edges)) == K


def edges_for(topology: str, K: int, sizes: list[int], rng: np.random.Generator | None = None) -> set[Edge]:
    """Which batch pairs keep their anchors."""
    if topology == 'complete':
        return set(combinations(range(K), 2))
    if topology in ('star', 'weak_star'):
        h = hub_of(topology, sizes)
        return {(min(h, k), max(h, k)) for k in range(K) if k != h}
    if topology == 'chain':
        return {(k, k + 1) for k in range(K - 1)}
    if topology == 'ring':
        return {(k, k + 1) for k in range(K - 1)} | {(0, K - 1)}
    if topology.startswith('chain+'):
        if rng is None:
            raise ValueError('chain+k needs an rng')
        return chain_with_chords(K, int(topology[6:]), rng)
    if topology.startswith('density'):
        if rng is None:
            raise ValueError('density sweeps need an rng')
        return thinned_complete(K, float(topology[7:]), removal_order(K, rng))
    if topology.startswith('er'):
        if rng is None:
            raise ValueError('random topologies need an rng')
        p = float(topology[2:])
        while True:  # redraw until connected
            e = {pair for pair in combinations(range(K), 2) if rng.random() < p}
            if _connected(K, e):
                return e
    raise ValueError(f'unknown topology {topology!r}')


def reachable(src: int, K: int, edges: set[Edge]) -> set[int]:
    adj = adjacency(K, edges)
    seen, todo = {src}, deque([src])
    while todo:
        u = todo.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                todo.append(v)
    return seen


def adjacency(K: int, edges: set[Edge]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(K)]
    for a, b in sorted(edges):
        adj[a].append(b)
        adj[b].append(a)
    return adj


def thin(anchors: Anchors, edges: set[Edge], budget: int | None, rng: np.random.Generator) -> np.ndarray:
    """Indices of the anchors kept: every anchor of a kept pair (budget None = all), or `budget`
    of them drawn uniformly without replacement within the pair."""
    keep = []
    for a, b in sorted(edges):
        idx = np.flatnonzero((anchors.ba == a) & (anchors.bb == b))
        if budget is not None and len(idx) > budget:
            idx = np.sort(rng.choice(idx, budget, replace=False))
        keep.append(idx)
    return np.concatenate(keep) if keep else np.empty(0, int)


def corrupt(anchors: Anchors, eps: float, rng: np.random.Generator) -> tuple[Anchors, np.ndarray]:
    """Shift t_b of a fraction eps of the anchors by ±U(15, 35) bins (the joint-align tutorial's
    corrupt() rule). Returns the new anchors and a boolean mask of the bad ones."""
    n = round(eps * len(anchors))
    bad_idx = rng.choice(len(anchors), n, replace=False)
    tb = anchors.tb.astype(float).copy()
    tb[bad_idx] += rng.choice([-1, 1], n) * rng.uniform(15, 35, n)
    bad = np.zeros(len(anchors), bool)
    bad[bad_idx] = True
    return Anchors(anchors.ba, anchors.bb, anchors.sa, anchors.sb, anchors.ta, tb, axis=anchors.axis), bad


def chain_with_chords(K: int, k: int, rng: np.random.Generator) -> set[Edge]:
    """The chain 0–1–…–(K−1) plus k distinct chords between batches at least 2 apart: each chord
    closes one loop. (The ring is the chain plus the single chord (0, K−1).)"""
    chain = {(i, i + 1) for i in range(K - 1)}
    candidates = [(a, b) for a, b in combinations(range(K), 2) if b - a >= 2]
    pick = rng.choice(len(candidates), k, replace=False)
    return chain | {candidates[i] for i in pick}


def removal_order(K: int, rng: np.random.Generator) -> list[Edge]:
    """A random order in which to delete the complete graph's edges (shared by every density of
    one study, so lower densities remove a superset of the edges higher ones remove)."""
    edges = list(combinations(range(K), 2))
    return [edges[i] for i in rng.permutation(len(edges))]


def thinned_complete(K: int, keep: float, order: list[Edge]) -> set[Edge]:
    """Delete edges of the complete graph in `order`, skipping any deletion that would disconnect
    it, until round(keep · K(K−1)/2) edges remain (never fewer than the K − 1 of a tree)."""
    edges = set(combinations(range(K), 2))
    target = max(round(keep * len(edges)), K - 1)
    for e in order:
        if len(edges) <= target:
            break
        if _connected(K, edges - {e}):
            edges.discard(e)
    return edges


def jitter(anchors: Anchors, sigma: float, rng: np.random.Generator) -> Anchors:
    """Gaussian timing noise of sd sigma (bins) on every anchor's t_b: ordinary measurement
    noise, as opposed to corrupt()'s gross mismatches."""
    if sigma <= 0:
        return anchors
    tb = anchors.tb.astype(float) + rng.normal(0.0, sigma, len(anchors))
    return Anchors(anchors.ba, anchors.bb, anchors.sa, anchors.sb, anchors.ta, tb, axis=anchors.axis)
