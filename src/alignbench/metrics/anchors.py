"""Held-out anchor error over RT-blocked folds, and λ selection by it (trustalign.metrics;
joint-align scripts/30_multibatch_align.py's rule).

RT blocks of RT_BLOCK bins, block b in fold b mod N_FOLDS; an anchor is held out if either of its
times falls in the fold and used for training only if neither does; the score is the median
|held-out error|. Neither TIC r nor a synthetic truth is ever used to choose anything.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..anchors import Anchors

MapFn = Callable[[int, int, np.ndarray], np.ndarray]

RT_BLOCK = 12
N_FOLDS = 5
LAMBDAS = (0.1, 1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)


def rt_folds(
    anchors: Anchors, rt_block: int = RT_BLOCK, n_folds: int = N_FOLDS
) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train, test) masks of the RT-blocked folds."""
    blk = lambda t: np.floor(t / rt_block).astype(int) % n_folds
    ba, bb = blk(anchors.ta), blk(anchors.tb)
    return [((ba != f) & (bb != f), (ba == f) | (bb == f)) for f in range(n_folds)]


def anchor_errors(g: MapFn, anchors: Anchors) -> np.ndarray:
    """Signed error in batch-bb bins of predicting every anchor's tb from its ta."""
    err = np.empty(len(anchors))
    for a, b in {(int(a), int(b)) for a, b in zip(anchors.ba, anchors.bb)}:
        m = (anchors.ba == a) & (anchors.bb == b)
        err[m] = g(a, b, anchors.ta[m]) - anchors.tb[m]
    return err


def heldout_errors(fit: Callable[[Anchors], Any], anchors: Anchors) -> np.ndarray:
    """Held-out errors (bins) of a method refitted on each fold's training anchors; fit(anchors)
    returns something with a .map (a Fit)."""
    errs = [anchor_errors(fit(anchors.subset(tr)).map, anchors.subset(te)) for tr, te in rt_folds(anchors)]
    return np.concatenate(errs)


def select_lambda(fit: Callable[[Anchors, float], Any], anchors: Anchors,
                  lambdas=LAMBDAS) -> tuple[float, dict[float, float]]:
    """The λ with the lowest median |held-out error| (bins) over the RT-blocked folds, and every
    λ's score. Ties go to the smaller λ. fit(anchors, lam) returns something with a .map."""
    folds = rt_folds(anchors)
    scores = {}
    for lam in lambdas:
        errs = [anchor_errors(fit(anchors.subset(tr), lam).map, anchors.subset(te)) for tr, te in folds]
        scores[lam] = float(np.median(np.abs(np.concatenate(errs))))
    return min(lambdas, key=lambda l: scores[l]), scores
