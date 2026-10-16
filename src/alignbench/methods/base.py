"""The method interface: an Aligner fits a study and returns a Fit holding a correspondence map."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..anchors import Anchors
from ..data import Study
from ..maps import CorrespondenceMap


@dataclass
class Fit:
    """What a method estimated: the map g(k, l, t; i, j) (RT bins in batch k, sample i → RT bins in
    batch l, sample j), and what produced it."""

    map: CorrespondenceMap
    method: str
    params: dict = field(default_factory=dict)
    used_anchors: np.ndarray | None = None  # mask over the anchors passed in, if the method used a subset
    model: object | None = None  # e.g. a JointWarp, for scales and robust weights
    extras: dict = field(default_factory=dict)
    per_sample: bool = False  # True if the map depends on the sample indices (COW-TIC, icoshift)


@runtime_checkable
class Aligner(Protocol):
    """A method. needs_anchors: the runner passes anchors.harvest(study) (sub-bin anchors on every
    cross-batch sample pair); methods that match their own anchors (the pooled ones) set it False."""

    name: str
    needs_anchors: bool

    def fit(self, study: Study, anchors: Anchors | None, **params: Any) -> Fit: ...
