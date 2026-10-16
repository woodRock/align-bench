"""The data model: one acquisition batch (Batch) and a set of batches on one time axis (Study).

Loaders live in alignbench.datasets; every loader takes explicit paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .axis import FISH_OIL_AXIS, TimeAxis


@dataclass(frozen=True, eq=False)  # eq on arrays is ambiguous: identity equality
class Batch:
    """One acquisition batch. Samples are in sorted filename order everywhere.

    warpalign.data.Batch, plus the time axis its bins live on."""

    name: str
    chromas: np.ndarray  # [N, T, M] float32, sqrt + L2 per bin
    tic_raw: np.ndarray  # [N, T]    float32, summed raw intensity of each binned apex scan
    sample_ids: tuple[str, ...]
    axis: TimeAxis = FISH_OIL_AXIS

    def __post_init__(self) -> None:
        n, t = np.shape(self.chromas)[:2]
        if np.shape(self.tic_raw) != (n, t) or len(self.sample_ids) != n:
            raise ValueError(
                f'batch {self.name!r}: chromas {np.shape(self.chromas)}, tic_raw '
                f'{np.shape(self.tic_raw)} and {len(self.sample_ids)} sample ids disagree'
            )
        if t != self.axis.n_bins:
            raise ValueError(f'batch {self.name!r} has {t} time bins, its axis {self.axis.n_bins}')

    def __len__(self) -> int:
        return len(self.chromas)

    @property
    def acquisition_start(self) -> np.ndarray:
        """[N] first bin with signal in each sample (the run's acquisition start)."""
        return acquisition_start(self.tic_raw)

    @property
    def tic(self) -> np.ndarray:
        """Normalised TIC [N, T]: the sum over m/z of the normalised chromatogram."""
        return self.chromas.sum(axis=-1)


def acquisition_start(tic_raw: np.ndarray) -> np.ndarray:
    """First bin with non-zero raw TIC in each row of tic_raw [N, T] (T if a row is empty)."""
    nz = np.asarray(tic_raw) > 0
    return np.where(nz.any(axis=1), nz.argmax(axis=1), nz.shape[1])


class Truth(Protocol):
    """Exact answers of a synthetic study (implemented in alignbench.datasets.synthetic)."""

    def true_map(self, k: int, l: int, t_min: np.ndarray) -> np.ndarray:
        """Minutes in batch k → the minutes in batch l of the same compound."""
        ...


@dataclass(frozen=True, eq=False, init=False)
class Study:
    """Batches on one time axis, and the exact answers if the study is synthetic."""

    batches: tuple[Batch, ...]
    axis: TimeAxis = FISH_OIL_AXIS
    name: str = ''
    truth: Truth | None = None
    meta: dict = field(default_factory=dict)  # dataset id, version, generator settings

    def __init__(
        self,
        batches: Sequence[Batch],
        axis: TimeAxis | None = None,
        name: str = '',
        truth: Truth | None = None,
        meta: dict | None = None,
    ) -> None:
        batches = tuple(batches)
        if not batches:
            raise ValueError('a Study needs at least one batch')
        axis = batches[0].axis if axis is None else axis
        if any(b.axis != axis for b in batches):
            raise ValueError(f'study {name!r}: every batch must be on the axis {axis}')
        names = [b.name for b in batches]
        if len(set(names)) != len(names):
            raise ValueError(f'study {name!r}: duplicate batch names {names}')
        object.__setattr__(self, 'batches', batches)
        object.__setattr__(self, 'axis', axis)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'truth', truth)
        object.__setattr__(self, 'meta', dict(meta or {}))

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, key: int | str) -> Batch:
        if isinstance(key, str):
            return self.batches[self.index(key)]
        return self.batches[key]

    def index(self, name: str) -> int:
        for k, b in enumerate(self.batches):
            if b.name == name:
                return k
        raise KeyError(f'study {self.name!r} has no batch {name!r}')

    @property
    def batch_names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.batches)
