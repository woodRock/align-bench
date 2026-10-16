"""The retention-time axis as a value: bin count, run length and evaluation window.

Every run is binned by retention time onto a fixed grid: bin i covers [i·bin_min, (i+1)·bin_min)
min and is placed at i·bin_min minutes everywhere. Metrics are computed on the common acquisition
window `eval_bins` (from eval_start_min to the end), because bins before every run has started
acquiring are shared zeros that inflate Pearson r.

Functions that need the axis take `axis: TimeAxis = FISH_OIL_AXIS`, so a call site that passes
nothing gets warpalign's fixed geometry (config.N_BINS, RUN_MIN, COMMON_START_MIN) exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class TimeAxis:
    n_bins: int
    run_min: float
    eval_start_min: float = 0.0

    def __post_init__(self) -> None:
        if self.n_bins < 2 or self.run_min <= 0:
            raise ValueError(f'TimeAxis needs n_bins ≥ 2 and run_min > 0, got {self}')
        if not 0.0 <= self.eval_start_min < self.last_min:
            raise ValueError(f'eval_start_min must lie in [0, last_min), got {self}')

    @property
    def bin_min(self) -> float:
        """Minutes per bin (0.225 for the fish-oil axis)."""
        return self.run_min / self.n_bins

    @cached_property
    def times(self) -> np.ndarray:
        """[n_bins] minutes of bin i (i · bin_min; spacing exactly bin_min). Read-only."""
        t = np.arange(self.n_bins) * self.bin_min
        t.flags.writeable = False
        return t

    @property
    def last_min(self) -> float:
        """Minutes of the last bin (44.775 for the fish-oil axis)."""
        return (self.n_bins - 1) * self.bin_min

    @property
    def eval_start_bin(self) -> int:
        """First bin of the evaluation window (22 for the fish-oil axis).

        warpalign computed int(COMMON_START_MIN // BIN_MIN); float floor division puts a start on
        a bin boundary one bin early for some geometries (2.0 // 0.1 == 19.0), so the quotient is
        floored with a small tolerance instead. Same value for the fish-oil axis (5.0 / 0.225)."""
        return math.floor(self.eval_start_min / self.bin_min + 1e-9)

    @property
    def eval_bins(self) -> slice:
        """Time bins every TIC metric is computed on."""
        return slice(self.eval_start_bin, self.n_bins)

    def to_dict(self) -> dict[str, float]:
        """For results files: the axis a number was computed on."""
        return {'n_bins': self.n_bins, 'run_min': self.run_min, 'eval_start_min': self.eval_start_min}


# The fish-oil study (warpalign 03_preprocess): 0–45 min in 200 bins of 0.225 min. Sep 2015 and Jan
# 2016 runs start acquiring at 5.0 min, Apr and Jul 2016 at 3.0 min, so evaluation starts at 5.0.
FISH_OIL_AXIS = TimeAxis(n_bins=200, run_min=45.0, eval_start_min=5.0)
