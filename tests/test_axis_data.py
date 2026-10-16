from __future__ import annotations

import numpy as np
import pytest

from alignbench.axis import FISH_OIL_AXIS, TimeAxis
from alignbench.data import Batch, Study


def test_fish_oil_axis_values():
    ax = FISH_OIL_AXIS
    assert (ax.n_bins, ax.bin_min, ax.last_min, ax.eval_start_bin) == (200, 0.225, 199 * 0.225, 22)
    assert ax.eval_bins == slice(22, 200)
    assert ax.to_dict() == {'n_bins': 200, 'run_min': 45.0, 'eval_start_min': 5.0}


def test_axis_times_read_only_and_other_geometry():
    ax = TimeAxis(n_bins=600, run_min=60.0, eval_start_min=2.0)
    assert ax.times.shape == (600,) and ax.eval_start_bin == 20
    with pytest.raises(ValueError):
        ax.times[0] = 1.0


@pytest.mark.parametrize('kw', [dict(n_bins=1, run_min=1.0), dict(n_bins=10, run_min=0.0),
                                dict(n_bins=10, run_min=10.0, eval_start_min=9.5)])
def test_axis_rejects_bad_geometry(kw):
    with pytest.raises(ValueError):
        TimeAxis(**kw)


def _batch(name, n=2, t=200, axis=FISH_OIL_AXIS):
    tic = np.ones((n, t), np.float32)
    tic[0, :5] = 0
    return Batch(name, np.ones((n, t, 3), np.float32), tic, tuple(map(str, range(n))), axis)


def test_batch_basics_and_validation():
    b = _batch('x')
    assert len(b) == 2 and b.tic.shape == (2, 200)
    assert b.acquisition_start.tolist() == [5, 0]
    with pytest.raises(ValueError, match='time bins'):
        _batch('x', t=100)
    with pytest.raises(ValueError, match='disagree'):
        Batch('x', np.ones((2, 200, 3)), np.ones((3, 200)), ('a', 'b'))


def test_study():
    s = Study([_batch('x'), _batch('y')], name='s')
    assert s.axis is FISH_OIL_AXIS and len(s) == 2 and s.batch_names == ('x', 'y')
    assert s['y'] is s.batches[1] and s[0] is s.batches[0] and s.index('y') == 1
    with pytest.raises(KeyError):
        s.index('z')
    with pytest.raises(ValueError, match='at least one'):
        Study([])
    with pytest.raises(ValueError, match='duplicate'):
        Study([_batch('x'), _batch('x')])
    other = TimeAxis(100, 45.0)
    with pytest.raises(ValueError, match='axis'):
        Study([_batch('x'), _batch('y', t=100, axis=other)])
