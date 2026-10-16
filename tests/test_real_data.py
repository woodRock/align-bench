"""The private fish-oil data: alignbench vs the pinned warpalign on every evaluation pair.

Skipped without the data (data/fish_oil) or without the [golden] extra (warpalign importable).
"""

from __future__ import annotations

import numpy as np
import pytest

from alignbench import anchors, warps
from alignbench.datasets import fish_oil
from alignbench.methods.tic import cow_tic_align, icoshift_align

pytestmark = pytest.mark.real


@pytest.fixture(scope='module')
def both(fish_oil_root):
    wa = pytest.importorskip('warpalign')
    from warpalign import data as wdata

    ours = fish_oil.load(fish_oil_root)
    theirs = {n: wdata.load_batch(n, fish_oil_root) for n in fish_oil.BATCHES}
    return wa, ours, theirs


def test_loader_matches_warpalign(both):
    _, ours, theirs = both
    for b in ours.batches:
        t = theirs[b.name]
        assert np.array_equal(b.chromas, t.chromas) and np.array_equal(b.tic_raw, t.tic_raw)
        assert b.sample_ids == t.sample_ids


@pytest.mark.parametrize('pair', fish_oil.EVAL_PAIRS + fish_oil.EXCLUDED_PAIRS)
def test_pooled_anchors_and_warps_match_warpalign(both, pair):
    from warpalign import anchors as wanchors
    from warpalign import warps as wwarps

    _, ours, _ = both
    a, b = ours[pair[0]], ours[pair[1]]
    got = anchors.match_pooled(a, b)
    want = wanchors.extract_anchors(a.chromas, b.chromas, a.tic_raw, b.tic_raw)
    assert vars(got) == vars(want)
    for mine, up in ((warps.pchip_warp, wwarps.pchip_warp), (warps.linear_warp, wwarps.linear_warp)):
        assert np.array_equal(warps.src_from_fn(mine(got.anchors)), wwarps.src_from_fn(up(want.anchors)))


def test_tic_baselines_match_warpalign(both):
    from warpalign import baselines

    _, ours, _ = both
    a, b = ours['batch_sep2015'], ours['batch_jul2016']
    for i in range(3):
        for mine, up in ((cow_tic_align, baselines.cow_tic_align), (icoshift_align, baselines.icoshift_align)):
            x, y = mine(a.tic[i], b.tic[0]), up(a.tic[i], b.tic[0])
            assert np.array_equal(x.src, y.src) and np.array_equal(x.warped, y.warped) and x.details == y.details


def test_harvest_matches_jointalign_on_real_samples(both):
    pytest.importorskip('jointalign')
    from jointalign.multibatch import harvest_anchors

    _, ours, _ = both
    chromas = [b.chromas[:2] for b in ours.batches]
    tics = [b.tic_raw[:2] for b in ours.batches]
    got, want = anchors.harvest_arrays(chromas, tics), harvest_anchors(chromas, tics)
    assert len(got) > 50
    for f in ('ba', 'bb', 'sa', 'sb', 'ta', 'tb'):
        assert np.array_equal(getattr(got, f), getattr(want, f))
