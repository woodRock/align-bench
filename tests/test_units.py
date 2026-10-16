"""Behaviour beyond bit-identity: new parameters, error paths, small helpers."""

from __future__ import annotations

import numpy as np
import pytest

from alignbench import anchors, warps
from alignbench.axis import FISH_OIL_AXIS, TimeAxis
from alignbench.data import Batch
from alignbench.methods.tic import cow_tic_align, icoshift_align
from alignbench.metrics import diagnostics, tic


def test_match_pooled_recovers_the_shift(batches):
    a, b = batches
    res = anchors.match_pooled(a, b)
    assert len(res.anchors) >= 8
    assert all(abs(q - r + 8) <= 1 for r, q in res.anchors)
    assert anchors.format_stages(res.stages).startswith('proposed ')
    assert np.abs(warps.src_from_fn(warps.pchip_warp(res.anchors))[30:180] - np.arange(30, 180) - 8).max() <= 1


def test_match_config_is_used(batches):
    a, b = batches
    assert anchors.match_pooled(a, b, anchors.MatchConfig(sim_threshold=1.01)).anchors == []
    one = anchors.match_pooled(a, b, anchors.MatchConfig(n_ref=1)).stages
    assert one['proposed'] < anchors.match_pooled(a, b).stages['proposed']
    # Drift limit in minutes: with a finer axis the same bins are fewer minutes apart
    assert anchors.match_sample_pair(
        a.chromas[0], b.chromas[0], a.tic_raw[0], b.tic_raw[0], anchors.MatchConfig(max_drift_min=1.0)
    )[0] == []


def test_match_pooled_needs_one_axis(batches):
    a, b = batches
    other = Batch('c', b.chromas, b.tic_raw, b.sample_ids, TimeAxis(200, 90.0))
    with pytest.raises(ValueError, match='different time axes'):
        anchors.match_pooled(a, other)


def test_too_few_peaks_gives_no_anchors():
    flat = np.zeros((200, 5), np.float32)
    assert anchors.match_sample_pair(flat, flat, np.zeros(200), np.zeros(200))[0] == []


def test_monotone_interp_rejects_bad_points():
    with pytest.raises(ValueError, match='non-empty'):
        warps.monotone_interp(np.array([]), np.array([]))
    with pytest.raises(ValueError, match='strictly increasing'):
        warps.monotone_interp(np.array([1.0, 1.0]), np.array([1.0, 2.0]))


def test_anchor_warps_without_anchors_are_identity():
    for fn in (warps.pchip_warp, warps.constant_shift_warp, warps.linear_warp):
        assert fn([]) is None
    assert warps.anchor_line([(10, 7)]) == (1.0, -3.0) and warps.anchor_drift([]) == 0.0


def test_axis_scales_minutes():
    ax = TimeAxis(400, 45.0, 5.0)  # half-width bins
    fn = warps.constant_shift_warp([(100, 90)], ax)
    assert np.allclose(warps.src_from_fn(fn, ax)[50:], np.arange(50, 400) + 10)
    assert diagnostics.warp_magnitude(np.arange(400) + 2.0, ax) == pytest.approx(2 * 45.0 / 400)
    assert tic.in_range(np.arange(400.0), np.array([3]), ax).sum() == 397


def test_tic_metrics_axis_and_sigmas():
    rng = np.random.default_rng(0)
    x = rng.random((2, 50))
    ax = TimeAxis(50, 10.0, 2.0)
    valid = np.ones((2, 50), bool)
    out, outside = tic.tic_metrics(x, x, x, x, valid, valid, axis=ax, sigmas=(3.0,))
    assert set(out) == set(tic.metric_suffixes((3.0,)).values()) and outside == 0.0
    assert np.allclose(out[''].reshape(2, 2).diagonal(), 1.0)


def test_tic_aligner_warp_mag_uses_its_axis():
    rng = np.random.default_rng(1)
    q, r = rng.random(200), rng.random(200)
    ax = TimeAxis(200, 90.0)
    for fn in (cow_tic_align, icoshift_align):
        a, b = fn(q, r), fn(q, r, axis=ax)
        assert b.warp_mag == pytest.approx(2 * a.warp_mag)
    assert cow_tic_align(q, r, axis=ax, slack=0, n_segs=250).axis is ax  # no valid path: identity


def test_within_batch_needs_two_rows():
    with pytest.raises(ValueError):
        tic.within_batch_r(np.ones((1, 200)))
    assert FISH_OIL_AXIS.eval_bins == tic.EVAL_BINS
