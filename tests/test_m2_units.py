"""M2 behaviour beyond bit-identity: Anchors conversions, harvest on a Study, graphs and maps
edge cases, synthetic datasets, truth metrics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alignbench import anchors, datasets, graphs, maps
from alignbench.anchors import POOLED, Anchors
from alignbench.axis import TimeAxis
from alignbench.data import Batch, Study
from alignbench.datasets import synthetic as syn
from alignbench.datasets.fish_oil import batch_dirs as fish_oil_dirs
from alignbench.metrics import anchors as manchors
from alignbench.metrics import truth as mtruth


@pytest.fixture(scope='module')
def k3():
    return datasets.load('synthetic:k3', seed=2, mz_bins=300)


@pytest.fixture(scope='module')
def k3_anchors(k3):
    return anchors.harvest(k3)


def test_anchors_from_pairs_round_trip_and_orientation():
    a = Anchors.from_pairs([(10, 8), (20, 17)], a=2, b=0, stages={'pooled': 2})
    assert a.ba.tolist() == [0, 0] and a.bb.tolist() == [2, 2] and a.sa.tolist() == [POOLED] * 2
    assert (
        a.ta.tolist() == [8, 17]
        and a.as_pairs(2, 0) == [(10, 8), (20, 17)]
        and a.as_pairs(0, 2) == [(8, 10), (17, 20)]
    )
    assert a.stages == {'pooled': 2} and a.subset(a.ta > 9).stages is None
    assert Anchors.from_pairs([]).as_pairs() == []
    with pytest.raises(ValueError, match='two different'):
        Anchors.from_pairs([(1, 2)], 1, 1)


def test_as_pairs_refuses_sub_bin_and_ambiguous(k3_anchors):
    with pytest.raises(ValueError, match='batch pairs'):
        k3_anchors.as_pairs()
    with pytest.raises(ValueError, match='sub-bin'):
        k3_anchors.as_pairs(0, 1)


def test_anchor_result_to_anchors(batches):
    a, b = batches
    res = anchors.match_pooled(a, b)
    anc = res.to_anchors()
    assert anc.as_pairs() == res.anchors and anc.stages == res.stages


def test_harvest_on_study_equals_arrays(k3, k3_anchors):
    arr = anchors.harvest_arrays([b.chromas for b in k3.batches], [b.tic_raw for b in k3.batches])
    for f in ('ba', 'bb', 'sa', 'sb', 'ta', 'tb'):
        assert np.array_equal(getattr(arr, f), getattr(k3_anchors, f))
    assert k3_anchors.axis is k3.axis and len(k3_anchors) > 100
    assert np.all(k3_anchors.ba < k3_anchors.bb)


def test_match_subbin_too_few_peaks():
    flat = np.zeros((200, 5), np.float32)
    assert anchors.match_subbin(flat, flat).shape == (0, 2)


def test_graph_errors_and_jitter_identity(k3_anchors):
    with pytest.raises(ValueError, match='unknown topology'):
        graphs.edges_for('tree', 3, [2, 2, 2])
    for t in ('chain+1', 'density0.5', 'er0.5'):
        with pytest.raises(ValueError, match='rng'):
            graphs.edges_for(t, 4, [2] * 4)
    assert graphs.jitter(k3_anchors, 0.0, np.random.default_rng(0)) is k3_anchors
    bad, mask = graphs.corrupt(k3_anchors, 0.1, np.random.default_rng(0))
    assert bad.axis is k3_anchors.axis and mask.sum() == round(0.1 * len(k3_anchors))


def test_maps_adapters():
    ident = maps.identity_map()
    t = np.array([1.0, 2.5])
    assert np.array_equal(ident(0, 1, t, i=3, j=4), t) and 'identity' in repr(ident)
    model = SimpleNamespace(latent=lambda k, t: t + k, inverse=lambda k, tau: tau - k)
    assert np.array_equal(maps.from_latent(model)(0, 2, t), t - 2)
    pf = maps.from_pair_fns({(0, 1): lambda x: x + 1})
    assert np.array_equal(pf(0, 1, t), t + 1) and np.array_equal(pf(1, 1, t), t)
    with pytest.raises(ValueError, match='no map'):
        pf(1, 0, t)
    g = maps.compose_paths({(0, 1): lambda x: x + 1, (1, 2): lambda x: 2 * x}, {(0, 2): [0, 1, 2]})
    assert np.array_equal(g(0, 2, t), 2 * (t + 1)) and np.array_equal(g(2, 2, t), t)
    with pytest.raises(ValueError, match='not connected'):
        g(2, 0, t)
    assert np.array_equal(maps.batch_map(lambda k, l, x: x * 0)(0, 1, t, i=0), [0, 0])


def test_composed_pchip_disconnected_pairs(k3_anchors):
    g, paths = maps.composed_pchip(k3_anchors, 3, {(0, 1)})
    assert set(paths) == {(0, 1), (1, 0)}
    with pytest.raises(ValueError):
        g(0, 2, np.array([50.0]))


def test_synthetic_registry_and_versions():
    assert {'synthetic:fishlike', 'synthetic:k5'} <= set(datasets.available())
    with pytest.raises(ValueError, match='version 2'):
        datasets.load('synthetic:k3@2')
    with pytest.raises(KeyError, match='unknown synthetic'):
        datasets.load('synthetic:k1')
    with pytest.raises(NotImplementedError, match='family'):
        datasets.load('synthetic:k3-ii', mz_bins=300)
    with pytest.raises(ValueError, match='mz_bins'):
        syn.make_world(mz_bins=200)
    with pytest.raises(ValueError, match='already'):
        datasets.register_family('synthetic', syn.load)


def test_fishlike_study_and_disk_round_trip(tmp_path):
    kw = dict(n_compounds=20, mz_bins=300, seed=4)
    study = datasets.load('synthetic:fishlike@1', **kw)
    assert study.batch_names == tuple(s.name for s in syn.DEFAULT_BATCHES)
    assert study.meta['dataset'] == 'synthetic:fishlike@1' and study.meta['seed'] == 4
    truth = syn.make_dataset(tmp_path, **kw)
    assert truth == study.truth.truth
    disk = datasets.load('fish_oil', root=tmp_path, batches=study.batch_names)
    assert fish_oil_dirs(tmp_path) == tuple(sorted(study.batch_names))
    for b, d in zip(study.batches, disk.batches):
        assert np.array_equal(b.chromas, d.chromas) and b.sample_ids == d.sample_ids
    syn.make_dataset(tmp_path, **kw)  # rewriting replaces the old files
    assert len(list((tmp_path / 'batch_sep2015' / 'chroma').glob('*.npz'))) == 10


def test_truth_metrics_need_truth_and_handle_empty(k3):
    plain = Study(list(k3.batches), name='no truth')
    with pytest.raises(ValueError, match='no truth'):
        mtruth.map_errors(plain, maps.identity_map())
    assert np.isnan(
        mtruth.cycle_inconsistency(Study(list(k3.batches[:2]), truth=k3.truth), maps.identity_map())[0]
    )
    errs = mtruth.map_errors(k3, maps.identity_map())
    assert set(errs) == {(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)}


def test_heldout_errors(k3, k3_anchors):
    fit = lambda anc: SimpleNamespace(map=maps.composed_pchip(anc, 3, {(0, 1), (1, 2), (0, 2)})[0])
    err = manchors.heldout_errors(fit, k3_anchors)
    assert len(err) == sum(te.sum() for _, te in manchors.rt_folds(k3_anchors))
    assert np.median(np.abs(err)) < 2.0


def test_synthetic_on_another_axis():
    ax = TimeAxis(300, 45.0, 5.0)
    s = syn.make_world_k(2, 0, axis=ax, mz_bins=300, n_compounds=10)
    assert s.axis is ax and s.batches[0].chromas.shape[1] == 300 and s.truth.axis is ax
    assert isinstance(s.batches[0], Batch)
