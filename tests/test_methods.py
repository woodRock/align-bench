"""Methods, the registry and the protocols beyond the script-32 acceptance test."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from alignbench import anchors, datasets, methods, warps
from alignbench.data import Study
from alignbench.methods import builtins
from alignbench.metrics import protocols as P

GRID = np.arange(200, dtype=float)


@pytest.fixture(scope='module')
def study(batches):
    return Study(list(batches), name='pair')


def test_registry():
    assert {'unaligned', 'pooled-pchip', 'linear', 'constant-shift', 'composed-pchip', 'cow-tic',
            'icoshift'} <= set(methods.available())
    assert isinstance(methods.get('cow-tic'), methods.Aligner)
    with pytest.raises(KeyError, match='unknown method'):
        methods.get('nope')
    with pytest.raises(ValueError, match='already'):
        methods.register(builtins.Unaligned)
    with pytest.raises(ValueError, match='name'):
        methods.register(lambda: None, name='')


def test_plugins_load_lazily_and_must_be_aligners(monkeypatch):
    loaded = []
    good = SimpleNamespace(name='mine', value='pkg:Mine', load=lambda: loaded.append(1) or builtins.Unaligned)
    bad = SimpleNamespace(name='bad', value='pkg:Bad', load=lambda: object)
    monkeypatch.setattr(methods, 'entry_points', lambda group: [good, bad])
    assert {'mine', 'bad'} <= set(methods.available()) and not loaded
    assert methods.get('mine').name == 'unaligned' and loaded
    with pytest.raises(TypeError, match='not an Aligner'):
        methods.get('bad')


def test_pooled_methods_equal_warpalign_warps(study):
    a, b = study.batches
    pooled = anchors.match_pooled(a, b).anchors
    for name, warp in (('pooled-pchip', warps.pchip_warp), ('linear', warps.linear_warp),
                       ('constant-shift', warps.constant_shift_warp)):
        fit = methods.get(name).fit(study, None)
        assert np.array_equal(fit.map(1, 0, GRID), warps.src_from_fn(warp(pooled)))
        assert np.array_equal(fit.map(0, 0, GRID), GRID)
        assert (0, 1) in fit.extras['pooled_anchors'] and fit.map(0, 1, GRID).shape == (200,)
    fit = methods.get('pooled-pchip').fit(study, None, n_ref=1)
    fit.map(1, 0, GRID)
    assert fit.extras['match_config'].n_ref == 1 and fit.params == {'n_ref': 1}


def test_per_sample_methods(study):
    fit = methods.get('cow-tic').fit(study, None)
    assert fit.per_sample and fit.params == {'signal': 'tic', 'n_segs': 4, 'slack': 22}
    with pytest.raises(ValueError, match='per sample pair'):
        fit.map(1, 0, GRID)
    m = fit.map(1, 0, GRID, i=0, j=0)
    assert np.array_equal(fit.map(1, 0, np.array([10.0, 10.5]), i=0, j=0), np.interp([10.0, 10.5], GRID, m))
    assert np.array_equal(fit.map(0, 0, GRID, i=1, j=1), GRID)
    ico = methods.get('icoshift').fit(study, None, signal='tic_raw', max_shift=5)
    assert ico.params['max_shift'] == 5 and np.all(np.abs(ico.map(1, 0, GRID, i=0, j=1) - GRID) <= 5)
    with pytest.raises(ValueError, match='signal'):
        methods.get('cow-tic').fit(study, None, signal='raw')


def test_composed_pchip_method():
    k3 = datasets.load('synthetic:k3', seed=2, mz_bins=300)
    anc = anchors.harvest(k3)
    with pytest.raises(ValueError, match='needs anchors'):
        methods.get('composed-pchip').fit(k3, None)
    fit = methods.get('composed-pchip').fit(k3, anc)
    assert fit.params['edges'] == [(0, 1), (0, 2), (1, 2)]
    chain = methods.get('composed-pchip').fit(k3, anc, edges={(0, 1), (1, 2)})
    assert chain.extras['paths'][(0, 2)] == [0, 1, 2]


def test_protocols_on_identical_batches(batches):
    a = batches[0]
    twin = Study([a, type(a)('a2', a.chromas, a.tic_raw, a.sample_ids)], name='twins')
    fit = methods.get('unaligned').fit(twin, None)
    for proto in ('tic_bframe', 'tic_midpoint'):
        rows = P.tic_protocol(twin, fit, proto)
        r = P.as_array(rows, 'a', 'a2', 'tic_r')
        assert np.allclose(np.diag(r), 1.0)
        assert {x['metric'] for x in rows} == {'tic_r', 'tic_r_s1', 'tic_r_s2', 'raw_r', 'raw_r_s1', 'raw_r_s2',
                                              'tic_r_valid', 'tic_r_nonempty', 'raw_r_nonempty', 'outside'}
        assert P.as_array(rows, 'a', 'a2', 'outside').max() == 0.0


def test_protocol_helpers(study):
    fit = methods.get('pooled-pchip').fit(study, None)
    with pytest.raises(ValueError, match='unknown TIC protocol'):
        P.tic_protocol(study, fit, 'nope')
    assert P.eval_pairs(study) == [(0, 1)]
    rows = P.tic_bframe(study, fit) + P.tic_midpoint(study, fit, sigmas=(3.0,))
    assert P.as_array(rows, 'a', 'b', 'tic_r', 'tic_bframe').mean() > P.as_array(
        P.tic_bframe(study, methods.get('unaligned').fit(study, None)), 'a', 'b').mean()
    assert {r['metric'] for r in rows if r['protocol'] == 'tic_midpoint'} >= {'tic_r_s3', 'raw_r_s3'}
    with pytest.raises(KeyError):
        P.as_array(rows, 'b', 'a')
    assert np.array_equal(P.monotone_map(np.array([5.0, 3.0, 900.0]), 200), [5.0, 5.0, 400.0])
    k5 = Study(list(study.batches), meta={'eval_pairs': [['b', 'a'], ['x', 'a']]})
    assert P.eval_pairs(k5) == [(1, 0)]


def test_other_protocols():
    k3 = datasets.load('synthetic:k3', seed=2, mz_bins=300)
    anc = anchors.harvest(k3)
    composed = methods.get('composed-pchip')
    fit = composed.fit(k3, anc, edges={(0, 1), (1, 2)})
    assert P.available() == sorted(P.PROTOCOLS)
    rows = P.run('map_error', k3, fit)
    summary = {r['metric']: r['value'] for r in rows if r['batch_a'] == '*'}
    assert summary['median_all_min'] < 0.2 and not np.isnan(summary['median_indirect_min'])
    assert len([r for r in rows if r['metric'] == 'median_min']) == 6
    cyc = {r['metric']: r['value'] for r in P.run('cycle_inconsistency', k3, fit)}
    assert cyc['median_min'] >= 0 and cyc['max_min'] >= cyc['median_min']
    refit = lambda a: composed.fit(k3, a, edges={(0, 1), (1, 2)})
    ho = P.run('heldout_anchor_error', k3, fit, anchors=anc, refit=refit)
    assert ho[0]['metric'] == 'median_bins' and ho[0]['value'] < 3 and len(ho) >= 3
    with pytest.raises(ValueError, match='refit'):
        P.run('heldout_anchor_error', k3, fit)
    wm = P.run('warp_magnitude', k3, fit)
    assert len(wm) == 3 and all(r['value'] > 0 for r in wm)
    sil = P.run('silhouette', k3, fit, pairs=[(0, 1)])
    assert -1 <= sil[0]['value'] <= 1
    cow = methods.get('cow-tic').fit(k3, None)
    assert np.isnan(P.run('silhouette', k3, cow, pairs=[(0, 1)])[0]['value'])
    for name in ('map_error', 'cycle_inconsistency'):
        with pytest.raises(ValueError, match='batch-level map'):
            P.run(name, k3, cow)
    plain = Study(list(k3.batches))
    with pytest.raises(ValueError, match='needs a study with truth'):
        P.run('map_error', plain, fit)
    with pytest.raises(KeyError, match='unknown protocol'):
        P.run('nope', k3, fit)
