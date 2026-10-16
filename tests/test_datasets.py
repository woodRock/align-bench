from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import synth
from alignbench import datasets
from alignbench.datasets import fish_oil


def test_fish_oil_loader_reads_the_written_format(tmp_path):
    arrays = synth.write_dataset(tmp_path)
    study = datasets.load('fish_oil', root=tmp_path)
    assert study.batch_names == fish_oil.BATCHES and study.meta['dataset'] == 'fish_oil'
    for b in study.batches:
        chromas, raws = arrays[b.name]
        assert np.array_equal(b.chromas, chromas) and np.array_equal(b.tic_raw, raws)
        assert b.chromas.dtype == np.float32 and b.sample_ids == tuple(f'sample_{i:02d}' for i in range(4))
    one = fish_oil.load(tmp_path, batches=('batch_jul2016',))
    assert one.batch_names == ('batch_jul2016',)


def test_fish_oil_loader_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        fish_oil.load_batch('batch_sep2015', tmp_path)
    d = tmp_path / 'b' / 'chroma'
    d.mkdir(parents=True)
    np.savez(d / 's.npz', chroma=np.zeros((200, 3), np.float32))
    with pytest.raises(ValueError, match='tic_raw'):
        fish_oil.load_batch('b', tmp_path)


def test_labels():
    assert fish_oil.pair_label('batch_sep2015', 'batch_jul2016') == 'Sep15×Jul16'
    assert fish_oil.pair_label('x', 'y') == 'x × y'
    assert fish_oil.batch_name('batch_jul2016') == 'Jul 2016' and fish_oil.batch_name('other') == 'other'


def test_registry(monkeypatch):
    assert 'fish_oil' in datasets.available()
    with pytest.raises(ValueError, match='already'):
        datasets.register('fish_oil', fish_oil.load)
    with pytest.raises(KeyError, match='unknown dataset'):
        datasets.get('nope')


def test_entry_point_plugins_are_discovered_lazily(monkeypatch):
    loaded = []
    ep = SimpleNamespace(name='mine', load=lambda: loaded.append(1) or (lambda **kw: kw))
    eps = lambda group: [ep] if group == datasets.ENTRY_POINT_GROUP else []
    monkeypatch.setattr(datasets, 'entry_points', eps)
    assert 'mine' in datasets.available() and not loaded
    assert datasets.load('mine', root='x') == {'root': 'x'} and loaded
