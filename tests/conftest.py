"""Shared fixtures: synthetic data, and the private fish-oil data when it is present."""

from __future__ import annotations

from pathlib import Path

import pytest

import synth

REPO = Path(__file__).resolve().parent.parent
FISH_OIL = REPO / 'data' / 'fish_oil'  # private; a symlink to warp-align's data, never committed


@pytest.fixture(scope='session')
def synthetic_pair():
    """((chromas_a, raw_a), (chromas_b, raw_b)): batch B shifted by synth.SHIFT bins."""
    return synth.pair_shift()


@pytest.fixture(scope='session')
def batches(synthetic_pair):
    from alignbench.data import Batch

    (ca, ra), (cb, rb) = synthetic_pair
    return (
        Batch('a', ca, ra, tuple(f'a{i}' for i in range(len(ca)))),
        Batch('b', cb, rb, tuple(f'b{i}' for i in range(len(cb)))),
    )


@pytest.fixture(scope='session')
def fish_oil_root() -> Path:
    if not (FISH_OIL / 'batch_sep2015' / 'chroma').is_dir():
        pytest.skip('private fish-oil data not present under data/fish_oil')
    return FISH_OIL
