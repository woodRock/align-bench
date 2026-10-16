"""The B-frame and midpoint protocols against joint-align's own scripts/32_fair_benchmark.py.

Acceptance (M3): on joint-align's synthetic dataset, alignbench's tic_r for Unaligned, Pooled PCHIP
and COW-TIC equals what script 32 produced at the pinned commit (fixture from
scripts/01_make_fair_benchmark_reference.py), for both protocols and every A_i × B_j of all ten
pairs. Strict in the environment that generated the fixture; across machines within the golden
tolerance (see test_golden.py).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest

import golden
from alignbench import datasets, methods
from alignbench.metrics import protocols as P
from test_golden import CROSS_PLATFORM_ATOL

GOLDEN = Path(__file__).parent / 'golden'
FIXTURES = {p.suffixes[-2].lstrip('.'): p for p in sorted(GOLDEN.glob('fair32_842ca595.*.npz'))}
JOINT_ALIGN_RESULTS = Path(__file__).resolve().parents[2] / 'joint-align' / 'results' / 'fair_benchmark.npz'
NAMES = {'Unaligned': 'unaligned', 'Pooled PCHIP': 'pooled-pchip', 'COW-TIC': 'cow-tic'}
PROTOCOLS = {'B-frame': 'tic_bframe', 'Midpoint': 'tic_midpoint'}


def score_like_script32(study, keys) -> dict[str, np.ndarray]:
    """alignbench's tic_r arrays under script 32's npz keys ('sep2015_jul2016|COW-TIC|B-frame')."""
    rows = []
    for name in {k.split('|')[1] for k in keys}:
        fit = methods.get(NAMES[name]).fit(study, None)
        for proto in PROTOCOLS.values():
            rows += P.tic_protocol(study, fit, proto)
    out = {}
    for k in keys:
        pair, meth, proto = k.split('|')
        a, b = pair.split('_')
        out[k] = P.as_array(rows, f'batch_{a}', f'batch_{b}', 'tic_r', PROTOCOLS[proto], NAMES[meth])
    return out


@cache
def synthetic_scores() -> dict[str, np.ndarray]:
    keys = [k for k in np.load(next(iter(FIXTURES.values()))).files if not k.startswith('#')]
    return score_like_script32(datasets.load('synthetic:fishlike', seed=0), keys)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.mark.slow
def test_matches_script32_bit_for_bit_on_synthetic():
    tag = golden.platform_tag()
    if tag not in FIXTURES:
        pytest.skip(f'no fair32 fixture for {tag}')
    want = _load(FIXTURES[tag])
    if str(want['#env']) != golden.environment():
        pytest.skip(f'fixture env {want["#env"]} != this env {golden.environment()}')
    want = {k: v for k, v in want.items() if not k.startswith('#')}
    assert len(want) == 60
    assert golden.compare(synthetic_scores(), want) == []


@pytest.mark.slow
@pytest.mark.parametrize('tag', sorted(FIXTURES))
def test_close_to_script32_on_every_platform(tag):
    want = {k: v for k, v in _load(FIXTURES[tag]).items() if not k.startswith('#')}
    assert golden.compare(synthetic_scores(), want, float_atol=CROSS_PLATFORM_ATOL) == []


@pytest.mark.slow
@pytest.mark.real
def test_matches_joint_align_real_results(fish_oil_root):
    """Jesse's real-data fair benchmark (joint-align results/fair_benchmark.npz, produced on the
    private data; warpalign 1ffaa24 = the pinned ea404e75 before its package rename)."""
    if not JOINT_ALIGN_RESULTS.exists():
        pytest.skip(f'{JOINT_ALIGN_RESULTS} not present')
    z = _load(JOINT_ALIGN_RESULTS)
    keys = [k for k in z if '|' in k and k.split('|')[1] in NAMES]
    assert len(keys) == 60
    got = score_like_script32(datasets.load('fish_oil', root=fish_oil_root), keys)
    assert golden.compare(got, {k: z[k] for k in keys}) == []
