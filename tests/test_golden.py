"""Bit-identity with the pinned upstream functions (fixtures from scripts/00_make_golden.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import golden

FIXTURES = sorted((Path(__file__).parent / 'golden').glob('warpalign_*.npz'))


@pytest.fixture(scope='module')
def warpalign_fixture() -> dict[str, np.ndarray]:
    assert len(FIXTURES) == 1, f'expected one warpalign fixture, found {FIXTURES}'
    with np.load(FIXTURES[0]) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope='module')
def alignbench_outputs() -> dict[str, np.ndarray]:
    return golden.flatten(golden.build_warpalign(golden.alignbench_api()))


def test_fixture_is_from_the_pinned_commit(warpalign_fixture):
    assert str(warpalign_fixture['#source']) == 'warpalign@ea404e753a0ac02c45918fbfe1fc58b8f0b25cd9'
    print('fixture env:', warpalign_fixture['#env'], '| this env:', golden.environment())


def test_synthetic_inputs_unchanged(warpalign_fixture, alignbench_outputs):
    # If numpy's random streams ever change, every comparison below would fail for that reason
    # alone; this says so directly.
    keys = [k for k in warpalign_fixture if '/inputs/' in k]
    assert keys
    assert golden.compare({k: alignbench_outputs[k] for k in keys}, {k: warpalign_fixture[k] for k in keys}) == []


@pytest.mark.parametrize(
    'group',
    ['axis', 'filters', 'lis_random', 'pool', 'warp_signal', 'warp_rows', 'monotone_interp', 'invert_src',
     'stats', 'metric_suffixes', 'pair_shift', 'pair_nonlinear', 'pair_sparse'],
)
def test_bit_identical_to_warpalign(group, warpalign_fixture, alignbench_outputs):
    mine = lambda k: k == group or k.startswith(f'{group}/')
    want = {k: v for k, v in warpalign_fixture.items() if mine(k)}
    got = {k: v for k, v in alignbench_outputs.items() if mine(k)}
    assert want, group
    assert golden.compare(got, want) == []


def test_every_fixture_key_is_covered(warpalign_fixture, alignbench_outputs):
    want = {k: v for k, v in warpalign_fixture.items() if k not in ('#source', '#env')}
    assert golden.compare(alignbench_outputs, want) == []


def test_bit_identical_to_live_warpalign(alignbench_outputs):
    """Same comparison against the pinned warpalign run now, in this process: strict on any
    platform (the committed fixture is strict on the platform that generated it). Needs the
    [golden] extra; CI's golden job installs it."""
    pytest.importorskip('warpalign')
    assert golden.compare(alignbench_outputs, golden.flatten(golden.build_warpalign(golden.upstream_api()))) == []


def test_compare_detects_a_one_ulp_change(warpalign_fixture):
    key = 'pair_shift/src/pchip'
    changed = dict(warpalign_fixture)
    changed[key] = warpalign_fixture[key].copy()
    changed[key][100] = np.nextafter(changed[key][100], np.inf)
    assert golden.compare(changed, warpalign_fixture) == [f'{key}: values differ']
    changed[key] = warpalign_fixture[key].astype(np.float32)
    assert 'float32' in golden.compare(changed, warpalign_fixture)[0]
