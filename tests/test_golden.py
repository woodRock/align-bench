"""Bit-identity with the pinned upstream functions (fixtures from scripts/00_make_golden.py).

Fixtures are per platform: the same upstream code gives last-bit different floats on macOS
(Accelerate BLAS) and Linux (OpenBLAS). On the platform that generated a fixture the comparison
is np.array_equal; against other platforms' fixtures integers and strings must still be equal
and floats agree within 1e-6 absolute (observed: ≤ 4e-7, in the float32 PCA of the silhouette).
CI's golden job additionally compares against the live pinned upstream, strict on any platform.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import golden

GOLDEN = Path(__file__).parent / 'golden'
PIN = 'ea404e75'
FIXTURES = {p.suffixes[-2].lstrip('.'): p for p in sorted(GOLDEN.glob(f'warpalign_{PIN}.*.npz'))}
META = ('#source', '#env')
CROSS_PLATFORM_ATOL = 1e-6
GROUPS = ['axis', 'filters', 'lis_random', 'pool', 'warp_signal', 'warp_rows', 'monotone_interp', 'invert_src',
          'stats', 'metric_suffixes', 'pair_shift', 'pair_nonlinear', 'pair_sparse']


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@pytest.fixture(scope='module')
def alignbench_outputs() -> dict[str, np.ndarray]:
    return golden.flatten(golden.build_warpalign(golden.alignbench_api()))


@pytest.fixture(scope='module')
def native() -> dict[str, np.ndarray]:
    """This platform's fixture."""
    if golden.platform_tag() not in FIXTURES:
        pytest.skip(f'no golden fixture for {golden.platform_tag()} (have {sorted(FIXTURES)})')
    return _load(FIXTURES[golden.platform_tag()])


def _group(d: dict, group: str) -> dict:
    return {k: v for k, v in d.items() if k == group or k.startswith(f'{group}/')}


def test_fixtures_are_from_the_pinned_commit():
    assert {'darwin-arm64', 'linux-x86_64'} <= set(FIXTURES)
    for path in FIXTURES.values():
        z = _load(path)
        assert str(z['#source']) == 'warpalign@ea404e753a0ac02c45918fbfe1fc58b8f0b25cd9'
        print(path.name, z['#env'])
    print('this env:', golden.environment())


def test_synthetic_inputs_unchanged(alignbench_outputs):
    # If numpy's random streams ever change, every comparison below would fail for that reason
    # alone; this says so directly. Inputs are the same on every platform.
    for path in FIXTURES.values():
        want = {k: v for k, v in _load(path).items() if '/inputs/' in k}
        assert want and golden.compare({k: alignbench_outputs[k] for k in want}, want) == []


@pytest.mark.parametrize('group', GROUPS)
def test_bit_identical_to_warpalign(group, native, alignbench_outputs):
    want, got = _group(native, group), _group(alignbench_outputs, group)
    assert want, group
    assert golden.compare(got, want) == [], f'fixture env {native["#env"]}; this env {golden.environment()}'


def test_every_fixture_key_is_covered(native, alignbench_outputs):
    assert golden.compare(alignbench_outputs, {k: v for k, v in native.items() if k not in META}) == []


@pytest.mark.parametrize('tag', sorted(FIXTURES))
def test_close_to_every_platforms_fixture(tag, alignbench_outputs):
    want = {k: v for k, v in _load(FIXTURES[tag]).items() if k not in META}
    assert golden.compare(alignbench_outputs, want, float_atol=CROSS_PLATFORM_ATOL) == []


def test_bit_identical_to_live_warpalign(alignbench_outputs):
    """The same comparison against the pinned warpalign run now, in this process: strict on any
    platform. Needs the [golden] extra; CI's golden job installs it."""
    pytest.importorskip('warpalign')
    live = golden.flatten(golden.build_warpalign(golden.upstream_api()))
    assert golden.compare(alignbench_outputs, live) == []


def test_compare_detects_a_one_ulp_change():
    fixture = _load(next(iter(FIXTURES.values())))
    key = 'pair_shift/src/pchip'
    changed = dict(fixture)
    changed[key] = fixture[key].copy()
    changed[key][100] = np.nextafter(changed[key][100], np.inf)
    assert golden.compare(changed, fixture) == [f'{key}: values differ']
    assert golden.compare(changed, fixture, float_atol=1e-6) == []
    changed[key] = fixture[key].astype(np.float32)
    assert 'float32' in golden.compare(changed, fixture)[0]
    changed[key] = fixture[key] + 1e-3
    assert 'max |diff|' in golden.compare(changed, fixture, float_atol=1e-6)[0]
