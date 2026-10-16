"""Bit-identity with the pinned upstream functions (fixtures from scripts/00_make_golden.py).

One fixture set per upstream package and commit: warpalign (M1), jointalign and trustalign (M2).
Fixtures are per platform: the same upstream code gives last-bit different floats on macOS
(Accelerate BLAS) and Linux (OpenBLAS), and OpenBLAS also differs between CPU models (it picks
SIMD kernels per CPU). In the exact environment that generated a fixture (its '#env') the
comparison is np.array_equal; against every fixture, integers and strings must be equal and floats
agree within 1e-6 absolute (observed: ≤ 4e-7, in the float32 PCA of the silhouette). CI's golden
job also compares against the live pinned upstreams, strict on any machine.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import numpy as np
import pytest

import golden

GOLDEN = Path(__file__).parent / 'golden'
META = ('#source', '#env')
CROSS_PLATFORM_ATOL = 1e-6

# stem → (pinned commit, case builder, alignbench API, upstream API)
SETS = {
    'warpalign': ('ea404e753a0ac02c45918fbfe1fc58b8f0b25cd9', golden.build_warpalign,
                  golden.alignbench_api, golden.upstream_api),
    'jointalign': ('842ca59535f5d18a3d9031aeb35c73ca34fc20f3', golden.build_jointalign,
                   golden.alignbench_api_m2, golden.upstream_api_m2),
    'trustalign': ('c7b3e42ae4f2031f6a33e649a378f35defb142b3', golden.build_trustalign,
                   golden.alignbench_api_m2, golden.upstream_api_m2),
}
DIST = {'warpalign': 'warpalign', 'jointalign': 'joint-align', 'trustalign': 'trustalign'}


def fixtures(stem: str) -> dict[str, Path]:
    """platform tag → fixture path for one set."""
    pin = SETS[stem][0][:8]
    return {p.suffixes[-2].lstrip('.'): p for p in sorted(GOLDEN.glob(f'{stem}_{pin}.*.npz'))}


@cache
def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


@cache
def outputs(stem: str) -> dict[str, np.ndarray]:
    """alignbench's flattened outputs for one set (computed once per session)."""
    _, build, api, _ = SETS[stem]
    return golden.flatten(build(api()))


def native(stem: str) -> dict[str, np.ndarray]:
    """This platform's fixture, if it was generated in exactly this environment (platform, BLAS
    kernel and threads, library versions): only then are BLAS-dependent floats reproducible."""
    have = fixtures(stem)
    if golden.platform_tag() not in have:
        pytest.skip(f'no {stem} fixture for {golden.platform_tag()} (have {sorted(have)})')
    fixture = _load(have[golden.platform_tag()])
    if str(fixture['#env']) != golden.environment():
        pytest.skip(f'fixture env {fixture["#env"]} != this env {golden.environment()}')
    return fixture


def _groups(stem: str) -> list[str]:
    any_fixture = _load(next(iter(fixtures(stem).values())))
    return sorted({k.split('/')[0] for k in any_fixture if not k.startswith('#')})


def _group(d: dict, group: str) -> dict:
    return {k: v for k, v in d.items() if k == group or k.startswith(f'{group}/')}


CASES = [(stem, group) for stem in SETS for group in _groups(stem)]


def test_fixtures_are_from_the_pinned_commits():
    assert set(fixtures('warpalign')) >= {'darwin-arm64', 'linux-x86_64'}
    for stem, (pin, *_) in SETS.items():
        assert fixtures(stem), stem
        for path in fixtures(stem).values():
            z = _load(path)
            assert str(z['#source']) == f'{DIST[stem]}@{pin}'
            print(path.name, z['#env'])
    print('this env:', golden.environment())


@pytest.mark.parametrize('stem', ['warpalign'])
def test_synthetic_inputs_unchanged(stem):
    # If numpy's random streams ever change, every comparison would fail for that reason alone;
    # this says so directly. Inputs are the same on every platform.
    for path in fixtures(stem).values():
        want = {k: v for k, v in _load(path).items() if '/inputs/' in k}
        assert want and golden.compare({k: outputs(stem)[k] for k in want}, want) == []


@pytest.mark.parametrize(('stem', 'group'), CASES, ids=[f'{s}-{g}' for s, g in CASES])
def test_bit_identical(stem, group):
    fixture = native(stem)
    want, got = _group(fixture, group), _group(outputs(stem), group)
    assert want, group
    assert golden.compare(got, want) == [], f'fixture env {fixture["#env"]}; this env {golden.environment()}'


@pytest.mark.parametrize('stem', list(SETS))
def test_every_fixture_key_is_covered(stem):
    fixture = native(stem)
    assert golden.compare(outputs(stem), {k: v for k, v in fixture.items() if k not in META}) == []


@pytest.mark.parametrize(
    ('stem', 'tag'), [(s, t) for s in SETS for t in fixtures(s)], ids=lambda x: str(x)
)
def test_close_to_every_platforms_fixture(stem, tag):
    want = {k: v for k, v in _load(fixtures(stem)[tag]).items() if k not in META}
    assert golden.compare(outputs(stem), want, float_atol=CROSS_PLATFORM_ATOL) == []


@pytest.mark.parametrize('stem', list(SETS))
def test_bit_identical_to_live_upstream(stem):
    """The same comparison against the pinned upstream run now, in this process: strict on any
    platform. Needs the [golden] extra; CI's golden job installs it."""
    pytest.importorskip('warpalign')
    pytest.importorskip('jointalign')
    pytest.importorskip('trustalign')
    _, build, _, upstream = SETS[stem]
    assert golden.compare(outputs(stem), golden.flatten(build(upstream()))) == []


def test_compare_detects_a_one_ulp_change():
    fixture = _load(next(iter(fixtures('warpalign').values())))
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
