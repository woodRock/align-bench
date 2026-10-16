"""Write the golden fixtures tests/golden/*.npz from the pinned upstream packages.

Run once per upstream pin, in the alignbench venv with the [golden] extra installed (never in an
environment with editable upstream checkouts):

    .venv/bin/python -m pip install -e ".[golden]"
    .venv/bin/python scripts/00_make_golden.py

Fixtures are per platform (tests/golden/<pkg>_<commit8>.<system>-<machine>.npz): BLAS differs
between macOS (Accelerate) and Linux (OpenBLAS) in the last bits. CI's golden job uploads the
Linux one as an artifact. Each fixture holds the flattened outputs of tests/golden.py's case
builder run against the upstream functions, plus the upstream commit it came from ('#source') and the platform and
numeric library versions it was computed with ('#env').
"""

from __future__ import annotations

import json
import sys
from importlib.metadata import distribution
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parent.parent / 'tests'
sys.path.insert(0, str(TESTS))

import golden  # noqa: E402


def commit(dist: str) -> str:
    """The pinned commit of an installed git+ distribution; refuse editable or local installs."""
    info = json.loads(distribution(dist).read_text('direct_url.json') or '{}')
    vcs = info.get('vcs_info', {})
    if 'commit_id' not in vcs:
        sys.exit(f'{dist} is not a pinned git install ({info}); install the [golden] extra')
    return vcs['commit_id']


def main() -> None:
    out = TESTS / 'golden'
    out.mkdir(exist_ok=True)
    rev = commit('warpalign')
    flat = golden.flatten(golden.build_warpalign(golden.upstream_api()))
    flat['#source'] = np.array(f'warpalign@{rev}')
    flat['#env'] = np.array(golden.environment())
    path = out / f'warpalign_{rev[:8]}.{golden.platform_tag()}.npz'
    np.savez_compressed(path, **flat)
    print(f'{path.relative_to(TESTS.parent)}: {len(flat)} arrays from warpalign@{rev}')


if __name__ == '__main__':
    main()
