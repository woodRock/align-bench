"""Write tests/golden/fair32_<commit8>.<platform>.npz: joint-align's own scripts/32_fair_benchmark.py
numbers for the methods alignbench reimplements, the acceptance reference for the B-frame and
midpoint protocols (M3).

Runs joint-align at the pinned commit end to end on its synthetic dataset, exactly as its
README's synthetic demo does: 00_make_synthetic_data.py → 30_multibatch_align.py →
32_fair_benchmark.py, in a temporary checkout, with the pinned packages of the [golden] extra.
Scripts 30 and 32 get small bootstrap and neural-step counts: those settings only affect the
Joint and Neural joint methods, never the arrays kept here (Unaligned, Pooled PCHIP, COW-TIC;
both protocols; every A_i × B_j of all ten pairs).

    .venv/bin/python scripts/01_make_fair_benchmark_reference.py [--keep DIR]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TESTS = Path(__file__).resolve().parent.parent / 'tests'
sys.path.insert(0, str(TESTS))

import golden  # noqa: E402

REPO = 'https://github.com/woodRock/joint-align.git'
COMMIT = '842ca59535f5d18a3d9031aeb35c73ca34fc20f3'
METHODS = ('Unaligned', 'Pooled PCHIP', 'COW-TIC')
ARGS_30 = ['--n-boot', '4', '--n-boot-cv', '2']
ARGS_32 = ['--neural-steps', '5']


def run(cmd: list[str], cwd: Path) -> None:
    print('$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def checkout(dest: Path) -> Path:
    run(['git', 'init', '-q', str(dest)], dest.parent)
    run(['git', 'fetch', '-q', '--depth', '1', REPO, COMMIT], dest)
    run(['git', 'checkout', '-q', 'FETCH_HEAD'], dest)
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=dest, capture_output=True, text=True).stdout.strip()
    assert head == COMMIT, head
    return dest


def main(keep: Path | None) -> None:
    from importlib.metadata import distribution

    pinned = distribution('joint-align').read_text('direct_url.json') or ''
    if COMMIT not in pinned:
        sys.exit('joint-align is not installed at the pinned commit: install the [golden] extra')
    tmp = Path(tempfile.mkdtemp(prefix='fair32-'))
    try:
        src = checkout(tmp / 'joint-align')
        data, res, figs = tmp / 'data', tmp / 'results', tmp / 'figs'
        py = sys.executable
        run([py, 'scripts/00_make_synthetic_data.py', '--out', str(data)], src)
        run([py, 'scripts/30_multibatch_align.py', '--data', str(data), '--results', str(res), '--figs', str(figs),
             *ARGS_30], src)
        run([py, 'scripts/32_fair_benchmark.py', '--data', str(data), '--results', str(res), *ARGS_32], src)
        with np.load(res / 'fair_benchmark.npz') as z:
            keep_keys = [k for k in z.files if '|' in k and k.split('|')[1] in METHODS]
            out = {k: z[k] for k in keep_keys}
        steps = f'scripts 00 → 30 {" ".join(ARGS_30)} → 32 {" ".join(ARGS_32)}'
        out['#source'] = np.array(f'joint-align@{COMMIT} {steps}')
        out['#env'] = np.array(golden.environment())
        path = TESTS / 'golden' / f'fair32_{COMMIT[:8]}.{golden.platform_tag()}.npz'
        np.savez_compressed(path, **out)
        print(f'{path.relative_to(TESTS.parent)}: {len(keep_keys)} arrays')
        if keep:
            shutil.copytree(tmp, keep, dirs_exist_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--keep', type=Path, help='copy the checkout, data and results here for inspection')
    main(ap.parse_args().keep)
