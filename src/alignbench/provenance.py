"""Which code produced a result: the lines every results file starts with.

One implementation for alignbench and every plugin package (replacing the provenance modules of
warpalign, jointalign and trustalign). For each installed distribution, revision() reports the
version and where the code came from, read from its install record (direct_url.json):

  pinned git+ install   'joint-align 1.1.0 @ 842ca59 (pinned https://github.com/…)'
  editable checkout     'joint-align 1.1.0 @ 842ca59 (src clean) from ~/Desktop/joint-align'
                        ('src dirty' if `git status --porcelain -- src` is non-empty)
  anything else         'joint-align 1.1.0 (installed, no git)'

Paths are home-relative (display_path), so committed reports never hold a user's absolute path.
environment() records the platform, BLAS and numeric library versions: last-bit float results
depend on them (see tests/test_golden.py).
"""

from __future__ import annotations

import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, distribution, entry_points, version
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np

PLUGIN_GROUPS = ('alignbench.methods', 'alignbench.datasets')


def _git(root: Path | str, *args: str) -> str:
    try:
        return subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ''


def display_path(p) -> str:
    """A path for reports that are committed: home-relative (~/...), never the user's absolute path."""
    p, home = Path(p).expanduser().resolve(), Path.home().resolve()
    try:
        return f'~/{p.relative_to(home)}'
    except ValueError:
        return str(p)


def direct_url(dist: str) -> dict:
    """The distribution's direct_url.json (PEP 610), or {} for an index or unrecorded install."""
    try:
        text = distribution(dist).read_text('direct_url.json')
    except PackageNotFoundError:
        return {}
    return json.loads(text) if text else {}


def checkout_revision(root: Path) -> str | None:
    """'@ <sha> (src clean|dirty)' if root is the top of a git checkout, else None."""
    top = _git(root, 'rev-parse', '--show-toplevel')
    if not top or Path(top).resolve() != Path(root).resolve():
        return None
    sha = _git(root, 'rev-parse', '--short', 'HEAD') or 'unknown commit'
    scope = ['--', 'src'] if (Path(root) / 'src').is_dir() else []
    state = 'src dirty' if _git(root, 'status', '--porcelain', *scope) else 'src clean'
    return f'@ {sha} ({state})'


def revision(dist: str) -> str:
    """One provenance line for an installed distribution (see the module docstring)."""
    try:
        ver = version(dist)
    except PackageNotFoundError:
        return f'{dist} (not installed)'
    info = direct_url(dist)
    commit = info.get('vcs_info', {}).get('commit_id')
    if commit:
        return f'{dist} {ver} @ {commit[:7]} (pinned {info.get("url", "?")})'
    url = info.get('url', '')
    if info.get('dir_info', {}).get('editable') and url.startswith('file://'):
        root = Path(unquote(urlparse(url).path))
        rev = checkout_revision(root)
        if rev:
            return f'{dist} {ver} {rev} from {display_path(root)}'
        return f'{dist} {ver} (editable, no git) from {display_path(root)}'
    return f'{dist} {ver} (installed, no git)'


def plugin_distributions() -> list[str]:
    """Distributions that register alignbench methods or datasets, sorted."""
    names = set()
    for group in PLUGIN_GROUPS:
        for ep in entry_points(group=group):
            if ep.dist is not None:
                names.add(ep.dist.metadata['Name'])
    names.discard('alignbench')
    return sorted(names)


def _blas() -> str:
    """BLAS library, and for OpenBLAS the kernel it picked for this CPU and its thread count:
    OpenBLAS chooses SIMD kernels per CPU model, so last bits differ between machines."""
    name = np.__config__.CONFIG['Build Dependencies']['blas']['name']
    try:
        from threadpoolctl import threadpool_info

        info = [i for i in threadpool_info() if i.get('user_api') == 'blas']
    except ImportError:  # pragma: no cover - threadpoolctl comes with scikit-learn
        info = []
    kern = ','.join(f'{i.get("architecture")}x{i.get("num_threads")}' for i in info)
    return f'{name}[{kern}]' if kern else name


def environment() -> str:
    """Platform, BLAS and numeric library versions: bit-identity is only expected within one."""
    libs = ' '.join(f'{d}={version(d)}' for d in ('numpy', 'scipy', 'scikit-learn'))
    return f'{platform.system()}-{platform.machine()} blas={_blas()} python={platform.python_version()} {libs}'


def header(extra: tuple[str, ...] = ()) -> list[str]:
    """The provenance lines for the top of a results file: alignbench, every plugin package, any
    extra distributions asked for, and the numeric environment."""
    dists = ['alignbench', *plugin_distributions()]
    dists += [d for d in extra if d not in dists]
    return [revision(d) for d in dists] + [f'environment: {environment()}']
