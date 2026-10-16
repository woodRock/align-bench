from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignbench import provenance


class FakeDist:
    def __init__(self, info):
        self.info = info

    def read_text(self, name):
        return None if self.info is None else json.dumps(self.info)


@pytest.fixture
def fake(monkeypatch):
    def install(info, ver='1.2.3'):
        monkeypatch.setattr(provenance, 'version', lambda d: ver)
        monkeypatch.setattr(provenance, 'distribution', lambda d: FakeDist(info))

    return install


def test_pinned_git_install(fake):
    fake({'url': 'https://github.com/x/y.git', 'vcs_info': {'vcs': 'git', 'commit_id': 'abcdef1234567'}})
    assert provenance.revision('y') == 'y 1.2.3 @ abcdef1 (pinned https://github.com/x/y.git)'


def test_editable_checkout_clean_and_dirty(fake, tmp_path):
    repo = tmp_path / 'repo'
    (repo / 'src').mkdir(parents=True)
    git = lambda *a: subprocess.run(['git', '-C', str(repo), *a], check=True, capture_output=True)
    git('init', '-q')
    (repo / 'src' / 'm.py').write_text('x = 1\n')
    git('add', '.')
    git('-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'init')
    fake({'url': repo.as_uri(), 'dir_info': {'editable': True}})
    assert re.fullmatch(r'y 1\.2\.3 @ \w+ \(src clean\) from .+repo', provenance.revision('y'))
    (repo / 'src' / 'm.py').write_text('x = 2\n')
    assert '(src dirty)' in provenance.revision('y')
    (repo / 'src' / 'm.py').write_text('x = 1\n')
    (repo / 'notes.txt').write_text('outside src: not dirty')
    assert '(src clean)' in provenance.revision('y')


def test_editable_without_git_and_inside_another_repo(fake, tmp_path):
    other = tmp_path / 'paper'
    pkg = other / 'vendor' / 'pkg'
    pkg.mkdir(parents=True)
    subprocess.run(['git', '-C', str(other), 'init', '-q'], check=True)
    fake({'url': pkg.as_uri(), 'dir_info': {'editable': True}})
    assert provenance.revision('y').startswith('y 1.2.3 (editable, no git) from')


def test_plain_install(fake):
    fake(None)
    assert provenance.revision('y') == 'y 1.2.3 (installed, no git)'
    fake({'url': 'file:///tmp/y-1.0.tar.gz', 'archive_info': {}})
    assert provenance.revision('y') == 'y 1.2.3 (installed, no git)'


def test_missing_distribution():
    assert provenance.revision('surely-not-installed-xyz') == 'surely-not-installed-xyz (not installed)'
    assert provenance.direct_url('surely-not-installed-xyz') == {}


def test_display_path_is_home_relative():
    assert provenance.display_path(Path.home() / 'x' / 'y') == '~/x/y'
    assert provenance.display_path('/') == '/'


def test_header_lists_alignbench_plugins_extras_and_environment(monkeypatch):
    ep = SimpleNamespace(dist=SimpleNamespace(metadata={'Name': 'joint-align'}))
    monkeypatch.setattr(provenance, 'entry_points', lambda group: [ep] if group == 'alignbench.methods' else [])
    lines = provenance.header(extra=('numpy', 'joint-align'))
    assert lines[0].startswith('alignbench ') and lines[1].startswith('joint-align ')
    assert lines[2].startswith('numpy ') and len(lines) == 4
    assert lines[-1].startswith('environment: ') and 'scipy=' in lines[-1]
