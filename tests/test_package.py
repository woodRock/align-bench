"""Packaging invariants: the core imports without torch or matplotlib."""
from __future__ import annotations

import subprocess
import sys

import alignbench


def test_version_is_set():
    assert alignbench.__version__ and alignbench.__version__ != "0+unknown"


def test_core_does_not_import_torch_or_matplotlib():
    # A fresh interpreter, so modules other tests imported do not leak in.
    code = (
        "import sys, alignbench; "
        "bad = [m for m in ('torch', 'matplotlib') if m in sys.modules]; "
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
