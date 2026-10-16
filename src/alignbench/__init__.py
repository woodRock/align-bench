"""
alignbench: shared core and benchmark for GC-MS retention-time alignment.

Data model and loaders, anchor matching, warp primitives, the classical baselines, every scoring
protocol, provenance, and a runner that scores any method on any dataset under the same protocols.
Every method is reduced to what it estimates: a correspondence map from retention time in one
batch to retention time in another. Methods that live in other packages register as plugins
(entry point group `alignbench.methods`); alignbench never imports them directly.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("alignbench")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0+unknown"

__all__ = ["__version__"]
