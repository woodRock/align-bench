"""Alignment methods: the Aligner protocol and Fit (base), the built-ins (builtins), the TIC-only
algorithms (tic), and a registry that also discovers plugins.

Other packages register methods through the entry point group `alignbench.methods`; the entry
point names a class (or factory) whose instances are Aligners:

    [project.entry-points."alignbench.methods"]
    joint = "jointalign.alignbench_plugin:Joint"

alignbench never imports a plugin package until its method is asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

from .base import Aligner, Fit
from .builtins import BUILTINS

ENTRY_POINT_GROUP = 'alignbench.methods'

_BUILTIN: dict[str, Callable[[], Aligner]] = {}


def register(factory: Callable[[], Aligner], name: str | None = None) -> None:
    """Register a built-in method (a class or zero-argument factory) under its name."""
    name = name or getattr(factory, 'name', '')
    if not name:
        raise ValueError('a method needs a name')
    if name in _BUILTIN:
        raise ValueError(f'method {name!r} is already registered')
    _BUILTIN[name] = factory


def _plugins() -> dict:
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def available() -> list[str]:
    """Names of every built-in and plugin method, sorted."""
    return sorted(set(_BUILTIN) | set(_plugins()))


def get(name: str) -> Aligner:
    """A new instance of the method registered as `name` (built-ins first, then plugins)."""
    if name in _BUILTIN:
        return _BUILTIN[name]()
    plugins = _plugins()
    if name in plugins:
        method = plugins[name].load()()
        if not isinstance(method, Aligner):
            raise TypeError(f'plugin method {name!r} ({plugins[name].value}) is not an Aligner')
        return method
    raise KeyError(f'unknown method {name!r}; available: {", ".join(available())}')


for _cls in BUILTINS:
    register(_cls)

__all__ = ['ENTRY_POINT_GROUP', 'Aligner', 'Fit', 'available', 'get', 'register']
