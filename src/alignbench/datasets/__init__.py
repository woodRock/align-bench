"""Named datasets: a registry of loaders that each return a Study.

A dataset is a callable `loader(**kwargs) -> Study`. Every path it needs is an explicit keyword
argument (no repo-relative paths, no environment variables). Built-in datasets register here;
other packages register theirs through the entry point group `alignbench.datasets`:

    [project.entry-points."alignbench.datasets"]
    my-study = "mypackage.loaders:load_my_study"

alignbench never imports a plugin package until its dataset is asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import entry_points

from ..data import Study

Loader = Callable[..., Study]
ENTRY_POINT_GROUP = 'alignbench.datasets'

_BUILTIN: dict[str, Loader] = {}


def register(name: str, loader: Loader) -> Loader:
    """Register a built-in dataset under `name` (lower-case; ':' separates a family and a variant,
    e.g. 'synthetic:k5'). Re-registering a name is an error."""
    if name in _BUILTIN:
        raise ValueError(f'dataset {name!r} is already registered')
    _BUILTIN[name] = loader
    return loader


def _plugins() -> dict[str, object]:
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def available() -> list[str]:
    """Names of every built-in and plugin dataset, sorted."""
    return sorted(set(_BUILTIN) | set(_plugins()))


def get(name: str) -> Loader:
    """The loader registered as `name` (built-ins first, then entry points)."""
    if name in _BUILTIN:
        return _BUILTIN[name]
    plugins = _plugins()
    if name in plugins:
        return plugins[name].load()  # type: ignore[attr-defined]
    raise KeyError(f'unknown dataset {name!r}; available: {", ".join(available()) or "none"}')


def load(name: str, **kwargs) -> Study:
    """Load the dataset registered as `name`; kwargs (e.g. root=Path(...)) go to its loader."""
    return get(name)(**kwargs)


from . import fish_oil  # noqa: E402  (registers the built-in datasets)

__all__ = ['ENTRY_POINT_GROUP', 'available', 'fish_oil', 'get', 'load', 'register']
