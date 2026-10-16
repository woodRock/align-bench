"""Named datasets: a registry of loaders that each return a Study.

A dataset is a callable `loader(**kwargs) -> Study`. A family is a callable
`loader(variant, version=None, **kwargs) -> Study` registered under a prefix, so
'synthetic:k5@1' resolves to the 'synthetic' family with variant 'k5' and generator version 1.
Every path a loader needs is an explicit keyword argument (no repo-relative paths, no
environment variables). Built-in datasets register here;
other packages register theirs through the entry point group `alignbench.datasets`:

    [project.entry-points."alignbench.datasets"]
    my-study = "mypackage.loaders:load_my_study"

alignbench never imports a plugin package until its dataset is asked for.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from importlib.metadata import entry_points

from ..data import Study

Loader = Callable[..., Study]
ENTRY_POINT_GROUP = 'alignbench.datasets'

_BUILTIN: dict[str, Loader] = {}
_FAMILIES: dict[str, tuple[Callable[..., Study], tuple[str, ...]]] = {}


def register(name: str, loader: Loader) -> Loader:
    """Register a built-in dataset under `name` (lower-case; ':' separates a family and a variant,
    e.g. 'synthetic:k5'). Re-registering a name is an error."""
    if name in _BUILTIN:
        raise ValueError(f'dataset {name!r} is already registered')
    _BUILTIN[name] = loader
    return loader


def register_family(prefix: str, loader: Callable[..., Study], examples: tuple[str, ...] = ()) -> None:
    """Register a parametric family: '<prefix>:<variant>[@<version>]' calls
    loader(variant, version=<version or None>, **kwargs). examples are listed by available()."""
    if prefix in _FAMILIES or prefix in _BUILTIN:
        raise ValueError(f'dataset family {prefix!r} is already registered')
    _FAMILIES[prefix] = (loader, examples)


def _plugins() -> dict[str, object]:
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def available() -> list[str]:
    """Names of every built-in and plugin dataset and the example ids of each family, sorted."""
    fams = {f'{p}:{ex}' for p, (_, examples) in _FAMILIES.items() for ex in examples}
    return sorted(set(_BUILTIN) | set(_plugins()) | fams)


def get(name: str) -> Loader:
    """The loader registered as `name`: built-ins first, then families, then entry points."""
    if name in _BUILTIN:
        return _BUILTIN[name]
    base, at, ver = name.partition('@')
    prefix, _, variant = base.partition(':')
    if prefix in _FAMILIES and variant:
        return partial(_FAMILIES[prefix][0], variant, version=int(ver) if at else None)
    plugins = _plugins()
    if name in plugins:
        return plugins[name].load()  # type: ignore[attr-defined]
    raise KeyError(f'unknown dataset {name!r}; available: {", ".join(available()) or "none"}')


def load(name: str, **kwargs) -> Study:
    """Load the dataset registered as `name`; kwargs (e.g. root=Path(...)) go to its loader."""
    return get(name)(**kwargs)


from . import fish_oil, synthetic  # noqa: E402  (register the built-in datasets)

register_family('synthetic', synthetic.load, ('fishlike', 'k3', 'k5', 'k10'))

__all__ = ['ENTRY_POINT_GROUP', 'available', 'fish_oil', 'get', 'load', 'register', 'register_family', 'synthetic']
