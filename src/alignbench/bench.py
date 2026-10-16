"""The benchmark runner: methods × datasets × seeds → cached cells → tidy results and reports.

A cell is one method × dataset × seed × parameters, scored under a list of protocols. Its result
is written to <out>/cells/<hash>.json: the spec, the tidy rows, provenance, the axis and timing.
The hash covers the spec only (not the code version), so `run` resumes by skipping cached cells;
each cell records the provenance it was computed with, and `report` lists every distinct
provenance it aggregates (rerun into a fresh directory after changing code).

collect() gathers every cell into results.csv; report() writes report.txt and table.tex. Both
are deterministic: rows are sorted, and timings never reach the report.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import shlex
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, datasets, methods
from .anchors import harvest
from .data import Study
from .metrics import protocols as P
from .provenance import header

DEFAULT_PROTOCOLS = ('tic_bframe', 'tic_midpoint')
RESULT_COLUMNS = (
    'cell',
    'method',
    'dataset',
    'seed',
    'protocol',
    'batch_a',
    'batch_b',
    'sample_a',
    'sample_b',
    'metric',
    'value',
)


@dataclass(frozen=True)
class CellSpec:
    method: str
    dataset: str
    seed: int = 0
    protocols: tuple[str, ...] = DEFAULT_PROTOCOLS
    method_params: dict = field(default_factory=dict)
    dataset_args: dict = field(default_factory=dict)  # e.g. {'root': 'data/fish_oil'} or {'mz_bins': 300}

    def to_dict(self) -> dict:
        d = asdict(self)
        d['protocols'] = list(self.protocols)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> CellSpec:
        return cls(
            d['method'],
            d['dataset'],
            int(d['seed']),
            tuple(d['protocols']),
            dict(d.get('method_params', {})),
            dict(d.get('dataset_args', {})),
        )

    @property
    def key(self) -> str:
        """Hash of the spec: canonical JSON, sha256, 16 hex digits."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @property
    def label(self) -> str:
        return f'{self.method}@{self.dataset}#{self.seed}'


def cell_path(out: Path, spec: CellSpec) -> Path:
    return Path(out) / 'cells' / f'{spec.key}.json'


def is_cached(out: Path, spec: CellSpec) -> bool:
    return cell_path(out, spec).exists()


# ── One cell ─────────────────────────────────────────────────────────────────


def _accepts(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
        return True
    return name in params or any(p.kind is p.VAR_KEYWORD for p in params.values())


def load_dataset(spec: CellSpec) -> Study:
    """The cell's study; the seed is passed to loaders that take one (synthetic), not to fixed data."""
    loader = datasets.get(spec.dataset)
    kw = dict(spec.dataset_args)
    if _accepts(loader, 'seed'):
        kw['seed'] = spec.seed
    return loader(**kw)


def compute_cell(spec: CellSpec, study: Study | None = None) -> dict:
    """Fit the method and run every protocol: the cell record (not yet written)."""
    t0 = time.perf_counter()
    study = load_dataset(spec) if study is None else study
    method = methods.get(spec.method)
    needs_refit = any(p in P.NEEDS_REFIT for p in spec.protocols)
    anchors = harvest(study) if method.needs_anchors or needs_refit else None
    t_fit = time.perf_counter()
    fit = method.fit(study, anchors if method.needs_anchors else None, **spec.method_params)
    rows, skipped = [], {}
    for name in spec.protocols:
        if name in P.NEEDS_TRUTH and study.truth is None:
            skipped[name] = 'needs a synthetic study (truth)'
            continue
        if name in P.NEEDS_BATCH_MAP and fit.per_sample:
            skipped[name] = f'{spec.method} is per sample pair (no batch-level map)'
            continue
        ctx: dict[str, Any] = {}
        if name in P.NEEDS_REFIT:
            if not method.needs_anchors:
                skipped[name] = f'{spec.method} does not fit to harvested anchors'
                continue
            ctx = {'anchors': anchors, 'refit': lambda a: method.fit(study, a, **spec.method_params)}
        rows += P.run(name, study, fit, **ctx)
    t_end = time.perf_counter()
    return {
        'spec': spec.to_dict(),
        'key': spec.key,
        'alignbench': __version__,
        'provenance': header(),
        'study': {
            'name': study.name,
            'batches': list(study.batch_names),
            'axis': study.axis.to_dict(),
            'meta': study.meta,
        },
        'fit': {'method': fit.method, 'params': fit.params, 'per_sample': fit.per_sample},
        'skipped': skipped,
        'rows': rows,
        'timing_s': {'prepare': t_fit - t0, 'fit_and_score': t_end - t_fit},
    }


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_json_default) + '\n')
    tmp.replace(path)  # atomic: a crashed cell leaves no partial result behind


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple | set):
        return sorted(o) if isinstance(o, set) else list(o)
    return str(o)


def run_cell(spec: CellSpec, out: Path, force: bool = False) -> Path:
    """Compute one cell and cache it (skipped if cached, unless force)."""
    path = cell_path(out, spec)
    if path.exists() and not force:
        return path
    _write_json(path, compute_cell(spec))
    return path


# ── Sweeps ───────────────────────────────────────────────────────────────────


def expand(
    methods_: Sequence[str],
    datasets_: Sequence[str],
    seeds: Iterable[int],
    protocols: Sequence[str] = DEFAULT_PROTOCOLS,
    method_params: dict[str, dict] | None = None,
    dataset_args: dict | None = None,
) -> list[CellSpec]:
    """Every cell of a sweep. method_params: {method name: params}; dataset_args go to every dataset."""
    method_params = method_params or {}
    return [
        CellSpec(m, d, int(s), tuple(protocols), dict(method_params.get(m, {})), dict(dataset_args or {}))
        for d in datasets_
        for s in seeds
        for m in methods_
    ]


def run(specs: Sequence[CellSpec], out: Path, log=print) -> list[Path]:
    """Compute every uncached cell, in order; return all cell paths. Resumable."""
    paths = []
    for spec in specs:
        cached = is_cached(out, spec)
        log(f'{"cached " if cached else "running"} {spec.label} [{spec.key}]')
        paths.append(run_cell(spec, out))
    return paths


def cell_command(spec: CellSpec, out: Path | str) -> list[str]:
    """The `alignbench cell` command line that computes one cell."""
    cmd = [
        'alignbench',
        'cell',
        '--method',
        spec.method,
        '--dataset',
        spec.dataset,
        '--seed',
        str(spec.seed),
        '--protocols',
        ','.join(spec.protocols),
        '--out',
        str(out),
    ]
    for k, v in sorted(spec.method_params.items()):
        cmd += ['--param', f'{k}={json.dumps(v)}']
    for k, v in sorted(spec.dataset_args.items()):
        cmd += ['--dataset-arg', f'{k}={json.dumps(v)}']
    return cmd


def task_lines(specs: Sequence[CellSpec], out: Path | str) -> list[str]:
    """One task-spooler submission per uncached cell (CPU-only: -G 0), for `run --emit-tasks`."""
    return [
        shlex.join(['task', '-G', '0', '-n', f'ab:{s.label}', *cell_command(s, out)])
        for s in specs
        if not is_cached(Path(out), s)
    ]


# ── Results and reports ──────────────────────────────────────────────────────


def load_cells(out: Path) -> list[dict]:
    """Every cached cell under out, sorted by (dataset, seed, method, key)."""
    cells = [json.loads(p.read_text()) for p in sorted((Path(out) / 'cells').glob('*.json'))]
    return sorted(cells, key=lambda c: (c['spec']['dataset'], c['spec']['seed'], c['spec']['method'], c['key']))


def collect(out: Path) -> Path:
    """Write out/results.csv: every row of every cell, with the cell's method, dataset and seed."""
    path = Path(out) / 'results.csv'
    with path.open('w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(RESULT_COLUMNS)
        for c in load_cells(out):
            s = c['spec']
            for r in c['rows']:
                w.writerow(
                    [
                        c['key'],
                        s['method'],
                        s['dataset'],
                        s['seed'],
                        r['protocol'],
                        r['batch_a'],
                        r['batch_b'],
                        r['sample_a'],
                        r['sample_b'],
                        r['metric'],
                        repr(float(r['value'])),
                    ]
                )
    return path


# Headline numbers of each protocol: (metric, label, how to reduce the rows of one cell)
SUMMARY = {
    'tic_bframe': ('tic_r', 'TIC r, B-frame', 'pairs'),
    'tic_midpoint': ('tic_r', 'TIC r, midpoint', 'pairs'),
    'map_error': ('median_all_min', 'map error (min), median', 'overall'),
    'cycle_inconsistency': ('median_min', 'cycle incons. (min)', 'overall'),
    'heldout_anchor_error': ('median_min', 'held-out anchor err. (min)', 'overall'),
    'warp_magnitude': ('mean_min', 'warp magnitude (min)', 'pairs'),
    'silhouette': ('silhouette', 'silhouette', 'pairs'),
}
DECIMALS = {'tic_r': 3, 'silhouette': 3}  # everything else (minutes): DEFAULT_DECIMALS
DEFAULT_DECIMALS = 2  # minutes; the brief asked for 1, which prints map errors of ~0.1 min as "0.1 ± 0.0"


def cell_summary(cell: dict, protocol: str) -> float:
    """A cell's headline number for one protocol: per pair the mean over sample pairs, then the mean
    over pairs ('pairs'); or the overall row ('overall'). NaN if the protocol did not run."""
    metric, _, how = SUMMARY[protocol]
    rows = [r for r in cell['rows'] if r['protocol'] == protocol and r['metric'] == metric]
    if not rows:
        return float('nan')
    if how == 'overall':
        return float(next(r['value'] for r in rows if r['batch_a'] == '*'))
    per_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        per_pair[(r['batch_a'], r['batch_b'])].append(r['value'])
    return float(np.nanmean([np.nanmean(v) for v in per_pair.values()]))


def summary_table(cells: list[dict]) -> tuple[list[str], list[tuple[str, str]], dict]:
    """(methods, [(dataset, protocol)] columns, {(method, dataset, protocol): [per-seed values]})."""
    vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    cols = set()
    for c in cells:
        s = c['spec']
        for p in s['protocols']:
            if p in SUMMARY and p not in c['skipped']:
                vals[(s['method'], s['dataset'], p)].append(cell_summary(c, p))
                cols.add((s['dataset'], p))
    order = {p: i for i, p in enumerate(SUMMARY)}
    columns = sorted(cols, key=lambda dp: (dp[0], order[dp[1]]))
    return sorted({c['spec']['method'] for c in cells}), columns, vals


def _fmt(v: list[float], decimals: int) -> str:
    v = [x for x in v if not np.isnan(x)]
    if not v:
        return '—'
    m = float(np.mean(v))
    return f'{m:.{decimals}f}' if len(v) == 1 else f'{m:.{decimals}f} ± {float(np.std(v, ddof=1)):.{decimals}f}'


def _decimals(protocol: str, decimals: int | None) -> int:
    return decimals if decimals is not None else DECIMALS.get(SUMMARY[protocol][0], DEFAULT_DECIMALS)


def report(out: Path, decimals: int | None = None) -> tuple[Path, Path]:
    """Write out/report.txt and out/table.tex from the cached cells (and refresh results.csv)."""
    out = Path(out)
    cells = load_cells(out)
    if not cells:
        raise FileNotFoundError(f'no cells under {out}/cells')
    collect(out)
    meths, columns, vals = summary_table(cells)
    seeds = sorted({c['spec']['seed'] for c in cells})

    provs = sorted({tuple(c['provenance']) for c in cells})
    lines = ['alignbench report', '']
    if len(provs) > 1:
        lines.append(f'WARNING: cells were computed with {len(provs)} different code versions:')
    for prov in provs:
        lines += list(prov) + ['']
    lines.append(f'Cells: {len(cells)} (results.csv; one JSON per cell under cells/), seeds {seeds}')
    for ds in sorted({c['spec']['dataset'] for c in cells}):
        c0 = next(c for c in cells if c['spec']['dataset'] == ds)
        lines.append(
            f'Dataset {ds}: axis {c0["study"]["axis"]}, batches {c0["study"]["batches"]}, '
            f'args {c0["spec"]["dataset_args"]}'
        )
    for m in meths:
        params = sorted(
            {
                json.dumps(c['fit']['params'], sort_keys=True, default=str)
                for c in cells
                if c['spec']['method'] == m
            }
        )
        lines.append(f'Method {m}: params {"; ".join(params)}')
    skipped = sorted(
        {(c['spec']['method'], c['spec']['dataset'], p, why) for c in cells for p, why in c['skipped'].items()}
    )
    for m, ds, p, why in skipped:
        lines.append(f'Skipped {p} for {m} on {ds}: {why}')
    lines += [
        '',
        'Headline numbers: mean over seeds (± s.d. over seeds). TIC r: mean over sample pairs, then over',
        'evaluation pairs; error metrics: the protocol\'s overall median. Source: the cells listed in',
        'results.csv (column "cell").',
        '',
    ]
    head = [f'{ds} | {SUMMARY[p][1]}' for ds, p in columns]
    width = max([len(m) for m in meths] + [6])
    cells_txt = [[_fmt(vals.get((m, ds, p), []), _decimals(p, decimals)) for ds, p in columns] for m in meths]
    colw = [max(len(h), *(len(r[i]) for r in cells_txt)) for i, h in enumerate(head)]
    lines.append(f'{"method":<{width}}  ' + '  '.join(f'{h:>{w}}' for h, w in zip(head, colw)))
    for m, r in zip(meths, cells_txt):
        lines.append(f'{m:<{width}}  ' + '  '.join(f'{v:>{w}}' for v, w in zip(r, colw)))
    txt = out / 'report.txt'
    txt.write_text('\n'.join(lines) + '\n')

    tex = out / 'table.tex'
    esc = lambda s: s.replace('_', r'\_').replace('%', r'\%').replace('&', r'\&').replace('#', r'\#')
    tl = [
        '% alignbench report: ' + ' | '.join(provs[0])
        if len(provs) == 1
        else '% alignbench report: mixed provenance',
        r'\begin{tabular}{l' + 'r' * len(columns) + '}',
        r'\toprule',
        'Method & ' + ' & '.join(esc(h) for h in head) + r' \\',
        r'\midrule',
    ]
    for m, r in zip(meths, cells_txt):
        tl.append(esc(m) + ' & ' + ' & '.join(v.replace('±', r'$\pm$').replace('—', '--') for v in r) + r' \\')
    tl += [r'\bottomrule', r'\end{tabular}']
    tex.write_text('\n'.join(tl) + '\n')
    return txt, tex
