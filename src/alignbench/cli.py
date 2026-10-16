"""The `alignbench` command.

    alignbench list methods|datasets|protocols
    alignbench run --methods pooled-pchip,cow-tic --dataset synthetic:k5 --seeds 0-4 \\
                   --protocols tic_bframe,map_error --out results/bench_k5/ [--emit-tasks]
    alignbench cell --method cow-tic --dataset synthetic:k5 --seed 3 --out results/bench_k5/
    alignbench report results/bench_k5/

Method parameters: --param name=VALUE (JSON value; with several methods, method.name=VALUE).
Dataset arguments: --dataset-arg name=VALUE (e.g. root='"data/fish_oil"', or --data-root PATH).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__, bench, datasets, methods
from .metrics import protocols as P


def parse_seeds(text: str) -> list[int]:
    """'0-4' → [0, 1, 2, 3, 4]; '0,2,5' → [0, 2, 5]; ranges and lists combine ('0-2,7')."""
    out: list[int] = []
    for part in text.split(','):
        lo, sep, hi = part.strip().partition('-')
        out += list(range(int(lo), int(hi) + 1)) if sep else [int(lo)]
    return sorted(set(out))


def _value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text  # bare strings need no quotes


def parse_kv(items: Sequence[str]) -> dict:
    out = {}
    for item in items:
        k, sep, v = item.partition('=')
        if not sep or not k:
            raise SystemExit(f'expected name=value, got {item!r}')
        out[k] = _value(v)
    return out


def method_params(items: Sequence[str], names: Sequence[str]) -> dict[str, dict]:
    """--param values per method: 'm.k=v' for method m only, 'k=v' for every method."""
    shared: dict = {}
    per: dict[str, dict] = {m: {} for m in names}
    for k, v in parse_kv(items).items():
        m, dot, key = k.rpartition('.')
        if dot and m in per:
            per[m][key] = v
        elif dot:
            raise SystemExit(f'--param {k}: {m!r} is not one of the methods {list(names)}')
        else:
            shared[key] = v
    return {m: {**shared, **p} for m, p in per.items()}


def _dataset_args(a) -> dict:
    args = parse_kv(a.dataset_arg)
    if a.data_root:
        args['root'] = str(a.data_root)
    return args


def _csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(',') if x.strip()]


def cmd_list(a) -> int:
    items = {'methods': methods.available, 'datasets': datasets.available, 'protocols': P.available}[a.what]()
    for name in items:
        note = ''
        if a.what == 'protocols':
            note = ' (synthetic: needs truth)' if name in P.NEEDS_TRUTH else ''
            note += ' (methods fitted to harvested anchors)' if name in P.NEEDS_REFIT else ''
            note += ' (headline for real data)' if name == P.HEADLINE else ''
        print(name + note)
    return 0


def cmd_run(a) -> int:
    meths, protocols = _csv(a.methods), _csv(a.protocols)
    specs = bench.expand(meths, _csv(a.dataset), parse_seeds(a.seeds), protocols,
                         method_params(a.param, meths), _dataset_args(a))
    if a.emit_tasks:
        for line in bench.task_lines(specs, a.out):
            print(line)
        return 0
    bench.run(specs, a.out, log=lambda s: print(s, file=sys.stderr, flush=True))
    txt, _ = bench.report(a.out, a.decimals)
    print(txt.read_text())
    return 0


def cmd_cell(a) -> int:
    spec = bench.CellSpec(a.method, a.dataset, a.seed, tuple(_csv(a.protocols)),
                          method_params(a.param, [a.method])[a.method], _dataset_args(a))
    print(bench.run_cell(spec, a.out, force=a.force))
    return 0


def cmd_report(a) -> int:
    txt, tex = bench.report(a.out, a.decimals)
    print(txt.read_text())
    print(f'{tex}', file=sys.stderr)
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument('--protocols', default=','.join(bench.DEFAULT_PROTOCOLS), help='comma-separated')
    p.add_argument('--param', action='append', default=[], metavar='[METHOD.]NAME=VALUE')
    p.add_argument('--dataset-arg', action='append', default=[], metavar='NAME=VALUE')
    p.add_argument('--data-root', type=Path, help="shorthand for --dataset-arg root=PATH (fish_oil)")
    p.add_argument('--out', type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='alignbench', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', action='version', version=f'alignbench {__version__}')
    sub = ap.add_subparsers(dest='command', required=True)

    p = sub.add_parser('list', help='list methods, datasets or protocols')
    p.add_argument('what', choices=('methods', 'datasets', 'protocols'))
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser('run', help='run every uncached cell of a sweep, then report')
    p.add_argument('--methods', required=True, help='comma-separated')
    p.add_argument('--dataset', required=True, help='comma-separated dataset ids')
    p.add_argument('--seeds', default='0', help="e.g. '0-4' or '0,2,5'")
    p.add_argument('--emit-tasks', action='store_true',
                   help='print one task-spooler submission per uncached cell instead of running')
    p.add_argument('--decimals', type=int, help='decimals in the report (default: 3 for r, 2 for minutes)')
    _common(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser('cell', help='compute one cell')
    p.add_argument('--method', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force', action='store_true', help='recompute even if cached')
    _common(p)
    p.set_defaults(fn=cmd_cell)

    p = sub.add_parser('report', help='results.csv, report.txt and table.tex from the cached cells')
    p.add_argument('out', type=Path)
    p.add_argument('--decimals', type=int)
    p.set_defaults(fn=cmd_report)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
