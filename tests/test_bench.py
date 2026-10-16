"""The runner and the CLI: cells, caching and resume, task lines, results and reports."""

from __future__ import annotations

import json
import shlex

import pytest

import synth
from alignbench import bench, cli

ARGS = ['--dataset', 'synthetic:k3', '--dataset-arg', 'mz_bins=300']


def test_cell_spec_key_and_round_trip():
    s = bench.CellSpec('pooled-pchip', 'synthetic:k3', 1, ('tic_bframe',), {'n_ref': 2}, {'mz_bins': 300})
    assert bench.CellSpec.from_dict(json.loads(json.dumps(s.to_dict()))) == s
    assert s.key == bench.CellSpec.from_dict(s.to_dict()).key and len(s.key) == 16
    assert (
        s.key
        != bench.CellSpec('pooled-pchip', 'synthetic:k3', 2, ('tic_bframe',), {'n_ref': 2}, {'mz_bins': 300}).key
    )
    assert s.label == 'pooled-pchip@synthetic:k3#1'
    specs = bench.expand(['a', 'b'], ['d'], [0, 1], method_params={'b': {'x': 1}})
    assert [(c.method, c.seed, c.method_params) for c in specs] == [
        ('a', 0, {}),
        ('b', 0, {'x': 1}),
        ('a', 1, {}),
        ('b', 1, {'x': 1}),
    ]


def test_cli_parsers():
    assert cli.parse_seeds('0-2,7,1') == [0, 1, 2, 7]
    assert cli.parse_kv(['a=1', 'b="x"', 'c=word', 'd=[1, 2]']) == {'a': 1, 'b': 'x', 'c': 'word', 'd': [1, 2]}
    with pytest.raises(SystemExit):
        cli.parse_kv(['novalue'])
    assert cli.method_params(['n_ref=2', 'cow-tic.slack=5'], ['pooled-pchip', 'cow-tic']) == {
        'pooled-pchip': {'n_ref': 2},
        'cow-tic': {'n_ref': 2, 'slack': 5},
    }
    with pytest.raises(SystemExit, match='not one of'):
        cli.method_params(['other.x=1'], ['cow-tic'])


def test_list(capsys):
    for what in ('methods', 'datasets', 'protocols'):
        assert cli.main(['list', what]) == 0
    out = capsys.readouterr().out
    assert 'cow-tic' in out and 'synthetic:k5' in out and 'tic_bframe (headline for real data)' in out


@pytest.mark.slow
def test_run_resume_report_end_to_end(tmp_path, capsys):
    out = tmp_path / 'bench'
    sweep = [
        'run',
        '--methods',
        'pooled-pchip,composed-pchip',
        *ARGS,
        '--seeds',
        '0-1',
        '--out',
        str(out),
        '--protocols',
        'tic_bframe,tic_midpoint,map_error,heldout_anchor_error',
    ]
    assert cli.main(sweep) == 0
    cells = sorted((out / 'cells').glob('*.json'))
    assert len(cells) == 4
    mtimes = [p.stat().st_mtime_ns for p in cells]
    report1 = (out / 'report.txt').read_bytes()
    tex1, csv1 = (out / 'table.tex').read_bytes(), (out / 'results.csv').read_bytes()
    capsys.readouterr()

    # Resume: every cell cached, nothing recomputed, identical report
    assert cli.main(sweep) == 0
    assert 'running' not in capsys.readouterr().err
    assert [p.stat().st_mtime_ns for p in cells] == mtimes
    assert cli.main(['report', str(out)]) == 0
    assert (out / 'report.txt').read_bytes() == report1 and (out / 'table.tex').read_bytes() == tex1
    assert (out / 'results.csv').read_bytes() == csv1

    text = report1.decode()
    assert text.startswith('alignbench report\n\nalignbench ') and 'environment: ' in text
    assert 'Skipped heldout_anchor_error for pooled-pchip' in text and 'args {\'mz_bins\': 300}' in text
    assert r'\toprule' in tex1.decode() and r'\bottomrule' in tex1.decode()
    cell = json.loads(cells[0].read_text())
    assert cell['study']['axis'] == {'n_bins': 200, 'run_min': 45.0, 'eval_start_min': 5.0}
    assert cell['provenance'][0].startswith('alignbench ') and set(cell['timing_s']) == {
        'prepare',
        'fit_and_score',
    }

    # Task lines: only the uncached cell of a larger sweep, and each one runs as a cell
    bigger = [*sweep[:2], 'pooled-pchip,composed-pchip,unaligned', *sweep[3:], '--emit-tasks']
    capsys.readouterr()
    assert cli.main(bigger) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2 and all(l.startswith("task -G 0 -n 'ab:unaligned@synthetic:k3#") for l in lines)
    argv = shlex.split(lines[0])[6:]  # after: task -G 0 -n LABEL alignbench
    assert argv[0] == 'cell' and cli.main(argv) == 0
    assert len(list((out / 'cells').glob('*.json'))) == 5


def test_cell_force_and_mixed_provenance(tmp_path, capsys):
    out = tmp_path / 'b'
    argv = ['cell', '--method', 'unaligned', *ARGS, '--protocols', 'tic_bframe,silhouette', '--out', str(out)]
    assert cli.main(argv) == 0
    path = next((out / 'cells').glob('*.json'))
    first = path.stat().st_mtime_ns
    assert cli.main(argv) == 0 and path.stat().st_mtime_ns == first
    assert cli.main([*argv, '--force']) == 0
    cell = json.loads(path.read_text())
    cell['provenance'] = ['alignbench 0.0.0 (installed, no git)', *cell['provenance'][1:]]
    other = path.with_name('0' * 16 + '.json')
    cell['key'] = other.stem
    other.write_text(json.dumps(cell))
    assert cli.main(['report', str(out), '--decimals', '1']) == 0
    text = capsys.readouterr().out
    assert 'WARNING: cells were computed with 2 different code versions' in text
    assert '% alignbench report: mixed provenance' in (out / 'table.tex').read_text()


def test_fixed_dataset_gets_no_seed_and_truth_protocols_skip(tmp_path, capsys):
    synth.write_dataset(tmp_path / 'data')
    out = tmp_path / 'b'
    assert (
        cli.main(
            [
                'cell',
                '--method',
                'unaligned',
                '--dataset',
                'fish_oil',
                '--data-root',
                str(tmp_path / 'data'),
                '--seed',
                '3',
                '--protocols',
                'tic_bframe,map_error',
                '--out',
                str(out),
            ]
        )
        == 0
    )
    cell = json.loads(next((out / 'cells').glob('*.json')).read_text())
    assert cell['skipped'] == {'map_error': 'needs a synthetic study (truth)'}
    assert cell['spec']['dataset_args']['root'] == str(tmp_path / 'data')
    assert cell['study']['batches'] == sorted(synth.BATCH_SHIFTS)


def test_report_needs_cells(tmp_path):
    with pytest.raises(FileNotFoundError, match='no cells'):
        bench.report(tmp_path)
