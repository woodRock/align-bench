# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`alignbench`: the shared, torch-free core of the GC-MS retention-time alignment work, and a benchmark
runner on top of it. It holds the data model and loaders, anchor matching, warp primitives, the
classical baselines, every scoring protocol and provenance. warp-align (`warpalign`), joint-align
(`jointalign`) and trust-align (`trustalign`) stay separate repositories, keep only what is new in
each paper, and register their methods as plugins (entry point group `alignbench.methods`).

Every method is reduced to what it estimates: a **correspondence map** from retention time (bins)
in batch k to retention time in batch l. Every protocol scores that map the same way.

## Commands

alignbench has its **own venv**. The global `python3.11` holds editable checkouts of warpalign,
joint-align and trustalign that Jesse's runs import; installing alignbench's `[golden]` extra there
would replace them with pinned commits. Never `pip install` into the global interpreter from here.

```bash
make venv                                         # python3.11 -m venv .venv + editable install [dev]
make check                                        # lint + types + tests with coverage (≥ 90 %), no real data
make test-fast                                    # not slow, not real: seconds
make test-real                                    # private fish-oil data (~3 min; COW-TIC on every sample pair)
.venv/bin/python -m pytest -m slow                # end-to-end runs on synthetic data
.venv/bin/python -m pytest tests/test_x.py::test_name
.venv/bin/python -m pip install -e ".[golden]"    # only to regenerate tests/golden/*.npz
```

Outside the venv, always call `python3.11` explicitly (`python3` is a zsh alias that is not
expanded under `nohup`). Markers: `slow` (end-to-end) and `real` (needs the private data under
`data/fish_oil`; skipped otherwise).

## Architecture (M1–M4 done)

- `axis.py`: `TimeAxis(n_bins, run_min, eval_start_min)`, `FISH_OIL_AXIS = (200, 45.0, 5.0)`. The
  time axis is a value, never a module global. Functions that used `N_BINS`/`BIN_MIN`/`EVAL_BINS`
  take `axis: TimeAxis = FISH_OIL_AXIS`.
- `data.py`: `Batch` (warpalign's, plus `axis`), `Study` (batches, axis, name, optional `Truth`).
- `datasets/`: `fish_oil` (private format), `synthetic` (jointalign worlds and trustalign worlds),
  and a registry of named datasets. Every loader takes explicit paths.
- `anchors.py`: columnar `Anchors(ba, bb, sa, sb, ta, tb)`; two named matchers, both bit-identical
  to their origin: `match_pooled` (= `warpalign.anchors.extract_anchors`, integer bins) and `harvest`
  (= `jointalign.multibatch.harvest_anchors`, sub-bin, every cross-batch sample pair). Settings in a
  frozen `MatchConfig` whose defaults are today's constants.
- `methods/tic.py`: COW-TIC, icoshift. Anchor warps (constant shift, linear, PCHIP) are in
  `warps.py`. warpalign's `evaluation.anchor_warp`/`unaligned` (PairEval) become the B-frame
  protocol in M3; until then `tests/golden.py` composes them from the moved pieces.
- `graphs.py`: anchor-graph topologies, thin / corrupt / jitter (trustalign.graphs).
- `maps.py`: `CorrespondenceMap` protocol (k, l, t; optional sample i, j) + adapters
  (`from_latent` for JointWarp-like models, `from_pair_fns`, `compose_paths` along
  `shortest_paths`), `pchip_pairwise` / `composed_pchip`.
- `datasets/synthetic.py`: jointalign `make_world` ('synthetic:fishlike') and trustalign worlds
  ('synthetic:k<K>'), as Studies with `SyntheticTruth` (true_map, observed_span); generator
  `VERSION` in the dataset id (`synthetic:k5@1`).
- `metrics/truth.py` (map error, summarise, cycle inconsistency, rejection),
  `metrics/anchors.py` (RT-block folds, held-out error, λ selection).
- `provenance.py`: one implementation from `direct_url.json` (pinned git+ / editable checkout /
  installed); `header()` = alignbench + plugin distributions + extras + environment.
- `methods/`: `Fit` and the `Aligner` protocol (`base.py`), registry + `alignbench.methods` entry
  points (`__init__.py`), built-ins (`builtins.py`): unaligned, constant-shift, linear,
  pooled-pchip (pooled anchors per ordered pair, lazily: g(k, l) uses match_pooled(study[l],
  study[k])), composed-pchip (harvested anchors), cow-tic and icoshift (per sample pair, `i`, `j`).
- `metrics/protocols.py`: tic_bframe (headline) and tic_midpoint = joint-align
  `32_fair_benchmark.py`'s score() (unmasked r via warpalign `pearson`, constant → 0; masked r via
  `pearson_paired`), plus map_error, cycle_inconsistency, heldout_anchor_error (needs anchors +
  refit), warp_magnitude, silhouette; all return tidy rows; `run(name, study, fit, **ctx)`.
  Study evaluation pairs: `meta['eval_pairs']` (fish oil + fishlike: joint-align's ten).
- `datasets.fish_oil.load(root)` loads every `batch_*` dir sorted (joint-align's order, with
  aug2015); pass `BATCHES` for warpalign's four.
- `bench.py`: CellSpec (method × dataset × seed × params × protocols; key = sha256 of the spec),
  cells cached as `<out>/cells/<key>.json` (rows, provenance, axis, timing), `run` resumes,
  `task_lines` for `--emit-tasks` (`task -G 0 -n ab:<label> alignbench cell …`; task-spooler has
  no -q), `collect` → results.csv, `report` → report.txt + table.tex (deterministic; 3 decimals
  for r, 2 for minutes by default, `--decimals`).
- `cli.py`: `alignbench list|run|cell|report` (entry point in pyproject).

## Golden tests

`tests/golden.py` holds one case builder per upstream (`build_warpalign`, `build_jointalign`, `build_trustalign`) and two APIs mapping the same
names to the pinned upstream functions or to alignbench's. `scripts/00_make_golden.py` runs it
against upstream and writes `tests/golden/<pkg>_<commit8>.<system>-<machine>.npz` (one per platform: macOS and Linux BLAS differ in the last bits; CI uploads the Linux one as an artifact) (flattened outputs, `#source`
commit, `#env` platform + numpy/scipy/sklearn versions). `tests/test_golden.py` runs it against
alignbench and compares with `np.array_equal`, dtype and shape included. Bit-identity is only
expected on the platform and versions in `#env`; CI pins the versions with
`tests/golden/constraints.txt`, and its `golden` job also compares against the live pinned
upstreams (strict on any platform). Across machines: integers/shapes/NaNs exact, floats within
1e-3 (observed ≤ 2.5e-4 bins, from sub-bin refinement amplifying BLAS last bits). Adding a moved function = add its calls to the builder (both
APIs), regenerate, commit the fixture. Synthetic inputs come from `tests/synth.py`; their digests
are in the fixture, so a numpy RNG change fails loudly.

M3 acceptance: `scripts/01_make_fair_benchmark_reference.py` runs joint-align's scripts 00 → 30 →
32 at the pinned commit on synthetic data (temporary checkout, ~1 min) and keeps the Unaligned /
Pooled PCHIP / COW-TIC arrays as `tests/golden/fair32_842ca595.<platform>.npz`;
`tests/test_protocols.py` compares (all 60 arrays bit-identical), and, when present, against
Jesse's real-data `../joint-align/results/fair_benchmark.npz` (also 60/60 bit-identical).

## Decisions (from Jesse, 2026-10-17)

- Package and import name `alignbench`; GitHub repo `woodRock/align-bench`.
- Headline protocol for real data is **B-frame** (`tic_bframe`); midpoint is reported alongside.
- MTBLS21/MTBLS288 loaders: **not planned** (Jesse, 2026-10-17). Both are single instrument
  sequences, not multi-batch studies: MTBLS288 ran 18–19 Dec 2013 (~32 h), MTBLS21 14–17 Mar 2008
  (~2.5 days; harvest years 2005/2006 and the `_run02` repeats are all one sequence, per the
  netCDF `experiment_date_time_stamp`). The dataset registry lets anyone add a loader later.
- Citable synthetic benchmark: undecided. Until then, give every synthetic dataset a version ID
  (e.g. `synthetic:k5@1`) and decide on freezing before M4.
- trust-align may migrate in M5 with the others: Experiment 1 is finished (Jesse, 2026-10-17).
- Still to ask before M5: whether Red Herring should consume `alignbench.data`/`alignbench.anchors`.

## Rules for this repo

- **Numbers must not change.** Every function moved from an upstream package gets a golden test:
  alignbench's output vs. the pinned upstream function's, on synthetic data, with `np.array_equal`
  (or `np.allclose(..., rtol=0, atol=0)`-level strictness; where float order genuinely differs, say
  why in a comment). Fixtures are generated once by a script importing the `[golden]` pins and
  committed under `tests/golden/*.npz`. New options default to existing behaviour, with a
  bit-identical test.
- Core dependencies are numpy, scipy and scikit-learn only. No torch, no matplotlib (plots go in the
  `[plots]` extra, imported lazily). alignbench never imports a plugin package.
- No repo-relative paths (`Path(__file__).parents[…]`) and no environment variables: paths are
  explicit arguments. The CI `package` job imports from outside the checkout to catch this.
- **`data/` is private and must never be committed or pushed** (the GitHub repo is public). Code is MIT.
- **Do not commit, tag or push results**; Jesse does. Code commits for alignbench go straight to `main` (Jesse, 2026-10-17); the M5 changes to the other repositories still go through branches and PRs.
- **Never edit an upstream package (warp-align, joint-align, trust-align) while one of Jesse's runs
  uses its editable install.** Check with Jesse or the repo's `logs/`. Don't touch those
  repositories at all before M5.
- Paper runs use CPU (bit-reproducible under a seed).
- Results files are self-describing: provenance header, protocol, axis, method parameters, and the
  source of each number. Papers are written separately (Claude web client) from these files.
- The README must not name a target venue; it keeps only the software citation (`CITATION.cff`).
