# alignbench

Shared core and benchmark for GC-MS retention-time alignment.

`alignbench` is a torch-free, installable Python package that holds what every alignment method
needs: the data model and loaders, spectral anchor matching, warp primitives, classical baselines,
scoring protocols and provenance. A benchmark runner scores any method on any dataset under the
same protocols. Every method is reduced to what it estimates, a correspondence map from retention
time in one batch to retention time in another, so methods with very different internals are
compared like for like.

> **Status:** under construction (scaffold). The API below is the target, not yet implemented.

## Install

```bash
python3.11 -m venv .venv
.venv/bin/pip install "alignbench @ git+https://github.com/woodRock/align-bench.git"
```

Development: `make venv && make check`.

## Usage (planned)

```bash
alignbench list methods|datasets|protocols
alignbench run --methods pooled-pchip,cow-tic --dataset synthetic:k5 --seeds 0-4 \
               --protocols tic_bframe,map_error --out results/bench_k5/
alignbench report results/bench_k5/
```

## Adding a method

Methods that live in other packages register through the `alignbench.methods` entry point group:

```toml
[project.entry-points."alignbench.methods"]
my-method = "mypackage.alignbench_plugin:MyMethod"
```

## Related repositories

- [warp-align](https://github.com/woodRock/warp-align): WarpTransformer (learned baseline).
- [joint-align](https://github.com/woodRock/joint-align): joint alignment across many batches.
- [trust-align](https://github.com/woodRock/trust-align): synthetic worlds and anchor-graph robustness.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Licence

MIT. No data is distributed with this repository.
