"""Synthetic GC-MS studies with exact ground truth (jointalign.synthetic and trustalign.worlds).

Generative model, per batch k and sample i:
  a compound at latent retention time τ elutes in batch k at  τ + d_k(τ) + ε_ki   (ε: jitter)
  its mass spectrum is shared by every batch; its abundance varies per sample; ~8 % of compounds
  are missing from any one sample; the run has column bleed late and no signal before the batch's
  acquisition start.
Stored like warpalign's preprocessing: chroma = sqrt-then-L2 per RT bin [T, M], tic_raw = the
summed raw intensity per bin [T].

Worlds:
  fishlike  jointalign's public stand-in for the fish-oil batches (DEFAULT_BATCHES: five batches
            with the fish-oil names and drift pattern). make_world / make / make_dataset.
  k<K>      trustalign Experiment 1: K batches with independent random drifts from family 'i'
            (shift ±3 min, ramp ±3 min, wiggle 0–0.3 min, phase U[0, 2π)), 2–8 samples each,
            acquisition start 3 or 5 min. draw_specs / make_world_k.
Same arguments and seed, same arrays and truth as the upstream generators (golden-tested).

Dataset ids (alignbench.datasets): 'synthetic:fishlike', 'synthetic:k<K>', optionally with the
generator version, e.g. 'synthetic:k5@1'. VERSION changes whenever a generator's output would.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..axis import FISH_OIL_AXIS, TimeAxis
from ..data import Batch, Study

VERSION = 1


@dataclass(frozen=True)
class BatchSpec:
    name: str
    n: int  # samples
    shift: float  # drift at the start of the run (min)
    ramp: float  # extra drift reached by the end of the run (min), smooth S-curve
    wiggle: float  # amplitude of a slow sinusoidal drift component (min)
    phase: float  # phase of the wiggle (rad)
    start: float  # acquisition start (min): no signal before this

    def drift(self, tau: np.ndarray, axis: TimeAxis = FISH_OIL_AXIS) -> np.ndarray:
        """d_k(τ) in minutes: where a compound at latent τ elutes in this batch, minus τ."""
        u = np.clip(np.asarray(tau, dtype=float) / axis.run_min, 0.0, 1.0)
        s = u * u * (3 - 2 * u)  # smoothstep 0 → 1
        return self.shift + self.ramp * s + self.wiggle * np.sin(2 * np.pi * u + self.phase)


# Same names and the same pattern of drifts as the fish-oil batches (sep2015 is the anchor-rich
# reference there); fewer samples so a demo runs in minutes.
DEFAULT_BATCHES = (
    BatchSpec('batch_apr2016', 5, -2.5, -2.3, 0.15, 0.4, 3.0),
    BatchSpec('batch_aug2015', 2, -0.2, 0.0, 0.20, 2.0, 5.0),
    BatchSpec('batch_jan2016', 8, 0.1, 0.1, 0.20, 1.1, 5.0),
    BatchSpec('batch_jul2016', 6, -2.6, -2.5, 0.15, 3.0, 3.0),
    BatchSpec('batch_sep2015', 10, 0.0, 0.0, 0.0, 0.0, 5.0),
)

BLEED_IONS = (73, 147, 207, 281)
FAMILY_I: dict[str, Any] = dict(shift=3.0, ramp=3.0, wiggle=(0.0, 0.3))
N_SAMPLES = (2, 8)  # samples per batch, inclusive
STARTS = (3.0, 5.0)  # acquisition start (min)


def _library(n: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """n EI-like spectra over m channels: a few shared fragment ions plus compound-specific lines."""
    common = rng.choice(np.arange(40, 120), 6, replace=False)
    lib = np.zeros((n, m))
    for i in range(n):
        own = rng.choice(np.arange(30, m), rng.integers(8, 16), replace=False)
        lib[i, own] = rng.gamma(1.5, 1.0, len(own))
        shared = rng.choice(common, 3, replace=False)
        lib[i, shared] += rng.uniform(0.3, 1.5, 3)
    return lib / lib.sum(axis=1, keepdims=True)


def _sample(lib, tau, spec: BatchSpec, abundance, rng, sigma_bins: float, jitter_min: float, axis: TimeAxis):
    """One chromatogram: (chroma [T, M] sqrt-L2 normalised, tic_raw [T])."""
    m = lib.shape[1]
    times = axis.times
    present = rng.random(len(tau)) > 0.08
    rt = tau + spec.drift(tau, axis) + rng.normal(0.0, jitter_min, len(tau))
    pos = rt / axis.bin_min
    t = np.arange(axis.n_bins, dtype=float)[:, None]
    h = abundance * rng.lognormal(0.0, 0.35, len(tau)) * present
    prof = h[None, :] * np.exp(-0.5 * ((t - pos[None, :]) / sigma_bins) ** 2)
    raw = prof @ lib
    bleed = np.clip((times - 36.0) / 9.0, 0, None) ** 2 * 0.25  # column bleed late in the run
    bleed_spec = np.zeros(m)
    bleed_spec[list(BLEED_IONS)] = [1.0, 0.6, 0.8, 0.5]  # siloxane-like ions
    raw = raw + bleed[:, None] * bleed_spec[None, :] + rng.exponential(2e-4, raw.shape)
    raw[times < spec.start] = 0.0
    tic_raw = raw.sum(axis=1)
    sq = np.sqrt(raw)
    norm = np.linalg.norm(sq, axis=1, keepdims=True)
    chroma = np.where(norm > 1e-12, sq / np.maximum(norm, 1e-12), 0.0)
    return chroma.astype(np.float32), tic_raw.astype(np.float32)


def _generate(batches, n_compounds: int, mz_bins: int, sigma_bins: float, jitter_min: float, seed: int,
              axis: TimeAxis):
    """Draw one world: per batch, chromas [n, T, M] and tic_raws [n, T] (float32), plus the latent
    compound times τ. The order of the random draws defines the world; keep it fixed."""
    if mz_bins <= max(BLEED_IONS):
        raise ValueError(f'mz_bins must exceed {max(BLEED_IONS)} (the column-bleed ions), got {mz_bins}')
    rng = np.random.default_rng(seed)
    lib = _library(n_compounds, mz_bins, rng)
    tau = np.sort(rng.uniform(5.5, 42.5, n_compounds))
    abundance = rng.lognormal(0.0, 0.8, n_compounds)
    chromas, tic_raws = [], []
    for spec in batches:
        pairs = [_sample(lib, tau, spec, abundance, rng, sigma_bins, jitter_min, axis) for _ in range(spec.n)]
        chromas.append(np.stack([c for c, _ in pairs]))
        tic_raws.append(np.stack([t for _, t in pairs]))
    return chromas, tic_raws, tau


def _truth(batches, tau: np.ndarray, seed: int, axis: TimeAxis) -> dict:
    times = axis.times
    return {
        'description': 'Synthetic stand-in for the fish oil batches (src/synthetic.py). A compound '
                       'at latent time tau elutes in batch k at tau + drift_k(tau) (minutes).',
        'seed': seed, 'bin_min': axis.bin_min, 'n_bins': axis.n_bins,
        'compound_tau_min': np.round(tau, 4).tolist(),
        'latent_grid_min': times.tolist(),
        'batches': {s.name: {**asdict(s), 'drift_min': np.round(s.drift(times, axis), 5).tolist()}
                    for s in batches},
    }


def make_world(batches=DEFAULT_BATCHES, n_compounds: int = 60, mz_bins: int = 1000, sigma_bins: float = 0.9,
               jitter_min: float = 0.04, seed: int = 0,
               axis: TimeAxis = FISH_OIL_AXIS) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    """One world in memory: (chromas, tic_raws, truth), chromas[k] batch k's [n_k, T, M] float32
    array and tic_raws[k] its [n_k, T] raw TICs, in the order of `batches` (jointalign.synthetic)."""
    chromas, tic_raws, tau = _generate(batches, n_compounds, mz_bins, sigma_bins, jitter_min, seed, axis)
    return chromas, tic_raws, _truth(batches, tau, seed, axis)


def make_dataset(out_dir: Path, batches=DEFAULT_BATCHES, n_compounds: int = 60, mz_bins: int = 1000,
                 sigma_bins: float = 0.9, jitter_min: float = 0.04, seed: int = 0,
                 axis: TimeAxis = FISH_OIL_AXIS) -> dict:
    """Write <out_dir>/<batch>/chroma/*.npz (chroma, tic_raw; readable by datasets.fish_oil.load)
    and <out_dir>/truth.json. Returns the truth dictionary."""
    out_dir = Path(out_dir)
    chromas, tic_raws, tau = _generate(batches, n_compounds, mz_bins, sigma_bins, jitter_min, seed, axis)
    for spec, chroma_k, tic_k in zip(batches, chromas, tic_raws):
        d = out_dir / spec.name / 'chroma'
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob('*.npz'):
            old.unlink()
        for i in range(spec.n):
            np.savez_compressed(d / f'{spec.name}_{i:02d}.npz', chroma=chroma_k[i], tic_raw=tic_k[i])
    truth = _truth(batches, tau, seed, axis)
    (out_dir / 'truth.json').write_text(json.dumps(truth, indent=1))
    return truth


def true_map(truth: dict, a: str, b: str, t_min: np.ndarray) -> np.ndarray:
    """The true correspondence: retention time (min) in batch a → the same compound's time in b."""
    grid = np.asarray(truth['latent_grid_min'])
    ra = grid + np.asarray(truth['batches'][a]['drift_min'])  # latent → time in a
    rb = grid + np.asarray(truth['batches'][b]['drift_min'])
    tau = np.interp(t_min, ra, grid)
    return np.interp(tau, grid, rb)


def draw_specs(K: int, rng: np.random.Generator, family: str = 'i') -> tuple[BatchSpec, ...]:
    """K random batch specifications from one drift family (trustalign.worlds.draw_specs)."""
    if family != 'i':
        raise NotImplementedError(f'drift family {family!r}: only family "i" exists yet')
    specs = []
    for k in range(K):
        specs.append(BatchSpec(
            name=f'batch_{k:02d}',
            n=int(rng.integers(N_SAMPLES[0], N_SAMPLES[1] + 1)),
            shift=float(rng.uniform(-FAMILY_I['shift'], FAMILY_I['shift'])),
            ramp=float(rng.uniform(-FAMILY_I['ramp'], FAMILY_I['ramp'])),
            wiggle=float(rng.uniform(*FAMILY_I['wiggle'])),
            phase=float(rng.uniform(0.0, 2 * np.pi)),
            start=float(rng.choice(STARTS)),
        ))
    return tuple(specs)


# ── Truth and Study ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SyntheticTruth:
    """Exact answers of a synthetic study (alignbench.data.Truth): batch k is specs[k]."""

    specs: tuple[BatchSpec, ...]
    truth: dict  # make_world's truth dictionary (latent grid, drifts, compound τ)
    axis: TimeAxis = FISH_OIL_AXIS

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.specs]

    def true_map(self, k: int, l: int, t_min: np.ndarray) -> np.ndarray:
        """Exact retention time (min) in batch l of whatever elutes at t_min in batch k."""
        return true_map(self.truth, self.specs[k].name, self.specs[l].name, t_min)

    def observed_span(self, k: int) -> tuple[float, float]:
        """Earliest and latest elution time (min) of the compounds batch k actually records:
        inside the run and after its acquisition start (jitter-free positions)."""
        tau = np.asarray(self.truth['compound_tau_min'])
        rt = tau + self.specs[k].drift(tau, self.axis)
        seen = rt[(rt >= self.specs[k].start) & (rt < self.axis.run_min)]
        return float(seen.min()), float(seen.max())


def to_study(specs, chromas, tic_raws, truth: dict, name: str, axis: TimeAxis = FISH_OIL_AXIS,
             meta: dict | None = None) -> Study:
    batches = [
        Batch(s.name, c, t, tuple(f'{s.name}_{i:02d}' for i in range(s.n)), axis)
        for s, c, t in zip(specs, chromas, tic_raws)
    ]
    return Study(batches, axis, name, SyntheticTruth(tuple(specs), truth, axis), meta)


def make(batches=DEFAULT_BATCHES, seed: int = 0, axis: TimeAxis = FISH_OIL_AXIS, **world_kw) -> Study:
    """make_world as a Study with exact truth."""
    chromas, tic_raws, truth = make_world(batches, seed=seed, axis=axis, **world_kw)
    meta = {'dataset': f'synthetic:fishlike@{VERSION}', 'seed': seed, **world_kw}
    return to_study(batches, chromas, tic_raws, truth, 'synthetic:fishlike', axis, meta)


def make_world_k(K: int, seed: int, family: str = 'i', axis: TimeAxis = FISH_OIL_AXIS, **world_kw) -> Study:
    """One trustalign Experiment 1 world (trustalign.worlds.make) as a Study. Every random draw
    comes from default_rng(seed): first the batch specs, then the seed handed to make_world."""
    rng = np.random.default_rng(seed)
    specs = draw_specs(K, rng, family)
    chromas, tic_raws, truth = make_world(batches=specs, seed=int(rng.integers(2**31)), axis=axis, **world_kw)
    name = f'synthetic:k{K}' + ('' if family == 'i' else f'-{family}')
    meta = {'dataset': f'{name}@{VERSION}', 'seed': seed, 'K': K, 'family': family, **world_kw}
    return to_study(specs, chromas, tic_raws, truth, name, axis, meta)


def load(variant: str, version: int | None = None, seed: int = 0, **kw) -> Study:
    """The registry entry for 'synthetic:<variant>[@version]': 'fishlike', 'k<K>' or 'k<K>-<family>'."""
    if version is not None and int(version) != VERSION:
        raise ValueError(f'synthetic generator version {version} requested; this alignbench has {VERSION}')
    if variant == 'fishlike':
        return make(seed=seed, **kw)
    if variant.startswith('k'):
        k, _, family = variant[1:].partition('-')
        if k.isdigit() and int(k) >= 2:
            return make_world_k(int(k), seed, family or 'i', **kw)
    raise KeyError(f"unknown synthetic dataset {variant!r}: use 'fishlike' or 'k<K>' (K ≥ 2)")
