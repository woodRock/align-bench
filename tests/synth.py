"""Synthetic GC-MS batches with known compound positions and drift, for the tests and the golden
fixtures (warpalign tests/synth.py, plus nonlinear drift, decoy compounds and empty scans).

Deterministic under a seed. The golden fixtures record a digest of every generated input, so a
change in numpy's random streams shows up as a clear failure, not as a silent mismatch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

T, M = 200, 1000


def compound_spectra(n: int, rng: np.random.Generator, n_peaks: int = 12) -> np.ndarray:
    """n distinct sparse intensity spectra [n, M] (L2 = 1), clearly separable by cosine."""
    specs = np.zeros((n, M))
    for i in range(n):
        mz = rng.choice(np.arange(40, 600), n_peaks, replace=False)
        specs[i, mz] = rng.uniform(0.2, 1.0, n_peaks)
    return specs / np.linalg.norm(specs, axis=1, keepdims=True)


def sample_intensity(positions, heights, specs, rng, sigma=0.8, noise=1e-4, start_bin=0):
    """Intensity chromatogram [T, M] with Gaussian elution profiles; zero before start_bin
    (as runs that start acquiring later)."""
    t = np.arange(T)[:, None]
    prof = heights[None, :] * np.exp(-0.5 * ((t - positions[None, :]) / sigma) ** 2)
    x = prof @ specs + rng.exponential(noise, size=(T, M))
    x[:start_bin] = 0.0
    return x


def normalise(x: np.ndarray) -> np.ndarray:
    """sqrt + L2 per bin, as warpalign 03_preprocess does."""
    s = np.sqrt(np.clip(x, 0, None))
    n = np.linalg.norm(s, axis=-1, keepdims=True)
    return (s / np.where(n > 1e-8, n, 1.0)).astype(np.float32)


def make_batch(
    n_samples: int,
    positions: np.ndarray,
    shift,
    specs: np.ndarray,
    rng: np.random.Generator,
    jitter: float = 0.0,
    start_bin: int = 0,
    empty: tuple[int, int] | None = None,
):
    """(chromas [N, T, M], tic_raw [N, T]) for a batch whose compounds sit at positions + shift.
    shift: a number, or a function of the positions (nonlinear drift). empty: (lo, hi) bins
    zeroed in every other sample (scans with no ions)."""
    chromas, raws = [], []
    heights = rng.uniform(1.0, 5.0, len(positions))
    off = shift(positions) if callable(shift) else shift
    for i in range(n_samples):
        pos = positions + off + rng.uniform(-jitter, jitter, len(positions))
        x = sample_intensity(
            pos, heights * rng.uniform(0.8, 1.2, len(positions)), specs, rng, start_bin=start_bin
        )
        if empty is not None and i % 2 == 0:
            x[empty[0] : empty[1]] = 0.0
        chromas.append(normalise(x))
        raws.append(x.sum(axis=1).astype(np.float32))
    return np.stack(chromas), np.stack(raws)


SHIFT = -8
POSITIONS = np.array([40, 52, 63, 75, 88, 101, 115, 128, 141, 155, 168, 181], dtype=float)


def pair_shift():
    """warpalign's conftest synthetic_pair: batch B shifted by SHIFT bins."""
    r = np.random.default_rng(7)
    specs = compound_spectra(len(POSITIONS), r)
    a = make_batch(4, POSITIONS, 0, specs, r, jitter=0.2)
    b = make_batch(4, POSITIONS, SHIFT, specs, r, jitter=0.2)
    return a, b


def pair_nonlinear():
    """Nonlinear drift, late-starting runs, empty scans, and decoys: compounds whose spectra are
    near copies of real ones, placed off the drift trend in batch B, so matching proposes wrong
    pairs that RANSAC and the pooled filters have to remove."""
    r = np.random.default_rng(11)
    n = len(POSITIONS)
    specs = compound_spectra(n, r)
    a = make_batch(5, POSITIONS, 0, specs, r, jitter=0.4, start_bin=22, empty=(60, 75))
    drift = lambda p: -3.0 - 0.04 * (p - 40) + 2.0 * np.sin(p / 30.0)
    decoy_specs = specs[[2, 7]] + 0.05 * compound_spectra(2, r)
    decoy_specs /= np.linalg.norm(decoy_specs, axis=1, keepdims=True)
    pos_b = np.concatenate([POSITIONS, [150.0, 45.0]])
    specs_b = np.concatenate([specs, decoy_specs])
    off = np.concatenate([drift(POSITIONS), [0.0, 0.0]])
    b = make_batch(4, pos_b, lambda p: off, specs_b, r, jitter=0.4, start_bin=13)
    return a, b


def pair_sparse():
    """Three compounds: few peaks, so the matcher and the pooled filters hit their small-n
    branches."""
    r = np.random.default_rng(23)
    pos = np.array([60.0, 110.0, 160.0])
    specs = compound_spectra(len(pos), r)
    a = make_batch(3, pos, 0, specs, r, jitter=0.2)
    b = make_batch(3, pos, 5, specs, r, jitter=0.2)
    return a, b


PAIRS = {'shift': pair_shift, 'nonlinear': pair_nonlinear, 'sparse': pair_sparse}

# Four batches mirroring the real layout: Sep/Jan unshifted, Apr/Jul shifted earlier by SHIFT bins
BATCH_SHIFTS = {
    'batch_sep2015': 0,
    'batch_jan2016': 0,
    'batch_apr2016': SHIFT,
    'batch_jul2016': SHIFT,
}


def write_dataset(root: Path, n_per_batch: int = 4, seed: int = 0) -> dict:
    """Write <root>/<batch>/chroma/*.npz in warpalign's preprocessed format. Returns the arrays."""
    rng = np.random.default_rng(seed)
    specs = compound_spectra(len(POSITIONS), rng)
    out = {}
    for name, shift in BATCH_SHIFTS.items():
        chromas, raws = make_batch(
            n_per_batch, POSITIONS, shift, specs, rng, jitter=0.3, start_bin=22 if shift == 0 else 13
        )
        d = Path(root) / name / 'chroma'
        d.mkdir(parents=True, exist_ok=True)
        for i, (c, raw) in enumerate(zip(chromas, raws)):
            np.savez_compressed(d / f'sample_{i:02d}.npz', chroma=c, tic_raw=raw)
        out[name] = (chromas, raws)
    return out
