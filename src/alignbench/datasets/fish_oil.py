"""The fish-oil GC-MS study (private): loader for warpalign's preprocessed on-disk format, and the
study's batch and pair definitions (formerly in warpalign.config).

Format (written by warpalign's 03_preprocess_fish_oil_batches.py; warp-align README, "Using your
own data"): <root>/<batch>/chroma/*.npz, one file per sample, each holding
  chroma   [T, M] float32  sqrt + L2-normalised intensity per time bin
  tic_raw  [T]    float32  summed raw intensity of each binned apex scan
Samples are in sorted filename order. T is FISH_OIL_AXIS.n_bins (200).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..axis import FISH_OIL_AXIS, TimeAxis
from ..data import Batch, Study
from . import register

BATCHES = ('batch_sep2015', 'batch_jan2016', 'batch_apr2016', 'batch_jul2016')
EVAL_PAIRS = (
    ('batch_sep2015', 'batch_jul2016'),
    ('batch_sep2015', 'batch_apr2016'),
    ('batch_jan2016', 'batch_apr2016'),
    ('batch_jan2016', 'batch_jul2016'),
)
EXCLUDED_PAIRS = (('batch_sep2015', 'batch_jan2016'), ('batch_apr2016', 'batch_jul2016'))
PAIR_LABELS = {
    ('batch_sep2015', 'batch_jul2016'): 'Sep15×Jul16',
    ('batch_sep2015', 'batch_apr2016'): 'Sep15×Apr16',
    ('batch_jan2016', 'batch_apr2016'): 'Jan16×Apr16',
    ('batch_jan2016', 'batch_jul2016'): 'Jan16×Jul16',
    ('batch_sep2015', 'batch_jan2016'): 'Sep15×Jan16',
    ('batch_apr2016', 'batch_jul2016'): 'Apr16×Jul16',
}


def pair_label(a: str, b: str) -> str:
    """'batch_sep2015', 'batch_jul2016' → 'Sep15×Jul16' (falls back to the raw names)."""
    return PAIR_LABELS.get((a, b), f'{a} × {b}')


def batch_name(label: str) -> str:
    """'batch_jul2016' → 'Jul 2016'."""
    s = label.replace('batch_', '')
    return f'{s[:3].capitalize()} {s[3:]}' if len(s) == 7 and s[3:].isdigit() else label


def chroma_files(chroma_dir: Path) -> list[Path]:
    files = sorted(Path(chroma_dir).glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'no .npz chromatograms in {chroma_dir}')
    return files


def load_chromas(chroma_dir: Path) -> np.ndarray:
    """[N, T, M] float32 chromatograms in sorted filename order."""
    return np.stack([np.load(p)['chroma'].astype(np.float32) for p in chroma_files(chroma_dir)])


def load_tic_raw(chroma_dir: Path) -> np.ndarray:
    """[N, T] raw TICs, in the same order as load_chromas."""
    out = []
    for p in chroma_files(chroma_dir):
        with np.load(p) as z:
            if 'tic_raw' not in z.files:
                raise ValueError(f'{p} has no tic_raw; rerun warpalign 03_preprocess_fish_oil_batches.py')
            out.append(z['tic_raw'].astype(np.float32))
    return np.stack(out)


def load_batch(name: str, root: Path, axis: TimeAxis = FISH_OIL_AXIS) -> Batch:
    """One batch from <root>/<name>/chroma (warpalign: load_batch(name, Paths.fish_oil))."""
    d = Path(root) / name / 'chroma'
    return Batch(
        name=name,
        chromas=load_chromas(d),
        tic_raw=load_tic_raw(d),
        sample_ids=tuple(p.stem for p in chroma_files(d)),
        axis=axis,
    )


def load(root: Path, batches: tuple[str, ...] = BATCHES, axis: TimeAxis = FISH_OIL_AXIS) -> Study:
    """The fish-oil study from <root>/<batch>/chroma for each batch (default: the four batches of
    the paper analyses, in BATCHES order)."""
    return Study(
        [load_batch(b, root, axis) for b in batches],
        axis=axis,
        name='fish_oil',
        meta={'dataset': 'fish_oil', 'root': str(Path(root).resolve()), 'batches': list(batches)},
    )


register('fish_oil', load)
