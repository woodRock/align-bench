"""Calibration anchors: the same compound located in batch A and batch B by spectral matching.

Two matchers, both used in published results and both reproduced bit for bit:

  match_pooled  warpalign.anchors.extract_anchors: integer-bin (r, q) anchors between two batches,
                pooled over a few batch-A samples matched against B[0], then filtered.
  harvest       jointalign.multibatch.harvest_anchors: sub-bin anchors on every cross-batch sample
                pair of a study, kept per sample pair (columnar Anchors).

Pooled matcher (match_pooled; warpalign.anchors.extract_anchors, reproduced bit for bit):
  For one sample pair (match_sample_pair): peaks on the raw TIC → spectral cosine between context
  fingerprints → Hungarian assignment within max_drift_min → similarity threshold → refine each
  B peak to the best-matching scan within ±refine_radius bins → RANSAC line (B → A, residual
  ransac_resid bins) → longest increasing subsequence in elution order.
  Pooled over n_ref batch-A samples matched against B[0]: median B bin per A bin → merge anchors
  ≤ merge_gap bins apart → drift outlier filter (leave-one-out residual from the straight-line
  drift trend of the other anchors) → local-slope plausibility filter (slope_range).

Pooled anchors are (r, q) integer bins: r in batch A, q in batch B (Anchors.from_pairs converts).

Harvest (match_subbin per sample pair; jointalign.multibatch.match_anchors): the same peak,
fingerprint, Hungarian and threshold steps, then parabolic sub-bin refinement on both sides
(the raw TIC around the A peak; the fingerprint similarity around the best B scan), RANSAC
(random_state 0, not the pooled matcher's 42) and a greedy monotone filter.

Peaks are found on the raw TIC (compound abundance); fingerprints and matching use the
normalised chromatogram.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor

from .axis import FISH_OIL_AXIS, TimeAxis
from .data import Batch, Study

Anchor = tuple[int, int]

MATCH_STAGES = ('proposed', 'above threshold', 'after RANSAC', 'after monotone')
POOL_STAGES = ('pooled', 'after merging', 'after outlier filter', 'after slope filter')


@dataclass(frozen=True)
class MatchConfig:
    """Anchor matching settings. Defaults are warpalign.config's values."""

    n_ref: int = 3  # batch-A samples matched against B[0] (pooled matcher)
    merge_gap: int = 2  # pooled anchors ≤ this many bins apart are one compound: merged
    outlier_sd: float = 1.5  # drop pooled anchors > outlier_sd · s.d. (≥ ransac_resid) off the trend
    slope_range: tuple[float, float] = (0.5, 2.0)  # plausible local Δq/Δr between consecutive anchors
    max_drift_min: float = 20.0  # candidate peak matches further apart than this are not considered
    sim_threshold: float = 0.30  # minimum spectral cosine for a match
    ransac_resid: float = 3.0  # bins
    refine_radius: int = 3  # bins searched around a query peak for the best-matching scan
    peak_height_pct: float = 80
    peak_min_dist: int = 5
    peak_max_n: int = 30


DEFAULT_MATCH = MatchConfig()


POOLED = -1  # sample index of anchors pooled over samples (match_pooled)
_COLS = ('ba', 'bb', 'sa', 'sb', 'ta', 'tb')


@dataclass
class Anchors:
    """Anchor pairs, one row per anchor (jointalign.multibatch.Anchors, plus keyword-only axis and stages).

    Rows are oriented ba < bb. sa, sb are sample indices (POOLED for pooled anchors); ta, tb are RT
    bins (fractional for harvest, integer-valued for match_pooled)."""

    ba: np.ndarray  # batch index of the first chromatogram
    bb: np.ndarray  # batch index of the second chromatogram (ba < bb)
    sa: np.ndarray  # sample index within batch ba
    sb: np.ndarray  # sample index within batch bb
    ta: np.ndarray  # RT bin in batch ba
    tb: np.ndarray  # RT bin in batch bb
    axis: TimeAxis = field(default=FISH_OIL_AXIS, kw_only=True)
    stages: dict[str, int] | None = field(default=None, kw_only=True)  # matcher stage counts, if known

    def __len__(self) -> int:
        return len(self.ta)

    def subset(self, mask: np.ndarray) -> Anchors:
        """The rows in mask (boolean or indices); stage counts no longer apply and are dropped."""
        return Anchors(*(getattr(self, f)[mask] for f in _COLS), axis=self.axis)

    def pair(self, a: int, b: int) -> Anchors:
        """Anchors between batches a and b, oriented so that ta is in a and tb is in b."""
        if a < b:
            return self.subset((self.ba == a) & (self.bb == b))
        s = self.subset((self.ba == b) & (self.bb == a))
        return Anchors(s.bb, s.ba, s.sb, s.sa, s.tb, s.ta, axis=self.axis)

    @classmethod
    def from_pairs(
        cls,
        pairs: Sequence[tuple[float, float]],
        a: int = 0,
        b: int = 1,
        axis: TimeAxis = FISH_OIL_AXIS,
        stages: dict[str, int] | None = None,
    ) -> Anchors:
        """warpalign's (r, q) list (r in batch a, q in batch b) as pooled Anchors rows."""
        if a == b:
            raise ValueError('anchors need two different batches')
        arr = np.asarray(pairs, dtype=float).reshape(-1, 2)
        n = len(arr)
        ta, tb = (arr[:, 0], arr[:, 1]) if a < b else (arr[:, 1], arr[:, 0])
        full = lambda v: np.full(n, v, dtype=int)
        return cls(full(min(a, b)), full(max(a, b)), full(POOLED), full(POOLED), ta, tb, axis=axis, stages=stages)

    def as_pairs(self, a: int | None = None, b: int | None = None) -> list[Anchor]:
        """warpalign's (r, q) integer-bin list, for one batch pair (oriented a → b; default: the only
        pair present). Fractional anchors raise: round them explicitly if that is intended."""
        if a is None or b is None:
            pairs = set(zip(self.ba.tolist(), self.bb.tolist()))
            if len(pairs) > 1:
                raise ValueError(f'anchors span {len(pairs)} batch pairs; pass a and b')
            a, b = next(iter(pairs)) if pairs else (0, 1)
        p = self.pair(a, b)
        if not (np.all(p.ta == np.round(p.ta)) and np.all(p.tb == np.round(p.tb))):
            raise ValueError('as_pairs needs integer-bin anchors (these are sub-bin)')
        return [(int(r), int(q)) for r, q in zip(p.ta, p.tb)]


@dataclass
class AnchorResult:
    anchors: list[Anchor]
    stages: dict[str, int] = field(default_factory=dict)  # stage → count (match stages summed)
    removed_by_outlier: list[Anchor] = field(default_factory=list)
    removed_by_slope: list[Anchor] = field(default_factory=list)

    def to_anchors(self, a: int = 0, b: int = 1, axis: TimeAxis = FISH_OIL_AXIS) -> Anchors:
        return Anchors.from_pairs(self.anchors, a, b, axis, dict(self.stages))


# ── Peaks and fingerprints ────────────────────────────────────────────────────


def detect_peaks(
    tic: np.ndarray,
    height_pct: float = DEFAULT_MATCH.peak_height_pct,
    min_dist: int = DEFAULT_MATCH.peak_min_dist,
    n_peaks: int = DEFAULT_MATCH.peak_max_n,
) -> np.ndarray:
    """Local maxima of a 1-D TIC above its height_pct percentile, at least min_dist bins apart;
    the n_peaks highest if more. Sorted bin indices."""
    tic = np.asarray(tic, dtype=np.float64)
    pks, props = find_peaks(tic, height=np.percentile(tic, height_pct), distance=min_dist)
    if len(pks) > n_peaks:
        pks = pks[np.argsort(props['peak_heights'])[-n_peaks:]]
    return np.sort(pks)


def fingerprint(chroma: np.ndarray, pk: int) -> np.ndarray:
    """L2-normalised mean spectrum of bins pk−1 … pk+1 (clipped to the run)."""
    v = chroma[max(0, pk - 1) : min(len(chroma), pk + 2)].mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n if n > 1e-8 else v).astype(np.float32)


def _fingerprints(chroma: np.ndarray, pks: np.ndarray) -> np.ndarray:
    return np.array([fingerprint(chroma, int(p)) for p in pks]).reshape(len(pks), chroma.shape[1])


# ── Order filters ─────────────────────────────────────────────────────────────


def longest_increasing_subsequence(values: Iterable[float]) -> list[int]:
    """Indices of a longest strictly increasing subsequence (O(n log n)); ties broken towards
    the earliest-ending subsequence. Empty input → []."""
    vals = list(values)
    tails: list[float] = []
    tail_idx: list[int] = []
    prev = [-1] * len(vals)
    for i, v in enumerate(vals):
        k = bisect_left(tails, v)
        if k == len(tails):
            tails.append(v)
            tail_idx.append(i)
        else:
            tails[k], tail_idx[k] = v, i
        prev[i] = tail_idx[k - 1] if k > 0 else -1
    out, i = [], tail_idx[-1] if tail_idx else -1
    while i != -1:
        out.append(i)
        i = prev[i]
    return out[::-1]


def monotone_subset(anchors: list[Anchor]) -> list[Anchor]:
    """Largest subset strictly increasing in both r and q (anchors sorted by r; ties in r keep
    one). Replaces the greedy keep-the-first filter, where one early mismatch with a large q
    removed every later anchor."""
    anchors = sorted(anchors)
    by_r: dict[int, Anchor] = {}
    for a in anchors:  # strictly increasing r: keep the first q per r
        by_r.setdefault(a[0], a)
    uniq = list(by_r.values())
    return [uniq[i] for i in longest_increasing_subsequence([q for _, q in uniq])]


def slope_filter(
    anchors: list[Anchor],
    lo: float = DEFAULT_MATCH.slope_range[0],
    hi: float = DEFAULT_MATCH.slope_range[1],
) -> tuple[list[Anchor], list[Anchor]]:
    """Drop anchors until every pair of consecutive anchors (sorted by r) has local slope
    Δq/Δr in [lo, hi]. Retention-time drift is smooth: a slope outside that range means one of
    the two is a misidentification. Of a violating pair, the anchor whose drift (q − r)
    deviates more from the median drift of the current set is dropped; repeat.
    Returns (kept, removed)."""
    kept, removed = sorted(anchors), []
    while len(kept) >= 2:
        r = np.array([a for a, _ in kept], dtype=float)
        q = np.array([b for _, b in kept], dtype=float)
        dr = np.diff(r)
        slope = np.where(dr > 0, np.diff(q) / np.where(dr > 0, dr, 1), -np.inf)
        bad = np.flatnonzero((slope < lo) | (slope > hi))
        if len(bad) == 0:
            break
        dev = np.abs((q - r) - np.median(q - r))
        # Worst violation first; within it, the anchor with the more atypical drift
        k = bad[np.argmax(np.maximum(dev[bad], dev[bad + 1]))]
        drop = k if dev[k] >= dev[k + 1] else k + 1
        removed.append(kept.pop(int(drop)))
    return kept, removed


# ── One sample pair ───────────────────────────────────────────────────────────


def match_sample_pair(
    chroma_a: np.ndarray,
    chroma_b: np.ndarray,
    tic_a: np.ndarray,
    tic_b: np.ndarray,
    cfg: MatchConfig = DEFAULT_MATCH,
    axis: TimeAxis = FISH_OIL_AXIS,
) -> tuple[list[Anchor], dict[str, int]]:
    """Anchors between one batch-A sample and one batch-B sample (warpalign.anchors.match_anchors).

    chroma_*: [T, M] normalised chromatograms (fingerprints); tic_*: [T] raw TICs (peaks).
    Returns ((r, q) anchors sorted by r, stage counts)."""
    counts = dict.fromkeys(MATCH_STAGES, 0)
    pk_a, pk_b = (
        detect_peaks(tic, cfg.peak_height_pct, cfg.peak_min_dist, cfg.peak_max_n) for tic in (tic_a, tic_b)
    )
    if len(pk_a) < 2 or len(pk_b) < 2:
        return [], counts
    fp_a, fp_b = _fingerprints(chroma_a, pk_a), _fingerprints(chroma_b, pk_b)
    sim = fp_b @ fp_a.T  # [n_b, n_a] cosine

    too_far = np.abs(pk_b[:, None] - pk_a[None, :]) * axis.bin_min > cfg.max_drift_min
    rows, cols = linear_sum_assignment(-np.where(too_far, -1.0, sim))
    ok = ~too_far[rows, cols]
    rows, cols = rows[ok], cols[ok]
    counts['proposed'] = len(rows)

    avail = sim[~too_far]
    thr = (
        max(float(np.percentile(avail, 70)), cfg.sim_threshold) if len(avail) >= 3 else cfg.sim_threshold
    )
    keep = sim[rows, cols] >= thr
    rows, cols = rows[keep], cols[keep]
    counts['above threshold'] = len(rows)
    if len(rows) < 2:
        return [], counts

    r = pk_a[cols].astype(np.int64)
    q = np.array(
        [
            _refine(chroma_b, int(pk_b[i]), fp_a[j], cfg.refine_radius)
            for i, j in zip(rows, cols, strict=True)
        ]
    )
    order = np.argsort(r, kind='stable')
    r, q = r[order], q[order]

    if len(r) >= 4:
        inl = _ransac_inliers(q, r, cfg.ransac_resid)
        if inl.sum() >= 2:
            r, q = r[inl], q[inl]
    counts['after RANSAC'] = len(r)

    anchors = monotone_subset(list(zip(r.tolist(), q.tolist(), strict=True)))
    counts['after monotone'] = len(anchors)
    return anchors, counts


def _refine(chroma_b: np.ndarray, q_pk: int, fp_ref: np.ndarray, radius: int) -> int:
    """The bin within ±radius of q_pk whose fingerprint best matches fp_ref (ties keep the peak
    itself, then the earliest bin)."""
    best, best_s = q_pk, float(fingerprint(chroma_b, q_pk) @ fp_ref)
    for cand in range(max(0, q_pk - radius), min(len(chroma_b), q_pk + radius + 1)):
        s = float(fingerprint(chroma_b, cand) @ fp_ref)
        if s > best_s:
            best, best_s = cand, s
    return best


def _ransac_inliers(q: np.ndarray, r: np.ndarray, resid: float) -> np.ndarray:
    """Inlier mask of a RANSAC straight line r ≈ a·q + b (residual threshold resid bins).
    All True if RANSAC cannot fit."""
    try:
        model = RANSACRegressor(residual_threshold=resid, min_samples=2, random_state=42)
        model.fit(q.reshape(-1, 1).astype(float), r.astype(float))
        return model.inlier_mask_
    except ValueError:
        return np.ones(len(r), dtype=bool)


# ── Pooled over batch-A samples ───────────────────────────────────────────────


def _round_half_up(x: float) -> int:
    """Round to the nearest integer, .5 upwards (Python's round() sends .5 to the even
    neighbour: 100.5 → 100 but 101.5 → 102)."""
    return int(np.floor(x + 0.5))


def merge_close(
    anchors: list[Anchor], max_gap: int = DEFAULT_MATCH.merge_gap, max_drift_diff: int = 1
) -> list[Anchor]:
    """Merge anchors that are the same compound seen in different A samples: r within max_gap
    bins of the group's *first* anchor (no chaining, so neighbouring compounds stay separate)
    *and* drift (q − r) within max_drift_diff bins of it. Merged anchors become their mean
    (r, q), rounded half up. Anything else stays separate, so a wrong match is left for the
    outlier and slope filters to judge rather than averaged into a right one.

    Why merge at all: with integer bins, the local slope between two anchors one or two bins
    apart is 0, 0.5, 1, 2 or 3 by rounding alone, which would make the slope filter's verdict
    on such a pair arbitrary. Sorted by r on return."""
    groups: list[list[Anchor]] = []
    for r, q in sorted(anchors):
        if groups:
            r0, q0 = groups[-1][0]
            if r - r0 <= max_gap and abs((q - r) - (q0 - r0)) <= max_drift_diff:
                groups[-1].append((r, q))
                continue
        groups.append([(r, q)])
    return [
        (
            _round_half_up(float(np.mean([r for r, _ in g]))),
            _round_half_up(float(np.mean([q for _, q in g]))),
        )
        for g in groups
    ]


def drift_outlier_filter(
    anchors: list[Anchor],
    n_sd: float = DEFAULT_MATCH.outlier_sd,
    floor: float = DEFAULT_MATCH.ransac_resid,
) -> tuple[list[Anchor], list[Anchor]]:
    """Drop anchors whose drift (q − r) is far from the drift *trend* of the other anchors.

    Each anchor is judged against a least-squares straight line of drift against r fitted to
    all the *other* anchors (a leave-one-out residual), so an outlier cannot bend the line it
    is judged by, even at the end of the range. Its deviation is compared with
    max(n_sd · s.d. of the others' residuals, floor): the floor (default ransac_resid, 3 bins)
    is the precision the matching stage already assumes, so scatter from binning is not
    mistaken for misidentification. The worst anchor beyond its limit is dropped and the rest
    re-judged; repeat. Drift trends smoothly along a run, so a constant-mean rule would wrongly
    remove the anchors at both ends. Needs ≥ 4 anchors (3 others to fit and spread); fewer are
    returned unchanged. Returns (kept, removed), both sorted by r."""
    kept, removed = sorted(anchors), []
    while len(kept) >= 4:
        r = np.array([a for a, _ in kept], dtype=float)
        drift = np.array([q for _, q in kept], dtype=float) - r
        excess = np.empty(len(kept))
        for i in range(len(kept)):
            others = np.arange(len(kept)) != i
            line = np.polyfit(r[others], drift[others], 1)
            spread = (drift[others] - np.polyval(line, r[others])).std()
            limit = max(n_sd * float(spread), floor)
            excess[i] = abs(drift[i] - np.polyval(line, r[i])) / limit
        worst = int(np.argmax(excess))
        if excess[worst] <= 1.0:
            break
        removed.append(kept.pop(worst))
    return kept, sorted(removed)


def pool_anchors(
    per_sample: list[list[Anchor]], cfg: MatchConfig = DEFAULT_MATCH
) -> tuple[list[Anchor], dict[str, int], list[Anchor], list[Anchor]]:
    """Median q per r across samples → merge close anchors → drift outlier filter → slope
    filter.
    Returns (anchors sorted by r, stage counts, removed by the outlier filter, removed by the
    slope filter)."""
    by_r: dict[int, list[int]] = defaultdict(list)
    for anchors in per_sample:
        for r, q in anchors:
            by_r[r].append(q)
    pooled = [(r, _round_half_up(float(np.median(qs)))) for r, qs in sorted(by_r.items())]
    counts = {'pooled': len(pooled)}
    pooled = merge_close(pooled, cfg.merge_gap)
    counts['after merging'] = len(pooled)
    pooled, removed_outlier = drift_outlier_filter(pooled, cfg.outlier_sd, cfg.ransac_resid)
    counts['after outlier filter'] = len(pooled)
    kept, removed_slope = slope_filter(pooled, *cfg.slope_range)
    counts['after slope filter'] = len(kept)
    return kept, counts, removed_outlier, removed_slope


def match_pooled_arrays(
    chromas_a: np.ndarray,
    chromas_b: np.ndarray,
    tic_raw_a: np.ndarray,
    tic_raw_b: np.ndarray,
    cfg: MatchConfig = DEFAULT_MATCH,
    axis: TimeAxis = FISH_OIL_AXIS,
) -> AnchorResult:
    """match_pooled on arrays ([N, T, M] chromatograms, [N, T] raw TICs): A[0 … n_ref−1] each
    matched against B[0], then pooled (warpalign.anchors.extract_anchors)."""
    per_sample, stages = [], dict.fromkeys(MATCH_STAGES, 0)
    for i in range(min(cfg.n_ref, len(chromas_a))):
        anchors, counts = match_sample_pair(
            chromas_a[i], chromas_b[0], tic_raw_a[i], tic_raw_b[0], cfg, axis
        )
        per_sample.append(anchors)
        for k, v in counts.items():
            stages[k] += v
    anchors, pool_counts, removed_outlier, removed_slope = pool_anchors(per_sample, cfg)
    stages.update(pool_counts)
    return AnchorResult(
        anchors=anchors,
        stages=stages,
        removed_by_outlier=removed_outlier,
        removed_by_slope=removed_slope,
    )


def match_pooled(a: Batch, b: Batch, cfg: MatchConfig = DEFAULT_MATCH) -> AnchorResult:
    """Calibration anchors between two batches on the same axis: integer-bin (r, q) anchors,
    pooled over cfg.n_ref batch-A samples matched against B[0]."""
    if a.axis != b.axis:
        raise ValueError(f'batches {a.name!r} and {b.name!r} are on different time axes')
    return match_pooled_arrays(a.chromas, b.chromas, a.tic_raw, b.tic_raw, cfg, a.axis)


# ── Sub-bin anchors on every cross-batch sample pair ─────────────────────────


def _parabolic(ym: float, y0: float, yp: float) -> float:
    """Sub-bin offset of the vertex of the parabola through three equally spaced points."""
    den = ym - 2.0 * y0 + yp
    if den >= 0:
        return 0.0
    return float(np.clip(0.5 * (ym - yp) / den, -0.5, 0.5))


def _fingerprints_raw(chroma: np.ndarray, pks: np.ndarray) -> np.ndarray:
    # jointalign's variant: no reshape, so an empty peak list gives shape (0,) (never reached:
    # match_subbin returns before fingerprinting fewer than two peaks).
    return np.array([fingerprint(chroma, int(p)) for p in pks])


def match_subbin(
    a: np.ndarray,
    b: np.ndarray,
    tic_a: np.ndarray | None = None,
    tic_b: np.ndarray | None = None,
    cfg: MatchConfig = DEFAULT_MATCH,
    axis: TimeAxis = FISH_OIL_AXIS,
) -> np.ndarray:
    """Spectral anchors between chromatograms a and b [T, M], as an (n, 2) array of sub-bin RT
    positions (t_a, t_b) (jointalign.multibatch.match_anchors). Peaks on the raw TICs tic_a, tic_b
    (the chromatogram sums if not given)."""
    empty = np.empty((0, 2))
    tic_a = a.sum(axis=1) if tic_a is None else np.asarray(tic_a, dtype=float)
    tic_b = b.sum(axis=1) if tic_b is None else np.asarray(tic_b, dtype=float)
    pa, pb = (detect_peaks(t, cfg.peak_height_pct, cfg.peak_min_dist, cfg.peak_max_n) for t in (tic_a, tic_b))
    if len(pa) < 2 or len(pb) < 2:
        return empty
    fa, fb = _fingerprints_raw(a, pa), _fingerprints_raw(b, pb)
    sim = fa @ fb.T
    far = np.abs(pa[:, None] - pb[None, :]) > cfg.max_drift_min / axis.bin_min
    ri, ci = linear_sum_assignment(-np.where(far, -1.0, sim))
    avail = sim[~far]
    thr = max(float(np.percentile(avail, 70)), cfg.sim_threshold) if avail.size >= 3 else cfg.sim_threshold
    keep = (sim[ri, ci] >= thr) & ~far[ri, ci]
    ri, ci = ri[keep], ci[keep]
    if len(ri) < 2:
        return empty

    n_bins, rad = axis.n_bins, cfg.refine_radius
    out = []
    for i, j in zip(ri, ci):
        p = int(pa[i])
        ta = p + (_parabolic(tic_a[p - 1], tic_a[p], tic_a[p + 1]) if 0 < p < n_bins - 1 else 0.0)
        cands = np.arange(max(0, pb[j] - rad), min(n_bins, pb[j] + rad + 1))
        s = np.array([float(fa[i] @ fingerprint(b, int(c))) for c in cands])
        k = int(np.argmax(s))
        tb = cands[k] + (_parabolic(s[k - 1], s[k], s[k + 1]) if 0 < k < len(s) - 1 else 0.0)
        out.append((ta, tb))
    anc = np.array(out, dtype=float)

    if len(anc) >= 4:
        try:
            rs = RANSACRegressor(residual_threshold=cfg.ransac_resid, min_samples=2, random_state=0)
            rs.fit(anc[:, :1], anc[:, 1])
            if rs.inlier_mask_.sum() >= 2:
                anc = anc[rs.inlier_mask_]
        except ValueError:
            pass

    anc = anc[np.argsort(anc[:, 0])]
    mono = [0]
    for i in range(1, len(anc)):
        if anc[i, 0] > anc[mono[-1], 0] and anc[i, 1] > anc[mono[-1], 1]:
            mono.append(i)
    return anc[mono] if len(mono) >= 2 else empty


def harvest_arrays(
    chromas: Sequence[np.ndarray],
    tics: Sequence[np.ndarray] | None = None,
    cfg: MatchConfig = DEFAULT_MATCH,
    axis: TimeAxis = FISH_OIL_AXIS,
) -> Anchors:
    """match_subbin on every cross-batch sample pair (jointalign.multibatch.harvest_anchors).
    chromas[k][i] is sample i of batch k; tics[k][i] its raw TIC for peak detection (default: the
    chromatogram sum; every published run passed the raw TICs)."""
    cols: list[list[np.ndarray]] = [[] for _ in range(6)]
    tic = (lambda k, i: None) if tics is None else (lambda k, i: tics[k][i])
    for a, b in combinations(range(len(chromas)), 2):
        for i, ca in enumerate(chromas[a]):
            for j, cb in enumerate(chromas[b]):
                anc = match_subbin(ca, cb, tic(a, i), tic(b, j), cfg, axis)
                n = len(anc)
                for col, v in zip(cols, (a, b, i, j)):
                    col.append(np.full(n, v, dtype=int))
                cols[4].append(anc[:, 0])
                cols[5].append(anc[:, 1])
    return Anchors(*(np.concatenate(c) for c in cols), axis=axis)


def harvest(study: Study, cfg: MatchConfig = DEFAULT_MATCH) -> Anchors:
    """Sub-bin anchors on every cross-batch sample pair of a study, peaks on the raw TICs."""
    return harvest_arrays(
        [b.chromas for b in study.batches], [b.tic_raw for b in study.batches], cfg, study.axis
    )


def format_stages(stages: dict[str, int]) -> str:
    return ' → '.join(f'{k} {v}' for k, v in stages.items())
