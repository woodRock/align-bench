"""Golden (bit-identity) cases: one case builder run against two APIs.

`build_warpalign(api)` calls every function M1 moved out of warpalign on the synthetic pairs in
synth.PAIRS (and on adversarial anchor lists and random vectors), through `api`: a namespace that
maps one set of names to either the pinned upstream functions (upstream_api; used by
scripts/00_make_golden.py to write tests/golden/*.npz) or alignbench's (alignbench_api; used by
tests/test_golden.py). Both sides therefore make exactly the same calls, and the outputs are
flattened to {key: array} the same way and compared with np.array_equal (NaN == NaN), dtype and
shape included.

Large outputs (warped [N, T, M] chromatograms) are stored as a digest of their bytes.
"""

from __future__ import annotations

import hashlib
import platform
from importlib.metadata import version
from types import SimpleNamespace

import numpy as np

import synth


def platform_tag() -> str:
    return f'{platform.system()}-{platform.machine()}'.lower()


def environment() -> str:
    """Platform and numeric library versions: bit-identity is only expected within one."""
    libs = ' '.join(f'{d}={version(d)}' for d in ('numpy', 'scipy', 'scikit-learn'))
    return f'{platform.system()}-{platform.machine()} python={platform.python_version()} {libs}'


# ── Flattening and comparison ────────────────────────────────────────────────


def digest(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    return f'{x.dtype.str}{x.shape}:' + hashlib.sha256(x.tobytes()).hexdigest()


def _scalar(x) -> bool:
    return isinstance(x, int | float | bool | str | np.generic)


def flatten(obj, prefix: str = '', out: dict | None = None) -> dict[str, np.ndarray]:
    """Nested dicts / lists / tuples / arrays / scalars → {'a/b/0': array}. Dict key order and
    list lengths are recorded too ('#keys', '#len'), so they are compared like values."""
    out = {} if out is None else out
    if isinstance(obj, dict):
        out[f'{prefix}#keys'] = np.array([str(k) for k in obj], dtype=str)
        for k, v in obj.items():
            flatten(v, f'{prefix}{k}/', out)
    elif isinstance(obj, list | tuple) and not all(_scalar(v) for v in obj):
        out[f'{prefix}#len'] = np.array(len(obj))
        for i, v in enumerate(obj):
            flatten(v, f'{prefix}{i}/', out)
    elif isinstance(obj, list | tuple):
        out[f'{prefix}#list'] = np.array(obj) if obj else np.zeros(0)
    else:
        out[prefix.rstrip('/') or '_'] = np.asarray(obj)
    return out


def compare(got: dict[str, np.ndarray], want: dict[str, np.ndarray], float_atol: float = 0.0) -> list[str]:
    """Every key whose value differs (missing, extra, dtype, shape or any element). float_atol > 0
    only for comparisons across platforms (see test_golden.py); integers and strings are always
    compared exactly."""
    bad = [f'missing: {k}' for k in sorted(set(want) - set(got))]
    bad += [f'extra: {k}' for k in sorted(set(got) - set(want))]
    for k in sorted(set(got) & set(want)):
        g, w = np.asarray(got[k]), np.asarray(want[k])
        if g.dtype != w.dtype or g.shape != w.shape:
            bad.append(f'{k}: {g.dtype}{g.shape} != {w.dtype}{w.shape}')
        elif g.dtype.kind in 'fc' and float_atol > 0:
            if not np.allclose(g, w, rtol=0.0, atol=float_atol, equal_nan=True):
                bad.append(f'{k}: max |diff| {np.nanmax(np.abs(g - w)):.3g} > {float_atol}')
        elif not np.array_equal(g, w, equal_nan=g.dtype.kind in 'fc'):
            bad.append(f'{k}: values differ')
    return bad


# ── The two APIs ─────────────────────────────────────────────────────────────


def upstream_api() -> SimpleNamespace:
    """The pinned warpalign functions (needs the [golden] extra)."""
    from warpalign import anchors, baselines, config, data, evaluation, metrics, warps

    return SimpleNamespace(
        source='warpalign',
        axis=dict(
            n_bins=config.N_BINS,
            run_min=config.RUN_MIN,
            bin_min=config.BIN_MIN,
            times=config.TIME_AXIS,
            last_min=config.LAST_MIN,
            eval_start_bin=config.EVAL_START_BIN,
            eval_bins=(config.EVAL_BINS.start, config.EVAL_BINS.stop),
        ),
        Batch=data.Batch,
        acquisition_start=data.acquisition_start,
        detect_peaks=anchors.detect_peaks,
        fingerprint=anchors.fingerprint,
        lis=anchors.longest_increasing_subsequence,
        monotone_subset=anchors.monotone_subset,
        slope_filter=anchors.slope_filter,
        merge_close=anchors.merge_close,
        drift_outlier_filter=anchors.drift_outlier_filter,
        pool_anchors=anchors.pool_anchors,
        match_sample_pair=anchors.match_anchors,
        match_pooled_arrays=anchors.extract_anchors,
        match_pooled_n=lambda ca, cb, ra, rb, n: anchors.extract_anchors(ca, cb, ra, rb, n_ref=n),
        warp_signal=warps.warp_signal,
        warp_rows=warps.warp_rows,
        src_from_fn=warps.src_from_fn,
        monotone_interp=warps.monotone_interp,
        pchip_warp=warps.pchip_warp,
        constant_shift_warp=warps.constant_shift_warp,
        linear_warp=warps.linear_warp,
        anchor_drift=warps.anchor_drift,
        anchor_line=warps.anchor_line,
        predict_query_bin=warps.predict_query_bin,
        invert_src=warps.invert_src,
        cow_tic_align=baselines.cow_tic_align,
        icoshift_align=baselines.icoshift_align,
        pearson_rows=metrics.pearson_rows,
        pearson_rows_all=lambda a, b: metrics.pearson_rows(a, b, None),
        smooth_rows=metrics.smooth_rows,
        pearson_rows_smoothed=metrics.pearson_rows_smoothed,
        all_pairs=metrics.all_pairs,
        pearson_paired=metrics.pearson_paired,
        pearson_rows_masked=metrics.pearson_rows_masked,
        pearson=metrics.pearson,
        tic_r_pairs=metrics.tic_r_pairs,
        within_batch_r=metrics.within_batch_r,
        study_silhouette=metrics.study_silhouette,
        warp_magnitude=metrics.warp_magnitude,
        per_a_mean=metrics.per_a_mean,
        holm=metrics.holm,
        mean_sd=metrics.mean_sd,
        paired_wilcoxon=metrics.paired_wilcoxon,
        metric_suffixes=evaluation.metric_suffixes,
        in_range=evaluation.in_range,
        tic_metrics=evaluation.tic_metrics,
        eval_unaligned=lambda a, b: _pair_eval(evaluation.unaligned(a, b)),
        eval_anchor_warp=lambda a, b, fn: _pair_eval(evaluation.anchor_warp(a, b, fn)),
    )


def _pair_eval(ev) -> dict:
    return {'metrics': ev.metrics, 'sil': ev.sil, 'warp_mag': ev.warp_mag, 'outside': ev.outside}


def alignbench_api() -> SimpleNamespace:
    from alignbench import anchors, data, warps
    from alignbench.axis import FISH_OIL_AXIS as ax
    from alignbench.methods import tic as tic_methods
    from alignbench.metrics import diagnostics, stats, tic

    def eval_anchor_warp(a, b, fn):
        # warpalign evaluation.anchor_warp, composed from the moved pieces. The B-frame protocol
        # (M3) replaces this; its own golden test compares against the same fixture keys.
        src = warps.src_from_fn(fn)
        met, outside = tic.tic_metrics(
            warps.warp_rows(a.tic, src),
            b.tic,
            warps.warp_rows(a.tic_raw, src),
            b.tic_raw,
            tic.in_range(src, a.acquisition_start),
            tic.in_range(warps.identity_src(), b.acquisition_start),
        )
        sil = diagnostics.study_silhouette(warps.warp_rows(a.chromas, src), b.chromas)
        return {'metrics': met, 'sil': sil, 'warp_mag': diagnostics.warp_magnitude(src), 'outside': outside}

    def eval_unaligned(a, b):
        ident = warps.identity_src()
        met, outside = tic.tic_metrics(
            a.tic,
            b.tic,
            a.tic_raw,
            b.tic_raw,
            tic.in_range(ident, a.acquisition_start),
            tic.in_range(ident, b.acquisition_start),
        )
        return {
            'metrics': met,
            'sil': diagnostics.study_silhouette(a.chromas, b.chromas),
            'warp_mag': 0.0,
            'outside': outside,
        }

    cfg = anchors.MatchConfig
    return SimpleNamespace(
        source='alignbench',
        axis=dict(
            n_bins=ax.n_bins,
            run_min=ax.run_min,
            bin_min=ax.bin_min,
            times=ax.times,
            last_min=ax.last_min,
            eval_start_bin=ax.eval_start_bin,
            eval_bins=(ax.eval_bins.start, ax.eval_bins.stop),
        ),
        Batch=data.Batch,
        acquisition_start=data.acquisition_start,
        detect_peaks=anchors.detect_peaks,
        fingerprint=anchors.fingerprint,
        lis=anchors.longest_increasing_subsequence,
        monotone_subset=anchors.monotone_subset,
        slope_filter=anchors.slope_filter,
        merge_close=anchors.merge_close,
        drift_outlier_filter=anchors.drift_outlier_filter,
        pool_anchors=anchors.pool_anchors,
        match_sample_pair=anchors.match_sample_pair,
        match_pooled_arrays=anchors.match_pooled_arrays,
        match_pooled_n=lambda ca, cb, ra, rb, n: anchors.match_pooled_arrays(ca, cb, ra, rb, cfg(n_ref=n)),
        warp_signal=warps.warp_signal,
        warp_rows=warps.warp_rows,
        src_from_fn=warps.src_from_fn,
        monotone_interp=warps.monotone_interp,
        pchip_warp=warps.pchip_warp,
        constant_shift_warp=warps.constant_shift_warp,
        linear_warp=warps.linear_warp,
        anchor_drift=warps.anchor_drift,
        anchor_line=warps.anchor_line,
        predict_query_bin=warps.predict_query_bin,
        invert_src=warps.invert_src,
        cow_tic_align=tic_methods.cow_tic_align,
        icoshift_align=tic_methods.icoshift_align,
        pearson_rows=tic.pearson_rows,
        pearson_rows_all=lambda a, b: tic.pearson_rows(a, b, None),
        smooth_rows=tic.smooth_rows,
        pearson_rows_smoothed=tic.pearson_rows_smoothed,
        all_pairs=tic.all_pairs,
        pearson_paired=tic.pearson_paired,
        pearson_rows_masked=tic.pearson_rows_masked,
        pearson=tic.pearson,
        tic_r_pairs=tic.tic_r_pairs,
        within_batch_r=tic.within_batch_r,
        study_silhouette=diagnostics.study_silhouette,
        warp_magnitude=diagnostics.warp_magnitude,
        per_a_mean=tic.per_a_mean,
        holm=stats.holm,
        mean_sd=stats.mean_sd,
        paired_wilcoxon=stats.paired_wilcoxon,
        metric_suffixes=tic.metric_suffixes,
        in_range=tic.in_range,
        tic_metrics=tic.tic_metrics,
        eval_unaligned=eval_unaligned,
        eval_anchor_warp=eval_anchor_warp,
    )


# ── The cases ────────────────────────────────────────────────────────────────


def _anchor_lists(rng: np.random.Generator) -> list[list[tuple[int, int]]]:
    """Adversarial (r, q) lists for the order and pooling filters: a smooth trend with
    misidentifications, duplicates in r, near neighbours (merging) and slope violations."""
    out = []
    for _ in range(6):
        n = int(rng.integers(3, 25))
        r = np.sort(rng.choice(np.arange(20, 190), n, replace=False))
        q = np.round(r - 6 - 0.03 * r + rng.normal(0, 0.7, n)).astype(int)
        bad = rng.random(n) < 0.2
        q[bad] += rng.integers(-25, 25, bad.sum())
        anchors = [(int(a), int(b)) for a, b in zip(r, q)]
        anchors += [(int(a) + 1, int(b) + 1) for a, b in anchors[:3]]  # merge candidates
        anchors += [(anchors[0][0], anchors[0][1] + 4)]  # duplicate r
        out.append(anchors)
    out += [[], [(50, 45)], [(50, 45), (80, 74)], [(50, 45), (51, 60), (52, 46), (90, 85)]]
    return out


def build_warpalign(api) -> dict:
    """Every M1 function from warpalign, on the synthetic pairs and adversarial inputs."""
    rng = np.random.default_rng(2026)
    out: dict = {'axis': api.axis}

    # Order and pooling filters on adversarial lists
    lists = _anchor_lists(rng)
    out['filters'] = [
        {
            'lis': api.lis([q for _, q in sorted(a)]),
            'monotone': api.monotone_subset(a),
            'slope': api.slope_filter(a),
            'merge': api.merge_close(a),
            'outlier': api.drift_outlier_filter(a),
        }
        for a in lists
    ]
    out['lis_random'] = [api.lis(rng.integers(0, 50, n).tolist()) for n in (0, 1, 7, 40)]
    out['pool'] = list(api.pool_anchors([lists[0], lists[1], lists[2]]))

    # Warp primitives on random signals, with near-integer and out-of-range positions
    x1, x2 = rng.normal(size=200), rng.normal(size=(200, 7)).astype(np.float32)
    src = np.cumsum(rng.uniform(0.6, 1.4, 200)) - 15.0
    src[50:55] = np.arange(50, 55) * 0.225 / 0.225 + 1e-12
    out['warp_signal'] = [api.warp_signal(x1, src), api.warp_signal(x2, src)]
    out['warp_rows'] = [
        api.warp_rows(np.stack([x1, x1[::-1]]), src),
        api.warp_rows(np.stack([x1, x1[::-1]]), np.stack([src, src + 0.5])),
    ]
    xs, ys = np.array([1.0, 3.0, 7.5, 20.0]), np.array([0.5, 3.2, 8.0, 18.0])
    t = np.linspace(-5.0, 50.0, 301)
    out['monotone_interp'] = [api.monotone_interp(xs, ys)(t), api.monotone_interp(xs[:1], ys[:1])(t)]
    out['invert_src'] = api.invert_src(np.sort(src), np.linspace(-30, 230, 97))

    # Stats on random vectors
    p = rng.uniform(0, 0.2, 9)
    xa, xb = rng.normal(size=12), rng.normal(size=12)
    out['stats'] = {
        'holm': api.holm(p),
        'mean_sd': [api.mean_sd(xa), api.mean_sd(xa[:1])],
        'wilcoxon': [api.paired_wilcoxon(xa, xb), api.paired_wilcoxon(xa, xa)],
    }
    out['metric_suffixes'] = [api.metric_suffixes(), api.metric_suffixes((0.5, 3.0))]

    for name, make in synth.PAIRS.items():
        (ca, ra), (cb, rb) = make()
        case: dict = {'inputs': [digest(v) for v in (ca, ra, cb, rb)]}
        a = api.Batch('a', ca, ra, tuple(f'a{i}' for i in range(len(ca))))
        b = api.Batch('b', cb, rb, tuple(f'b{i}' for i in range(len(cb))))
        case['acq_start'] = [api.acquisition_start(ra), api.acquisition_start(rb)]
        case['peaks'] = [api.detect_peaks(x) for x in np.concatenate([ra, rb])]
        case['fingerprints'] = [api.fingerprint(ca[0], k) for k in (0, 1, 57, 198, 199)]

        # Anchors: every sample pair, the pooled matcher at the default and all-sample n_ref
        case['match_pairs'] = [
            list(api.match_sample_pair(ca[i], cb[j], ra[i], rb[j]))
            for i in range(len(ca))
            for j in range(min(2, len(cb)))
        ]
        res = api.match_pooled_arrays(ca, cb, ra, rb)
        pooled = res.anchors
        case['pooled'] = vars(res)
        case['pooled_all'] = vars(api.match_pooled_n(ca, cb, ra, rb, len(ca)))

        # Anchor warps through the pooled anchors
        fns = {
            'pchip': api.pchip_warp(pooled),
            'constant': api.constant_shift_warp(pooled),
            'linear': api.linear_warp(pooled),
        }
        srcs = {k: api.src_from_fn(fn) for k, fn in fns.items()}
        case['src'] = srcs
        case['src_identity'] = api.src_from_fn(None)
        case['anchor_drift'] = api.anchor_drift(pooled)
        case['anchor_line'] = list(api.anchor_line(pooled))
        if pooled:
            case['predict'] = [api.predict_query_bin(pooled, r) for r in (0.0, 33.3, 100.0, 199.0)]
        case['invert'] = api.invert_src(srcs['pchip'], np.array([-4.0, 0.0, 20.5, 150.0, 210.0]))
        case['warped_chromas'] = digest(api.warp_rows(ca, srcs['pchip']))
        case['warp_mag'] = {k: api.warp_magnitude(s) for k, s in srcs.items()}

        # TIC baselines on every (A_i, B_j)
        tic_a, tic_b = ca.sum(-1), cb.sum(-1)
        cow, ico = [], []
        for i in range(len(ca)):
            for j in range(len(cb)):
                for fn, sink in ((api.cow_tic_align, cow), (api.icoshift_align, ico)):
                    al = fn(tic_a[i], tic_b[j])
                    sink.append({'warped': al.warped, 'src': al.src, 'details': al.details, 'mag': al.warp_mag})
        case['cow'], case['icoshift'] = cow, ico
        al = api.cow_tic_align(tic_a[0], tic_b[0], n_segs=7, slack=3)
        case['cow_narrow'] = {'warped': al.warped, 'src': al.src, 'details': al.details}

        # Metrics
        case['pearson_rows'] = [api.pearson_rows(tic_a, tic_b), api.pearson_rows_all(ra, rb)]
        case['pearson'] = api.pearson(tic_a[0], tic_b[0])
        case['smooth'] = [api.smooth_rows(ra, 0.0), api.smooth_rows(ra, 1.5)]
        case['smoothed'] = api.pearson_rows_smoothed(ra, rb, (1.0, 2.0))
        va = api.in_range(srcs['pchip'], api.acquisition_start(ra))
        vb = api.in_range(np.arange(200.0), api.acquisition_start(rb))
        case['in_range'] = [va, vb]
        case['masked'] = api.pearson_rows_masked(tic_a, tic_b, va, vb)
        xa, xb = api.all_pairs(tic_a, tic_b)
        case['paired'] = [api.pearson_paired(xa, xb), api.pearson_paired(xa[:, :5], xb[:, :5], None, None)]
        case['tic_r_pairs'] = api.tic_r_pairs(ca, cb)
        case['within'] = list(api.within_batch_r(tic_a))
        case['per_a'] = api.per_a_mean(case['tic_r_pairs'], len(ca), len(cb))
        case['silhouette'] = api.study_silhouette(ca, cb)
        cow_src = np.stack([c['src'] for c in cow])
        n_b = len(cb)
        case['tic_metrics_paired'] = list(
            api.tic_metrics(
                np.stack([c['warped'] for c in cow]),
                np.repeat(tic_b[None], len(ca), 0).reshape(-1, 200),
                api.warp_rows(np.repeat(ra, n_b, 0), cow_src),
                np.tile(rb, (len(ca), 1)),
                api.in_range(cow_src, np.repeat(api.acquisition_start(ra), n_b)),
                api.in_range(np.arange(200.0), np.tile(api.acquisition_start(rb), len(ca))),
                paired=True,
            )
        )
        case['eval'] = {
            'unaligned': api.eval_unaligned(a, b),
            **{k: api.eval_anchor_warp(a, b, fn) for k, fn in fns.items()},
        }
        out[f'pair_{name}'] = case
    return out
