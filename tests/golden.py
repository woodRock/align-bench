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
from types import SimpleNamespace

import numpy as np

import synth


def platform_tag() -> str:
    return f'{platform.system()}-{platform.machine()}'.lower()


def environment() -> str:
    """Platform, BLAS and numeric library versions (alignbench.provenance.environment)."""
    from alignbench.provenance import environment as env

    return env()


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


# ── jointalign (842ca595) and trustalign (c7b3e42a) ──────────────────────────

ANCHOR_COLS = ('ba', 'bb', 'sa', 'sb', 'ta', 'tb')


def _anc(a) -> dict:
    return {f: getattr(a, f) for f in ANCHOR_COLS}


def upstream_api_m2() -> SimpleNamespace:
    """The pinned jointalign and trustalign functions (needs the [golden] extra)."""
    from dataclasses import asdict

    from jointalign import multibatch as mb
    from jointalign import synthetic as syn
    from trustalign import baselines, graphs, metrics, worlds

    return SimpleNamespace(
        source='jointalign+trustalign',
        Anchors=mb.Anchors,
        DEFAULT_BATCHES=syn.DEFAULT_BATCHES,
        BatchSpec=syn.BatchSpec,
        make_world=syn.make_world,
        true_map=syn.true_map,
        match_subbin=mb.match_anchors,
        harvest=mb.harvest_anchors,
        pchip_pairwise=mb.pchip_pairwise,
        world=worlds.make,
        world_parts=lambda w: dict(
            specs=[asdict(s) for s in w.specs],
            chromas=w.chromas,
            tic_raws=w.tic_raws,
            truth=w.truth,
            sizes=w.sizes,
        ),
        observed_span=lambda w, k: w.observed_span(k),
        world_true_map=lambda w, k, l, t: w.true_map(k, l, t),
        draw_specs=lambda K, rng: [asdict(s) for s in worlds.draw_specs(K, rng)],
        g=graphs,
        shortest_paths=baselines.shortest_paths,
        counts=baselines._counts,
        composed_pchip=lambda anc, K, edges: baselines.composed_pchip(anc, K, edges).map,
        rt_folds=metrics.rt_folds,
        anchor_errors=metrics.anchor_errors,
        select_lambda=metrics.select_lambda,
        inside_mask=metrics.inside_mask,
        map_errors=metrics.map_errors,
        summarise=metrics.summarise,
        cycle_inconsistency=metrics.cycle_inconsistency,
        rejection=metrics.rejection,
    )


def alignbench_api_m2() -> SimpleNamespace:
    from dataclasses import asdict

    from alignbench import anchors, graphs, maps
    from alignbench.datasets import synthetic as syn
    from alignbench.metrics import anchors as manchors
    from alignbench.metrics import truth as mtruth

    return SimpleNamespace(
        source='alignbench',
        Anchors=anchors.Anchors,
        DEFAULT_BATCHES=syn.DEFAULT_BATCHES,
        BatchSpec=syn.BatchSpec,
        make_world=syn.make_world,
        true_map=syn.true_map,
        match_subbin=anchors.match_subbin,
        harvest=anchors.harvest_arrays,
        pchip_pairwise=maps.pchip_pairwise,
        world=syn.make_world_k,
        world_parts=lambda s: dict(
            specs=[asdict(x) for x in s.truth.specs],
            chromas=[b.chromas for b in s.batches],
            tic_raws=[b.tic_raw for b in s.batches],
            truth=s.truth.truth,
            sizes=[len(b) for b in s.batches],
        ),
        observed_span=lambda s, k: s.truth.observed_span(k),
        world_true_map=lambda s, k, l, t: s.truth.true_map(k, l, t),
        draw_specs=lambda K, rng: [asdict(x) for x in syn.draw_specs(K, rng)],
        g=graphs,
        shortest_paths=maps.shortest_paths,
        counts=maps.edge_counts,
        composed_pchip=lambda anc, K, edges: maps.composed_pchip(anc, K, edges)[0],
        rt_folds=manchors.rt_folds,
        anchor_errors=manchors.anchor_errors,
        select_lambda=manchors.select_lambda,
        inside_mask=mtruth.inside_mask,
        map_errors=mtruth.map_errors,
        summarise=mtruth.summarise,
        cycle_inconsistency=mtruth.cycle_inconsistency,
        rejection=mtruth.rejection,
    )


def build_jointalign(api) -> dict:
    """jointalign: synthetic worlds and truth, sub-bin matching, harvest, Anchors, pairwise PCHIP."""
    out: dict = {}
    grid = np.linspace(-2.0, 47.0, 211)
    worlds = {
        'default': api.make_world(),
        'custom': api.make_world(
            batches=(
                api.BatchSpec('x', 3, 1.0, -0.5, 0.3, 0.2, 4.0),
                api.BatchSpec('y', 2, -1.5, 1.0, 0.0, 0.0, 3.0),
                api.BatchSpec('z', 3, 0.3, 2.0, 0.1, 5.0, 5.0),
            ),
            n_compounds=40,
            mz_bins=400,
            sigma_bins=1.1,
            jitter_min=0.08,
            seed=5,
        ),
    }
    for wname, (chromas, tics, truth) in worlds.items():
        names = list(truth['batches'])
        case: dict = {
            'chromas': [digest(c) for c in chromas],
            'tic_raws': [digest(t) for t in tics],
            'truth': truth,
            'true_map': {f'{a}>{b}': api.true_map(truth, a, b, grid) for a in names for b in names if a != b},
        }
        anc = api.harvest(chromas, tics)
        case['harvest'] = _anc(anc)
        case['pair'] = [_anc(anc.pair(0, 1)), _anc(anc.pair(2, 0))]
        case['subset'] = _anc(anc.subset(anc.ta > 100))
        bins = np.linspace(-5.0, 205.0, 97)
        case['pchip'] = {
            f'{a}>{b}': api.pchip_pairwise(anc, a, b)(bins)
            for a in range(len(names))
            for b in range(len(names))
            if a != b
        }
        out[f'world_{wname}'] = case
    small = worlds['custom'][0]
    out['harvest_no_tics'] = _anc(api.harvest([c[:2] for c in small]))
    one = api.Anchors(*(np.array([v]) for v in (0, 1, 0, 0, 50.0, 47.0)))
    out['pchip_one_anchor'] = api.pchip_pairwise(one, 0, 1)(np.array([0.0, 50.0, 199.0]))
    for name, make in synth.PAIRS.items():
        (ca, ra), (cb, rb) = make()
        out[f'subbin_{name}'] = [
            api.match_subbin(ca[i], cb[j], ra[i], rb[j]) for i in range(len(ca)) for j in range(2)
        ]
        out[f'subbin_sum_{name}'] = [api.match_subbin(ca[i], cb[0]) for i in range(len(ca))]
    return out


def build_trustalign(api) -> dict:
    """trustalign: K-batch worlds, graph topologies and perturbations, shortest paths, composed
    PCHIP, held-out folds and λ selection, truth metrics."""
    out: dict = {'draw_specs': api.draw_specs(6, np.random.default_rng(3))}
    g = api.g
    for K, seed in ((4, 0), (5, 3)):
        w = api.world(K, seed)
        parts = api.world_parts(w)
        case: dict = {
            'specs': parts['specs'],
            'chromas': [digest(c) for c in parts['chromas']],
            'tic_raws': [digest(t) for t in parts['tic_raws']],
            'truth': parts['truth'],
            'span': [list(api.observed_span(w, k)) for k in range(K)],
        }
        times = np.arange(200) * 0.225
        case['true_map'] = {
            f'{k}>{l}': api.world_true_map(w, k, l, times) for k in range(K) for l in range(K) if k != l
        }
        anc = api.harvest(parts['chromas'], parts['tic_raws'])
        case['harvest'] = _anc(anc)
        sizes = parts['sizes']
        case['hubs'] = [g.weak_hub(sizes), g.hub_of('weak_star', sizes), g.hub_of('star', sizes)]
        topo = {}
        for t in g.TOPOLOGIES:
            if t.startswith('chain+') and int(t[6:]) > K * (K - 1) // 2 - (K - 1):
                continue
            topo[t] = sorted(g.edges_for(t, K, sizes, np.random.default_rng(7)))
        case['topologies'] = topo
        rng = np.random.default_rng(11)
        edges = set(map(tuple, topo['ring']))
        kept = g.thin(anc, edges, 5, rng)
        case['thin'] = [kept, g.thin(anc, edges, None, rng), g.thin(anc, set(), 3, rng)]
        bad_anc, bad = g.corrupt(anc.subset(kept), 0.25, rng)
        case['corrupt'] = [_anc(bad_anc), bad]
        case['jitter'] = [_anc(g.jitter(anc, 0.5, rng)), _anc(g.jitter(anc, 0.0, rng))]
        case['chords'] = sorted(g.chain_with_chords(K, 1, rng))
        order = g.removal_order(K, rng)
        case['removal'] = [order, sorted(g.thinned_complete(K, 0.5, order))]
        case['reachable'] = sorted(g.reachable(1, K, {(0, 1), (2, 3)}))
        case['adjacency'] = g.adjacency(K, edges)
        for t in ('complete', 'chain', 'star', 'ring'):
            e = set(map(tuple, topo[t]))
            case[f'paths_{t}'] = {
                f'{k}>{l}': p for (k, l), p in api.shortest_paths(K, e, api.counts(anc, e)).items()
            }
        gmap = api.composed_pchip(anc, K, set(map(tuple, topo['chain'])))
        bins = np.linspace(-3.0, 202.0, 83)
        case['composed'] = {f'{k}>{l}': gmap(k, l, bins) for k in range(K) for l in range(K)}
        case['folds'] = api.rt_folds(anc)
        case['anchor_errors'] = api.anchor_errors(gmap, anc)
        complete = set(map(tuple, topo['complete']))
        fit = lambda a, lam, K=K, e=complete: SimpleNamespace(map=api.composed_pchip(a, K, e))
        case['select_lambda'] = list(api.select_lambda(fit, anc, (1.0, 10.0)))
        case['inside'] = {
            f'{k}>{l}': api.inside_mask(w, k, l, times) for k in range(K) for l in range(K) if k != l
        }
        errs = api.map_errors(w, gmap)
        case['map_errors'] = {f'{k}>{l}': v for (k, l), v in errs.items()}
        case['map_errors_some'] = {
            f'{k}>{l}': v for (k, l), v in api.map_errors(w, gmap, [(0, 1), (2, 0)]).items()
        }
        case['summarise'] = [api.summarise(errs, set(map(tuple, topo['chain']))), api.summarise({}, set())]
        case['cycle'] = list(api.cycle_inconsistency(w, gmap))
        wts = np.where(rng.random(len(bad)) < 0.3, 0.0, rng.random(len(bad)))
        case['rejection'] = [list(api.rejection(wts, bad)), list(api.rejection(wts, np.zeros(len(bad), bool)))]
        out[f'k{K}_s{seed}'] = case
    return out
