"""Point-wise, event-level and inferential metrics."""
import numpy as np, pandas as pd
from scipy import stats
from .detectors import auprc

def pointwise(S, label, thr, mask):
    s = S[:, mask]; y = label[:, mask]; m = ~np.isnan(s)
    flag = (s > thr) & m
    tp = np.sum(flag & y); fp = np.sum(flag & ~y); fn = np.sum(~flag & y & m)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fpr = fp / max(np.sum(~y & m), 1)
    return dict(precision=prec, recall=rec, f1=f1, fpr=fpr, auprc=auprc(s.ravel(), y.ravel()), flags=flag)

def matched_threshold(S, label, mask, fpr=0.01):
    """Diagnostic: threshold giving a realized `fpr` on the clean test readings (uses test labels)."""
    s = S[:, mask]; y = label[:, mask]
    clean = s[~y & ~np.isnan(s)]
    if clean.size == 0:  # detector not applicable (no finite score on the test window)
        return np.nan
    return float(np.percentile(clean, 100 * (1 - fpr)))

def per_family(S, label_by_family, clean_label, thr, mask):
    """F1 per family: family readings positive, all clean readings negative."""
    out = {}
    s = S[:, mask]; m = ~np.isnan(s); flag = (s > thr) & m
    neg = ~clean_label[:, mask] & m
    for fam, lab in label_by_family.items():
        y = lab[:, mask]
        tp = np.sum(flag & y); fn = np.sum(~flag & y & m); fp = np.sum(flag & neg & ~y)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        out[fam] = 2 * prec * rec / max(prec + rec, 1e-12)
    return out

def events(flags, segs, mask_start, mask_end, min_hours=3):
    """Event-level detection for segments inside [mask_start, mask_end): any-flag criterion and a persistence criterion
    (>= min(min_hours, dur) flagged hours), delay to first flag, and detection within 24/72/168 h."""
    rows = []
    for r in segs.itertuples():
        if r.start < mask_start or r.end > mask_end:
            continue
        f = flags[r.sidx, r.start - mask_start:r.end - mask_start]
        n = int(f.sum()); dur = r.end - r.start
        first = int(np.argmax(f)) if n > 0 else np.nan
        rows.append(dict(var=r.var, family=r.family, dur_h=dur, any=n > 0, persistent=n >= min(min_hours, dur), delay=first,
                         within24=(n > 0 and first < 24), within72=(n > 0 and first < 72), within168=(n > 0 and first < 168)))
    return pd.DataFrame(rows)

def chance_events(flags, segs, mask_start, mask_end, clean_label, rng, min_hours=3, n_rep=20):
    """Chance level: apply the same flags to randomly placed windows of the same lengths on clean readings of the same station."""
    rows = []
    T = mask_end - mask_start
    for r in segs.itertuples():
        if r.start < mask_start or r.end > mask_end:
            continue
        dur = r.end - r.start
        for _ in range(n_rep):
            s = int(rng.integers(0, max(T - dur, 1)))
            win = flags[r.sidx, s:s + dur]
            cl = ~clean_label[r.sidx, mask_start + s:mask_start + s + dur]
            if cl.mean() < 0.9:
                continue
            n = int(win.sum())
            rows.append(dict(family=r.family, any=n > 0, persistent=n >= min(min_hours, dur)))
    return pd.DataFrame(rows)

def seed_paired(diffs):
    """Paired seed-level inference on differences (array over seeds): mean, sd, t-CI, p (two-sided)."""
    d = np.asarray(diffs, float); n = len(d)
    if n < 2:
        return dict(n=n, mean=float(d.mean()) if n else np.nan, sd=np.nan, ci_lo=np.nan, ci_hi=np.nan, p=np.nan)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n); tq = stats.t.ppf(0.975, n - 1)
    p = stats.ttest_1samp(d, 0.0).pvalue if sd > 0 else (0.0 if m != 0 else 1.0)
    return dict(n=n, mean=float(m), sd=float(sd), ci_lo=float(m - tq * se), ci_hi=float(m + tq * se), p=float(p))

def tost(diffs, margin):
    """Two one-sided tests for equivalence within +/- margin; returns the larger of the two one-sided p-values."""
    d = np.asarray(diffs, float); n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return np.nan
    se = d.std(ddof=1) / np.sqrt(n)
    t1 = (d.mean() + margin) / se; t2 = (d.mean() - margin) / se
    p1 = 1 - stats.t.cdf(t1, n - 1); p2 = stats.t.cdf(t2, n - 1)
    return float(max(p1, p2))

def block_bootstrap_f1_diff(flagsA, flagsB, label, valid, block_h=720, n_boot=2000, seed=0):
    """Paired moving-block bootstrap of pooled F1 (over all seeds' stacked test windows) and of the difference A-B.
    flagsA/B, label, valid: arrays (S, N, T_test) stacked over seeds; blocks are calendar blocks of block_h hours shared across seeds.
    Implemented with per-block TP/FP/FN counts weighted by the block multiplicity of each resample (identical to re-indexing)."""
    rng = np.random.default_rng(seed)
    S_, N, T = label.shape
    nb = int(np.ceil(T / block_h))
    def counts(fl):
        f = fl & valid
        tp = (f & label).sum(axis=(0, 1)); fp = (f & ~label).sum(axis=(0, 1)); fn = (~f & label & valid).sum(axis=(0, 1))  # per hour
        c = np.zeros((3, nb))
        for b in range(nb):
            sl = slice(b * block_h, min((b + 1) * block_h, T))
            c[0, b] = tp[sl].sum(); c[1, b] = fp[sl].sum(); c[2, b] = fn[sl].sum()
        return c
    cA, cB = counts(flagsA), counts(flagsB)
    def f1(c, w):
        tp, fp, fn = (c * w).sum(1)
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); return 2 * p * r / max(p + r, 1e-12)
    stats_ = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        stats_.append((f1(cA, w), f1(cB, w)))
    stats_ = np.array(stats_)
    one = np.ones(nb)
    full = (f1(cA, one), f1(cB, one))
    d = stats_[:, 0] - stats_[:, 1]
    return dict(f1A=full[0], f1B=full[1], diff=full[0] - full[1], ci_lo=float(np.percentile(d, 2.5)), ci_hi=float(np.percentile(d, 97.5)), n_blocks=nb)
