"""Classical detectors, no-training controls, residual channels, two-timescale scoring and threshold calibration.
All detectors consume standardized anomalies a (N,T) per variable (NaN where missing) and return raw residual channels.
"""
import numpy as np
from .data import haversine_km

# ---------------------------------------------------------------- residual channels
def ema(x, horizon):
    """Causal exponential moving average with horizon h (alpha = 2/(h+1)); NaNs are carried (ignored in the update)."""
    if horizon <= 1:
        return x.copy()
    alpha = 2.0 / (horizon + 1.0)
    out = np.empty_like(x); out[:] = np.nan
    s = np.full(x.shape[0], np.nan)
    for t in range(x.shape[1]):
        v = x[:, t]
        m = ~np.isnan(v)
        s = np.where(m, np.where(np.isnan(s), v, alpha * v + (1 - alpha) * s), s)
        out[:, t] = s
    return out

def persistence_innovation(a):
    r = np.full_like(a, np.nan); r[:, 1:] = a[:, 1:] - a[:, :-1]; return r

def kalman_innovation(a, q):
    """Local-level model: level_t = level_{t-1} + eta (var q), obs = level + eps (var 1). Returns standardized innovation."""
    N, T = a.shape
    out = np.full_like(a, np.nan)
    m = np.zeros(N); P = np.ones(N) * 10.0; init = np.zeros(N, dtype=bool)
    for t in range(T):
        y = a[:, t]; obs = ~np.isnan(y)
        P = P + q
        S = P + 1.0
        innov = (y - m) / np.sqrt(S)
        out[obs, t] = innov[obs]
        out[obs & ~init, t] = np.nan  # no innovation before the first observation
        K = P / S
        m = np.where(obs, m + K * (y - m), m)
        P = np.where(obs, (1 - K) * P, P)
        init = init | obs
    return out

def clim_z(a):
    return a.copy()

def train_correlations(a, train_mask):
    """Pairwise correlation of anomalies on the training window (pairwise-complete)."""
    x = a[:, train_mask]
    N = x.shape[0]
    C = np.full((N, N), np.nan)
    for i in range(N):
        xi = x[i]
        for j in range(i, N):
            m = ~np.isnan(xi) & ~np.isnan(x[j])
            if m.sum() > 500:
                c = np.corrcoef(xi[m], x[j][m])[0, 1]
                C[i, j] = C[j, i] = c
    np.fill_diagonal(C, np.nan)
    return C

def sct_weights(C, kind="corr2", k=None):
    """Leave-one-out neighbour weights from training correlations (squared, non-negative), optionally top-k."""
    W = np.where(np.isnan(C), 0.0, np.clip(C, 0, None) ** 2)
    np.fill_diagonal(W, 0.0)
    if k is not None:
        for i in range(W.shape[0]):
            thr = np.sort(W[i])[-k] if (W[i] > 0).sum() >= k else 0
            W[i, W[i] < thr] = 0.0
    return W

def sct_estimate(a, W, train_mask, calibrate=True):
    """Weighted leave-one-out neighbour estimate of the anomaly, with an optional training-window regression slope per station
    (the spatial-regression idea: a_i ~ beta_i * neighbour_mean). Residual = a - estimate."""
    N, T = a.shape
    A = np.where(np.isnan(a), 0.0, a); M = (~np.isnan(a)).astype(float)
    num = W @ A; den = W @ M
    est = np.where(den > 1e-9, num / np.maximum(den, 1e-9), np.nan)
    if calibrate:
        beta = np.ones(N)
        for i in range(N):
            m = train_mask & ~np.isnan(a[i]) & ~np.isnan(est[i])
            if m.sum() > 500:
                xx = est[i][m]; yy = a[i][m]
                beta[i] = float(np.dot(xx, yy) / max(np.dot(xx, xx), 1e-9))
        est = est * beta[:, None]
    return est

def oi_estimate(a, meta, train_mask, L_km=60.0, H_m=300.0, eps=0.1, kmax=20, rmax_km=200.0, allowed=None):
    """Leave-one-out optimal-interpolation estimate with a Gaussian covariance in horizontal distance and elevation
    (the structure of operational spatial consistency tests), climatological background (zero anomaly), k nearest neighbours.
    eps = observation-error to background-error variance ratio. allowed: optional (N,N) boolean mask of admissible neighbours
    (e.g. same silo); None admits every station."""
    N, T = a.shape
    lat, lon, elev = meta.lat.to_numpy(), meta.lon.to_numpy(), meta.elev.to_numpy()
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    Z = np.abs(elev[:, None] - elev[None, :])
    Cfull = np.exp(-0.5 * (D / L_km) ** 2 - 0.5 * (Z / H_m) ** 2)
    est = np.full_like(a, np.nan)
    A = np.where(np.isnan(a), 0.0, a); M = ~np.isnan(a)
    for i in range(N):
        cand = np.argsort(D[i])
        cand = [j for j in cand if j != i and D[i, j] <= rmax_km and (allowed is None or allowed[i, j])][:kmax]
        if len(cand) < 2:
            continue
        cand = np.array(cand)
        Cnn = Cfull[np.ix_(cand, cand)] + eps * np.eye(len(cand))
        c_i = Cfull[i, cand]
        # handle missing neighbours per time step by grouping identical availability patterns
        Mi = M[cand]  # (k, T)
        patterns = {}
        keys = np.packbits(Mi, axis=0).T  # (T, bytes)
        for t in range(T):
            key = keys[t].tobytes()
            if key not in patterns:
                avail = Mi[:, t]
                if avail.sum() < 2:
                    patterns[key] = None
                else:
                    idx = np.where(avail)[0]
                    w = np.linalg.solve(Cnn[np.ix_(idx, idx)], c_i[idx])
                    patterns[key] = (idx, w)
            pw = patterns[key]
            if pw is not None:
                idx, w = pw
                est[i, t] = float(np.dot(w, A[cand[idx], t]))
    return est

def run_length(x):
    """Length of the run of identical consecutive values ending at t (per station)."""
    N, T = x.shape
    out = np.zeros((N, T))
    for t in range(1, T):
        same = (x[:, t] == x[:, t - 1]) & ~np.isnan(x[:, t])
        out[:, t] = np.where(same, out[:, t - 1] + 1, 0)
    return out

def variance_ratio(x, sh, window=24):
    """Rolling standard deviation of hourly differences over `window` hours, relative to the training sigma_hf."""
    d = np.diff(x, axis=1, prepend=np.nan)
    N, T = x.shape
    out = np.full((N, T), np.nan)
    c1 = np.nancumsum(np.nan_to_num(d), axis=1); c2 = np.nancumsum(np.nan_to_num(d) ** 2, axis=1); cn = np.cumsum(~np.isnan(d), axis=1)
    for t in range(window, T):
        n = cn[:, t] - cn[:, t - window]
        s1 = c1[:, t] - c1[:, t - window]; s2 = c2[:, t] - c2[:, t - window]
        var = np.where(n > 6, s2 / np.maximum(n, 1) - (s1 / np.maximum(n, 1)) ** 2, np.nan)
        out[:, t] = np.sqrt(np.maximum(var, 0)) / (np.sqrt(2) * sh[:, None])[:, 0] if False else np.sqrt(np.maximum(var, 0)) / (np.sqrt(2) * sh)
    return out

# ---------------------------------------------------------------- scoring
def robust_standardize(r, val_mask):
    """|r| standardized per station with median and IQR of |r| on the validation stream (labels not used)."""
    s = np.abs(r)
    med = np.nanmedian(np.where(val_mask[None, :], s, np.nan), axis=1)
    q75 = np.nanpercentile(np.where(val_mask[None, :], s, np.nan), 75, axis=1)
    q25 = np.nanpercentile(np.where(val_mask[None, :], s, np.nan), 25, axis=1)
    iqr = np.maximum(q75 - q25, 1e-6)
    return (s - med[:, None]) / iqr[:, None]

def auprc(score, label):
    """Area under the precision-recall curve (step-wise, like average precision); NaN scores are ignored."""
    m = ~np.isnan(score)
    s, y = score[m], label[m].astype(bool)
    if y.sum() == 0:
        return np.nan
    order = np.argsort(-s, kind="stable")
    y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(~y)
    prec = tp / (tp + fp); rec = tp / y.sum()
    # average precision = sum over positives of precision at that rank
    return float(np.sum(prec[y]) / y.sum())

GRID_TWO = [(w, ma, mb) for w in (0.0, 1 / 3, 2 / 3, 1.0) for ma in (1, 3) for mb in (12, 24, 72)]

def two_timescale_scores(s, grid=GRID_TWO):
    """Return dict (w, ma, mb) -> combined score w*EMA_ma(s) + (1-w)*EMA_mb(s)."""
    cache = {}
    out = {}
    for w, ma, mb in grid:
        if ma not in cache: cache[ma] = ema(s, ma)
        if mb not in cache: cache[mb] = ema(s, mb)
        out[(w, ma, mb)] = w * cache[ma] + (1 - w) * cache[mb]
    return out

def select_on_validation(score_dict, label, val_mask):
    """Pick the grid key with the highest validation AUPRC; if no combination yields a finite AUPRC (no validation positives or
    all-NaN scores), fall back to the first key so that downstream code never receives None."""
    best, bestv = None, -1
    for k, S in score_dict.items():
        v = auprc(S[:, val_mask].ravel(), label[:, val_mask].ravel())
        if np.isfinite(v) and v > bestv:
            best, bestv = k, v
    if best is None:
        best, bestv = next(iter(score_dict)), np.nan
    return best, bestv

def calibrate_threshold(S, label, val_mask, fpr=0.01, label_free=False):
    """Per-variable threshold: (1-fpr) quantile of validation scores on clean hours (label-assisted) or on all hours (label-free)."""
    v = S[:, val_mask]
    if not label_free:
        v = np.where(label[:, val_mask], np.nan, v)
    return float(np.nanpercentile(v, 100 * (1 - fpr)))
