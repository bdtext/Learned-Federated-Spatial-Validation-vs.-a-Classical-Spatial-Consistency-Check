"""Network construction, plausibility screen, duplicate audit, splits, climatology and scales."""
import os, json
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
PROC = os.path.join(ROOT, "data", "processed")
VARS = ["TEMP", "RH", "DEWP", "WSPM", "PRES"]
# WMO-style plausibility limits (Guide to Instruments and Methods of Observation, No. 8; broad limits for mid-latitude surface stations)
LIMITS = {"TEMP": (-50.0, 50.0), "RH": (0.0, 100.0), "DEWP": (-60.0, 40.0), "WSPM": (0.0, 60.0), "PRES": (870.0, 1085.0)}
# maximum plausible hourly change (step test), in physical units
STEP = {"TEMP": 15.0, "RH": 80.0, "DEWP": 15.0, "WSPM": 40.0, "PRES": 15.0}

SPLITS = {
    # name: (train_start, train_end, val_start, val_end, test_start, test_end), inclusive hours in UTC
    "EU": ("2023-01-01 00:00", "2024-12-31 23:00", "2025-01-01 00:00", "2025-12-31 23:00", "2026-01-01 00:00", "2026-10-12 23:00"),
    # GHCNh near-real-time files for 2026 are only about half complete, so the NA corpus ends in 2025 (validation still a full year)
    "NA": ("2023-01-01 00:00", "2024-06-30 23:00", "2024-07-01 00:00", "2025-06-30 23:00", "2025-07-01 00:00", "2025-12-31 23:00"),
}
RANGE = {"EU": ("2023-01-01 00:00", "2026-10-12 23:00"), "NA": ("2023-01-01 00:00", "2025-12-31 23:00")}

class Network:
    """A set of stations with aligned hourly arrays X[var] of shape (N, T) and metadata."""
    def __init__(self, name, times, stations, meta, X, q=None):
        self.name, self.times, self.stations, self.meta, self.X, self.q = name, times, stations, meta, X, (q or {})
        self.N, self.T = len(stations), len(times)

    def subset(self, keep):
        keep = np.asarray(keep)
        return Network(self.name, self.times, [self.stations[i] for i in keep], self.meta.iloc[keep].reset_index(drop=True),
                       {v: a[keep] for v, a in self.X.items()}, {v: a[keep] for v, a in self.q.items()})

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    d = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(d))

def load_long(source):
    df = pd.read_parquet(os.path.join(PROC, f"{source}_hourly.parquet"))
    meta = pd.read_csv(os.path.join(PROC, f"{source}_stations.csv"), dtype={"id": str})
    return df, meta

def build_network(name, sources, min_cov=0.95, box=None, t_range=None):
    """Combine one or more processed sources into aligned arrays. Stations are kept if TEMP coverage >= min_cov within t_range."""
    parts, metas = [], []
    for s in sources:
        df, meta = load_long(s)
        meta["source"] = s
        parts.append(df); metas.append(meta)
    df = pd.concat(parts, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)
    if box is not None:
        meta = meta[(meta.lat.between(box[0], box[1])) & (meta.lon.between(box[2], box[3]))]
    if t_range is not None:
        a, b = pd.Timestamp(t_range[0], tz="UTC"), pd.Timestamp(t_range[1], tz="UTC")
        df = df[(df["time"] >= a) & (df["time"] <= b)]
    times = np.sort(df["time"].unique())
    tpos = {t: i for i, t in enumerate(times)}
    keep, X, Q = [], {v: [] for v in VARS}, {}
    for _, m in meta.iterrows():
        d = df[df.station == m["id"]]
        if len(d) == 0:
            continue
        d = d.set_index("time").reindex(times)
        if d["TEMP"].notna().mean() < min_cov:
            continue
        keep.append(m)
        for v in VARS:
            X[v].append(d[v].to_numpy(dtype=float) if v in d else np.full(len(times), np.nan))
    meta = pd.DataFrame(keep).reset_index(drop=True)
    X = {v: np.vstack(X[v]) for v in VARS}
    return Network(name, times, list(meta["id"]), meta, X)

def plausibility_screen(net):
    """Set physically implausible values and impossible hourly jumps to NaN; return counts per variable."""
    removed = {}
    for v in VARS:
        a = net.X[v]
        lo, hi = LIMITS[v]
        bad = (a < lo) | (a > hi)
        if v == "DEWP":
            bad |= a > net.X["TEMP"] + 0.5  # dew point above temperature (allow 0.5 degC rounding)
        # step test: both neighbours differ by more than STEP -> isolated spike beyond physical limits
        d1 = np.abs(np.diff(a, axis=1, prepend=np.nan)); d2 = np.abs(np.diff(a[:, ::-1], axis=1, prepend=np.nan))[:, ::-1]
        bad |= (d1 > STEP[v]) & (d2 > STEP[v])
        removed[v] = int(np.nansum(bad))
        a[bad] = np.nan
    return removed

def duplicate_audit(net, var="TEMP", thresh=0.99):
    """Pairs with > thresh fraction of exactly equal jointly observed hours are duplicates; also report the max equality fraction."""
    a = net.X[var]
    dup, maxeq = [], np.zeros((net.N, net.N))
    for i in range(net.N):
        for j in range(i + 1, net.N):
            m = ~np.isnan(a[i]) & ~np.isnan(a[j])
            if m.sum() < 1000:
                continue
            eq = np.mean(a[i][m] == a[j][m])
            maxeq[i, j] = maxeq[j, i] = eq
            if eq > thresh:
                dup.append((net.stations[i], net.stations[j], float(eq)))
    return dup, maxeq

def split_masks(net, key):
    t = pd.DatetimeIndex(net.times)
    a, b, c, d, e, f = [pd.Timestamp(x, tz="UTC") for x in SPLITS[key]]
    return {"train": (t >= a) & (t <= b), "val": (t >= c) & (t <= d), "test": (t >= e) & (t <= f)}

def harmonic_design(times, n_annual=3, n_diurnal=2):
    t = pd.DatetimeIndex(times)
    doy = (t.dayofyear - 1 + t.hour / 24.0) / 365.25 * 2 * np.pi
    hod = t.hour / 24.0 * 2 * np.pi
    cols = [np.ones(len(t))]
    for k in range(1, n_annual + 1):
        cols += [np.sin(k * doy), np.cos(k * doy)]
    for k in range(1, n_diurnal + 1):
        cols += [np.sin(k * hod), np.cos(k * hod)]
    # interactions annual x diurnal (first harmonics)
    for ka in range(1, n_annual + 1):
        for kd in range(1, n_diurnal + 1):
            cols += [np.sin(ka * doy) * np.sin(kd * hod), np.sin(ka * doy) * np.cos(kd * hod),
                     np.cos(ka * doy) * np.sin(kd * hod), np.cos(ka * doy) * np.cos(kd * hod)]
    return np.column_stack(cols)

def fit_climatology(X, times, train_mask):
    """Least-squares harmonic climatology per station on the training window of the (possibly contaminated) stream.
    Returns clim (N,T), sigma_a (N,), sigma_hf (N,)."""
    D = harmonic_design(times)
    N, T = X.shape
    clim = np.full((N, T), np.nan); sa = np.full(N, np.nan); sh = np.full(N, np.nan)
    for i in range(N):
        m = train_mask & ~np.isnan(X[i])
        if m.sum() < 1000:
            continue
        beta, *_ = np.linalg.lstsq(D[m], X[i][m], rcond=None)
        clim[i] = D @ beta
        r = X[i][m] - D[m] @ beta
        sa[i] = np.std(r)
        dx = np.diff(X[i][train_mask]); dx = dx[~np.isnan(dx)]
        sh[i] = np.std(dx) / np.sqrt(2)
    return clim, sa, sh

def standardize(X, clim, sa):
    return (X - clim) / sa[:, None]
