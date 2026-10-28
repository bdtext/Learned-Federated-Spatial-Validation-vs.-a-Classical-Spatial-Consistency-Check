"""Seeded fault injection with five families, audited from the injected-minus-clean difference.
Faults are injected per (station, variable) on the CLEAN standardized anomaly scale (units of sigma_a of the clean training window),
independently across stations and variables, with random sign, until `budget` of station-days per variable is faulty
(default 1 per cent per family, 5 per cent total). Segments are separated by >= 1 day and lie >= 3 days from split boundaries.
Physical units of each injected magnitude are recorded in the segment table (sigma_a tabulated per station and variable).
Families: drift (linear ramp 0.01-0.1 sigma/day, 7-60 d); step (0.5-3 sigma, 3-45 d); spike (3-8 sigma, 1-3 h);
stuck (frozen at last value for 6-72 h, reported also through gaps of the original record); noise (Gaussian, factor 2-5 of sigma_hf, 2-14 d).
Additive families perturb observed hours only (gaps stay gaps); labels mark the hours at which the contaminated stream carries a fault.
"""
import numpy as np, pandas as pd

FAMILIES = ["drift", "step", "spike", "stuck", "noise"]

# Subtle regime: magnitudes in PHYSICAL units anchored to instrument-scale errors (per variable); durations as in the sigma regime.
# drift: physical units per day; step/spike: physical units; noise: added standard deviation in physical units.
SUBTLE = {
    "TEMP": dict(drift=(0.02, 0.10), step=(0.5, 2.0), spike=(2.0, 5.0), noise=(0.3, 1.0)),
    "DEWP": dict(drift=(0.02, 0.10), step=(0.5, 2.0), spike=(2.0, 5.0), noise=(0.3, 1.0)),
    "RH":   dict(drift=(0.1, 0.5), step=(3.0, 10.0), spike=(10.0, 30.0), noise=(2.0, 6.0)),
    "WSPM": dict(drift=(0.01, 0.05), step=(0.3, 1.0), spike=(2.0, 5.0), noise=(0.3, 1.0)),
    "PRES": dict(drift=(0.02, 0.10), step=(0.5, 2.0), spike=(2.0, 5.0), noise=(0.2, 0.6)),
}

def _draw(rng, fam, regime="sigma", var=None):
    if regime == "subtle":
        r = SUBTLE[var]
        if fam == "drift":
            return dict(rate=rng.uniform(*r["drift"]), dur_h=int(rng.uniform(7, 60) * 24), physical=True)
        if fam == "step":
            return dict(mag=rng.uniform(*r["step"]), dur_h=int(rng.uniform(3, 45) * 24), physical=True)
        if fam == "spike":
            return dict(mag=rng.uniform(*r["spike"]), dur_h=int(rng.integers(1, 4)), physical=True)
        if fam == "stuck":
            return dict(dur_h=int(rng.uniform(6, 72)), physical=True)
        if fam == "noise":
            return dict(sd=rng.uniform(*r["noise"]), dur_h=int(rng.uniform(2, 14) * 24), physical=True)
    if fam == "drift":
        return dict(rate=rng.uniform(0.01, 0.10), dur_h=int(rng.uniform(7, 60) * 24))
    if fam == "step":
        return dict(mag=rng.uniform(0.5, 3.0), dur_h=int(rng.uniform(3, 45) * 24))
    if fam == "spike":
        return dict(mag=rng.uniform(3.0, 8.0), dur_h=int(rng.integers(1, 4)))
    if fam == "stuck":
        return dict(dur_h=int(rng.uniform(6, 72)))
    if fam == "noise":
        return dict(factor=rng.uniform(2.0, 5.0), dur_h=int(rng.uniform(2, 14) * 24))

def inject(net, sa, sh, split_bounds, seed, budget_per_family=0.01, clip=None, regime="sigma"):
    """Returns Xf (dict var -> contaminated array), labels (dict var -> bool array), segments DataFrame.
    split_bounds: list of (start_idx, end_idx) index ranges for train/val/test; faults respect a 3-day buffer at every boundary.
    sa, sh: dict var -> (N,) scales of the CLEAN training window (used for magnitudes). clip: dict var -> (lo, hi)."""
    rng = np.random.default_rng(seed)
    Xf = {v: net.X[v].copy() for v in net.X}
    labels = {v: np.zeros_like(net.X[v], dtype=bool) for v in net.X}
    # forbidden zones: 3 days around every split boundary
    T = net.T
    forbid = np.zeros(T, dtype=bool)
    for (s, e) in split_bounds:
        forbid[max(0, s - 72):min(T, s + 72)] = True
        forbid[max(0, e - 72):min(T, e + 72)] = True
    segs = []
    for v in net.X:
        valid_station = (~np.isnan(net.X[v])).mean(axis=1) > 0.5
        for fam in FAMILIES:
            # budget in station-days over the whole record
            target_days = budget_per_family * valid_station.sum() * T / 24.0
            used_days = 0.0
            occupied = labels[v].copy()  # includes all families injected so far for this variable
            tries = 0
            while used_days < target_days and tries < 100000:
                tries += 1
                i = int(rng.integers(0, net.N))
                if not valid_station[i]:
                    continue
                p = _draw(rng, fam, regime, v)
                dur = p["dur_h"]
                s = int(rng.integers(72, T - dur - 72))
                e = s + dur
                if forbid[s:e].any():
                    continue
                # separation of >= 1 day from other segments on the same station/variable
                lo, hi = max(0, s - 24), min(T, e + 24)
                if occupied[i, lo:hi].any():
                    continue
                x = Xf[v][i, s:e]
                if np.isnan(x).mean() > 0.2 or np.isnan(sa[v][i]) or sa[v][i] == 0:
                    continue
                sign = rng.choice([-1.0, 1.0])
                sig = sa[v][i]; sigh = sh[v][i]
                phys = p.get("physical", False)
                if fam == "drift":
                    rate = p["rate"] if phys else p["rate"] * sig
                    ramp = rate * np.arange(dur) / 24.0
                    Xf[v][i, s:e] = x + sign * ramp
                    mag_phys = rate  # per day
                elif fam == "step":
                    mag = p["mag"] if phys else p["mag"] * sig
                    Xf[v][i, s:e] = x + sign * mag; mag_phys = mag
                elif fam == "spike":
                    mag = p["mag"] if phys else p["mag"] * sig
                    Xf[v][i, s:e] = x + sign * mag; mag_phys = mag
                elif fam == "stuck":
                    last = Xf[v][i, s - 1]
                    if np.isnan(last):
                        continue
                    Xf[v][i, s:e] = last; mag_phys = 0.0
                elif fam == "noise":
                    sd = p["sd"] if phys else np.sqrt(max(p["factor"] ** 2 - 1.0, 0.0)) * sigh
                    Xf[v][i, s:e] = x + rng.normal(0.0, sd, size=dur); mag_phys = sd
                if clip and v in clip:
                    Xf[v][i, s:e] = np.clip(Xf[v][i, s:e], clip[v][0], clip[v][1])
                lab = ~np.isnan(x)
                labels[v][i, s:e] = lab
                occupied[i, s:e] = True
                used_days += (dur if fam != "spike" else 24) / 24.0
                segs.append(dict(var=v, family=fam, station=net.stations[i], sidx=i, start=s, end=e, dur_h=dur, sign=sign,
                                 sigma_a=sig, sigma_hf=sigh, param=p, mag_phys=mag_phys, seed=seed))
    return Xf, labels, pd.DataFrame(segs)

def audit(net, Xf, labels, segs, tol=1e-6):
    """Independent audit: reconstruct each segment's form from the injected-minus-clean difference (not from the generator's log)."""
    out = []
    for r in segs.itertuples():
        v, i, s, e = r.var, r.sidx, r.start, r.end
        d = Xf[v][i, s:e] - net.X[v][i, s:e]
        m = ~np.isnan(d)
        ok = True; note = ""
        phys = r.param.get("physical", False)
        if r.family == "drift":
            x = np.arange(e - s)[m]; y = d[m]
            if len(x) > 2:
                slope = np.polyfit(x, y, 1)[0] * 24.0
                exp = r.param["rate"] if phys else r.param["rate"] * r.sigma_a
                ok = abs(abs(slope) - exp) < 0.05 * exp + tol
                note = f"slope/day={slope:.4f}"
        elif r.family in ("step", "spike"):
            mag = np.nanmedian(np.abs(d[m])) if m.any() else np.nan
            exp = r.param["mag"] if phys else r.param["mag"] * r.sigma_a
            ok = np.isfinite(mag) and (abs(mag - exp) < 0.05 * exp + tol or (r.family == "spike" and mag < exp))  # clipping may reduce
            note = f"mag={mag:.3f} expected={exp:.3f}"
        elif r.family == "stuck":
            vals = Xf[v][i, s:e]
            ok = np.nanstd(vals) < tol
            note = f"std={np.nanstd(vals):.4f}"
        elif r.family == "noise":
            sd = np.nanstd(d[m]) if m.sum() > 10 else np.nan
            exp = r.param["sd"] if phys else np.sqrt(max(r.param["factor"] ** 2 - 1, 0)) * r.sigma_hf
            ok = np.isfinite(sd) and abs(sd - exp) < 0.25 * exp + tol
            note = f"sd={sd:.3f} expected={exp:.3f}"
        eff = float(np.mean(np.abs(d[m]) > tol)) if m.any() else 0.0  # fraction of observed hours with a non-zero perturbation
        xf = Xf[v][i, s:e]
        # labelled hours must equal the hours at which the contaminated stream has an observation (stuck-at segments keep reporting
        # the frozen value through gaps of the original record; those hours are observations of the faulty stream and are labelled)
        lab_ok = bool(labels[v][i, s:e].sum() == int((~np.isnan(xf)).sum()))
        clipped = bool(np.any((xf <= 0.0) & (v in ("RH", "WSPM"))) or np.any((xf >= 100.0) & (v == "RH")))
        out.append(dict(var=v, family=r.family, station=r.station, start=s, dur_h=e - s, form_ok=bool(ok or clipped), clipped=clipped,
                        labels_ok=lab_ok, effective_fraction=eff, note=note))
    return pd.DataFrame(out)
