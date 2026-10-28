"""Smoke test: build the DE network from whatever has been parsed, inject faults (seed 0), audit, run classical detectors, score."""
import os, sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D, inject as I, detectors as Dt, metrics as M

t0 = time.time()
net = D.build_network("DE", ["dwd"], min_cov=0.95)
print("network", net.N, "stations", net.T, "hours", f"{time.time()-t0:.1f}s")
removed = D.plausibility_screen(net); print("screen removed", removed)
dup, maxeq = D.duplicate_audit(net); print("duplicates", dup, "max equality", np.nanmax(maxeq).round(3))
masks = D.split_masks(net, "EU")
tr, va, te = masks["train"], masks["val"], masks["test"]
print("split hours", tr.sum(), va.sum(), te.sum())
# clean-window scales for injection magnitudes
sa, sh = {}, {}
for v in ["TEMP", "RH", "DEWP"]:
    _, sa[v], sh[v] = D.fit_climatology(net.X[v], net.times, tr)
print("sigma_a TEMP median", np.nanmedian(sa["TEMP"]).round(2), "sigma_hf", np.nanmedian(sh["TEMP"]).round(2))
bounds = [(np.where(tr)[0][0], np.where(tr)[0][-1] + 1), (np.where(va)[0][0], np.where(va)[0][-1] + 1), (np.where(te)[0][0], np.where(te)[0][-1] + 1)]
sub = D.Network(net.name, net.times, net.stations, net.meta, {v: net.X[v] for v in ["TEMP", "RH", "DEWP"]})
Xf, labels, segs = I.inject(sub, sa, sh, bounds, seed=0, clip={"RH": (0, 100), "WSPM": (0, None)})
print("segments", len(segs), segs.family.value_counts().to_dict(), "labelled fraction", {v: round(float(labels[v].mean()), 4) for v in labels})
aud = I.audit(sub, Xf, labels, segs); print("audit form_ok", aud.form_ok.mean().round(4), "labels_ok", aud.labels_ok.mean().round(4), "effective fraction mean", aud.effective_fraction.mean().round(3))
print(aud.groupby("family").form_ok.mean().round(3).to_dict())
# detectors on TEMP
v = "TEMP"
t1 = time.time()
clim, sa_c, sh_c = D.fit_climatology(Xf[v], net.times, tr)  # fitted on the contaminated stream, as an agency would
a = D.standardize(Xf[v], clim, sa_c)
C = Dt.train_correlations(a, tr); print("corr median", np.nanmedian(C).round(3), f"{time.time()-t1:.1f}s")
W = Dt.sct_weights(C)
chan = {}
chan["CLIM-Z"] = Dt.clim_z(a)
chan["Persist"] = Dt.persistence_innovation(a)
chan["Kalman"] = Dt.kalman_innovation(a, q=0.1)
chan["SCT"] = a - Dt.sct_estimate(a, W, tr)
t2 = time.time(); chan["SCT-OI"] = a - Dt.oi_estimate(a, net.meta, tr); print(f"OI {time.time()-t2:.1f}s")
res = {}
for name, r in chan.items():
    s = Dt.robust_standardize(r, va)
    scores = Dt.two_timescale_scores(s)
    key, vbest = Dt.select_on_validation(scores, labels[v], va)
    S = scores[key]
    thr = Dt.calibrate_threshold(S, labels[v], va)
    pw = M.pointwise(S, labels[v], thr, te)
    res[name] = dict(grid=key, val_auprc=round(vbest, 4), f1=round(pw["f1"], 4), fpr=round(pw["fpr"], 4), auprc=round(pw["auprc"], 4))
    print(name, res[name])
# SCT+P control: spatial channel (two timescales) + persistence channel
s_sp = Dt.robust_standardize(chan["SCT"], va); s_fc = Dt.robust_standardize(chan["Persist"], va)
best = None
e = {h: Dt.ema(s_sp, h) for h in (3, 24, 72)}; ef = {h: Dt.ema(s_fc, h) for h in (1, 3)}
for w1 in (0, 1/3, 2/3, 1):
    for w2 in (0, 1/3, 2/3, 1):
        w3 = 1 - w1 - w2
        if w3 < -1e-9: continue
        for m1 in (1, 3):
            for m2 in (24, 72):
                S = w1 * ef[m1] + w2 * e[3] + w3 * e[m2]
                vv = Dt.auprc(S[:, va].ravel(), labels[v][:, va].ravel())
                if best is None or vv > best[0]: best = (vv, (w1, w2, w3, m1, m2), S)
S = best[2]; thr = Dt.calibrate_threshold(S, labels[v], va); pw = M.pointwise(S, labels[v], thr, te)
print("SCT+P", best[1], dict(val_auprc=round(best[0], 4), f1=round(pw["f1"], 4), fpr=round(pw["fpr"], 4), auprc=round(pw["auprc"], 4)))
ev = M.events(pw["flags"], segs[segs["var"] == v], np.where(te)[0][0], np.where(te)[0][-1] + 1)
print(ev.groupby("family")[["any", "persistent", "within72"]].mean().round(2))
print(f"total {time.time()-t0:.1f}s")
