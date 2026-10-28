"""Post-hoc evaluation of a cached tuning configuration that was not part of the selection grid (DE width 64):
loads results/<corpus>/seed0/model_ch<ch>_lr<lr>.pt, recomputes the validation AUPRC of the temperature spatial innovation
exactly as in run_detection.py, and appends the entry to tuning.json with post_hoc=True (the selection itself is not changed).
Usage: python posthoc_cfg.py DE 64 0.002
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D, inject as I, detectors as Dt, model as Mo
from run_detection import CORPORA, CLIP, VARS, RES, jdump
torch.set_num_threads(int(os.environ.get("QC_THREADS", 4)))

corpus, ch, lr = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
sources, splitkey = CORPORA[corpus]
sd = os.path.join(RES, corpus, "seed0"); ckey = f"ch{ch}_lr{lr}"
net = D.build_network(corpus, sources, min_cov=0.9, t_range=D.RANGE[splitkey]); D.plausibility_screen(net)
masks = D.split_masks(net, splitkey); tr, va, te = masks["train"], masks["val"], masks["test"]
vars_present = [v for v in VARS if (~np.isnan(net.X[v])).mean() > 0.3]
sa, sh = {}, {}
for v in vars_present:
    _, sa[v], sh[v] = D.fit_climatology(net.X[v], net.times, tr)
bounds = [(int(np.where(m)[0][0]), int(np.where(m)[0][-1]) + 1) for m in (tr, va, te)]
sub = D.Network(net.name, net.times, net.stations, net.meta, {v: net.X[v] for v in vars_present})
Xf, labels, segs = I.inject(sub, sa, sh, bounds, seed=0, clip=CLIP)
A = {}
for v in vars_present:
    clim, sa_c, sh_c = D.fit_climatology(Xf[v], net.times, tr); A[v] = D.standardize(Xf[v], clim, sa_c)
a = np.stack([A[v] for v in vars_present]); K, N, T = a.shape
prior = np.stack([Dt.sct_weights(Dt.train_correlations(a[k], tr)) for k in range(K)])
tf = Mo.time_features(net.times); fetch = Mo.make_windows(a, None, tf)
torch.manual_seed(0); model = Mo.LSOSTGNN(N, K, prior, ch=ch)
model.load_state_dict(torch.load(os.path.join(sd, f"model_{ckey}.pt"))); hist = json.load(open(os.path.join(sd, f"history_{ckey}.json")))
rfc, rsp = Mo.residuals(model, fetch, a, np.arange(24, T - 1))
k0 = vars_present.index("TEMP"); s = Dt.robust_standardize(rsp[k0], va)
vv = Dt.auprc(Dt.ema(s, 3)[:, va].ravel(), labels["TEMP"][:, va].ravel())
entry = dict(cfg=dict(ch=ch, lr=lr), epochs=len(hist), best_val_loss=min(h["val_loss"] for h in hist), val_auprc_sp_TEMP=vv, history=hist, post_hoc=True)
tfile = os.path.join(RES, corpus, "tuning.json"); t = json.load(open(tfile))
t["log"] = [e for e in t["log"] if e["cfg"] != entry["cfg"]] + [entry]
jdump(t, tfile)
print(f"{corpus} {ckey}: epochs {len(hist)}, best val loss {entry['best_val_loss']:.4f}, val AUPRC(TEMP sp) {vv:.4f}; selected {t['best']} with",
      [round(e["val_auprc_sp_TEMP"], 4) for e in t["log"] if not e.get("post_hoc")])
