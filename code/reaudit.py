"""Regenerate audit.csv for every detection seed directory with the current audit criterion (injection is deterministic per seed)."""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D, inject as I
from run_detection import CORPORA, CLIP, VARS, RES

for corpus, regime in (("DE", "sigma"), ("NL", "sigma"), ("NA", "sigma"), ("DE_subtle", "subtle")):
    seeds = sorted(glob.glob(os.path.join(RES, corpus, "seed*", "segments.csv")))
    if not seeds: continue
    base = corpus.split("_")[0]; sources, splitkey = CORPORA[base]
    net = D.build_network(base, sources, min_cov=0.9, t_range=D.RANGE[splitkey]); D.plausibility_screen(net)
    masks = D.split_masks(net, splitkey); tr, va, te = masks["train"], masks["val"], masks["test"]
    vars_present = [v for v in VARS if (~np.isnan(net.X[v])).mean() > 0.3]
    sa, sh = {}, {}
    for v in vars_present:
        _, sa[v], sh[v] = D.fit_climatology(net.X[v], net.times, tr)
    bounds = [(int(np.where(m)[0][0]), int(np.where(m)[0][-1]) + 1) for m in (tr, va, te)]
    sub = D.Network(net.name, net.times, net.stations, net.meta, {v: net.X[v] for v in vars_present})
    for f in seeds:
        sd = os.path.dirname(f); seed = int(os.path.basename(sd)[4:])
        Xf, labels, segs = I.inject(sub, sa, sh, bounds, seed=seed, clip=CLIP, regime=regime)
        old = pd.read_csv(f, dtype={"station": str})
        assert len(old) == len(segs) and (old.start.to_numpy() == segs.start.to_numpy()).all(), f"segments differ for {sd}"
        aud = I.audit(sub, Xf, labels, segs); aud.to_csv(os.path.join(sd, "audit.csv"), index=False)
        print(corpus, "seed", seed, "form_ok", round(aud.form_ok.mean(), 4), "labels_ok", round(aud.labels_ok.mean(), 4), flush=True)
