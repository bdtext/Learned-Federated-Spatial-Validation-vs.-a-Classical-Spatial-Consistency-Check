"""Mechanism test: residual variance of the leave-one-out spatial estimate on CLEAN (uninjected) test-window data for the
correlation-weighted SCT, the OI estimate (validation-selected parameters of the detection benchmark) and the trained
leave-self-out validator (cached seed-0 weights), per station, stratified by nearest-neighbour distance and by the mean
correlation with the five most correlated neighbours.
Outputs results/summary/mechanism_<corpus>.csv (per station), results/summary/view_mechanism.csv (binned summary) and figures/mechanism.png.
Usage: python mechanism.py
"""
import os, sys, json
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D, detectors as Dt, model as Mo
from run_detection import CORPORA, VARS, RES
from paper_tables import haversine
torch.set_num_threads(int(os.environ.get("QC_THREADS", 4)))
SUM = os.path.join(RES, "summary"); FIG = os.path.join(RES, "..", "figures")

def main():
    rows_all, per_station = [], {}
    for corpus in ("DE", "NL", "NA"):
        sources, splitkey = CORPORA[corpus]
        net = D.build_network(corpus, sources, min_cov=0.9, t_range=D.RANGE[splitkey]); D.plausibility_screen(net)
        masks = D.split_masks(net, splitkey); tr, va, te = masks["train"], masks["val"], masks["test"]
        vars_present = [v for v in VARS if (~np.isnan(net.X[v])).mean() > 0.3]
        A = {}
        for v in vars_present:
            clim, sa, sh = D.fit_climatology(net.X[v], net.times, tr); A[v] = D.standardize(net.X[v], clim, sa)
        a = np.stack([A[v] for v in vars_present]); K, N, T = a.shape
        C = {v: Dt.train_correlations(A[v], tr) for v in vars_present}
        # learned validator: cached seed-0 weights of the selected configuration (prior buffer restored from the checkpoint)
        best = json.load(open(os.path.join(RES, corpus, "tuning.json")))["best"]
        ckey = f"ch{best['ch']}_lr{best['lr']}"
        prior = np.stack([Dt.sct_weights(C[v]) for v in vars_present])
        model = Mo.LSOSTGNN(N, K, prior, ch=best["ch"]); model.load_state_dict(torch.load(os.path.join(RES, corpus, "seed0", f"model_{ckey}.pt")))
        tf = Mo.time_features(net.times); fetch = Mo.make_windows(a, None, tf)
        te_idx = np.where(te)[0]; te_idx = te_idx[(te_idx >= 24) & (te_idx < T - 1)]
        _, rsp = Mo.residuals(model, fetch, a, te_idx)
        params = json.load(open(os.path.join(RES, corpus, "seed0", "metrics.json")))["_params"]
        lat, lon = net.meta.lat.to_numpy(), net.meta.lon.to_numpy()
        Dm = haversine(lat[:, None], lon[:, None], lat[None, :], lon[None, :]); np.fill_diagonal(Dm, np.inf); nn_km = Dm.min(1)
        for k, v in enumerate(vars_present):
            W = Dt.sct_weights(C[v])
            r_sct = A[v] - Dt.sct_estimate(A[v], W, tr)
            L, eps = params.get(v, {}).get("oi", [60.0, 0.1])
            est_oi = Dt.oi_estimate(A[v], net.meta, tr, L_km=L, eps=eps) if corpus != "NA" else np.full_like(A[v], np.nan)
            r_oi = A[v] - est_oi
            r_lso = rsp[k]
            Cv = np.where(np.isnan(C[v]), -1.0, C[v]); corr5 = np.array([np.mean(np.sort(Cv[i])[-5:]) for i in range(N)])
            for i in range(N):
                m = te.copy(); m &= ~np.isnan(A[v])[i]
                def var(r):
                    x = r[i][m]; x = x[np.isfinite(x)]
                    return float(np.var(x)) if len(x) > 100 else float("nan")
                rows_all.append(dict(corpus=corpus, variable=v, station=net.stations[i], nn_km=round(float(nn_km[i]), 1), corr5=round(float(corr5[i]), 3),
                                     var_anom=var(A[v]), var_sct=var(r_sct), var_oi=var(r_oi), var_lso=var(r_lso)))
        print(corpus, "done", flush=True)
    df = pd.DataFrame(rows_all); df["ratio_lso_sct"] = df.var_lso / df.var_sct; df["ratio_oi_sct"] = df.var_oi / df.var_sct
    for corpus in df.corpus.unique():
        df[df.corpus == corpus].to_csv(os.path.join(SUM, f"mechanism_{corpus}.csv"), index=False)
    # binned summary (temperature) by neighbour correlation and by spacing
    t = df[df.variable == "TEMP"].copy()
    t["corr_bin"] = pd.cut(t.corr5, [-1, 0.3, 0.6, 0.8, 0.9, 1.0], labels=["<0.3", "0.3-0.6", "0.6-0.8", "0.8-0.9", ">0.9"])
    t["nn_bin"] = pd.cut(t.nn_km, [0, 15, 30, 60, 200, 10000], labels=["<15", "15-30", "30-60", "60-200", ">200"])
    out = []
    for key in ("corr_bin", "nn_bin"):
        for (corpus, b), g in t.groupby(["corpus", key], observed=True):
            out.append(dict(corpus=corpus, stratum=f"{'corr5' if key == 'corr_bin' else 'nn_km'} {b}", n_stations=len(g),
                            var_anomaly=round(float(g.var_anom.median()), 3), var_SCT=round(float(g.var_sct.median()), 3),
                            var_OI=round(float(g.var_oi.median()), 3) if g.var_oi.notna().any() else "n/a", var_learned=round(float(g.var_lso.median()), 3),
                            ratio_learned_SCT=round(float(g.ratio_lso_sct.median()), 3), ratio_OI_SCT=round(float(g.ratio_oi_sct.median()), 3) if g.ratio_oi_sct.notna().any() else "n/a"))
    pd.DataFrame(out).to_csv(os.path.join(SUM, "view_mechanism.csv"), index=False)
    allv = df.groupby("corpus").agg(var_SCT=("var_sct", "median"), var_OI=("var_oi", "median"), var_learned=("var_lso", "median"),
                                    ratio_learned_SCT=("ratio_lso_sct", "median"), ratio_OI_SCT=("ratio_oi_sct", "median"), n=("station", "size")).round(3).reset_index()
    allv.to_csv(os.path.join(SUM, "view_mechanism_overall.csv"), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for corpus, g in t.groupby("corpus"):
        axes[0].scatter(g.corr5, g.ratio_lso_sct, s=12, label=corpus, alpha=0.7); axes[1].scatter(g.nn_km, g.ratio_lso_sct, s=12, label=corpus, alpha=0.7)
    for ax, xl in zip(axes, ("mean correlation with the 5 most correlated neighbours", "nearest-neighbour distance (km)")):
        ax.axhline(1.0, color="grey", ls="--", lw=0.8); ax.set_xlabel(xl); ax.set_ylabel("residual variance, learned / SCT (TEMP, clean test data)")
    axes[1].set_xscale("log"); axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, "mechanism.png"), dpi=200); plt.close(fig)
    print(pd.DataFrame(out).to_string()); print(allv.to_string())

if __name__ == "__main__":
    main()
