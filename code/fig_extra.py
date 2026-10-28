"""Supplementary figures: station maps with silos, example injected segments, sigma tables in physical units."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D, inject as I
from run_detection import CORPORA, CLIP

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

def station_maps():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, corpus in zip(axes, ("DE", "NL", "NA")):
        p = os.path.join(ROOT, "results", corpus, "stations.csv")
        if not os.path.exists(p): continue
        st = pd.read_csv(p)
        groups = st["state"].astype(str)
        for g in sorted(groups.unique()):
            m = groups == g
            ax.scatter(st.lon[m], st.lat[m], s=18, label=g)
        ax.set_title(f"{corpus}: {len(st)} stations"); ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "station_maps.png"), dpi=200); plt.close(fig)

def example_segments(corpus="DE", seed=0):
    sources, splitkey = CORPORA[corpus]
    net = D.build_network(corpus, sources, min_cov=0.9, t_range=D.RANGE[splitkey]); D.plausibility_screen(net)
    masks = D.split_masks(net, splitkey); tr, va, te = masks["train"], masks["val"], masks["test"]
    sa, sh = {}, {}
    for v in ("TEMP", "RH", "DEWP"):
        _, sa[v], sh[v] = D.fit_climatology(net.X[v], net.times, tr)
    bounds = [(int(np.where(m)[0][0]), int(np.where(m)[0][-1]) + 1) for m in (tr, va, te)]
    sub = D.Network(net.name, net.times, net.stations, net.meta, {v: net.X[v] for v in ("TEMP", "RH", "DEWP")})
    Xf, labels, segs = I.inject(sub, sa, sh, bounds, seed=seed, clip=CLIP)
    fig, axes = plt.subplots(5, 1, figsize=(9, 10))
    for ax, fam in zip(axes, I.FAMILIES):
        s = segs[(segs.family == fam) & (segs["var"] == "TEMP") & (segs.start >= bounds[2][0])]
        if len(s) == 0: s = segs[(segs.family == fam) & (segs["var"] == "TEMP")]
        r = s.iloc[0]; pad = 48 if fam != "spike" else 12
        a, b = max(0, r.start - pad), min(net.T, r.end + pad)
        t = pd.DatetimeIndex(net.times[a:b])
        ax.plot(t, net.X["TEMP"][r.sidx, a:b], color="grey", lw=1, label="clean")
        ax.plot(t, Xf["TEMP"][r.sidx, a:b], color="red", lw=1, label="injected")
        ax.axvspan(pd.Timestamp(net.times[r.start]), pd.Timestamp(net.times[r.end - 1]), color="orange", alpha=0.2)
        ax.set_title(f"{fam}: station {r.station}, {r.dur_h} h, sigma_a={r.sigma_a:.2f} degC, magnitude={r.mag_phys:.2f} degC" + ("/day" if fam == "drift" else ""), fontsize=9)
        ax.set_ylabel("TEMP (degC)")
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, f"example_segments_{corpus}.png"), dpi=200); plt.close(fig)
    # realized magnitude distribution in physical units
    segs.to_csv(os.path.join(ROOT, "results", f"segments_{corpus}_seed{seed}.csv"), index=False)
    summ = segs.groupby(["var", "family"]).agg(n=("dur_h", "size"), dur_h_mean=("dur_h", "mean"), mag_phys_median=("mag_phys", "median"),
                                              mag_phys_min=("mag_phys", "min"), mag_phys_max=("mag_phys", "max")).reset_index()
    summ.to_csv(os.path.join(ROOT, "results", f"segment_summary_{corpus}_seed{seed}.csv"), index=False)
    print(summ.round(3).to_string())

if __name__ == "__main__":
    station_maps()
    for c in ("DE", "NA"):
        try: example_segments(c)
        except Exception as e: print("skip", c, e)
