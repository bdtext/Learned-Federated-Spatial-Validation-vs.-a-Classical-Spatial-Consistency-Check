"""Calibration under distribution shift: realized false-positive rate per calendar month of the test window, from the saved
test flags (results/<corpus>/seed*/flags_test.npz), pooled over variables and seeds, for the main detectors.
Outputs results/summary/view_monthly_far_<corpus>.csv, results/summary/view_monthly_far_summary.csv and figures/monthly_far.png.
Usage: python monthly_far.py
"""
import os, sys, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D

ROOT = os.path.join(os.path.dirname(__file__), ".."); RES = os.path.join(ROOT, "results"); SUM = os.path.join(RES, "summary"); FIG = os.path.join(ROOT, "figures")
DETS = ["SCT+P", "SCT-OI+P", "LSO-STGNN", "SCT+P+PV", "LSO-STGNN+PV"]
SPLITKEY = {"DE": "EU", "NL": "EU", "NA": "NA", "DE_subtle": "EU"}

def main():
    summary, panels = [], {}
    for corpus in ("DE", "NL", "NA", "DE_subtle"):
        files = sorted(glob.glob(os.path.join(RES, corpus, "seed*", "flags_test.npz")))
        if not files: continue
        t0 = pd.Timestamp(D.SPLITS[SPLITKEY[corpus]][4])
        acc = {}  # (det, month) -> [fp, neg]
        for f in files:
            z = np.load(f)
            vars_ = sorted({k.split("__")[1] for k in z.files if k.startswith("label__")})
            T = z[f"label__{vars_[0]}"].shape[1]
            months = pd.date_range(t0, periods=T, freq="h").strftime("%Y-%m").to_numpy()
            for det in DETS:
                if not any(f"{det}__{v}" in z.files and z[f"{det}__{v}"].any() for v in vars_): continue  # detector not applicable here
                for v in vars_:
                    if f"{det}__{v}" not in z.files: continue
                    fl, y, val = z[f"{det}__{v}"], z[f"label__{v}"], z[f"valid__{v}"]
                    clean = ~y & val; fp = fl & clean
                    for m in np.unique(months):
                        sel = months == m
                        a = acc.setdefault((det, m), [0, 0]); a[0] += int(fp[:, sel].sum()); a[1] += int(clean[:, sel].sum())
        months = sorted({m for _, m in acc}); rows = []
        for m in months:
            r = dict(month=m)
            for det in DETS:
                fp, neg = acc.get((det, m), (0, 0)); r[det] = round(fp / neg, 4) if neg else float("nan")
            rows.append(r)
        df = pd.DataFrame(rows); df.to_csv(os.path.join(SUM, f"view_monthly_far_{corpus}.csv"), index=False); panels[corpus] = df
        for det in DETS:
            if det in df and df[det].notna().any():
                summary.append(dict(corpus=corpus, detector=det, months=len(df), fpr_min=round(float(df[det].min()), 4), fpr_max=round(float(df[det].max()), 4),
                                    month_max=df.loc[df[det].idxmax(), "month"], fpr_mean=round(float(df[det].mean()), 4)))
    pd.DataFrame(summary).to_csv(os.path.join(SUM, "view_monthly_far_summary.csv"), index=False)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.3), squeeze=False)
    for ax, (corpus, df) in zip(axes[0], panels.items()):
        for det in DETS:
            if det in df: ax.plot(df.month, df[det], marker=".", lw=1, label=det)
        ax.axhline(0.01, color="grey", ls="--", lw=0.8); ax.set_title(corpus); ax.set_ylabel("realized FPR on clean readings")
        ax.tick_params(axis="x", rotation=60, labelsize=6)
    axes[0][0].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(FIG, "monthly_far.png"), dpi=200); plt.close(fig)
    print(pd.DataFrame(summary).to_string())

if __name__ == "__main__":
    main()
