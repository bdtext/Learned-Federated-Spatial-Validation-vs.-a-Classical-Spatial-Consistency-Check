"""Descriptive tables for the paper, generated from results/: corpora, clean-window scales, realized fault magnitudes
in physical units, injection audit, learned-validator tuning, and the verification-by-recomputation simulation.
Usage: python paper_tables.py   -> results/summary/table_*.csv (skips what does not exist yet)
"""
import os, sys, json, glob, ast
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results"); SUM = os.path.join(RES, "summary"); os.makedirs(SUM, exist_ok=True)
AGENCY = {"DE": "DWD (Germany)", "NL": "KNMI (Netherlands)", "NA": "NOAA GHCNh (USA, Canada, Israel)", "EU": "DWD + KNMI"}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0; p1, p2 = np.radians(lat1), np.radians(lat2); dp = p2 - p1; dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def corpora():
    rows = []
    for c in ("DE", "NL", "NA"):
        nf = os.path.join(RES, c, "network.json"); sf = os.path.join(RES, c, "stations.csv")
        if not (os.path.exists(nf) and os.path.exists(sf)): continue
        n = json.load(open(nf)); st = pd.read_csv(sf, dtype={"id": str})
        D = haversine(st.lat.to_numpy()[:, None], st.lon.to_numpy()[:, None], st.lat.to_numpy()[None, :], st.lon.to_numpy()[None, :])
        np.fill_diagonal(D, np.inf); nn = D.min(1)
        rows.append(dict(corpus=c, agency=AGENCY[c], stations=n["N"], median_nn_km=round(float(np.median(nn)), 1), max_nn_km=round(float(nn.max()), 1),
                         elev_min_m=float(st.elev.min()), elev_max_m=float(st.elev.max()), hours_train=n["split_hours"]["train"], hours_val=n["split_hours"]["val"],
                         hours_test=n["split_hours"]["test"], median_corr_TEMP=round(n["median_corr_TEMP"], 3),
                         with_wind_pressure=int(st.has_WSPM.sum()) if "has_WSPM" in st else n["N"]))
    pd.DataFrame(rows).to_csv(os.path.join(SUM, "table_corpora.csv"), index=False); return rows

def sigma():
    rows = []
    for c in ("DE", "NL", "NA"):
        f = os.path.join(RES, c, "sigma_table.csv")
        if not os.path.exists(f): continue
        t = pd.read_csv(f, index_col=0)
        r = dict(corpus=c)
        for v in ("TEMP", "RH", "DEWP", "WSPM", "PRES"):
            if f"sigma_a_{v}" in t:
                r[f"sigma_a_{v}"] = round(float(t[f"sigma_a_{v}"].median()), 2); r[f"sigma_hf_{v}"] = round(float(t[f"sigma_hf_{v}"].median()), 2)
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(SUM, "table_sigma.csv"), index=False)

def segments():
    rows = []
    for c in ("DE", "NL", "NA", "DE_subtle"):
        f = os.path.join(RES, c, "seed0", "segments.csv")
        if not os.path.exists(f): continue
        s = pd.read_csv(f, dtype={"station": str})
        for (v, fam), g in s.groupby(["var", "family"]):
            unit = {"TEMP": "degC", "DEWP": "degC", "RH": "%", "WSPM": "m/s", "PRES": "hPa"}[v]
            rows.append(dict(corpus=c, variable=v, family=fam, n_segments=len(g), mean_duration_h=round(float(g.dur_h.mean()), 1),
                             magnitude_median=round(float(g.mag_phys.abs().median()), 2), magnitude_min=round(float(g.mag_phys.abs().min()), 2),
                             magnitude_max=round(float(g.mag_phys.abs().max()), 2), unit=unit + ("/day" if fam == "drift" else "")))
    pd.DataFrame(rows).to_csv(os.path.join(SUM, "table_segments.csv"), index=False)

def audit():
    rows = []
    for c in ("DE", "NL", "NA", "DE_subtle"):
        for f in sorted(glob.glob(os.path.join(RES, c, "seed*", "audit.csv"))):
            a = pd.read_csv(f)
            rows.append(dict(corpus=c, seed=os.path.basename(os.path.dirname(f)), n_segments=len(a), form_ok=round(float(a.form_ok.mean()), 4),
                             labels_ok=round(float(a.labels_ok.mean()), 4), clipped_fraction=round(float(a.clipped.mean()), 4),
                             effective_fraction_mean=round(float(a.effective_fraction.mean()), 4)))
    pd.DataFrame(rows).to_csv(os.path.join(SUM, "table_audit.csv"), index=False)

def labelled():
    rows = []
    for c in ("DE", "NL", "NA", "DE_subtle"):
        f = os.path.join(RES, c, "seed0", "flags_test.npz")
        if not os.path.exists(f): continue
        z = np.load(f)
        for v in sorted({k.split("__")[1] for k in z.files if k.startswith("label__")}):
            lab = z[f"label__{v}"]; val = z[f"valid__{v}"]
            rows.append(dict(corpus=c, variable=v, test_readings=int(val.sum()), labelled_fraction=round(float(lab[val].mean()), 4)))
    pd.DataFrame(rows).to_csv(os.path.join(SUM, "table_labelled.csv"), index=False)

def tuning():
    rows = []
    for c in ("DE", "NL", "NA"):
        f = os.path.join(RES, c, "tuning.json")
        if not os.path.exists(f): continue
        t = json.load(open(f))
        for e in t["log"]:
            rows.append(dict(corpus=c, width=e["cfg"]["ch"], lr=e["cfg"]["lr"], epochs=e["epochs"], best_val_loss=round(e["best_val_loss"], 4),
                             val_auprc_TEMP=round(e["val_auprc_sp_TEMP"], 4), selected=(e["cfg"] == t["best"])))
    pd.DataFrame(rows).to_csv(os.path.join(SUM, "table_tuning.csv"), index=False)

def govsim():
    f = os.path.join(RES, "govsim", "govsim.json")
    if not os.path.exists(f): return
    g = json.load(open(f)); rows = []
    for k, r in g.items():
        rows.append(dict(parameters=r["n_params"], K=r["K"], f=r["f"], rule=r["rule"], verifiers_identical=r["verifiers_identical"],
                         verify_ms_median=round(r["verify_ms_median"], 1), verify_ms_max=round(r["verify_ms_max"], 1),
                         tampered_rejected=r["tampered_proposal_rejected"], attributed=r["coordinator_attributed"], corrupted_detected=r["corrupted_artifact_detected"]))
    pd.DataFrame(rows).to_csv(os.path.join(SUM, "table_govsim.csv"), index=False)

def model_size():
    rows = []
    for c in ("DE", "NL", "NA"):
        for f in sorted(glob.glob(os.path.join(RES, c, "seed*", "model_training.json"))):
            m = json.load(open(f)); rows.append(dict(corpus=c, seed=os.path.basename(os.path.dirname(f)), cfg=json.dumps(m["cfg"]), epochs=len(m["history"]),
                                                     best_val_loss=round(min(h["val_loss"] for h in m["history"]), 4), n_params=m.get("n_params")))
    pd.DataFrame(rows).to_csv(os.path.join(SUM, "table_models.csv"), index=False)

# ---------------------------------------------------------------- compact views for the paper (rounded, mean +/- SD strings)
ORDER = ["CLIM-Z", "EWMA-SPC", "Kalman", "SCT", "SCT-OI", "SCT+P", "SCT+K", "SCT-OI+P", "LSO-sp-only", "LSO-STGNN",
         "SCT+PV", "SCT+P+PV", "SCT-OI+P+PV", "LSO-STGNN+PV"]
FAMS = ["drift", "step", "spike", "stuck", "noise"]

def pm(m, sd, nd=3):
    try:
        m = float(m)
    except (TypeError, ValueError):
        return "n/a"
    if m != m: return "n/a"
    try:
        sd = float(sd)
    except (TypeError, ValueError):
        sd = float("nan")  # single seed: no spread
    return f"{m:.{nd}f}" if sd != sd else f"{m:.{nd}f} +/- {sd:.{nd}f}"

def f3(x):
    try:
        x = float(x); return "n/a" if x != x else f"{x:.3f}"
    except (TypeError, ValueError):
        return "n/a"

def views():
    def rd(name):
        f = os.path.join(SUM, name)
        return pd.read_csv(f, keep_default_na=False) if os.path.exists(f) and os.path.getsize(f) > 0 else None
    pw, pc, pv, ev = rd("pointwise.csv"), rd("paired_contrasts.csv"), rd("per_variable.csv"), rd("events.csv")
    if pw is not None:
        for c in sorted(pw.corpus.unique()):
            t = pw[pw.corpus == c].set_index("detector"); rows = []
            for det in ORDER:
                if det not in t.index: continue
                r = t.loc[det]
                if str(r.get("applicable", "True")) == "False":
                    rows.append(dict(detector=det, F1="n/a (no neighbours within the OI radius)", AUPRC="n/a", FPR="n/a", F1_label_free="n/a", F1_matched="n/a", F1_drift="n/a")); continue
                rows.append(dict(detector=det, F1=pm(r.f1_mean, r.f1_sd), AUPRC=pm(r.auprc_mean, r.auprc_sd), FPR=f3(r.fpr_mean),
                                 F1_label_free=f3(r.f1_labelfree_mean), F1_matched=f3(r.f1_matched_mean), F1_drift=f3(r.drift_f1_mean)))
            pd.DataFrame(rows).to_csv(os.path.join(SUM, f"view_detection_{c}.csv"), index=False)
    if pc is not None:
        rows = []
        for r in pc.itertuples():
            rows.append(dict(corpus=r.corpus, contrast=r.contrast, dF1=f3(r.f1_mean), CI95=f"[{f3(r.f1_ci_lo)}, {f3(r.f1_ci_hi)}]", p=f"{float(r.f1_p):.3g}" if str(r.f1_p) not in ("", "nan") else "n/a",
                             TOST_p_1pt=f"{float(r.f1_tost_p):.3g}" if str(r.f1_tost_p) not in ("", "nan") else "n/a", wins=f"{int(r.wins)}/{int(r.f1_n)}",
                             dAUPRC=f3(r.auprc_mean), AUPRC_CI95=f"[{f3(r.auprc_ci_lo)}, {f3(r.auprc_ci_hi)}]"))
        pd.DataFrame(rows).to_csv(os.path.join(SUM, "view_contrasts.csv"), index=False)
        main = ["LSO-STGNN - SCT+P", "LSO-STGNN - SCT-OI+P", "LSO-STGNN - LSO-sp-only", "SCT-OI+P - SCT+P", "SCT+P+PV - SCT+P", "LSO-STGNN+PV - SCT+P+PV"]
        mrows = [dict(corpus=r["corpus"], contrast=r["contrast"], dF1_points=f"{100 * float(r['dF1']):.1f}" if r["dF1"] != "n/a" else "n/a",
                      CI95_points="[" + ", ".join(f"{100 * float(x):.1f}" for x in r["CI95"].strip("[]").split(", ")) + "]" if r["CI95"] != "[n/a, n/a]" else "n/a",
                      p=r["p"], wins=r["wins"], dAUPRC=r["dAUPRC"]) for r in rows if r["contrast"] in main]
        pd.DataFrame(mrows).to_csv(os.path.join(SUM, "view_contrasts_main.csv"), index=False)
    bf = os.path.join(SUM, "bootstrap.json")
    if os.path.exists(bf):
        b = json.load(open(bf)); rows = []
        for c, d in b.items():
            for k, v in d.items():
                for bl, x in v.items():
                    rows.append(dict(corpus=c, contrast=k, block_h=bl, n_blocks=x["n_blocks"], dF1=f3(x["diff"]), CI95=f"[{f3(x['ci_lo'])}, {f3(x['ci_hi'])}]"))
        pd.DataFrame(rows).to_csv(os.path.join(SUM, "view_bootstrap.csv"), index=False)
    if pv is not None:
        for c in sorted(pv.corpus.unique()):
            u = pv[(pv.corpus == c) & pv.detector.isin(["SCT", "SCT-OI", "SCT+P", "SCT-OI+P", "LSO-sp-only", "LSO-STGNN"])]
            if not len(u): continue
            piv = u.pivot(index="variable", columns="detector", values="f1").reindex(["TEMP", "RH", "DEWP", "WSPM", "PRES"])
            piv = piv[[d for d in ["SCT", "SCT-OI", "SCT+P", "SCT-OI+P", "LSO-sp-only", "LSO-STGNN"] if d in piv.columns]]
            piv.map(f3).reset_index().to_csv(os.path.join(SUM, f"view_pervar_{c}.csv"), index=False)
            fam = u.groupby("detector")[[f"f1_{f}" for f in FAMS]].mean().reindex([d for d in ORDER if d in set(u.detector)])
            fam.columns = FAMS; fam.map(f3).reset_index().to_csv(os.path.join(SUM, f"view_family_{c}.csv"), index=False)
    if ev is not None:
        for c in sorted(ev.corpus.unique()):
            u = ev[(ev.corpus == c) & ev.detector.isin(["SCT+P", "SCT-OI+P", "LSO-STGNN", "SCT+P+PV", "LSO-STGNN+PV"])]
            if not len(u): continue
            rows = [dict(detector=r.detector, family=r.family, EDR_any=f3(r.edr_any), chance_any=f3(r.chance_any), EDR_persistent=f3(r.edr_persistent),
                         chance_persistent=f3(r.chance_persistent), within_72h=f3(r.within72), median_delay_h=f"{float(r.median_delay_h):.0f}" if str(r.median_delay_h) not in ("", "nan") else "n/a")
                    for r in u.itertuples()]
            pd.DataFrame(rows).to_csv(os.path.join(SUM, f"view_events_{c}.csv"), index=False)
    rt, rc, rb = rd("residency.csv"), rd("residency_contrasts.csv"), rd("robustness.csv")
    if rt is not None:
        order = ["SCT-all", "SCT-silo", "SCT-OI-all", "SCT-OI-silo", "centralized", "centralized-confined", "federated-fullgraph", "federated", "federated-matched", "local-only"]
        rows = []
        for c in sorted(rt.corpus.unique()):
            t = rt[rt.corpus == c].set_index("arm")
            for a in order:
                if a in t.index: rows.append(dict(corpus=c, arm=a, n_seeds=int(t.loc[a].n_seeds), F1=pm(t.loc[a].f1_mean, t.loc[a].f1_sd), F1_by_seed=t.loc[a].f1_by_seed))
        pd.DataFrame(rows).to_csv(os.path.join(SUM, "view_residency.csv"), index=False)
    if rc is not None:
        rows = [dict(corpus=r.corpus, contrast=r.contrast, dF1=f3(r.mean), CI95=f"[{f3(r.ci_lo)}, {f3(r.ci_hi)}]", p=f"{float(r.p):.3g}" if str(r.p) not in ("", "nan") else "n/a",
                     TOST_p_2pt=f"{float(r.tost_p_2pt):.3g}" if str(r.tost_p_2pt) not in ("", "nan") else "n/a") for r in rc.itertuples()]
        pd.DataFrame(rows).to_csv(os.path.join(SUM, "view_residency_contrasts.csv"), index=False)
    if rb is not None:
        rows = []
        for r in rb.itertuples():
            if r.rule == "SCT-silo" or r.key == "SCT-silo": continue
            rows.append(dict(rule=r.rule, adversaries=r.adversaries, attack=r.attack, F1=pm(r.f1_mean, r.f1_sd), retention=f3(r.retention_mean), floor_SCT_silo=f3(r.floor_f1), n_seeds=int(r.n_seeds)))
        pd.DataFrame(rows).to_csv(os.path.join(SUM, "view_robustness.csv"), index=False)

if __name__ == "__main__":
    corpora(); sigma(); segments(); audit(); labelled(); tuning(); govsim(); model_size(); views()
    print("tables:", sorted(f for f in os.listdir(SUM) if f.startswith("table_")))
