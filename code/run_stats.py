"""Aggregate results over seeds: tables (mean +/- SD, seed-level paired differences with t-intervals, TOST, block bootstrap),
per-variable and per-family slices, event-level rates with chance levels, residency and robustness summaries, and figures.
Usage: python run_stats.py
Outputs: results/summary/*.csv|json and figures/*.png
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from qc import metrics as M

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results"); SUM = os.path.join(RES, "summary"); FIG = os.path.join(ROOT, "figures")
os.makedirs(SUM, exist_ok=True); os.makedirs(FIG, exist_ok=True)
DETS = ["CLIM-Z", "EWMA-SPC", "Kalman", "SCT", "SCT-OI", "SCT+P", "SCT+K", "SCT-OI+P", "LSO-sp-only", "LSO-STGNN"]
CONTROL, LEARNED = "SCT+P", "LSO-STGNN"

def load_corpus(corpus):
    d = os.path.join(RES, corpus)
    seeds = sorted(int(p.split("seed")[-1]) for p in glob.glob(os.path.join(d, "seed*")) if os.path.exists(os.path.join(p, "metrics.json")))
    mets = {s: json.load(open(os.path.join(d, f"seed{s}", "metrics.json"))) for s in seeds}
    return d, seeds, mets

def applicable(mets, seeds, det):
    """A detector is applicable on a corpus when it produced finite test scores for at least one variable in every seed."""
    return all(any(mets[s][det][v].get("applicable", True) for v in mets[s][det]) for s in seeds if det in mets[s])

def table_pointwise(corpus, d, seeds, mets):
    rows = []
    dets = [x for x in DETS if all(x in mets[s]["_pooled"] for s in seeds)]
    for det in dets + [x + "+PV" for x in dets if all(x + "+PV" in mets[s]["_pooled"] for s in seeds)]:
        if not applicable(mets, seeds, det):
            rows.append(dict(corpus=corpus, detector=det, n_seeds=len(seeds), applicable=False)); continue
        f1 = [mets[s]["_pooled"][det]["f1"] for s in seeds]; fpr = [mets[s]["_pooled"][det]["fpr"] for s in seeds]
        vars_ = [v for v in mets[seeds[0]] if not v.startswith("_")]
        au = [np.nanmean([mets[s][det][v]["auprc"] for v in mets[s][det]]) for s in seeds]
        f1m = [np.nanmean([mets[s][det][v]["f1_matched"] for v in mets[s][det]]) for s in seeds]
        f1lf = [np.nanmean([mets[s][det][v]["f1_labelfree"] for v in mets[s][det]]) for s in seeds]
        dr = [np.nanmean([mets[s][det][v]["per_family"]["drift"] for v in mets[s][det]]) for s in seeds]
        rows.append(dict(corpus=corpus, detector=det, n_seeds=len(seeds), applicable=True, f1_mean=np.mean(f1), f1_sd=np.std(f1, ddof=1) if len(f1) > 1 else np.nan,
                         auprc_mean=np.mean(au), auprc_sd=np.std(au, ddof=1) if len(au) > 1 else np.nan, fpr_mean=np.mean(fpr),
                         f1_matched_mean=np.mean(f1m), f1_labelfree_mean=np.mean(f1lf), drift_f1_mean=np.mean(dr), f1_by_seed=f1))
    return pd.DataFrame(rows)

def paired(corpus, d, seeds, mets, a, b, margin=0.01):
    """Seed-level paired difference a - b in pooled F1 and mean AUPRC, t-interval, p, TOST."""
    if not all(a in mets[s]["_pooled"] and b in mets[s]["_pooled"] for s in seeds) or not (applicable(mets, seeds, a) and applicable(mets, seeds, b)):
        return None
    df1 = [mets[s]["_pooled"][a]["f1"] - mets[s]["_pooled"][b]["f1"] for s in seeds]
    dau = [np.nanmean([mets[s][a][v]["auprc"] for v in mets[s][a]]) - np.nanmean([mets[s][b][v]["auprc"] for v in mets[s][b]]) for s in seeds]
    r = dict(corpus=corpus, contrast=f"{a} - {b}", **{f"f1_{k}": v for k, v in M.seed_paired(df1).items()}, f1_tost_p=M.tost(df1, margin),
             **{f"auprc_{k}": v for k, v in M.seed_paired(dau).items()}, wins=int(np.sum(np.array(df1) > 0)))
    return r

def bootstrap(corpus, d, seeds, a, b):
    FA, FB, L, V = [], [], [], []
    for s in seeds:
        z = np.load(os.path.join(d, f"seed{s}", "flags_test.npz"))
        vars_ = sorted({k.split("__")[1] for k in z.files if k.startswith("label__")})
        fa = np.concatenate([z[f"{a}__{v}"] for v in vars_ if f"{a}__{v}" in z.files], 0)
        fb = np.concatenate([z[f"{b}__{v}"] for v in vars_ if f"{b}__{v}" in z.files], 0)
        lab = np.concatenate([z[f"label__{v}"] for v in vars_ if f"{a}__{v}" in z.files], 0)
        val = np.concatenate([z[f"valid__{v}"] for v in vars_ if f"{a}__{v}" in z.files], 0)
        FA.append(fa); FB.append(fb); L.append(lab); V.append(val)
    FA, FB, L, V = (np.stack(x) for x in (FA, FB, L, V))
    out = {}
    for bl in (360, 720, 1440):
        out[bl] = M.block_bootstrap_f1_diff(FA, FB, L, V, block_h=bl, n_boot=1000)
    return out

def per_variable(corpus, d, seeds, mets):
    rows = []
    for det in DETS:
        if not all(det in mets[s] for s in seeds) or not applicable(mets, seeds, det): continue
        for v in mets[seeds[0]][det]:
            rows.append(dict(corpus=corpus, detector=det, variable=v,
                             f1=np.mean([mets[s][det][v]["f1"] for s in seeds]), auprc=np.nanmean([mets[s][det][v]["auprc"] for s in seeds]),
                             fpr=np.mean([mets[s][det][v]["fpr"] for s in seeds]),
                             **{f"f1_{fam}": np.mean([mets[s][det][v]["per_family"][fam] for s in seeds]) for fam in ("drift", "step", "spike", "stuck", "noise")}))
    return pd.DataFrame(rows)

def events_table(corpus, d, seeds, mets):
    rows = []
    for det in DETS + ["SCT+P+PV", "LSO-STGNN+PV"]:
        if not all(det in mets[s] for s in seeds) or not applicable(mets, seeds, det): continue
        for fam in ("drift", "step", "spike", "stuck", "noise"):
            anyr, pers, w72, w168, ch_any, ch_pers, delay, n = [], [], [], [], [], [], [], 0
            for s in seeds:
                for v in mets[s][det]:
                    e = mets[s][det][v]["events"].get(fam); c = mets[s][det][v]["chance"].get(fam)
                    if e:
                        anyr.append(e["any"]); pers.append(e["persistent"]); w72.append(e["within72"]); w168.append(e["within168"])
                        delay.append(mets[s][det][v]["median_delay"].get(fam, np.nan))
                    if c: ch_any.append(c["any"]); ch_pers.append(c["persistent"])
            rows.append(dict(corpus=corpus, detector=det, family=fam, edr_any=np.mean(anyr) if anyr else np.nan, edr_persistent=np.mean(pers) if pers else np.nan,
                             within72=np.mean(w72) if w72 else np.nan, within168=np.mean(w168) if w168 else np.nan, chance_any=np.mean(ch_any) if ch_any else np.nan,
                             chance_persistent=np.mean(ch_pers) if ch_pers else np.nan, median_delay_h=np.nanmedian(delay) if delay else np.nan))
    return pd.DataFrame(rows)

def residency_table(corpus):
    d = os.path.join(RES, f"residency_{corpus}")
    files = sorted(glob.glob(os.path.join(d, "seed*", "residency.json")))
    if not files: return None, None
    R = [json.load(open(f)) for f in files]
    arms = [k for k in R[0] if k != "history" and not k.startswith("_")]
    rows = [dict(corpus=corpus, arm=a, n_seeds=len(R), f1_mean=np.mean([r[a]["f1"] for r in R]), f1_sd=np.std([r[a]["f1"] for r in R], ddof=1) if len(R) > 1 else np.nan,
                 f1_by_seed=[r[a]["f1"] for r in R]) for a in arms]
    contrasts = []
    for a, b in (("centralized", "federated"), ("centralized", "centralized-confined"), ("centralized-confined", "federated"), ("federated", "local-only"), ("centralized-confined", "local-only"), ("local-only", "federated"), ("centralized", "local-only"),
                 ("centralized", "federated-matched"), ("local-only", "federated-matched"), ("federated-matched", "federated"), ("centralized-confined", "federated-matched"),
                 ("SCT-OI-all", "SCT-OI-silo"), ("federated", "SCT-OI-silo"), ("federated-matched", "SCT-OI-silo"), ("centralized", "SCT-OI-all"), ("local-only", "SCT-OI-silo"),
                 ("federated-fullgraph", "federated"), ("SCT-all", "SCT-silo"), ("federated", "SCT-silo"), ("local-only", "SCT-silo")):
        if a in arms and b in arms:
            dd = [r[a]["f1"] - r[b]["f1"] for r in R]
            contrasts.append(dict(corpus=corpus, contrast=f"{a} - {b}", **M.seed_paired(dd), tost_p_2pt=M.tost(dd, 0.02)))
    return pd.DataFrame(rows), pd.DataFrame(contrasts)

def robustness_table():
    d = os.path.join(RES, "robustness_DE")
    R = []
    for sdir in sorted(glob.glob(os.path.join(d, "seed*"))):
        merged = {}
        for f in sorted(glob.glob(os.path.join(sdir, "robustness*.json"))):
            merged.update(json.load(open(f)))
        if any(not k.startswith("_") and k != "SCT-silo" for k in merged): R.append(merged)
    if not R: return None
    keys = sorted(k for k in set.intersection(*[set(r.keys()) for r in R]) if not k.startswith("_"))
    rows = []
    for k in keys:
        f1 = [r[k]["f1"] for r in R]
        rule, adv, att = (k.split("|") + ["", ""])[:3] if "|" in k else (k, "", "")
        clean_key = f"{rule}|adv1|clean"
        ret = [r[k]["f1"] / max(r[clean_key]["f1"], 1e-9) for r in R] if clean_key in R[0] and "|" in k else [np.nan]
        floor = [r["SCT-silo"]["f1"] for r in R]
        rows.append(dict(key=k, rule=rule, adversaries=adv, attack=att, n_seeds=len(R), f1_mean=np.mean(f1), f1_sd=np.std(f1, ddof=1) if len(f1) > 1 else np.nan,
                         retention_mean=np.nanmean(ret), floor_f1=np.mean(floor), f1_by_seed=f1))
    return pd.DataFrame(rows)

def fig_pointwise(tab, corpus):
    t = tab[(tab.corpus == corpus) & (~tab.detector.str.endswith("+PV")) & (tab.applicable)].sort_values("f1_mean")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(t.detector, t.f1_mean, xerr=t.f1_sd.fillna(0), color=["#4c72b0" if d != CONTROL else "#dd8452" for d in t.detector])
    ax.set_xlabel("Pooled F1 at the calibrated 1% budget (mean ± SD over seeds)"); ax.set_title(corpus)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"f1_{corpus}.png"), dpi=200); plt.close(fig)

def fig_paired(corpus, d, seeds, mets):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, det in enumerate([x for x in DETS if x != CONTROL and all(x in mets[s]["_pooled"] for s in seeds) and applicable(mets, seeds, x)]):
        diffs = [mets[s]["_pooled"][det]["f1"] - mets[s]["_pooled"][CONTROL]["f1"] for s in seeds]
        ax.scatter(diffs, [i] * len(diffs), color="k", s=14); ax.plot([np.mean(diffs)] * 2, [i - 0.3, i + 0.3], color="r")
        ax.text(0.0, i + 0.35, det, fontsize=7, ha="center")
    ax.axvline(0, color="grey", lw=0.8); ax.set_yticks([]); ax.set_xlabel(f"F1 difference vs {CONTROL} (per seed; red = mean)"); ax.set_title(corpus)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"paired_{corpus}.png"), dpi=200); plt.close(fig)

def fig_learning_curves():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for corpus in ("DE", "NL", "NA", "EU"):
        for f in sorted(glob.glob(os.path.join(RES, corpus, "seed*", "model_training.json"))):
            h = json.load(open(f))["history"]
            ax.plot([x["epoch"] for x in h], [x["val_loss"] for x in h], marker=".", lw=0.8, label=f"{corpus} {os.path.basename(os.path.dirname(f))}")
    ax.set_xlabel("epoch"); ax.set_ylabel("validation task loss (Huber)"); ax.legend(fontsize=6, ncol=2); fig.tight_layout()
    fig.savefig(os.path.join(FIG, "learning_curves.png"), dpi=200); plt.close(fig)

def fig_residency(rt):
    """Grouped bars of pooled F1 per arm and corpus (mean +/- SD over seeds)."""
    fig, axes = plt.subplots(1, len(rt), figsize=(4.5 * len(rt), 3.6), squeeze=False)
    for ax, t in zip(axes[0], rt):
        t = t.copy(); order = ["SCT-all", "SCT-silo", "SCT-OI-all", "SCT-OI-silo", "centralized", "centralized-confined", "federated-fullgraph", "federated", "federated-matched", "local-only"]
        t["o"] = t.arm.map({a: i for i, a in enumerate(order)}); t = t.sort_values("o")
        ax.bar(range(len(t)), t.f1_mean, yerr=t.f1_sd.fillna(0), color=["#dd8452" if a.startswith("SCT") else ("#55a868" if "matched" in a else "#4c72b0") for a in t.arm])
        ax.set_xticks(range(len(t))); ax.set_xticklabels(t.arm, rotation=35, ha="right", fontsize=7); ax.set_ylabel("pooled F1 (spatial channel)"); ax.set_title(f"residency: {t.corpus.iloc[0]}")
        lo = max(0.0, t.f1_mean.min() - 0.1); ax.set_ylim(lo, min(1.0, t.f1_mean.max() + 0.08))
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "residency.png"), dpi=200); plt.close(fig)

def fig_robustness(rb):
    """F1 under each attack per rule and adversarial fraction, with the clean run and the no-training floor."""
    rules = ["mean", "median", "multikrum", "fltrust"]; attacks = ["clean", "signflip", "alie", "minmax"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, adv in zip(axes, ["adv1", "adv2"]):
        w = 0.2
        for j, att in enumerate(attacks):
            vals, errs = [], []
            for r in rules:
                row = rb[(rb.rule == r) & (rb.adversaries == adv) & (rb.attack == att)]
                vals.append(row.f1_mean.iloc[0] if len(row) else np.nan); errs.append(row.f1_sd.iloc[0] if len(row) and np.isfinite(row.f1_sd.iloc[0]) else 0)
            ax.bar(np.arange(len(rules)) + (j - 1.5) * w, vals, w, yerr=errs, label=att)
        floor = rb.floor_f1.mean()
        ax.axhline(floor, color="grey", ls="--", lw=0.8, label="SCT-silo floor")
        ax.set_xticks(range(len(rules))); ax.set_xticklabels(rules); ax.set_title(f"{adv.replace('adv', '')} of 8 silos adversarial"); ax.set_ylabel("pooled F1 (spatial channel)")
    axes[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, "robustness.png"), dpi=200); plt.close(fig)

def main():
    tabs, pairs, boots, pervar, evs = [], [], {}, [], []
    for corpus in ("DE", "NL", "EU", "NA", "DE_subtle"):
        if not os.path.exists(os.path.join(RES, corpus)): continue
        d, seeds, mets = load_corpus(corpus)
        if not seeds: continue
        tabs.append(table_pointwise(corpus, d, seeds, mets))
        for a, b in ((LEARNED, CONTROL), (LEARNED, "SCT"), (LEARNED, "SCT-OI+P"), (CONTROL, "SCT"), ("SCT-OI", "SCT"), ("SCT-OI+P", CONTROL), ("LSO-sp-only", "SCT"), (LEARNED, "LSO-sp-only"), (CONTROL + "+PV", CONTROL), ("SCT", "Kalman"), (LEARNED + "+PV", CONTROL + "+PV"), (LEARNED + "+PV", "SCT-OI+P+PV")):
            p = paired(corpus, d, seeds, mets, a, b)
            if p: pairs.append(p)
        if all(os.path.exists(os.path.join(d, f"seed{s}", "flags_test.npz")) for s in seeds) and LEARNED in mets[seeds[0]]["_pooled"]:
            boots[corpus] = {f"{LEARNED} - {CONTROL}": bootstrap(corpus, d, seeds, LEARNED, CONTROL), f"{CONTROL} - SCT": bootstrap(corpus, d, seeds, CONTROL, "SCT")}
        pervar.append(per_variable(corpus, d, seeds, mets)); evs.append(events_table(corpus, d, seeds, mets))
        fig_pointwise(tabs[-1], corpus); fig_paired(corpus, d, seeds, mets)
    if tabs:
        T = pd.concat(tabs); T.to_csv(os.path.join(SUM, "pointwise.csv"), index=False)
        pd.DataFrame(pairs).to_csv(os.path.join(SUM, "paired_contrasts.csv"), index=False)
        pd.concat(pervar).to_csv(os.path.join(SUM, "per_variable.csv"), index=False)
        pd.concat(evs).to_csv(os.path.join(SUM, "events.csv"), index=False)
        json.dump(boots, open(os.path.join(SUM, "bootstrap.json"), "w"), indent=1, default=float)
    rt, rc = [], []
    for corpus in ("DE", "EU"):
        a, b = residency_table(corpus)
        if a is not None: rt.append(a); rc.append(b)
    if rt:
        pd.concat(rt).to_csv(os.path.join(SUM, "residency.csv"), index=False); pd.concat(rc).to_csv(os.path.join(SUM, "residency_contrasts.csv"), index=False)
        fig_residency(rt)
    rb = robustness_table()
    if rb is not None:
        rb.to_csv(os.path.join(SUM, "robustness.csv"), index=False); fig_robustness(rb)
    fig_learning_curves()
    print("summary written:", os.listdir(SUM))

if __name__ == "__main__":
    main()
