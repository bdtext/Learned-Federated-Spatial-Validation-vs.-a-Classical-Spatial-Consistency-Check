"""Main detection experiment: classical detectors, no-training controls and the learned validator on one corpus, several seeds.
Usage: python run_detection.py <corpus> <seeds comma-separated> [--tune]
Corpora: DE (DWD box, 92 stations), NL (KNMI, 34), EU (DE+NL), NA (GHCNh 34 airports).
Outputs: results/<corpus>/seed<k>/*.json|npz and results/<corpus>/tuning.json
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D, inject as I, detectors as Dt, metrics as M, model as Mo

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
CORPORA = {"DE": (["dwd"], "EU"), "NL": (["knmi"], "EU"), "EU": (["dwd", "knmi"], "EU"), "NA": (["ghcnh"], "NA")}
VARS = ["TEMP", "RH", "DEWP", "WSPM", "PRES"]
CLIP = {"RH": (0.0, 100.0), "WSPM": (0.0, None)}
torch.set_num_threads(int(os.environ.get("QC_THREADS", max(4, os.cpu_count() // 2))))

def jdump(o, p):
    def conv(x):
        if isinstance(x, (np.floating, np.integer)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, dict): return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [conv(v) for v in x]
        return x
    json.dump(conv(o), open(p, "w"), indent=1)

def two_scale_select(s_dict, label, val_mask, weights_grid=None):
    """Select two-timescale combination(s) for a set of standardized channels s_dict (name -> (N,T)).
    Single channel: w*EMA_ma + (1-w)*EMA_mb. Two channels ('fc','sp'): w1*EMA_m1(fc) + w2*EMA_3(sp) + w3*EMA_m2(sp)."""
    if set(s_dict) == {"x"}:
        scores = Dt.two_timescale_scores(s_dict["x"])
        key, v = Dt.select_on_validation(scores, label, val_mask)
        return scores[key], dict(grid=key, val_auprc=v)
    fc, sp = s_dict["fc"], s_dict["sp"]
    ef = {h: Dt.ema(fc, h) for h in (1, 3)}; es = {h: Dt.ema(sp, h) for h in (3, 24, 72)}
    best = None
    for w1 in (0, 1 / 3, 2 / 3, 1):
        for w2 in (0, 1 / 3, 2 / 3, 1):
            w3 = 1 - w1 - w2
            if w3 < -1e-9: continue
            for m1 in (1, 3):
                for m2 in (24, 72):
                    S = w1 * ef[m1] + w2 * es[3] + w3 * es[m2]
                    vv = Dt.auprc(S[:, val_mask].ravel(), label[:, val_mask].ravel())
                    if np.isfinite(vv) and (best is None or vv > best[0]): best = (vv, (w1, w2, w3, m1, m2), S)
    if best is None:  # no validation positives for this variable: default combination
        S = (1 / 3) * ef[1] + (1 / 3) * es[3] + (1 / 3) * es[24]
        best = (np.nan, (1 / 3, 1 / 3, 1 / 3, 1, 24), S)
    return best[2], dict(grid=best[1], val_auprc=best[0])

def evaluate(name, S, label, labels_fam, va, te, segs_v, net, seed_rng, out, var, obs=None):
    if obs is not None:  # score only where the (contaminated) observation exists: identical negative pool for every detector
        S = np.where(obs, S, np.nan)
    applicable = bool(np.isfinite(S[:, te]).any())
    thr = Dt.calibrate_threshold(S, label, va)
    thr_free = Dt.calibrate_threshold(S, label, va, label_free=True)
    pw = M.pointwise(S, label, thr, te); pwf = M.pointwise(S, label, thr_free, te)
    thr_m = M.matched_threshold(S, label, te); pwm = M.pointwise(S, label, thr_m, te)
    fam = M.per_family(S, labels_fam, label, thr, te)
    t0 = int(np.where(te)[0][0]); t1 = int(np.where(te)[0][-1]) + 1
    ev = M.events(pw["flags"], segs_v, t0, t1)
    ch = M.chance_events(pw["flags"], segs_v, t0, t1, label, seed_rng)
    evs = ev.groupby("family")[["any", "persistent", "within24", "within72", "within168"]].mean().to_dict("index") if len(ev) else {}
    med = ev[ev["any"]].groupby("family")["delay"].median().to_dict() if len(ev) else {}
    chs = ch.groupby("family")[["any", "persistent"]].mean().to_dict("index") if len(ch) else {}
    out.setdefault(name, {})[var] = dict(threshold=thr, f1=pw["f1"], precision=pw["precision"], recall=pw["recall"], fpr=pw["fpr"], auprc=pw["auprc"],
                                      f1_labelfree=pwf["f1"], fpr_labelfree=pwf["fpr"], f1_matched=pwm["f1"], per_family=fam,
                                      events=evs, median_delay=med, chance=chs, n_events=int(len(ev)), applicable=applicable)
    return pw["flags"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("corpus"); ap.add_argument("seeds"); ap.add_argument("--tune", action="store_true")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--ch", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-model", action="store_true"); ap.add_argument("--tag", default=""); ap.add_argument("--regime", default="sigma")
    ap.add_argument("--only-cfg", default="", help="ch,lr: train and cache only this configuration for the listed seed(s), then stop (parallel tuning)")
    ap.add_argument("--grid", default="32,0.001;32,0.003;64,0.002", help="tuning grid as 'ch,lr;ch,lr;...' (used with --tune)")
    args = ap.parse_args()
    sources, splitkey = CORPORA[args.corpus]
    seeds = [int(s) for s in args.seeds.split(",")]
    outdir = os.path.join(RES, args.corpus + args.tag); os.makedirs(outdir, exist_ok=True)
    net = D.build_network(args.corpus, sources, min_cov=0.9, t_range=D.RANGE[splitkey])
    removed = D.plausibility_screen(net)
    dup, maxeq = D.duplicate_audit(net)
    masks = D.split_masks(net, splitkey); tr, va, te = masks["train"], masks["val"], masks["test"]
    vars_present = [v for v in VARS if (~np.isnan(net.X[v])).mean() > 0.3]
    # clean-window scales for injection magnitudes (tabulated)
    sa, sh = {}, {}
    for v in vars_present:
        _, sa[v], sh[v] = D.fit_climatology(net.X[v], net.times, tr)
    sig_tab = pd.DataFrame({f"sigma_a_{v}": sa[v] for v in vars_present} | {f"sigma_hf_{v}": sh[v] for v in vars_present}, index=net.stations)
    sig_tab.to_csv(os.path.join(outdir, "sigma_table.csv"))
    jdump(dict(N=net.N, T=net.T, stations=net.stations, vars=vars_present, removed=removed, duplicates=dup, max_equality=float(np.nanmax(maxeq)),
               split_hours=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
               median_corr_TEMP=float(np.nanmedian(Dt.train_correlations(D.standardize(net.X["TEMP"], *D.fit_climatology(net.X["TEMP"], net.times, tr)[:2]), tr)))),
          os.path.join(outdir, "network.json"))
    net.meta.to_csv(os.path.join(outdir, "stations.csv"), index=False)
    bounds = [(int(np.where(m)[0][0]), int(np.where(m)[0][-1]) + 1) for m in (tr, va, te)]
    sub = D.Network(net.name, net.times, net.stations, net.meta, {v: net.X[v] for v in vars_present})
    tfile = os.path.join(outdir, "tuning.json") if os.path.exists(os.path.join(outdir, "tuning.json")) else os.path.join(RES, args.corpus, "tuning.json")
    tuning = json.load(open(tfile)) if os.path.exists(tfile) else None
    for seed in seeds:
        t_seed = time.time()
        sd = os.path.join(outdir, f"seed{seed}"); os.makedirs(sd, exist_ok=True)
        if os.path.exists(os.path.join(sd, "metrics.json")) and (tuning is not None or not args.tune or seed != seeds[0]):
            print(f"seed {seed} already done, skipping", flush=True); continue
        Xf, labels, segs = I.inject(sub, sa, sh, bounds, seed=seed, clip=CLIP, regime=args.regime)
        aud = I.audit(sub, Xf, labels, segs)
        segs.to_csv(os.path.join(sd, "segments.csv"), index=False); aud.to_csv(os.path.join(sd, "audit.csv"), index=False)
        # per-variable standardized anomalies from the contaminated stream
        A, chan, labels_fam = {}, {}, {}
        for v in vars_present:
            clim, sa_c, sh_c = D.fit_climatology(Xf[v], net.times, tr)
            A[v] = D.standardize(Xf[v], clim, sa_c)
            C = Dt.train_correlations(A[v], tr); W = Dt.sct_weights(C)
            chan[v] = {"CLIM-Z": Dt.clim_z(A[v]), "Persist": Dt.persistence_innovation(A[v]), "SCT": A[v] - Dt.sct_estimate(A[v], W, tr),
                       "EWMA": None, "Kalman": None, "SCT-OI": None, "RunLen": Dt.run_length(Xf[v]), "VarRatio": Dt.variance_ratio(Xf[v], sh_c) - 1.0}
            # validation-selected parameters for EWMA (span), Kalman (q) and OI (L, eps)
            best = None
            for q in (0.01, 0.1, 1.0):
                r = Dt.kalman_innovation(A[v], q); s = Dt.robust_standardize(r, va)
                vv = Dt.auprc(Dt.ema(s, 3)[:, va].ravel(), labels[v][:, va].ravel())
                if best is None or vv > best[0]: best = (vv, q, r)
            chan[v]["Kalman"] = best[2]; chan[v]["Kalman_q"] = best[1]
            best = None
            for L in (30.0, 60.0, 120.0):
                for eps in (0.05, 0.2):
                    r = A[v] - Dt.oi_estimate(A[v], net.meta, tr, L_km=L, eps=eps); s = Dt.robust_standardize(r, va)
                    vv = Dt.auprc(Dt.ema(s, 3)[:, va].ravel(), labels[v][:, va].ravel())
                    if best is None or vv > best[0]: best = (vv, (L, eps), r)
            chan[v]["SCT-OI"] = best[2]; chan[v]["OI_params"] = best[1]
            best = None
            for span in (3, 12, 24):
                r = Dt.ema(A[v], span); s = Dt.robust_standardize(r, va)
                vv = Dt.auprc(s[:, va].ravel(), labels[v][:, va].ravel())
                if best is None or vv > best[0]: best = (vv, span, r)
            chan[v]["EWMA"] = best[2]; chan[v]["EWMA_span"] = best[1]
            labels_fam[v] = {fam: np.zeros_like(labels[v]) for fam in I.FAMILIES}
            for r in segs[segs["var"] == v].itertuples():
                labels_fam[v][r.family][r.sidx, r.start:r.end] = labels[v][r.sidx, r.start:r.end]
        # learned validator (centralized)
        model_res = {}
        if not args.no_model:
            a = np.stack([A[v] for v in vars_present]); K, N, T = a.shape
            prior = np.stack([Dt.sct_weights(Dt.train_correlations(a[k], tr)) for k in range(K)])
            tf = Mo.time_features(net.times); fetch = Mo.make_windows(a, None, tf)
            train_ts = np.where(tr)[0]; train_ts = train_ts[(train_ts >= 24) & (train_ts < T - 1)]
            val_ts = np.where(va)[0]; val_ts = val_ts[val_ts < T - 1][::4]
            cfgs = [dict(ch=args.ch, lr=args.lr)]
            if args.only_cfg:
                ch_, lr_ = args.only_cfg.split(","); cfgs = [dict(ch=int(ch_), lr=float(lr_))]
            elif args.tune and seed == seeds[0]:
                cfgs = [dict(ch=int(c.split(",")[0]), lr=float(c.split(",")[1])) for c in args.grid.split(";") if c]
            elif tuning is not None:
                cfgs = [tuning["best"]]
            tune_log = []
            best_cfg = None
            for cfg in cfgs:
                ckey = f"ch{cfg['ch']}_lr{cfg['lr']}"; mfile = os.path.join(sd, f"model_{ckey}.pt"); hfile = os.path.join(sd, f"history_{ckey}.json")
                torch.manual_seed(seed)
                model = Mo.LSOSTGNN(N, K, prior, ch=cfg["ch"])
                if os.path.exists(mfile) and os.path.exists(hfile):  # resume: weights and learning curve cached by an earlier (interrupted) run
                    model.load_state_dict(torch.load(mfile)); hist = json.load(open(hfile))
                    print(f"seed {seed}: loaded cached model {ckey} ({len(hist)} epochs)", flush=True)
                else:
                    t_tr = time.time()
                    hist = Mo.train_model(model, fetch, a, None, train_ts, val_ts, epochs=args.epochs, lr=cfg["lr"], batch=128, patience=3, seed=seed, verbose=False)
                    torch.save(model.state_dict(), mfile); json.dump(hist, open(hfile, "w"))
                    print(f"seed {seed}: trained {ckey} in {time.time()-t_tr:.0f}s ({len(hist)} epochs)", flush=True)
                if args.only_cfg:
                    continue
                rfc, rsp = Mo.residuals(model, fetch, a, np.arange(24, T - 1))
                # validation AUPRC of the plain spatial innovation (TEMP) for model selection, no test data touched
                k0 = vars_present.index("TEMP"); s = Dt.robust_standardize(rsp[k0], va)
                vv = Dt.auprc(Dt.ema(s, 3)[:, va].ravel(), labels["TEMP"][:, va].ravel())
                tune_log.append(dict(cfg=cfg, epochs=len(hist), best_val_loss=min(h["val_loss"] for h in hist), val_auprc_sp_TEMP=vv, history=hist))
                if best_cfg is None or vv > best_cfg[0]: best_cfg = (vv, cfg, rfc, rsp, hist, model)
            if args.only_cfg:
                print(f"seed {seed}: configuration {args.only_cfg} cached; stopping as requested", flush=True); continue
            if args.tune and seed == seeds[0]:
                jdump(dict(best=best_cfg[1], log=tune_log), os.path.join(outdir, "tuning.json")); tuning = dict(best=best_cfg[1])
            vv, cfg, rfc, rsp, hist, model = best_cfg
            jdump(dict(cfg=cfg, history=hist, n_params=sum(p.numel() for p in model.parameters())), os.path.join(sd, "model_training.json"))
            # learned adjacency diagnostics
            with torch.no_grad():
                Aadj = model.adjacency().numpy(); shr = model.shrink.numpy()
            jdump(dict(adj_vs_prior_mean_abs_diff=float(np.mean(np.abs(Aadj - prior / np.maximum(prior.sum(-1, keepdims=True), 1e-9)))), shrink_mean=float(shr.mean()), shrink_sd=float(shr.std())), os.path.join(sd, "model_diag.json"))
            for k, v in enumerate(vars_present):
                chan[v]["LSO_fc"] = rfc[k].astype(float); chan[v]["LSO_sp"] = rsp[k].astype(float)
        # scoring, thresholds, metrics
        out, flags_store = {}, {}
        rng = np.random.default_rng(1000 + seed)
        for v in vars_present:
            lab = labels[v]; segs_v = segs[segs["var"] == v]; obs_v = ~np.isnan(A[v])
            std = {n: Dt.robust_standardize(chan[v][n], va) for n in ("CLIM-Z", "EWMA", "Kalman", "Persist", "SCT", "SCT-OI") if chan[v].get(n) is not None}
            std["RunLen"] = chan[v]["RunLen"] / 6.0; std["VarRatio"] = np.nan_to_num(chan[v]["VarRatio"], nan=0.0)
            pv = np.maximum(std["RunLen"], std["VarRatio"])  # persistence/variance pre-test channel
            singles = {"CLIM-Z": std["CLIM-Z"], "EWMA-SPC": std["EWMA"], "Kalman": std["Kalman"], "SCT": std["SCT"], "SCT-OI": std["SCT-OI"]}
            for n, s in singles.items():
                S, sel = two_scale_select({"x": s}, lab, va); out.setdefault("_sel", {}).setdefault(v, {})[n] = sel
                flags_store[(n, v)] = evaluate(n, S, lab, labels_fam[v], va, te, segs_v, net, rng, out, v, obs=obs_v)
                S2 = np.maximum(S, pv); flags_store[(n + "+PV", v)] = evaluate(n + "+PV", S2, lab, labels_fam[v], va, te, segs_v, net, rng, out, v, obs=obs_v)
            for n, fcname, spname in (("SCT+P", "Persist", "SCT"), ("SCT+K", "Kalman", "SCT"), ("SCT-OI+P", "Persist", "SCT-OI")):
                S, sel = two_scale_select({"fc": std[fcname], "sp": std[spname]}, lab, va); out["_sel"][v][n] = sel
                flags_store[(n, v)] = evaluate(n, S, lab, labels_fam[v], va, te, segs_v, net, rng, out, v, obs=obs_v)
                S2 = np.maximum(S, pv); flags_store[(n + "+PV", v)] = evaluate(n + "+PV", S2, lab, labels_fam[v], va, te, segs_v, net, rng, out, v, obs=obs_v)
            if not args.no_model:
                sfc = Dt.robust_standardize(chan[v]["LSO_fc"], va); ssp = Dt.robust_standardize(chan[v]["LSO_sp"], va)
                S, sel = two_scale_select({"fc": sfc, "sp": ssp}, lab, va); out["_sel"][v]["LSO-STGNN"] = sel
                flags_store[("LSO-STGNN", v)] = evaluate("LSO-STGNN", S, lab, labels_fam[v], va, te, segs_v, net, rng, out, v, obs=obs_v)
                S2 = np.maximum(S, pv); flags_store[("LSO-STGNN+PV", v)] = evaluate("LSO-STGNN+PV", S2, lab, labels_fam[v], va, te, segs_v, net, rng, out, v, obs=obs_v)
                Ssp, sel = two_scale_select({"x": ssp}, lab, va); out["_sel"][v]["LSO-sp-only"] = sel
                flags_store[("LSO-sp-only", v)] = evaluate("LSO-sp-only", Ssp, lab, labels_fam[v], va, te, segs_v, net, rng, out, v, obs=obs_v)
            out.setdefault("_params", {})[v] = dict(kalman_q=chan[v]["Kalman_q"], oi=chan[v]["OI_params"], ewma_span=chan[v]["EWMA_span"])
        # pooled metrics across variables (per detector): weighted by readings
        pooled = {}
        for n in {k[0] for k in flags_store}:
            tp = fp = fn = neg = 0
            for v in vars_present:
                f = flags_store[(n, v)]; y = labels[v][:, te]; m = ~np.isnan(A[v][:, te])
                tp += np.sum(f & y); fp += np.sum(f & ~y & m); fn += np.sum(~f & y & m); neg += np.sum(~y & m)
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            pooled[n] = dict(f1=2 * p * r / max(p + r, 1e-12), precision=p, recall=r, fpr=fp / max(neg, 1))
        out["_pooled"] = pooled
        jdump(out, os.path.join(sd, "metrics.json"))
        np.savez_compressed(os.path.join(sd, "flags_test.npz"), **{f"{n}__{v}": flags_store[(n, v)] for (n, v) in flags_store},
                            **{f"label__{v}": labels[v][:, te] for v in vars_present}, **{f"valid__{v}": ~np.isnan(A[v][:, te]) for v in vars_present})
        print(f"seed {seed} done in {time.time()-t_seed:.0f}s; pooled F1:", {k: round(x["f1"], 3) for k, x in sorted(pooled.items())}, flush=True)

if __name__ == "__main__":
    main()
