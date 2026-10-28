"""Residency experiment: what does strict data residency cost spatial QC, and is the cost a neighbour effect or a training effect?
Silos: DE federal states (NRW, Hessen, Niedersachsen, Rheinland-Pfalz; Bayern/Bremen stations merged into the nearest state silo) or
countries (DE, NL) for the cross-border corpus. Arms, all evaluated on the TEMP/RH/DEWP spatial channel with identical scoring
(2 x 2 design: {pooled, silo} training x {full, confined} neighbour graph, plus references):
  centralized          : pooled training, full graph
  centralized-confined : pooled training, graph confined to within-silo neighbours
  federated (FedAvg)   : silo training (loss on own stations only), confined graph, shared backbone averaged
  federated-fullgraph  : silo training (loss on own stations only), full graph as INPUT (cross-silo neighbour anomalies are visible,
                         parameters are exchanged) -- isolates the neighbour effect from the training effect
  local-only           : silo training, confined graph, no exchange
  SCT-all / SCT-silo   : classical check with all neighbours / within-silo neighbours
Every arm is written to residency_partial.json as soon as it finishes (resumable); residency.json marks completion.
Usage: python run_residency.py <corpus: DE|EU> <seeds> [--rounds 10 --epochs 15 --stride 2 --fullgraph]
The federated-fullgraph diagnostic is optional (--fullgraph): it costs as much as all other arms together.
"""
import os, sys, json, time, argparse
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D, inject as I, detectors as Dt, metrics as M, model as Mo, fl as FL
from run_detection import CORPORA, CLIP, jdump, two_scale_select

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
VARS3 = ["TEMP", "RH", "DEWP"]
torch.set_num_threads(int(os.environ.get("QC_THREADS", max(4, os.cpu_count() // 2))))

def silo_assignment(net, corpus):
    st = net.meta["state"].tolist()
    if corpus == "EU":
        return ["NL" if s == "Netherlands" else "DE" for s in st]
    merge = {"Bayern": "Hessen", "Bremen": "Niedersachsen"}
    return [merge.get(s, s) for s in st]

def confine_prior(prior, silos):
    """Zero out cross-silo prior weights so that the spatial head and attention bias see only in-silo neighbours."""
    P = prior.copy()
    same = np.array([[si == sj for sj in silos] for si in silos])
    P[:, ~same] = 0.0
    return P

def masked_fetch(fetch_full, node_mask):
    """Inputs from the full network, spatial-head targets restricted to the silo's own nodes (loss mask)."""
    nm = torch.from_numpy(node_mask.astype(np.float32))[None, :, None]
    def f(ts):
        xw, al, ml = fetch_full(ts)
        return xw, al, ml * nm
    return f

def eval_sp(rsp, labels, va, te, vars3, obs=None):
    """Score the spatial innovation per variable with the single-channel two-timescale grid (validation-selected), threshold at the
    calibrated 1 per cent budget, and pool TP/FP/FN over variables. obs: (K,N,T) observation-validity mask (identical negative pool)."""
    tp = fp = fn = neg = 0; per = {}
    for k, v in enumerate(vars3):
        s = Dt.robust_standardize(rsp[k], va); S, sel = two_scale_select({"x": s}, labels[v], va)
        if obs is not None:
            S = np.where(obs[k], S, np.nan)
        thr = Dt.calibrate_threshold(S, labels[v], va); pw = M.pointwise(S, labels[v], thr, te)
        thr_m = M.matched_threshold(S, labels[v], te); pwm = M.pointwise(S, labels[v], thr_m, te)
        per[v] = dict(f1=pw["f1"], auprc=pw["auprc"], fpr=pw["fpr"], f1_matched=pwm["f1"], grid=list(sel["grid"]))
        f = pw["flags"]; y = labels[v][:, te]; m = obs[k][:, te] if obs is not None else ~np.isnan(rsp[k][:, te])
        tp += np.sum(f & y); fp += np.sum(f & ~y & m); fn += np.sum(~f & y & m); neg += np.sum(~y & m)
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(f1=2 * p * r / max(p + r, 1e-12), precision=p, recall=r, fpr=fp / max(neg, 1), per_var=per)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("corpus"); ap.add_argument("seeds"); ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--ch", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--fullgraph", action="store_true", help="also run the federated-fullgraph diagnostic arm (expensive)")
    args = ap.parse_args()
    sources, splitkey = CORPORA[args.corpus]
    outdir = os.path.join(RES, f"residency_{args.corpus}"); os.makedirs(outdir, exist_ok=True)
    net = D.build_network(args.corpus, sources, min_cov=0.9, t_range=D.RANGE[splitkey]); D.plausibility_screen(net)
    masks = D.split_masks(net, splitkey); tr, va, te = masks["train"], masks["val"], masks["test"]
    silos = silo_assignment(net, args.corpus); names = sorted(set(silos))
    nodes = {s: [i for i, x in enumerate(silos) if x == s] for s in names}
    jdump(dict(silos={s: [net.stations[i] for i in nodes[s]] for s in names}), os.path.join(outdir, "silos.json"))
    sa, sh = {}, {}
    for v in VARS3:
        _, sa[v], sh[v] = D.fit_climatology(net.X[v], net.times, tr)
    bounds = [(int(np.where(m)[0][0]), int(np.where(m)[0][-1]) + 1) for m in (tr, va, te)]
    sub = D.Network(net.name, net.times, net.stations, net.meta, {v: net.X[v] for v in VARS3})
    tune = os.path.join(RES, "DE", "tuning.json")
    cfg = json.load(open(tune))["best"] if os.path.exists(tune) else dict(ch=args.ch, lr=args.lr)
    print(f"corpus {args.corpus}: N={net.N}, silos={ {s: len(nodes[s]) for s in names} }, cfg={cfg}", flush=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        t0 = time.time(); sd = os.path.join(outdir, f"seed{seed}"); os.makedirs(sd, exist_ok=True)
        final = os.path.join(sd, "residency.json"); partial = os.path.join(sd, "residency_partial.json")
        ARMS = ["SCT-all", "SCT-silo", "SCT-OI-all", "SCT-OI-silo", "centralized", "centralized-confined", "local-only", "federated", "federated-matched"]
        if os.path.exists(final) and all(a in json.load(open(final)) for a in ARMS):
            print(f"seed {seed} already done, skipping", flush=True); continue
        res = json.load(open(final)) if os.path.exists(final) else (json.load(open(partial)) if os.path.exists(partial) else {})
        def save(): jdump(res, partial)
        Xf, labels, segs = I.inject(sub, sa, sh, bounds, seed=seed, clip=CLIP)
        A = []
        for v in VARS3:
            clim, sa_c, _ = D.fit_climatology(Xf[v], net.times, tr); A.append(D.standardize(Xf[v], clim, sa_c))
        a = np.stack(A); K, N, T = a.shape; obs = np.isfinite(a)
        prior_full = np.stack([Dt.sct_weights(Dt.train_correlations(a[k], tr)) for k in range(K)])
        prior_conf = confine_prior(prior_full, silos)
        tf = Mo.time_features(net.times); fetch = Mo.make_windows(a, None, tf)
        train_ts = np.where(tr)[0]; train_ts = train_ts[(train_ts >= 24) & (train_ts < T - 1)][::args.stride]
        val_ts = np.where(va)[0]; val_ts = val_ts[val_ts < T - 1][::4]
        res.setdefault("_meta", dict(cfg=cfg, rounds=args.rounds, epochs=args.epochs, stride=args.stride, n_train_windows=int(len(train_ts)), silos={s: len(nodes[s]) for s in names}))
        # classical references
        for name, P in (("SCT-all", prior_full), ("SCT-silo", prior_conf)):
            if name in res: continue
            rsp = np.stack([a[k] - Dt.sct_estimate(a[k], P[k], tr) for k in range(K)])
            res[name] = eval_sp(rsp, labels, va, te, VARS3, obs); save()
            print(f"  {name}: F1 {res[name]['f1']:.3f}", flush=True)
        # operational-style classical references: OI with distance and elevation, all neighbours / within-silo neighbours,
        # with the OI parameters selected on validation in the DE detection benchmark (per variable)
        same = np.array([[si == sj for sj in silos] for si in silos])
        pfile = os.path.join(RES, "DE", "seed0", "metrics.json")
        oi_params = json.load(open(pfile))["_params"] if os.path.exists(pfile) else {}
        for name, allowed in (("SCT-OI-all", None), ("SCT-OI-silo", same)):
            if name in res: continue
            t1 = time.time(); rsp = []
            for k, v in enumerate(VARS3):
                L, eps = oi_params.get(v, {}).get("oi", [60.0, 0.1])
                rsp.append(a[k] - Dt.oi_estimate(a[k], net.meta, tr, L_km=L, eps=eps, allowed=allowed))
            res[name] = eval_sp(np.stack(rsp), labels, va, te, VARS3, obs); res[name]["oi_params"] = {v: oi_params.get(v, {}).get("oi", [60.0, 0.1]) for v in VARS3}; save()
            print(f"  {name}: F1 {res[name]['f1']:.3f} ({time.time()-t1:.0f}s)", flush=True)
        # pooled training: full graph and confined graph
        for name, P in (("centralized", prior_full), ("centralized-confined", prior_conf)):
            if name in res: continue
            t1 = time.time(); torch.manual_seed(seed); model = Mo.LSOSTGNN(N, K, P, ch=cfg["ch"])
            hist = Mo.train_model(model, fetch, a, None, train_ts, val_ts, epochs=args.epochs, lr=cfg["lr"], batch=128, patience=3, seed=seed, verbose=False)
            rfc, rsp = Mo.residuals(model, fetch, a, np.arange(24, T - 1))
            res[name] = eval_sp(rsp, labels, va, te, VARS3, obs); res[name]["history"] = hist; res[name]["seconds"] = time.time() - t1; save()
            print(f"  {name}: F1 {res[name]['f1']:.3f} ({len(hist)} epochs, {time.time()-t1:.0f}s)", flush=True)
        # silo-local networks (confined graph): each silo sees only its own stations
        def build_silos():
            S = []
            for j, s in enumerate(names):
                idx = np.array(nodes[s]); a_s = a[:, idx]; P_s = prior_full[:, idx][:, :, idx]
                torch.manual_seed(seed * 100 + j); m = Mo.LSOSTGNN(len(idx), K, P_s, ch=cfg["ch"])
                S.append(dict(name=s, nodes=idx, model=m, a=a_s, fetch=Mo.make_windows(a_s, None, tf)))
            return S
        if "local-only" not in res:
            t1 = time.time(); S = build_silos(); rsp = np.full((K, N, T), np.nan, dtype=np.float32); curves = {}
            for s in S:
                hist = Mo.train_model(s["model"], s["fetch"], s["a"], None, train_ts, val_ts, epochs=args.epochs, lr=cfg["lr"], batch=128, patience=3, seed=seed, verbose=False)
                _, r = Mo.residuals(s["model"], s["fetch"], s["a"], np.arange(24, T - 1)); rsp[:, s["nodes"]] = r; curves[s["name"]] = hist
            res["local-only"] = eval_sp(rsp, labels, va, te, VARS3, obs); res["local-only"]["history"] = curves; res["local-only"]["seconds"] = time.time() - t1; save()
            print(f"  local-only: F1 {res['local-only']['f1']:.3f} ({time.time()-t1:.0f}s)", flush=True)
        if "federated" not in res:
            t1 = time.time(); S = build_silos()
            g, hist = FL.federated_train(None, S, a, None, train_ts, val_ts, rounds=args.rounds, local_epochs=1, lr=cfg["lr"], batch=128, seed=seed,
                                         rule="mean", verbose=False, fetch_by_silo=[s["fetch"] for s in S])
            rsp = np.full((K, N, T), np.nan, dtype=np.float32)
            for s in S:
                _, r = Mo.residuals(s["model"], s["fetch"], s["a"], np.arange(24, T - 1)); rsp[:, s["nodes"]] = r
            res["federated"] = eval_sp(rsp, labels, va, te, VARS3, obs); res["federated"]["history"] = hist; res["federated"]["seconds"] = time.time() - t1; save()
            print(f"  federated: F1 {res['federated']['f1']:.3f} ({time.time()-t1:.0f}s)", flush=True)
        if "federated-matched" not in res:
            # budget-matched federated arm: as many rounds of one local epoch as the epoch cap of the centralized and local-only arms
            t1 = time.time(); S = build_silos()
            g, hist = FL.federated_train(None, S, a, None, train_ts, val_ts, rounds=args.epochs, local_epochs=1, lr=cfg["lr"], batch=128, seed=seed,
                                         rule="mean", verbose=False, fetch_by_silo=[s["fetch"] for s in S])
            rsp = np.full((K, N, T), np.nan, dtype=np.float32)
            for s in S:
                _, r = Mo.residuals(s["model"], s["fetch"], s["a"], np.arange(24, T - 1)); rsp[:, s["nodes"]] = r
            res["federated-matched"] = eval_sp(rsp, labels, va, te, VARS3, obs); res["federated-matched"]["history"] = hist
            res["federated-matched"]["rounds"] = args.epochs; res["federated-matched"]["seconds"] = time.time() - t1; save()
            print(f"  federated-matched ({args.epochs} rounds): F1 {res['federated-matched']['f1']:.3f} ({time.time()-t1:.0f}s)", flush=True)
        if args.fullgraph and "federated-fullgraph" not in res:
            # diagnostic: silo models consume the full graph as input but their loss covers only their own stations; parameters are exchanged
            t1 = time.time(); Sfull = []
            for j, s in enumerate(names):
                idx = np.array(nodes[s]); nm = np.zeros(N, dtype=bool); nm[idx] = True
                a_loss = a.copy(); a_loss[:, ~nm] = np.nan  # forecast-head targets restricted to own nodes
                torch.manual_seed(seed * 100 + j); m = Mo.LSOSTGNN(N, K, prior_full, ch=cfg["ch"])
                Sfull.append(dict(name=s, nodes=idx, model=m, a=a_loss, fetch=masked_fetch(fetch, nm)))
            g, hist = FL.federated_train(None, Sfull, a, fetch, train_ts, val_ts, rounds=args.rounds, local_epochs=1, lr=cfg["lr"], batch=128, seed=seed,
                                         rule="mean", verbose=False, fetch_by_silo=[s["fetch"] for s in Sfull])
            rsp = np.full((K, N, T), np.nan, dtype=np.float32)
            for s in Sfull:
                _, r = Mo.residuals(s["model"], fetch, a, np.arange(24, T - 1)); rsp[:, s["nodes"]] = r[:, s["nodes"]]
            res["federated-fullgraph"] = eval_sp(rsp, labels, va, te, VARS3, obs); res["federated-fullgraph"]["history"] = hist; res["federated-fullgraph"]["seconds"] = time.time() - t1; save()
            print(f"  federated-fullgraph: F1 {res['federated-fullgraph']['f1']:.3f} ({time.time()-t1:.0f}s)", flush=True)
        jdump(res, final)
        print(f"seed {seed} done in {time.time()-t0:.0f}s:", {k: round(v["f1"], 3) for k, v in res.items() if not k.startswith("_")}, flush=True)

if __name__ == "__main__":
    main()
