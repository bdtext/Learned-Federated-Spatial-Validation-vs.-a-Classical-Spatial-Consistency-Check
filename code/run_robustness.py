"""Byzantine robustness of federated training of the learned validator on the DE corpus.
Silos: K=8 station-level silos (contiguous longitude bands of the DE box) so that every silo keeps a few neighbours.
Rules: mean, coordinate median, multi-Krum (f = number of adversaries, admissible when K >= 2f+3), FLTrust (root set = silo 0's
first 90 training days, held by the server). Attacks: scaled sign-flip (-5u), a-little-is-enough (z=1.5), min-max
(Shejwalkar & Houmansadr). Adversarial fractions: 1/8 and 2/8. Rounds: 5 x 1 local epoch (stride-4 training windows).
Reported: absolute pooled F1 (spatial channel, TEMP/RH/DEWP) under attack and clean, retention, and the no-training floor (SCT-silo).
Each (seed, rule) can run as its own process (--rules) and writes robustness_<rules>.json; every finished key is saved at once (resumable);
done_<rules>.flag marks completion. run_stats.py merges all robustness*.json files of a seed.
Usage: python run_robustness.py <seeds> [--rules mean,median,multikrum,fltrust] [--K 8 --rounds 6 --stride 2 --corpus DE]
"""
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D, inject as I, detectors as Dt, metrics as M, model as Mo, fl as FL
from run_detection import CORPORA, CLIP, jdump
from run_residency import eval_sp, confine_prior

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
VARS3 = ["TEMP", "RH", "DEWP"]
ALL_RULES = ("mean", "median", "multikrum", "fltrust")
torch.set_num_threads(int(os.environ.get("QC_THREADS", max(4, os.cpu_count() // 2))))

def group_stations(meta, K):
    """Deterministic geographic grouping: sort by longitude and split into K contiguous longitude bands."""
    order = np.argsort(meta.lon.to_numpy())
    bands = np.array_split(order, K)
    silos = np.zeros(len(meta), dtype=int)
    for g, b in enumerate(bands):
        silos[b] = g
    return silos

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seeds"); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5); ap.add_argument("--corpus", default="DE"); ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--rules", default=",".join(ALL_RULES))
    args = ap.parse_args()
    rules = [r for r in args.rules.split(",") if r]; tag = "" if rules == list(ALL_RULES) else "_" + "-".join(rules)
    sources, splitkey = CORPORA[args.corpus]
    outdir = os.path.join(RES, f"robustness_{args.corpus}"); os.makedirs(outdir, exist_ok=True)
    net = D.build_network(args.corpus, sources, min_cov=0.9, t_range=D.RANGE[splitkey]); D.plausibility_screen(net)
    masks = D.split_masks(net, splitkey); tr, va, te = masks["train"], masks["val"], masks["test"]
    sa, sh = {}, {}
    for v in VARS3:
        _, sa[v], sh[v] = D.fit_climatology(net.X[v], net.times, tr)
    bounds = [(int(np.where(m)[0][0]), int(np.where(m)[0][-1]) + 1) for m in (tr, va, te)]
    sub = D.Network(net.name, net.times, net.stations, net.meta, {v: net.X[v] for v in VARS3})
    tune = os.path.join(RES, "DE", "tuning.json"); cfg = json.load(open(tune))["best"] if os.path.exists(tune) else dict(ch=32, lr=2e-3)
    K_silo = args.K
    silos = group_stations(net.meta, K_silo)
    jdump(dict(silos={int(g): [net.stations[i] for i in np.where(silos == g)[0]] for g in range(K_silo)}), os.path.join(outdir, "silos.json"))
    print(f"corpus {args.corpus}: N={net.N}, K={K_silo}, rules={rules}, cfg={cfg}", flush=True)
    for seed in [int(s) for s in args.seeds.split(",")]:
        t0 = time.time(); sd = os.path.join(outdir, f"seed{seed}"); os.makedirs(sd, exist_ok=True)
        rfile = os.path.join(sd, f"robustness{tag}.json"); flag = os.path.join(sd, f"done{tag}.flag")
        if os.path.exists(flag):
            print(f"seed {seed} rules {rules} already done, skipping", flush=True); continue
        res = json.load(open(rfile)) if os.path.exists(rfile) else {}
        Xf, labels, segs = I.inject(sub, sa, sh, bounds, seed=seed, clip=CLIP)
        A = []
        for v in VARS3:
            clim, sa_c, _ = D.fit_climatology(Xf[v], net.times, tr); A.append(D.standardize(Xf[v], clim, sa_c))
        a = np.stack(A); K, N, T = a.shape; obs = np.isfinite(a)
        prior_full = np.stack([Dt.sct_weights(Dt.train_correlations(a[k], tr)) for k in range(K)])
        prior_conf = confine_prior(prior_full, silos.tolist())
        tf = Mo.time_features(net.times)
        train_ts = np.where(tr)[0]; train_ts = train_ts[(train_ts >= 24) & (train_ts < T - 1)][::args.stride]
        val_ts = np.where(va)[0]; val_ts = val_ts[val_ts < T - 1][::4]
        res["_meta"] = dict(cfg=cfg, rounds=args.rounds, stride=args.stride, K=K_silo, n_train_windows=int(len(train_ts)))
        if "SCT-silo" not in res:
            res["SCT-silo"] = eval_sp(np.stack([a[k] - Dt.sct_estimate(a[k], prior_conf[k], tr) for k in range(K)]), labels, va, te, VARS3, obs)
        jdump(res, rfile)
        def build():
            S = []
            for g in range(K_silo):
                idx = np.where(silos == g)[0]; a_s = a[:, idx]; P_s = prior_full[:, idx][:, :, idx]
                torch.manual_seed(seed * 100 + g); m = Mo.LSOSTGNN(len(idx), K, P_s, ch=cfg["ch"])
                S.append(dict(name=str(g), nodes=idx, model=m, a=a_s, fetch=Mo.make_windows(a_s, None, tf)))
            return S
        # root set for FLTrust: the server holds silo 0's first 90 training days and trains the same architecture on it
        def root_tuple():
            idx = np.where(silos == 0)[0]; a_r = a[:, idx]; P_r = prior_full[:, idx][:, :, idx]
            torch.manual_seed(seed * 100 + 99); m = Mo.LSOSTGNN(len(idx), K, P_r, ch=cfg["ch"])
            ts = train_ts[:(90 * 24) // args.stride]
            return (m, Mo.make_windows(a_r, None, tf), ts, a_r)
        rng = np.random.default_rng(seed)
        for n_adv in (1, 2):
            adv = sorted(rng.choice(K_silo, n_adv, replace=False).tolist())
            for rule in rules:
                f = n_adv
                if rule == "multikrum" and K_silo < 2 * f + 3:
                    continue
                for attack in (None, "signflip", "alie", "minmax"):
                    key = f"{rule}|adv{n_adv}|{attack or 'clean'}"
                    if key in res and "f1" in res[key]:
                        continue
                    if attack is None and n_adv == 2 and f"{rule}|adv1|clean" in res:  # the clean run does not depend on n_adv
                        res[key] = res[f"{rule}|adv1|clean"]; jdump(res, rfile); continue
                    t1 = time.time(); S = build()
                    rt = root_tuple() if rule == "fltrust" else None
                    g, hist = FL.federated_train(None, S, a, None, train_ts, val_ts, rounds=args.rounds, local_epochs=1, lr=cfg["lr"], batch=128, seed=seed,
                                                 rule=rule, f=f, adversaries=adv if attack else (), attack=attack, attack_kw={"z": 1.5} if attack == "alie" else None,
                                                 root=rt, verbose=False, fetch_by_silo=[s["fetch"] for s in S])
                    rsp = np.full((K, N, T), np.nan, dtype=np.float32)
                    for s in S:
                        _, r = Mo.residuals(s["model"], s["fetch"], s["a"], np.arange(24, T - 1)); rsp[:, s["nodes"]] = r
                    res[key] = eval_sp(rsp, labels, va, te, VARS3, obs); res[key]["adversaries"] = adv; res[key]["history"] = hist; res[key]["seconds"] = time.time() - t1
                    print(f"  {key}: F1 {res[key]['f1']:.3f} ({time.time()-t1:.0f}s)", flush=True)
                    jdump(res, rfile)
        open(flag, "w").write("ok")
        print(f"seed {seed} rules {rules} done in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
