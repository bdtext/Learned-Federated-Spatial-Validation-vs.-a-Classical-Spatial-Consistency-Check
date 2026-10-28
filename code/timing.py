"""Compute and latency table: per corpus, one timed training epoch of the learned validator and the inference time per hour of
network data on CPU (idle host, QC_THREADS threads), plus the logged wall-clock of the full training runs (which ran under
concurrent load). Outputs results/summary/view_compute.csv.
Usage: python timing.py   (run on an otherwise idle machine)
"""
import os, sys, re, glob, json, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(__file__))
from qc import data as D, detectors as Dt, model as Mo
from run_detection import CORPORA, VARS, RES
THREADS = int(os.environ.get("QC_THREADS", 8)); torch.set_num_threads(THREADS)

def logged_training_times(corpus):
    """(median seconds, median epochs, n runs) from 'seed k: trained ... in Ns (E epochs)' lines of the run logs."""
    secs, eps = [], []
    for f in glob.glob(os.path.join(RES, f"log_{corpus}_*.txt")) + glob.glob(os.path.join(RES, f"log_{corpus}*.txt")):
        for line in open(f, encoding="utf-8", errors="ignore"):
            m = re.search(r"trained \S+ in (\d+)s \((\d+) epochs\)", line)
            if m: secs.append(int(m.group(1))); eps.append(int(m.group(2)))
    return (float(np.median(secs)), float(np.median(eps)), len(secs)) if secs else (float("nan"), float("nan"), 0)

def main():
    rows = []
    for corpus in ("DE", "NL", "NA"):
        sources, splitkey = CORPORA[corpus]
        net = D.build_network(corpus, sources, min_cov=0.9, t_range=D.RANGE[splitkey]); D.plausibility_screen(net)
        masks = D.split_masks(net, splitkey); tr, va = masks["train"], masks["val"]
        vars_present = [v for v in VARS if (~np.isnan(net.X[v])).mean() > 0.3]
        A = []
        for v in vars_present:
            clim, sa, sh = D.fit_climatology(net.X[v], net.times, tr); A.append(D.standardize(net.X[v], clim, sa))
        a = np.stack(A); K, N, T = a.shape
        prior = np.stack([Dt.sct_weights(Dt.train_correlations(a[k], tr)) for k in range(K)])
        best = json.load(open(os.path.join(RES, corpus, "tuning.json")))["best"]
        torch.manual_seed(0); model = Mo.LSOSTGNN(N, K, prior, ch=best["ch"])
        n_params = sum(p.numel() for p in model.parameters()); n_shared = sum(p.numel() for p in model.shared_parameters())
        tf = Mo.time_features(net.times); fetch = Mo.make_windows(a, None, tf)
        train_ts = np.where(tr)[0]; train_ts = train_ts[(train_ts >= 24) & (train_ts < T - 1)]
        val_ts = np.where(va)[0]; val_ts = val_ts[val_ts < T - 1][::4]
        t0 = time.time(); Mo.train_model(model, fetch, a, None, train_ts, val_ts, epochs=1, lr=best["lr"], batch=128, patience=3, seed=0, verbose=False)
        epoch_s = time.time() - t0
        ts = np.arange(T - 24 * 7 - 2, T - 2)  # one week of hourly inference
        t0 = time.time(); Mo.residuals(model, fetch, a, ts); infer_s = time.time() - t0
        # classical references for scale: SCT and OI over the whole record (single-threaded numpy)
        t0 = time.time(); Dt.sct_estimate(a[0], prior[0], tr); sct_s = time.time() - t0
        L, eps = json.load(open(os.path.join(RES, corpus, "seed0", "metrics.json")))["_params"]["TEMP"]["oi"]
        t0 = time.time(); Dt.oi_estimate(a[0], net.meta, tr, L_km=L, eps=eps) if corpus != "NA" else None; oi_s = time.time() - t0
        lsec, leps, nruns = logged_training_times(corpus)
        rows.append(dict(corpus=corpus, stations=N, variables=K, parameters=n_params, shared_parameters=n_shared, threads=THREADS,
                         train_windows=len(train_ts), epoch_seconds=round(epoch_s, 1), epochs_per_run=int(leps) if leps == leps else "n/a",
                         logged_run_minutes_under_load=round(lsec / 60, 1) if lsec == lsec else "n/a", logged_runs=nruns,
                         inference_ms_per_network_hour=round(1000 * infer_s / len(ts), 1),
                         SCT_seconds_full_record_one_variable=round(sct_s, 2), OI_seconds_full_record_one_variable=round(oi_s, 1) if corpus != "NA" else "n/a"))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "summary", "view_compute.csv"), index=False)

if __name__ == "__main__":
    main()
