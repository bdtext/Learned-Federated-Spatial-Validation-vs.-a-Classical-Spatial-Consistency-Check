"""Cross-silo federated training of the shared backbone, aggregation rules, and model-poisoning attacks.
Silos train local epochs on their own stations with their own graph (confined); only shared parameters are exchanged.
"""
import copy
import numpy as np
import torch
from .model import LSOSTGNN, train_model

def shared_names(model):
    return [n for n, _ in model.named_parameters() if n not in ("emb", "logits", "shrink")]

def get_shared(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if n not in ("emb", "logits", "shrink")}

def set_shared(model, sd):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in sd:
                p.copy_(sd[n])

def flatten(sd, names):
    return torch.cat([sd[n].reshape(-1) for n in names])

def unflatten(vec, template, names):
    out, i = {}, 0
    for n in names:
        k = template[n].numel(); out[n] = vec[i:i + k].reshape(template[n].shape); i += k
    return out

# ---------------------------------------------------------------- aggregation rules (on update vectors)
def agg_mean(U, weights=None):
    if weights is None:
        return U.mean(0)
    w = torch.tensor(weights, dtype=U.dtype) / sum(weights)
    return (U * w[:, None]).sum(0)

def agg_median(U):
    return U.median(0).values

def agg_multikrum(U, f):
    """Multi-Krum: score each update by the sum of squared distances to its K-f-2 nearest peers; average the K-f lowest."""
    K = U.shape[0]
    m = K - f - 2
    if m < 1:
        raise ValueError("multi-Krum requires K >= f + 3")
    D = torch.cdist(U, U) ** 2
    scores = torch.stack([torch.sort(D[k][torch.arange(K) != k]).values[:m].sum() for k in range(K)])
    sel = torch.argsort(scores)[: K - f]
    return U[sel].mean(0), sel

def agg_fltrust(U, u0):
    """FLTrust: trust = ReLU(cos(u_k, u0)); updates normalized to |u0|; weighted mean."""
    n0 = u0.norm().clamp_min(1e-12)
    ts = torch.stack([torch.relu(torch.dot(u, u0) / (u.norm().clamp_min(1e-12) * n0)) for u in U])
    if ts.sum() <= 0:
        return torch.zeros_like(u0), ts
    Un = torch.stack([u * (n0 / u.norm().clamp_min(1e-12)) for u in U])
    return (Un * ts[:, None]).sum(0) / ts.sum(), ts

# ---------------------------------------------------------------- attacks (on the adversary's update vectors)
def attack_signflip(u, scale=5.0):
    return -scale * u

def attack_alie(U_honest_est, z=1.0):
    """'A little is enough' (Baruch et al. 2019): mu - z*sigma per coordinate, using the attacker's estimate of honest statistics."""
    mu = U_honest_est.mean(0); sd = U_honest_est.std(0)
    return mu - z * sd

def attack_minmax(U_honest_est, dev="sd", gamma_max=20.0, iters=20):
    """Min-max (Shejwalkar & Houmansadr 2021): mu - gamma * dev, with the largest gamma such that the malicious update is
    no farther from any honest update than the maximum pairwise honest distance."""
    mu = U_honest_est.mean(0)
    d = U_honest_est.std(0) if dev == "sd" else torch.sign(mu)
    maxd = torch.cdist(U_honest_est, U_honest_est).max()
    lo, hi = 0.0, gamma_max
    for _ in range(iters):
        g = (lo + hi) / 2
        m = mu - g * d
        if torch.cdist(m[None], U_honest_est).max() <= maxd:
            lo = g
        else:
            hi = g
    return mu - lo * d

# ---------------------------------------------------------------- federated round loop
def federated_train(make_model, silos, a, fetch, train_ts, val_ts, rounds=15, local_epochs=1, lr=2e-3, batch=128, seed=0,
                    rule="mean", f=0, adversaries=(), attack=None, attack_kw=None, root=None, verbose=True, log=None, fetch_by_silo=None):
    """silos: list of dicts {name, nodes (indices into the global network), model (LSOSTGNN built on the silo's own graph)}.
    Shared parameters are averaged; local parameters stay. Returns the global shared state and per-round history.
    The adversary set is a list of silo indices; attack(u_k, U_honest_est) returns the malicious update.
    root: optional (model, fetch, train_ts, a_root) tuple for FLTrust's server update."""
    names = shared_names(silos[0]["model"])
    g = get_shared(silos[0]["model"])  # common init: all silos start from silo 0's shared init
    for s in silos:
        set_shared(s["model"], g)
    hist = []
    for r in range(rounds):
        updates, nks = [], []
        for k, s in enumerate(silos):
            set_shared(s["model"], g)
            before = flatten(get_shared(s["model"]), names)
            fs = fetch_by_silo[k] if fetch_by_silo is not None else fetch
            train_model(s["model"], fs, s["a"], None, train_ts, val_ts, epochs=local_epochs, lr=lr, batch=batch, patience=10 ** 6,
                        seed=seed * 1000 + r, verbose=False, shared_only=False)
            after = flatten(get_shared(s["model"]), names)
            updates.append(after - before); nks.append(len(s["nodes"]))
        U = torch.stack(updates)
        honest = [k for k in range(len(silos)) if k not in adversaries]
        if attack is not None and len(adversaries):
            est = U[honest]  # omniscient adversary estimates honest statistics from honest updates
            for k in adversaries:
                if attack == "signflip":
                    U[k] = attack_signflip(U[k], **(attack_kw or {}))
                elif attack == "alie":
                    U[k] = attack_alie(est, **(attack_kw or {}))
                elif attack == "minmax":
                    U[k] = attack_minmax(est, **(attack_kw or {}))
        info = {}
        if rule == "mean":
            agg = agg_mean(U, nks)
        elif rule == "median":
            agg = agg_median(U)
        elif rule == "multikrum":
            agg, sel = agg_multikrum(U, f); info["selected"] = sel.tolist()
        elif rule == "fltrust":
            rm, rfetch, rts = root[:3]  # (root model, fetch, train hours, root anomalies)
            set_shared(rm, g); b0 = flatten(get_shared(rm), names)
            train_model(rm, rfetch, root[3], None, rts, val_ts, epochs=local_epochs, lr=lr, batch=batch, patience=10 ** 6, seed=seed * 1000 + r, verbose=False)
            u0 = flatten(get_shared(rm), names) - b0
            agg, ts = agg_fltrust(U, u0); info["trust"] = ts.tolist()
        g = {n: g[n] + v for n, v in unflatten(agg, g, names).items()}
        hist.append(dict(round=r + 1, update_norms=[float(u.norm()) for u in U], **info))
        if verbose: print(f"  round {r+1}: |u| = {[round(float(u.norm()),3) for u in U]}", flush=True)
        if log is not None: log(hist[-1])
    for s in silos:
        set_shared(s["model"], g)
    return g, hist
