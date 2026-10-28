"""Leave-self-out spatiotemporal graph validator (LSO-STGNN), PyTorch, CPU.
Inputs per station and hour: a 24-h window of standardized anomalies of the K validated variables (NaN -> 0 plus a missing mask),
sine/cosine of hour-of-day and day-of-year. Backbone: dilated causal temporal convolution -> graph attention over a hybrid
adjacency (correlation prior x learnable logits) with station embeddings -> forecast head (residual around persistence) and a
leave-self-out spatial head (weighted neighbour mean with learnable logits and shrinkage + attention correction whose diagonal is
masked and whose query is the station embedding alone, consuming pre-attention encodings only).
Shared parameters = encoder, attention, heads; local parameters = station embeddings, adjacency logits, shrinkage.
"""
import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

class TCN(nn.Module):
    def __init__(self, fin, ch=32, dil=(1, 2, 4, 8), k=3):
        super().__init__()
        self.inp = nn.Conv1d(fin, ch, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(ch, ch, k, dilation=d, padding=(k - 1) * d) for d in dil])
        self.dil = dil; self.k = k
    def forward(self, x):  # x: (B, F, W)
        h = self.inp(x)
        for conv, d in zip(self.blocks, self.dil):
            y = conv(h)[:, :, : h.shape[-1]]  # causal crop
            h = h + F.gelu(y)
        return h[:, :, -1]  # (B, ch) encoding at the last hour

class LSOSTGNN(nn.Module):
    def __init__(self, N, K, prior, ch=32, heads=4, emb=16, hidden=32):
        super().__init__()
        self.N, self.K = N, K
        fin = 2 * K + 4
        self.tcn = TCN(fin, ch)
        self.emb = nn.Parameter(torch.randn(N, emb) * 0.1)            # local
        self.logits = nn.Parameter(torch.zeros(K, N, N))               # local (per-variable adjacency refinement)
        self.shrink = nn.Parameter(torch.ones(K, N))                   # local
        self.register_buffer("prior", torch.tensor(prior, dtype=torch.float32))  # (K, N, N), zero diagonal
        d = ch + emb
        self.q = nn.Linear(d, heads * 8); self.k = nn.Linear(d, heads * 8); self.v = nn.Linear(ch, heads * 8)
        self.heads = heads
        self.out = nn.Linear(heads * 8, ch)
        self.fc_head = nn.Sequential(nn.Linear(ch, hidden), nn.GELU(), nn.Linear(hidden, K))
        # leave-self-out correction: query from embedding only, keys/values from neighbours' pre-attention encodings
        self.sq = nn.Linear(emb, heads * 8); self.sk = nn.Linear(d, heads * 8); self.sv = nn.Linear(ch, heads * 8)
        self.sp_head = nn.Sequential(nn.Linear(heads * 8, hidden), nn.GELU(), nn.Linear(hidden, K))

    def adjacency(self):
        A = self.prior * torch.exp(self.logits)
        A = A / A.sum(-1, keepdim=True).clamp_min(1e-9)
        return A  # (K, N, N)

    def shared_parameters(self):
        return [p for n, p in self.named_parameters() if n not in ("emb", "logits", "shrink")]

    def forward(self, xw, a_last, mask_last):
        """xw: (B, N, F, W) windows; a_last: (B, N, K) anomalies at the last hour (NaN->0); mask_last: (B, N, K) observed.
        Returns forecast (B,N,K) of the next-hour anomaly and spatial estimate (B,N,K) of the current anomaly."""
        B, N, Fd, W = xw.shape
        h = self.tcn(xw.reshape(B * N, Fd, W)).reshape(B, N, -1)                    # (B, N, ch)
        A = self.adjacency()                                                           # (K, N, N)
        bias = torch.log(A.mean(0) + 1e-4)                                             # (N, N)
        e = self.emb.unsqueeze(0).expand(B, N, -1)
        he = torch.cat([h, e], -1)
        q = self.q(he).reshape(B, N, self.heads, 8); k = self.k(he).reshape(B, N, self.heads, 8); v = self.v(h).reshape(B, N, self.heads, 8)
        att = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(8) + bias.unsqueeze(0).unsqueeze(0)
        att = att.softmax(-1)
        z = torch.einsum("bhnm,bmhd->bnhd", att, v).reshape(B, N, -1)
        z = h + F.gelu(self.out(z))
        fc = a_last + self.fc_head(z)                                                  # residual around persistence
        # leave-self-out spatial head
        Am = a_last * mask_last                                                        # (B, N, K)
        num = torch.einsum("knm,bmk->bnk", A, Am)
        den = torch.einsum("knm,bmk->bnk", A, mask_last).clamp_min(1e-6)
        lin = self.shrink.t().unsqueeze(0) * num / den
        sq = self.sq(self.emb).reshape(N, self.heads, 8).unsqueeze(0).expand(B, -1, -1, -1)
        sk = self.sk(he).reshape(B, N, self.heads, 8); sv = self.sv(h).reshape(B, N, self.heads, 8)
        satt = torch.einsum("bnhd,bmhd->bhnm", sq, sk) / math.sqrt(8) + bias.unsqueeze(0).unsqueeze(0)
        eye = torch.eye(N, dtype=torch.bool, device=xw.device)
        satt = satt.masked_fill(eye.unsqueeze(0).unsqueeze(0), float("-inf")).softmax(-1)
        c = torch.einsum("bhnm,bmhd->bnhd", satt, sv).reshape(B, N, -1)
        sp = lin + self.sp_head(c)
        return fc, sp

def make_windows(a, mask, times_feats, W=24):
    """a: (K, N, T) standardized anomalies with NaN; returns function to fetch batch windows ending at hour t."""
    K, N, T = a.shape
    A = np.nan_to_num(a, nan=0.0).astype(np.float32)
    Mk = (~np.isnan(a)).astype(np.float32)
    def fetch(ts):
        xw = np.stack([np.concatenate([A[:, :, t - W + 1:t + 1], Mk[:, :, t - W + 1:t + 1], np.broadcast_to(times_feats[t - W + 1:t + 1].T[:, None, :], (4, N, W))], 0) for t in ts])  # (B, F, N, W)
        xw = np.transpose(xw, (0, 2, 1, 3))  # (B, N, F, W)
        a_last = np.transpose(A[:, :, ts], (2, 1, 0)); m_last = np.transpose(Mk[:, :, ts], (2, 1, 0))
        return torch.from_numpy(xw), torch.from_numpy(a_last), torch.from_numpy(m_last)
    return fetch

def time_features(times):
    import pandas as pd
    t = pd.DatetimeIndex(times)
    hod = t.hour / 24.0 * 2 * np.pi; doy = (t.dayofyear - 1) / 365.25 * 2 * np.pi
    return np.stack([np.sin(hod), np.cos(hod), np.sin(doy), np.cos(doy)], 1).astype(np.float32)  # (T, 4)

def huber(x, y, m):
    l = F.huber_loss(x, y, reduction="none", delta=1.0) * m
    return l.sum() / m.sum().clamp_min(1.0)

def train_model(model, fetch, a, mask, train_ts, val_ts, W=24, epochs=30, lr=2e-3, wd=1e-4, batch=128, patience=4, seed=0,
                r_sp=1.0, verbose=True, log=None, shared_only=False, params=None):
    """Train with AdamW, cosine decay, grad clipping, early stopping on validation task loss. Returns history."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    K, N, T = a.shape
    A = torch.from_numpy(np.nan_to_num(a, nan=0.0).astype(np.float32)); Mk = torch.from_numpy((~np.isnan(a)).astype(np.float32))
    ps = params if params is not None else (model.shared_parameters() if shared_only else list(model.parameters()))
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=wd)
    steps = epochs * int(np.ceil(len(train_ts) / batch))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    best, best_state, bad, hist = np.inf, None, 0, []
    def eval_loss(ts):
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ts), batch):
                b = ts[i:i + batch]
                xw, al, ml = fetch(b)
                fc, sp = model(xw, al, ml)
                y1 = A[:, :, b + 1].permute(2, 1, 0); m1 = Mk[:, :, b + 1].permute(2, 1, 0)
                l = huber(fc, y1, m1) + r_sp * huber(sp, al, ml)
                tot += float(l) * len(b); n += len(b)
        return tot / max(n, 1)
    for ep in range(epochs):
        model.train(); order = rng.permutation(train_ts); tl, nb = 0.0, 0
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            xw, al, ml = fetch(b)
            fc, sp = model(xw, al, ml)
            y1 = A[:, :, b + 1].permute(2, 1, 0); m1 = Mk[:, :, b + 1].permute(2, 1, 0)
            loss = huber(fc, y1, m1) + r_sp * huber(sp, al, ml)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(ps, 1.0); opt.step(); sched.step()
            tl += float(loss); nb += 1
        vl = eval_loss(val_ts)
        hist.append(dict(epoch=ep + 1, train_loss=tl / max(nb, 1), val_loss=vl))
        if verbose: print(f"  epoch {ep+1} train {tl/max(nb,1):.4f} val {vl:.4f}", flush=True)
        if log is not None: log(hist[-1])
        if vl < best - 1e-5:
            best, bad = vl, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return hist

def residuals(model, fetch, a, ts_all, batch=256):
    """Forecast innovation r_fc[t+1] = a[t+1] - fc(t) and spatial innovation r_sp[t] = a[t] - sp(t); returns (K,N,T) arrays."""
    K, N, T = a.shape
    rfc = np.full((K, N, T), np.nan, dtype=np.float32); rsp = np.full((K, N, T), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(ts_all), batch):
            b = ts_all[i:i + batch]
            xw, al, ml = fetch(b)
            fc, sp = model(xw, al, ml)
            fc = fc.numpy().transpose(2, 1, 0); sp = sp.numpy().transpose(2, 1, 0)  # (K,N,B)
            nxt = b + 1; ok = nxt < T
            rfc[:, :, nxt[ok]] = a[:, :, nxt[ok]] - fc[:, :, ok]
            rsp[:, :, b] = a[:, :, b] - sp
    return rfc, rsp
