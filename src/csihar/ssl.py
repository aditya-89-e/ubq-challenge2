"""Self-supervised (SimCLR-style) training of the CSI encoder.

No activity or room labels are used. Sampling is done per *capture* (capture
identity is provenance, not a label): each step draws captures uniformly and
takes two windows of the same capture (different temporal positions when the
capture has several) as the positive pair, then augments each view:

  * random temporal crop (same length for the batch, 96..150 rows)
  * random global gain and smooth per-subcarrier gain tilt (AGC / multipath
    level changes that should not change the activity)
  * additive Gaussian noise
  * random contiguous subcarrier-band masking and time masking

Time reversal is deliberately NOT used: SIT-DOWN and STAND-UP are near time
reversals of each other.
"""
import numpy as np
import torch

from .models import CSIEncoder, ProjectionHead, nt_xent


def _augment(x, rng, crop_len):
    B, T, S = x.shape
    out = np.empty((B, crop_len, S), np.float32)
    for i in range(B):
        s = rng.integers(0, T - crop_len + 1)
        v = x[i, s:s + crop_len].copy()
        v *= rng.uniform(0.8, 1.25)
        tilt = np.interp(np.arange(S), np.linspace(0, S - 1, 6), rng.normal(0, 0.15, 6))
        v += tilt[None, :].astype(np.float32)
        v += rng.normal(0, 0.1, v.shape).astype(np.float32)
        w = rng.integers(8, 48)
        b0 = rng.integers(0, S - w)
        v[:, b0:b0 + w] = 0.0
        tw = rng.integers(0, crop_len // 6)
        t0 = rng.integers(0, crop_len - tw + 1)
        v[t0:t0 + tw] = 0.0
        out[i] = v
    return out


def train_ssl(Z, capture_ids, cfg, log=print):
    """Z: (N, T, S) normalised development windows; capture_ids: (N,)."""
    sc = cfg["ssl"]
    torch.manual_seed(sc["seed"])
    rng = np.random.default_rng(sc["seed"])
    caps = np.asarray(capture_ids)
    uniq, inv = np.unique(caps, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]

    enc = CSIEncoder(Z.shape[-1], sc["width"], sc["emb_dim"])
    head = ProjectionHead(sc["emb_dim"])
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=sc["lr"], weight_decay=sc["weight_decay"])
    steps_per_epoch = max(1, len(uniq) // sc["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, sc["lr"], total_steps=sc["epochs"] * steps_per_epoch)
    history = []
    for ep in range(sc["epochs"]):
        enc.train(); head.train()
        perm = rng.permutation(len(uniq))
        losses = []
        for b in range(steps_per_epoch):
            ks = perm[b * sc["batch_size"]:(b + 1) * sc["batch_size"]]
            i1 = np.array([rng.choice(members[k]) for k in ks])
            i2 = np.array([rng.choice(members[k]) for k in ks])
            L = int(rng.integers(96, Z.shape[1] + 1))
            v1 = torch.from_numpy(_augment(Z[i1], rng, L))
            v2 = torch.from_numpy(_augment(Z[i2], rng, L))
            loss = nt_xent(head(enc(v1)), head(enc(v2)), sc["temperature"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
        if ep % 5 == 0 or ep == sc["epochs"] - 1:
            log(f"  ssl epoch {ep:3d} loss {history[-1]:.4f}")
    enc.eval()
    return enc, history


@torch.no_grad()
def embed(enc, Z, batch=256):
    enc.eval()
    out = [enc(torch.from_numpy(np.ascontiguousarray(Z[i:i + batch]))).numpy()
           for i in range(0, len(Z), batch)]
    return np.concatenate(out).astype(np.float32)
