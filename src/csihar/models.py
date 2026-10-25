"""CSI encoder and self-supervised heads."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout, k, stride):
    return nn.Sequential(
        nn.Conv1d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
        nn.BatchNorm1d(cout),
        nn.GELU(),
    )


class CSIEncoder(nn.Module):
    """Temporal 1-D CNN over a (T, 256) amplitude window.

    Subcarriers are treated as input channels so the first layer learns
    spectral filters; the stack then models temporal dynamics. Mean and
    standard-deviation pooling over time give a length-agnostic embedding
    (static channel shape + motion energy).
    """

    def __init__(self, n_sub=256, width=128, emb_dim=128):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv1d(n_sub, width, 1, bias=False), nn.BatchNorm1d(width), nn.GELU())
        self.temporal = nn.Sequential(
            _block(width, width, 7, 1),
            _block(width, width, 5, 2),
            _block(width, width, 5, 2),
            _block(width, width, 3, 2),
        )
        self.out = nn.Linear(2 * width, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):            # x: (B, T, S)
        h = self.temporal(self.spectral(x.transpose(1, 2)))
        pooled = torch.cat([h.mean(-1), h.std(-1)], dim=1)
        return self.out(pooled)


class ProjectionHead(nn.Module):
    def __init__(self, emb_dim=128, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, proj_dim))

    def forward(self, z):
        return self.net(z)


def nt_xent(z1, z2, tau=0.2):
    """SimCLR NT-Xent loss for two batches of paired projections."""
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / tau
    n = z1.shape[0]
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)
