"""Leakage-safe normalisation of amplitude windows.

Stages (configurable in experiment.yaml -> normalization):
 1. per-packet gain: divide each observation by its RMS over data bins. The
    4366c0 export rescales every packet by a power of two ("autoscale") and
    the receiver AGC also varies, so absolute packet level is not a stable
    physical quantity. Stateless, so no leakage risk.
 2. optional log1p.
 3. per-subcarrier standardisation with mean/std fitted on DEVELOPMENT
    windows only -- pooled over both rooms ("global") or per room ("room").
 4. null bins (guard band, DC) -- identified from development data -- are
    set to 0 after standardisation so their noise is not amplified to unit
    variance. The array keeps all 256 columns.
"""
import json

import numpy as np


class Normalizer:
    def __init__(self, per_packet_gain=True, log=False, scope="global", null_bin_rel_level=0.1):
        self.per_packet_gain = per_packet_gain
        self.log = log
        self.scope = scope
        self.null_bin_rel_level = null_bin_rel_level
        self.null_mask = None
        self.stats = {}

    # -- stateless part -------------------------------------------------
    def _pre(self, X):
        X = X.astype(np.float32, copy=True)
        if self.per_packet_gain:
            data = ~self.null_mask if self.null_mask is not None else slice(None)
            rms = np.sqrt((X[..., data] ** 2).mean(-1, keepdims=True))
            X /= np.maximum(rms, 1e-6)
        if self.log:
            X = np.log1p(X)
        return X

    # -- fitting on development data only -------------------------------
    def fit(self, X, rooms):
        """X: (N, T, 256) development windows; rooms: (N,) array of 'A'/'B'."""
        rooms = np.asarray(rooms)
        med = {}
        for r in np.unique(rooms):
            prof = np.median(X[rooms == r].mean(1), axis=0)
            med[r] = prof / np.median(prof)
        self.null_mask = np.zeros(X.shape[-1], bool)
        for prof in med.values():
            self.null_mask |= prof < self.null_bin_rel_level
        Z = self._pre(X)
        groups = {"global": np.ones(len(rooms), bool)}
        for r in np.unique(rooms):
            groups[r] = rooms == r
        for key, m in groups.items():
            flat = Z[m].reshape(-1, Z.shape[-1])
            self.stats[key] = (flat.mean(0), flat.std(0) + 1e-6)
        return self

    def transform(self, X, rooms=None, scope=None):
        scope = scope or self.scope
        Z = self._pre(X)
        if scope == "global" or rooms is None:
            mu, sd = self.stats["global"]
            Z = (Z - mu) / sd
        else:
            rooms = np.asarray(rooms)
            for r in np.unique(rooms):
                mu, sd = self.stats[r]
                Z[rooms == r] = (Z[rooms == r] - mu) / sd
        Z[..., self.null_mask] = 0.0
        return Z.astype(np.float32)

    # -- persistence -----------------------------------------------------
    def to_dict(self):
        return dict(per_packet_gain=self.per_packet_gain, log=self.log, scope=self.scope,
                    null_bin_rel_level=self.null_bin_rel_level,
                    null_mask=self.null_mask.tolist(),
                    stats={k: [m.tolist(), s.tolist()] for k, (m, s) in self.stats.items()})

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            d = json.load(fh)
        n = cls(d["per_packet_gain"], d["log"], d["scope"], d["null_bin_rel_level"])
        n.null_mask = np.array(d["null_mask"], bool)
        n.stats = {k: (np.array(m, np.float32), np.array(s, np.float32)) for k, (m, s) in d["stats"].items()}
        return n
