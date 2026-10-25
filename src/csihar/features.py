"""Handcrafted CSI window features (baseline for Tasks 1-3)."""
import numpy as np


def handcrafted(Z: np.ndarray, n_bands: int = 32) -> np.ndarray:
    """Features from normalised windows Z (N, T, 256).

    Subcarriers are averaged into `n_bands` contiguous groups to keep the
    feature count moderate. Per band: temporal mean (static channel shape),
    std and mean |first difference| (motion energy), and the max-min range.
    Global descriptors: dominant-PC variance ratio and spectral energy of the
    band-averaged signal in four Doppler-like frequency ranges.
    """
    N, T, S = Z.shape
    B = Z.reshape(N, T, n_bands, S // n_bands).mean(-1)          # (N, T, n_bands)
    mean = B.mean(1)
    std = B.std(1)
    diff = np.abs(np.diff(B, axis=1)).mean(1)
    rng = B.max(1) - B.min(1)
    centred = B - mean[:, None, :]
    spec = np.abs(np.fft.rfft(centred, axis=1)) ** 2              # (N, F, n_bands)
    F = spec.shape[1]
    edges = np.linspace(1, F, 5).astype(int)
    bands = np.stack([spec[:, a:b].sum(1).mean(-1) for a, b in zip(edges[:-1], edges[1:])], 1)
    bands = np.log1p(bands)
    sv = np.linalg.svd(centred, compute_uv=False)                 # (N, min(T, n_bands))
    pc_ratio = (sv[:, :1] ** 2) / np.maximum((sv ** 2).sum(1, keepdims=True), 1e-9)
    return np.concatenate([mean, std, diff, rng, bands, pc_ratio], 1).astype(np.float32)
