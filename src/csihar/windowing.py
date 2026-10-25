"""Deterministic 150 x 256 window extraction from common-form captures.

Policy (documented in the report):
  * validation captures -> exactly one window, centred in the capture
    (start = (n - L) // 2). One window per capture keeps the validation set
    capture-level and independent of capture length.
  * development captures -> sliding windows of length L with a fixed stride,
    starting at row 0. Extra windows are augmentation for training only; K
    in Task 3 still counts captures, not windows.
  * captures shorter than L (all of Room A after removing legacy frames:
    ~70-127 valid rows) -> one window obtained by linear interpolation along
    time onto L evenly spaced points ("resample"), or by repeating the first/
    last row ("edge"). Resampling is the default because edge padding leaves
    a constant block whose length encodes capture length, which is a
    forbidden metadata cue for the room classifier.
Windows never mix rows from different captures.
"""
import numpy as np


def window_starts(n, length, stride, split):
    if n < length:
        return []
    if split == "validation":
        return [(n - length) // 2]
    return list(range(0, n - length + 1, stride))


def fit_short(amp, length, policy):
    n = amp.shape[0]
    if policy == "edge":
        before = (length - n) // 2
        return np.pad(amp, ((before, length - n - before), (0, 0)), mode="edge")
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, length)
    out = np.empty((length, amp.shape[1]), np.float32)
    for j in range(amp.shape[1]):
        out[:, j] = np.interp(dst, src, amp[:, j])
    return out


def extract_windows(amp, length, stride, split, short_policy):
    """Return list of (window, start, end, method) for one capture."""
    n = amp.shape[0]
    if n >= length:
        return [(amp[s:s + length].astype(np.float32), s, s + length, "slice")
                for s in window_starts(n, length, stride, split)]
    return [(fit_short(amp, length, short_policy), 0, n, short_policy)]
