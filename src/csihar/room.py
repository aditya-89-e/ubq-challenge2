"""Task 1: recorded-environment fingerprint (Room A vs Room B) / router.

Feature: the time-averaged, gain-normalised, globally standardised amplitude
profile over the 256 subcarriers (static multipath + frequency-selective
fading signature), plus its temporal spread. It is computed purely from the
window content; capture length, file type and path are never seen.
"""
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def room_features(Z):
    """Z: (N, T, 256) globally normalised windows of any length T >= 1."""
    mean = Z.mean(1)
    spread = Z.std(1) if Z.shape[1] > 1 else np.zeros_like(mean)
    return np.concatenate([mean, spread], 1)


class RoomRouter:
    def __init__(self, C=0.05):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))

    def fit(self, Z, rooms):
        self.model.fit(room_features(Z), rooms)
        return self

    def predict(self, Z):
        F = room_features(Z)
        p = self.model.predict_proba(F)
        return self.model.classes_[p.argmax(1)], p.max(1)

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


def subwindows(Z, length, centre=True, rng=None):
    """Take one sub-window of `length` rows from each window (centred or random)."""
    T = Z.shape[1]
    if length >= T:
        return Z
    if centre:
        s = (T - length) // 2
        return Z[:, s:s + length]
    starts = rng.integers(0, T - length + 1, len(Z))
    return np.stack([z[s:s + length] for z, s in zip(Z, starts)])
