"""Metric helpers and result logging (experiment, target_room, K, metric, value)."""
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def macro_f1(y, p, labels=None):
    return float(f1_score(y, p, labels=labels, average="macro", zero_division=0))


def balanced_acc(y, p):
    return float(balanced_accuracy_score(y, p))


def accuracy(y, p):
    return float(np.mean(np.asarray(y) == np.asarray(p)))


class ResultLog:
    """Append-only CSV with the required columns plus free-form extras."""
    FIELDS = ["experiment", "target_room", "K", "metric", "value", "seed", "notes"]

    def __init__(self, path):
        self.path = Path(path)
        self.rows = []

    def add(self, experiment, metric, value, target_room="", K="", seed="", notes=""):
        self.rows.append(dict(experiment=experiment, target_room=target_room, K=K,
                              metric=metric, value=round(float(value), 4), seed=seed, notes=notes))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.FIELDS)
            w.writeheader()
            w.writerows(self.rows)
