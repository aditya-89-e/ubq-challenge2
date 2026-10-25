"""Task 2: frozen-representation linear probes.

Representations: SSL encoder, the same architecture with random weights, and
handcrafted features. A linear (logistic-regression) classifier is fitted on
labelled DEVELOPMENT windows and scored on the fixed VALIDATION windows:
  * per room   (Room A: 6 activities, Room B: 5 activities)
  * joint      (both rooms, 8 activity labels)
  * cross-room shared (train SIT/STAND/WALK on one room, test on the other)
Also writes 2-D embeddings for the encoder-diagnosis figure.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csihar.config import SHARED, load_config, resolve  # noqa: E402
from csihar.metrics import ResultLog, balanced_acc, macro_f1  # noqa: E402
from csihar.pipeline import load_normalized, representations  # noqa: E402


def probe(Etr, ytr, Ete):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))
    return clf.fit(Etr, ytr).predict(Ete)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = resolve(cfg, "results")
    log = ResultLog(res_dir / "task2_results.csv")

    Zd, md, Zv, mv, _ = load_normalized(cfg)
    reps = representations(cfg, Zd, Zv, md)
    for kind, (Ed, Ev) in reps.items():
        settings = {"room_A": ("A", "A"), "room_B": ("B", "B"), "joint": (None, None),
                    "shared_A_to_B": ("A", "B"), "shared_B_to_A": ("B", "A")}
        for name, (src, tgt) in settings.items():
            d = np.ones(len(md), bool) if src is None else (md.room == src).values
            v = np.ones(len(mv), bool) if tgt is None else (mv.room == tgt).values
            if name.startswith("shared"):
                d = d & md.label.isin(SHARED).values
                v = v & mv.label.isin(SHARED).values
            p = probe(Ed[d], md.label.values[d], Ev[v])
            y = mv.label.values[v]
            f1, ba = macro_f1(y, p), balanced_acc(y, p)
            log.add(f"task2_probe_{kind}", "macro_f1", f1, target_room=name)
            log.add(f"task2_probe_{kind}", "balanced_accuracy", ba, target_room=name)
            print(f"{kind:12s} {name:14s} macroF1 {f1:.3f}  balAcc {ba:.3f}")
    log.save()

    # 2-D embedding of the validation windows (qualitative diagnosis only).
    rows = []
    for kind in ("ssl", "random"):
        Ev = StandardScaler().fit_transform(reps[kind][1])
        try:
            import umap  # noqa: F401
            from umap import UMAP
            xy = UMAP(n_neighbors=15, min_dist=0.2, random_state=0).fit_transform(Ev)
            method = "umap"
        except Exception:
            xy = TSNE(perplexity=30, random_state=0, init="pca").fit_transform(Ev)
            method = "tsne"
        rows.append(pd.DataFrame(dict(kind=kind, method=method, x=xy[:, 0], y=xy[:, 1],
                                      room=mv.room, label=mv.label, window_id=mv.window_id)))
    pd.concat(rows).to_csv(res_dir / "task2_embedding_2d.csv", index=False)


if __name__ == "__main__":
    main()
