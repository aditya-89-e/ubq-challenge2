"""Task 1: room identification on fixed 150x256 validation windows.

Outputs
  workspace/artifacts/router.pkl          router used by Task 3 and run_episode
  results/task1_results.csv               experiment,target_room,K,metric,value
  results/task1_window_length.csv         accuracy vs window length
  results/task1_predictions.csv           window_id,room,label,confidence
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csihar.config import load_config, resolve  # noqa: E402
from csihar.metrics import ResultLog, accuracy  # noqa: E402
from csihar.pipeline import artifacts_dir, load_normalized, representations  # noqa: E402
from csihar.room import RoomRouter, subwindows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res_dir = resolve(cfg, "results")
    res_dir.mkdir(parents=True, exist_ok=True)
    log = ResultLog(res_dir / "task1_results.csv")

    Zd, md, Zv, mv, _ = load_normalized(cfg)
    yd, yv = md.room.values, mv.room.values

    # Headline: static-fingerprint router on full 150-row windows.
    router = RoomRouter().fit(Zd, yd)
    pred, conf = router.predict(Zv)
    acc = accuracy(yv, pred)
    log.add("task1_room_fingerprint", "accuracy", acc, notes="profile-LR, 150x256 val windows")
    for r in ("A", "B"):
        m = yv == r
        log.add("task1_room_fingerprint", "accuracy", accuracy(yv[m], pred[m]), target_room=r)
    router.save(artifacts_dir(cfg) / "router.pkl")
    pd.DataFrame(dict(window_id=mv.window_id, room=pred, label="", confidence=conf.round(4),
                      true_room=yv)).to_csv(res_dir / "task1_predictions.csv", index=False)
    print(f"router accuracy on validation: {acc:.4f}")

    # Same task from frozen representations (linear probe on room labels).
    for kind, (Ed, Ev) in representations(cfg, Zd, Zv, md).items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=5000)).fit(Ed, yd)
        a = accuracy(yv, clf.predict(Ev))
        log.add(f"task1_room_probe_{kind}", "accuracy", a)
        print(f"room probe on {kind}: {a:.4f}")

    # Accuracy vs window length, using the fixed validation capture pool:
    # train on random sub-windows of development windows, test on the centred
    # sub-window of each validation window.
    rng = np.random.default_rng(0)
    rows = []
    for L in cfg["task1"]["window_lengths"]:
        r = RoomRouter().fit(subwindows(Zd, L, centre=False, rng=rng), yd)
        p, _ = r.predict(subwindows(Zv, L, centre=True))
        rows.append(dict(window_length=L, accuracy=accuracy(yv, p),
                         acc_A=accuracy(yv[yv == "A"], p[yv == "A"]),
                         acc_B=accuracy(yv[yv == "B"], p[yv == "B"])))
        log.add("task1_window_length", "accuracy", rows[-1]["accuracy"], notes=f"L={L}")
        print(rows[-1])
    pd.DataFrame(rows).to_csv(res_dir / "task1_window_length.csv", index=False)
    log.save()


if __name__ == "__main__":
    main()
