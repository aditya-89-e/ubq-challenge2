"""Window export (window_XXXXXX.npy + provenance index) and loading."""
from pathlib import Path

import numpy as np
import pandas as pd

from .amplitude import read_common_csv
from .config import resolve, section_hash
from .windowing import extract_windows

SPLIT_DIR = {"development": "development", "validation": "local_validation"}


def build_windows(cfg, manifest=None):
    wcfg = cfg["window"]
    work = resolve(cfg, "work")
    out = resolve(cfg, "windows")
    if manifest is None:
        manifest = pd.read_csv(work / "manifest.csv", keep_default_na=False)
    chash = section_hash(cfg, "qc", "window")
    rows, arrays = [], {"development": [], "validation": []}
    wid = 0
    for rec in manifest.itertuples(index=False):
        if rec.error or rec.n_valid < wcfg["min_valid"]:
            continue
        amp = read_common_csv(rec.intermediate)
        for win, start, end, method in extract_windows(
                amp, wcfg["length"], wcfg["dev_stride"], rec.split, wcfg["short_policy"]):
            wid += 1
            name = f"window_{wid:06d}"
            path = out / SPLIT_DIR[rec.split] / f"{name}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, win.astype(np.float32))
            arrays[rec.split].append(win.astype(np.float32))
            rows.append(dict(window_id=name, capture_id=rec.capture_id, room=rec.room, label=rec.label,
                             split=rec.split, row_start=start, row_end=end, n_valid=rec.n_valid,
                             method=method, source=rec.source, intermediate=rec.intermediate,
                             config_hash=chash, path=str(path)))
    index = pd.DataFrame(rows)
    index.to_csv(out / "index.csv", index=False)
    for split, arr in arrays.items():
        np.save(out / f"cache_{split}.npy", np.stack(arr))
    return index


def load_split(cfg, split):
    """Return (X, meta) for 'development' or 'validation' from the stacked cache."""
    out = resolve(cfg, "windows")
    index = pd.read_csv(out / "index.csv")
    meta = index[index.split == split].reset_index(drop=True)
    X = np.load(out / f"cache_{split}.npy", mmap_mode="r")
    assert len(X) == len(meta)
    return np.asarray(X), meta
