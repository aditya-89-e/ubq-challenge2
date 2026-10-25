"""Stage 1: raw sources -> intermediate CSVs -> 150x256 windows (+ provenance).

    python scripts/build_dataset.py [--config configs/experiment.yaml] [--skip-ingest]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csihar.config import load_config, resolve  # noqa: E402
from csihar.dataset import build_windows  # noqa: E402
from csihar.ingest import build_intermediate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.skip_ingest:
        manifest = pd.read_csv(resolve(cfg, "work") / "manifest.csv", keep_default_na=False)
    else:
        manifest = build_intermediate(cfg, args.workers)
    ok = manifest[manifest.error == ""]
    print("captures per room/split/label:")
    print(ok.groupby(["room", "split", "label"]).size().unstack(fill_value=0))
    print("errors:", (manifest.error != "").sum())
    print(manifest[manifest.error != ""][["capture_id", "error"]].to_string())
    qc_cols = [c for c in manifest.columns if c.startswith("qc_")]
    print("QC rejections (frames/rows) per room:")
    print(manifest.groupby("room")[qc_cols].sum().T)
    print("valid rows per capture:")
    print(ok.groupby("room").n_valid.describe())

    index = build_windows(cfg, manifest)
    print("windows per room/split:")
    print(index.groupby(["room", "split"]).size())
    print("window method counts:")
    print(index.groupby(["room", "method"]).size())


if __name__ == "__main__":
    main()
