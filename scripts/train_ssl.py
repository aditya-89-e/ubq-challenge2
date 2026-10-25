"""Stage 2: fit the normaliser on development windows and train the SSL encoder.

Uses only development windows of both rooms, no activity or room labels.
    python scripts/train_ssl.py [--config ...]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csihar.config import load_config  # noqa: E402
from csihar.pipeline import get_encoder, load_normalized, representations  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    Zd, md, Zv, mv, _ = load_normalized(cfg, refit=True)
    print(f"normalised dev {Zd.shape} val {Zv.shape} ({time.time() - t0:.0f}s)")
    get_encoder(cfg, Zd, md, retrain=True, log=lambda s: print(s, flush=True))
    print(f"SSL trained ({time.time() - t0:.0f}s)")
    representations(cfg, Zd, Zv, md, refresh=True)
    print(f"representations cached ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
