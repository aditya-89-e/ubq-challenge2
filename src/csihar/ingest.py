"""Source captures -> common intermediate CSVs + capture manifest.

Room A: raw Nexmon pcap -> UDP payload -> complex CSI -> amplitude -> QC.
Room B: amplitude CSV -> schema check -> QC.
Both are written as one CSV per capture with columns subcarrier_001..256 and
one row per valid observation in original temporal order. The manifest keeps
capture identity, label, split, source path, and QC counts for provenance.
"""
import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .amplitude import read_common_csv, to_amplitude, to_frame
from .cleaning import qc_amplitude_table, qc_room_a
from .config import ROOM_A_FOLDERS, ROOM_B_PREFIXES, resolve
from .csi_decode import decode_payloads
from .pcap_reader import read_udp_payloads
from .splits import group_by_label, room_b_validation

ROOM_B_RE = re.compile(r"^(Empty|Lying|Sitting|Standing|Walking)_")


def discover_room_a(exp2_dir: Path):
    rows = []
    for folder, label in ROOM_A_FOLDERS.items():
        for split, base in (("development", exp2_dir), ("validation", exp2_dir / "validation")):
            for f in sorted((base / folder).glob("*.pcap")):
                rows.append(dict(capture_id=f"A_{split[:3]}_{f.stem}", room="A", label=label,
                                 split=split, source=str(f)))
    return rows


def discover_room_b(exp3_dir: Path, per_class=25, seed=42):
    items = []
    for f in sorted(exp3_dir.glob("*.csv")):
        m = ROOM_B_RE.match(f.name)
        if not m or f.name.startswith("._"):
            continue
        items.append((f.name, ROOM_B_PREFIXES[m.group(1)]))
    val = room_b_validation(group_by_label(items), per_class=per_class, seed=seed)
    return [dict(capture_id=f"B_{Path(name).stem}", room="B", label=label,
                 split="validation" if name in val else "development",
                 source=str(exp3_dir / name)) for name, label in items]


def _process(job):
    rec, out_dir, max_subband_frac = job
    rec = dict(rec)
    try:
        if rec["room"] == "A":
            payloads, stamps, n_total = read_udp_payloads(rec["source"])
            frames = decode_payloads(payloads, stamps)
            keep, counts, diag = qc_room_a(frames, max_subband_frac)
            amp = to_amplitude(frames.csi)[keep]
            ts = frames.timestamp[keep]
            rec.update(n_raw=n_total, n_payload=len(payloads),
                       qc_malformed=len(payloads) - frames.csi.shape[0],  # bad magic / length
                       **{f"qc_{k}": v for k, v in counts.items()},
                       **diag, duration_s=float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
                       ts_monotonic=bool(np.all(np.diff(ts) >= 0)))
        else:
            raw = read_common_csv(rec["source"])
            keep, counts = qc_amplitude_table(raw, max_subband_frac)
            amp = raw[keep]
            rec.update(n_raw=raw.shape[0], **{f"qc_{k}": v for k, v in counts.items()})
        rec["n_valid"] = int(amp.shape[0])
        rec["content_md5"] = hashlib.md5(np.ascontiguousarray(amp).tobytes()).hexdigest() if len(amp) else ""
        out = Path(out_dir) / f"room_{rec['room']}" / f"{rec['capture_id']}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        to_frame(amp).to_csv(out, index=False, float_format="%.4f")
        rec["intermediate"] = str(out)
        rec["error"] = ""
    except Exception as e:  # recorded in manifest, capture excluded downstream
        rec.update(n_valid=0, intermediate="", error=repr(e))
    return rec


def build_intermediate(cfg, workers=None):
    exp2 = resolve(cfg, "exp2_dir")
    exp3 = resolve(cfg, "exp3_dir")
    out_dir = resolve(cfg, "intermediate")
    sp = cfg["splits"]
    records = discover_room_a(exp2) + discover_room_b(exp3, sp["room_b_val_per_class"], sp["seed"])
    jobs = [(r, str(out_dir), cfg["qc"]["max_subband_frac"]) for r in records]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_process, jobs, chunksize=8))
    manifest = pd.DataFrame(results)
    qc_cols = sorted(c for c in manifest.columns if c.startswith("qc_"))
    manifest[qc_cols] = manifest[qc_cols].fillna(0).astype(int)
    # Capture-level duplicates would leak across the dev/val boundary; flag
    # every copy after the first so it is excluded from windowing.
    dup = (manifest["content_md5"].fillna("") != "") & manifest.duplicated("content_md5", keep="first")
    manifest.loc[dup, "error"] = "duplicate_capture"
    path = resolve(cfg, "work") / "manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return manifest
