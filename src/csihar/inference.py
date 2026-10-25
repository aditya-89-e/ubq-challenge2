"""End-to-end inference on arbitrary captures (used by run_episode.py).

Any input goes through exactly the same stages as the development data:
  .pcap -> UDP payload -> 4366c0 CSI -> amplitude -> QC (Room A rules)
  .csv  -> 256-column amplitude table -> QC (amplitude-table rules)
  .npy  -> an already-built (150, 256) amplitude window (used as is)
then windowing (validation policy: one centred window per capture, or the
development sliding policy for support captures), normalisation with the
stored development statistics, router, SSL encoder, few-shot head.
"""
from pathlib import Path

import numpy as np

from .amplitude import read_common_csv, to_amplitude
from .cleaning import qc_amplitude_table, qc_room_a
from .csi_decode import decode_payloads
from .pcap_reader import read_udp_payloads
from .windowing import extract_windows

LABEL_ALIASES = {
    "fall": "FALL", "sit": "SIT", "sitting": "SIT", "sitdown": "SITDOWN", "sit-down": "SITDOWN",
    "sit_down": "SITDOWN", "stand": "STAND", "standing": "STAND", "standup": "STANDUP",
    "stand-up": "STANDUP", "stand_up": "STANDUP", "walk": "WALK", "walking": "WALK",
    "empty": "EMPTY", "lying": "LYING", "lie": "LYING",
}
INPUT_SUFFIXES = {".pcap", ".pcapng", ".csv", ".npy"}


def canonical_label(name):
    key = name.strip().lower()
    if key not in LABEL_ALIASES:
        raise ValueError(f"unknown activity folder name: {name!r}")
    return LABEL_ALIASES[key]


def load_capture(path, cfg):
    """Return the common-form (n, 256) amplitude matrix of one capture (or a window)."""
    path = Path(path)
    if path.suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != 256:
            raise ValueError(f"{path}: expected (T, 256) array, got {arr.shape}")
        return arr
    if path.suffix in (".pcap", ".pcapng"):
        payloads, stamps, _ = read_udp_payloads(path)
        frames = decode_payloads(payloads, stamps)
        keep, _, _ = qc_room_a(frames, cfg["qc"]["max_subband_frac"])
        return to_amplitude(frames.csi)[keep]
    raw = read_common_csv(path)
    keep, _ = qc_amplitude_table(raw, cfg["qc"]["max_subband_frac"])
    return raw[keep]


def capture_windows(amp, cfg, split):
    w = cfg["window"]
    return [win for win, *_ in extract_windows(amp, w["length"], w["dev_stride"], split, w["short_policy"])]


def find_inputs(folder):
    return sorted(p for p in Path(folder).rglob("*")
                  if p.is_file() and p.suffix.lower() in INPUT_SUFFIXES and not p.name.startswith("."))
