"""Hint 1: inspect the source files before building anything.

Reports file types/counts, packet and payload structure, Nexmon header
fields, sequence ordering, timestamps, and Room B CSV schema. Writes
results/data_inspection.json and prints a summary.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from csihar.cleaning import subband_energy_fraction  # noqa: E402
from csihar.config import load_config, resolve  # noqa: E402
from csihar.csi_decode import decode_payloads  # noqa: E402
from csihar.pcap_reader import read_udp_payloads  # noqa: E402


def inspect_room_a(exp2):
    files = sorted(exp2.rglob("*.pcap"))
    per_folder = collections.Counter(str(f.parent.relative_to(exp2)).replace("\\", "/") for f in files)
    other = [str(p.relative_to(exp2)) for p in exp2.rglob("*") if p.is_file() and p.suffix != ".pcap"]
    payload_lens, chips, chanspecs, cores = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    n_pkts, n_seq, durations, rates, seq_steps = [], [], [], [], collections.Counter()
    legacy_agree = []
    for f in files:
        payloads, stamps, n_total = read_udp_payloads(f)
        payload_lens.update(len(p) for p in payloads)
        fr = decode_payloads(payloads, stamps)
        chips.update(hex(c) for c in fr.chip)
        chanspecs.update(hex(c) for c in fr.chanspec)
        cores.update(f"core{c}/ss{s}" for c, s in zip(fr.core, fr.stream))
        seqd = fr.seq != 0xFFFF
        frac = subband_energy_fraction(np.abs(fr.csi))
        legacy_agree.append(np.mean((frac.max(1) > 0.9) == ~seqd))
        n_pkts.append(n_total)
        n_seq.append(int(seqd.sum()))
        dur = fr.timestamp[-1] - fr.timestamp[0]
        durations.append(dur)
        rates.append(seqd.sum() / dur if dur > 0 else 0)
        seq_steps.update(np.diff(fr.seq[seqd].astype(int)).tolist())
    q = lambda a: dict(min=float(np.min(a)), median=float(np.median(a)), max=float(np.max(a)))
    return dict(
        n_files=len(files), per_folder=dict(per_folder), non_pcap_files=other,
        udp_payload_lengths=dict(payload_lens), chip=dict(chips), chanspec=dict(chanspecs),
        core_stream=dict(cores), packets_per_capture=q(n_pkts),
        sequenced_frames_per_capture=q(n_seq), duration_s=q(durations),
        sequenced_rate_hz=q(rates), seq_step_top=dict(seq_steps.most_common(6)),
        legacy_content_rule_vs_seq0xffff_agreement=q(legacy_agree),
    )


def inspect_room_b(exp3):
    files = sorted(exp3.glob("*.csv"))
    prefixes = collections.Counter(f.name.rsplit("_", 1)[0] for f in files)
    rows, headers, dtypes, nonfinite = [], collections.Counter(), collections.Counter(), 0
    for f in files:
        df = pd.read_csv(f)
        rows.append(len(df))
        headers[(df.columns[0], df.columns[-1], df.shape[1])] += 1
        dtypes.update(str(t) for t in set(df.dtypes))
        nonfinite += int((~np.isfinite(df.to_numpy(float)).all(1)).sum())
    return dict(
        n_files=len(files), filename_prefixes=dict(prefixes),
        rows_per_capture=dict(collections.Counter(rows)),
        header_first_last_ncols={f"{a}..{b} ({n})": c for (a, b, n), c in headers.items()},
        dtypes=dict(dtypes), nonfinite_rows=nonfinite,
        other_files=[p.name for p in exp3.iterdir() if p.suffix != ".csv"],
    )


def main():
    cfg = load_config()
    report = dict(room_A=inspect_room_a(resolve(cfg, "exp2_dir")),
                  room_B=inspect_room_b(resolve(cfg, "exp3_dir")))
    out = resolve(cfg, "results") / "data_inspection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
