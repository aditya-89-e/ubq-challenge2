"""CSI quality control.

Every rule returns a boolean keep-mask plus a reason label so that each
capture's rejection counts can be written to the manifest (what we reject
and why is part of the provenance trail).

Room A (decoded PCAP) rules, applied in this order:
  header      - chip/chanspec/core/stream differ from the capture's dominant
                configuration (a different radio setting is not the same
                observation type)
  legacy20    - energy confined to one 20 MHz sub-band (> max_subband_frac of
                the frame energy). These are legacy 20 MHz (non-HT) frames
                captured alongside the VHT80 frames. Every frame with sequence
                number 0xFFFF is of this kind, but so are ~38% of sequenced
                frames, so the decision is made on CSI content, not on the
                sequence field. The statistic is cleanly bimodal (VHT80 frames
                <= 0.8, legacy frames >= 0.95), so the threshold is not
                sensitive.
  zero        - all-zero or constant amplitude (zero-information CSI)
  nonfinite   - NaN/Inf amplitude
  dup_seq     - repeated sequence number (MAC retransmission of the same frame)
  dup_row     - bit-identical amplitude row to the previous kept row

Room B (amplitude CSV) rules: nonfinite/missing, zero, negative, legacy20
(same threshold as Room A; ~5% of Room B rows), dup_row.
"""
from collections import OrderedDict

import numpy as np

NFFT = 256
SEQ_NONE = 0xFFFF


def subband_energy_fraction(amp: np.ndarray) -> np.ndarray:
    """Fraction of per-frame energy in each 20 MHz sub-band (FFT bin order).

    Sub-bands are taken in frequency order after an fftshift: bins
    [0..63], [64..127], [128..191], [192..255] of the shifted spectrum.
    Returns (n, 4).
    """
    shifted = np.fft.fftshift(amp.astype(np.float64) ** 2, axes=1)
    e = shifted.reshape(amp.shape[0], 4, NFFT // 4).sum(-1)
    return e / np.maximum(e.sum(1, keepdims=True), 1e-12)


def qc_room_a(frames, max_subband_frac=0.9):
    """QC on decoded Nexmon frames. Returns (keep_mask, counts, diagnostics)."""
    n = frames.csi.shape[0]
    amp = np.abs(frames.csi).astype(np.float32)
    keep = np.ones(n, bool)
    counts = OrderedDict()

    def reject(name, bad):
        bad = bad & keep
        counts[name] = int(bad.sum())
        keep[bad] = False

    if n:
        def mode(x):
            vals, c = np.unique(x, return_counts=True)
            return vals[c.argmax()]
        hdr_bad = ((frames.chip != mode(frames.chip)) | (frames.chanspec != mode(frames.chanspec))
                   | (frames.core != mode(frames.core)) | (frames.stream != mode(frames.stream)))
    else:
        hdr_bad = np.zeros(0, bool)
    reject("header", hdr_bad)

    frac = subband_energy_fraction(amp) if n else np.zeros((0, 4))
    legacy = frac.max(1) > max_subband_frac
    reject("legacy20", legacy)

    reject("zero", (amp.max(1) == 0) | (amp.std(1) == 0))
    reject("nonfinite", ~np.isfinite(amp).all(1))

    seen = set()
    dup_seq = np.zeros(n, bool)
    for i in np.where(keep)[0]:
        s = int(frames.seq[i])
        if s in seen:
            dup_seq[i] = True
        seen.add(s)
    reject("dup_seq", dup_seq)
    reject("dup_row", _dup_consecutive(amp, keep))

    diag = {
        "seq_none_total": int((frames.seq == SEQ_NONE).sum()),
        # agreement between the content rule and the sequence-number marker
        "legacy_vs_seqnone_agree": float(np.mean(legacy == (frames.seq == SEQ_NONE))) if n else 1.0,
        "seq_none_kept": int(((frames.seq == SEQ_NONE) & keep).sum()),
    }
    return keep, counts, diag


def qc_amplitude_table(amp: np.ndarray, max_subband_frac=0.9):
    """QC on an amplitude matrix (Room B CSV). Returns (keep_mask, counts).

    The legacy-20 MHz content rule is the same one applied to Room A, so both
    rooms pass an identical observation-validity test.
    """
    keep = np.ones(amp.shape[0], bool)
    counts = OrderedDict()

    def reject(name, bad):
        bad = bad & keep
        counts[name] = int(bad.sum())
        keep[bad] = False

    reject("nonfinite", ~np.isfinite(amp).all(1))
    safe = np.where(np.isfinite(amp), amp, 0)
    reject("zero", (safe.max(1) == 0) | (safe.std(1) == 0))
    reject("negative", (safe < 0).any(1))
    reject("legacy20", subband_energy_fraction(safe).max(1) > max_subband_frac)
    reject("dup_row", _dup_consecutive(safe, keep))
    return keep, counts


def _dup_consecutive(amp, keep):
    """Mark rows identical to the previous kept row."""
    dup = np.zeros(amp.shape[0], bool)
    prev = None
    for i in np.where(keep)[0]:
        if prev is not None and np.array_equal(amp[i], amp[prev]):
            dup[i] = True
        else:
            prev = i
    return dup
