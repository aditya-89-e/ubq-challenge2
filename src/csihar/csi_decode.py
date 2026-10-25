"""Nexmon CSI payload parsing and BCM4366c0 packed-CSI reconstruction.

Each Nexmon CSI UDP payload (port 5500) is an 18-byte little-endian header
followed by NFFT 32-bit words, one per subcarrier:

    offset  size  field
    0       2     magic (0x1111)
    2       1     RSSI (int8)
    3       1     802.11 frame-control byte
    4       6     source MAC
    10      2     sequence number
    12      2     core (bits 0-2) / spatial stream (bits 3-5)
    14      2     chanspec
    16      2     chip version
    18      4*N   packed CSI

The BCM4366c0 does not export int16 I/Q. Each 32-bit word holds a shared-
exponent float pair (format documented in the public Nexmon CSI project):

    bit 29        sign of I
    bits 18..28   11-bit magnitude of I
    bit 17        sign of Q
    bits 6..16    11-bit magnitude of Q
    bits 0..5     6-bit two's-complement exponent shared by I and Q

value = sign * mantissa * 2**exponent. The firmware-side convention also
applies a per-packet power-of-two scale ("autoscale") so that the largest
component occupies `nbits` bits, flushing components whose shifted exponent
falls below -nman to zero. We implement that vectorised over all packets so
the amplitude scale is comparable to amplitude data exported by Nexmon tools.
"""
from dataclasses import dataclass

import numpy as np

MAGIC = 0x1111
HEADER_LEN = 18

# 4366c0 packed-float layout
NMAN = 12   # mantissa field width incl. sign bit position bookkeeping
NEXP = 6
NBITS = 10  # autoscale target bit width

_HEADER_DTYPE = np.dtype([
    ("magic", "<u2"), ("rssi", "i1"), ("fctl", "u1"), ("mac", "u1", (6,)),
    ("seq", "<u2"), ("core_ss", "<u2"), ("chanspec", "<u2"), ("chip", "<u2"),
])


@dataclass
class CSIFrames:
    """Decoded CSI for one capture, in original packet order."""
    csi: np.ndarray        # (n, nfft) complex64, FFT bin order
    rssi: np.ndarray       # (n,) int8
    seq: np.ndarray        # (n,) uint16
    core: np.ndarray       # (n,) uint8
    stream: np.ndarray     # (n,) uint8
    chanspec: np.ndarray   # (n,) uint16
    chip: np.ndarray       # (n,) uint16
    timestamp: np.ndarray  # (n,) float64, pcap timestamp


def parse_headers(payloads):
    """Parse the 18-byte Nexmon header of each payload into a structured array."""
    raw = np.frombuffer(b"".join(p[:HEADER_LEN] for p in payloads), dtype=_HEADER_DTYPE)
    return raw


def unpack_4366c0(words: np.ndarray, nbits: int = NBITS, autoscale: bool = True) -> np.ndarray:
    """Reconstruct complex CSI from packed 4366c0 words.

    words: (n_packets, nfft) uint32. Returns (n_packets, nfft) complex64.
    """
    w = words.astype(np.uint32)
    mag_mask = np.uint32((1 << (NMAN - 1)) - 1)
    exp_mask = np.uint32((1 << NEXP) - 1)
    sign_i_bit = np.uint32(1 << (NEXP + 2 * NMAN - 1))
    sign_q_bit = np.uint32(1 << (NEXP + NMAN - 1))

    mi = ((w >> np.uint32(NEXP + NMAN)) & mag_mask).astype(np.int64)
    mq = ((w >> np.uint32(NEXP)) & mag_mask).astype(np.int64)
    e = (w & exp_mask).astype(np.int64)
    e = np.where(e >= (1 << (NEXP - 1)), e - (1 << NEXP), e)  # two's complement
    si = np.where(w & sign_i_bit, -1, 1)
    sq = np.where(w & sign_q_bit, -1, 1)

    if autoscale:
        # Highest occupied bit per subcarrier = exponent + bit length of (mi | mq);
        # the packet is shifted so its maximum lands on `nbits`.
        x = mi | mq
        bitlen = np.where(x > 0, np.floor(np.log2(np.maximum(x, 1))).astype(np.int64) + 1, 0)
        top = np.where(x > 0, e + bitlen, np.iinfo(np.int64).min)
        maxbit = top.max(axis=1, keepdims=True)
        maxbit = np.where(maxbit == np.iinfo(np.int64).min, -(1 << (NEXP - 1)), maxbit)
        shift = e + (nbits - maxbit)
    else:
        shift = e

    def apply(m):
        out = np.zeros_like(m)
        flush = shift < -NMAN
        right = (shift < 0) & ~flush
        left = shift >= 0
        out[right] = m[right] >> (-shift[right])
        out[left] = m[left] << shift[left]
        return out

    i_val = si * apply(mi)
    q_val = sq * apply(mq)
    return (i_val + 1j * q_val).astype(np.complex64)


def decode_payloads(payloads, timestamps, nfft: int = 256) -> CSIFrames:
    """Decode a list of raw Nexmon UDP payloads (bytes) into complex CSI.

    Payloads with a bad magic number or the wrong length are dropped here;
    content-level validity is judged later in cleaning.py.
    """
    expected = HEADER_LEN + 4 * nfft
    keep = [i for i, p in enumerate(payloads)
            if len(p) == expected and p[0] == 0x11 and p[1] == 0x11]
    payloads = [payloads[i] for i in keep]
    ts = np.asarray(timestamps, dtype=np.float64)[keep]
    if not payloads:
        empty = np.zeros(0)
        return CSIFrames(np.zeros((0, nfft), np.complex64), empty.astype(np.int8),
                         empty.astype(np.uint16), empty.astype(np.uint8), empty.astype(np.uint8),
                         empty.astype(np.uint16), empty.astype(np.uint16), empty)
    hdr = parse_headers(payloads)
    words = np.frombuffer(b"".join(p[HEADER_LEN:] for p in payloads), dtype="<u4").reshape(-1, nfft)
    csi = unpack_4366c0(words)
    return CSIFrames(
        csi=csi,
        rssi=hdr["rssi"].copy(),
        seq=hdr["seq"].copy(),
        core=(hdr["core_ss"] & 0x7).astype(np.uint8),
        stream=((hdr["core_ss"] >> 3) & 0x7).astype(np.uint8),
        chanspec=hdr["chanspec"].copy(),
        chip=hdr["chip"].copy(),
        timestamp=ts,
    )
