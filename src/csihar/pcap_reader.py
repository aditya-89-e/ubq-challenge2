"""Read Nexmon CSI UDP payloads from a pcap file, preserving packet order.

dpkt handles the generic Ethernet/IPv4/UDP layers; everything after the UDP
payload boundary is decoded by our own code in csi_decode.py.
"""
from pathlib import Path

import dpkt

NEXMON_PORT = 5500


def read_udp_payloads(path, port: int = NEXMON_PORT):
    """Return (payloads, timestamps, n_total_packets) for UDP packets to `port`.

    Packets that are not Ethernet/IPv4/UDP on the Nexmon port are skipped and
    counted only in n_total_packets.
    """
    payloads, stamps = [], []
    n_total = 0
    with open(Path(path), "rb") as fh:
        try:
            reader = dpkt.pcap.Reader(fh)
        except ValueError:
            fh.seek(0)
            reader = dpkt.pcapng.Reader(fh)
        for ts, buf in reader:
            n_total += 1
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except (dpkt.UnpackError, dpkt.NeedData):
                continue
            ip = eth.data
            if not isinstance(ip, dpkt.ip.IP):
                continue
            udp = ip.data
            if not isinstance(udp, dpkt.udp.UDP) or udp.dport != port:
                continue
            payloads.append(bytes(udp.data))
            stamps.append(ts)
    return payloads, stamps, n_total
