"""Post-processing to remove bias from a raw (possibly biased) source."""
import hashlib
import numpy as np


def von_neumann(bits):
    """01 -> 0, 10 -> 1, 00/11 discarded. Removes bias for independent bits; ~75% loss at p=0.5."""
    a, b = bits[0:-1:2], bits[1::2]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    keep = a != b
    return a[keep].astype(np.uint8)


def sha256_condition(bits, block_in=512):
    """Hash each 512-bit raw block into 256 output bits (conditioning, not entropy creation)."""
    n = len(bits) // block_in
    out = bytearray()
    for i in range(n):
        out += hashlib.sha256(np.packbits(bits[i * block_in:(i + 1) * block_in]).tobytes()).digest()
    return np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))
