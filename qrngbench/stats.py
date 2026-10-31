"""Statistical tests (NIST SP 800-22 style) + entropy estimates.
Every test returns (statistic, p_value). Fail if p < ALPHA."""
from math import erfc, sqrt, log
import numpy as np
from scipy.special import gammaincc
from scipy.stats import chi2

ALPHA = 0.01


def monobit(bits):
    n = len(bits)
    s = int(np.sum(2 * bits.astype(np.int64) - 1))
    z = abs(s) / sqrt(n)
    return z, erfc(z / sqrt(2))


def block_frequency(bits, M=128):
    N = len(bits) // M
    pi = bits[: N * M].reshape(N, M).mean(axis=1)
    x = 4 * M * float(np.sum((pi - 0.5) ** 2))
    return x, float(gammaincc(N / 2, x / 2))


def runs(bits):
    n = len(bits)
    pi = bits.mean()
    if abs(pi - 0.5) >= 2 / sqrt(n):  # prerequisite fails
        return float("nan"), 0.0
    v = 1 + int(np.count_nonzero(bits[1:] != bits[:-1]))
    p = erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * sqrt(2 * n) * pi * (1 - pi)))
    return v, p


def spectral_dft(bits):
    n = len(bits)
    x = 2 * bits.astype(float) - 1
    m = np.abs(np.fft.fft(x))[: n // 2]
    T = sqrt(log(1 / 0.05) * n)
    n0, n1 = 0.95 * n / 2, np.count_nonzero(m < T)
    d = (n1 - n0) / sqrt(n * 0.95 * 0.05 / 4)
    return d, erfc(abs(d) / sqrt(2))


def byte_chi_square(bits):
    """Chi-square over the 256 byte values (df=255). Replaces the old 0/1 chi-square,
    which is algebraically just monobit z^2 with the same p-value."""
    nb = len(bits) // 8
    by = np.packbits(bits[: nb * 8])
    obs = np.bincount(by, minlength=256)
    x = float(np.sum((obs - nb / 256) ** 2 / (nb / 256)))
    return x, float(chi2.sf(x, 255))


def serial_corr(bits, lag=1):
    x = bits.astype(float)
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def shannon_per_bit(bits, block=8):
    nb = len(bits) // block
    w = bits[: nb * block].reshape(nb, block) @ (1 << np.arange(block - 1, -1, -1))
    p = np.bincount(w, minlength=2**block) / nb
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / block)


def min_entropy_per_byte(bits):
    nb = len(bits) // 8
    p = np.bincount(np.packbits(bits[: nb * 8]), minlength=256) / nb
    return float(-np.log2(p.max()))


TESTS = {
    "Monobit": monobit,
    "Block Freq (M=128)": block_frequency,
    "Runs": runs,
    "DFT Spectral": spectral_dft,
    "Byte Chi-Square": byte_chi_square,
}


def run_suite(bits):
    out = {}
    for name, fn in TESTS.items():
        s, p = fn(bits)
        out[name] = {"stat": s, "p": p, "pass": p >= ALPHA}
    out["ones_ratio"] = float(bits.mean())
    out["shannon_bits_per_bit"] = shannon_per_bit(bits)
    out["min_entropy_bits_per_byte"] = min_entropy_per_byte(bits)
    out["autocorr_lag1"] = serial_corr(bits, 1)
    out["autocorr_lag5"] = serial_corr(bits, 5)
    return out
