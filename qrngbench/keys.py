"""256-bit key simulation from a bitstream."""
import numpy as np


def key_quality(bits, key_bits=256):
    k = len(bits) // key_bits
    keys = bits[: k * key_bits].reshape(k, key_bits)
    bias = float(np.mean(np.abs(keys.mean(axis=1) - 0.5)))
    lsb = float(keys[:, -1].mean())
    uniq = len({row.tobytes() for row in keys}) / k
    # Expected mean |bias| for ideal keys ~ sqrt(2/pi)*0.5/sqrt(key_bits)
    ideal = float(np.sqrt(2 / np.pi) * 0.5 / np.sqrt(key_bits))
    return {"keys": k, "mean_bias": bias, "ideal_mean_bias": ideal, "lsb_balance": lsb, "uniqueness": uniq}
