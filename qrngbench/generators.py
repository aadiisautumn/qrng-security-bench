"""Bit sources: Qiskit-Aer 'QRNG', Mersenne Twister, ChaCha20, OS CSPRNG.

All functions return a numpy uint8 array of 0/1 values with length n_bits.
"""
import os
import random
import numpy as np
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms


def _bytes_to_bits(b: bytes, n_bits: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))[:n_bits]


def qrng_aer(n_bits: int, shots_per_circuit: int = 1000, seed=None) -> np.ndarray:
    """H|0> then measure. NOTE: AerSimulator draws its 'measurement outcomes'
    from a classical PRNG, so this is a *simulated* QRNG, not true entropy."""
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    sim = AerSimulator()
    out = []
    n_circ = -(-n_bits // shots_per_circuit)
    for i in range(n_circ):
        kw = {} if seed is None else {"seed_simulator": seed + i}
        res = sim.run(qc, shots=shots_per_circuit, memory=True, **kw).result()
        out.extend(int(s) for s in res.get_memory())
    return np.array(out[:n_bits], dtype=np.uint8)


def mersenne_twister(n_bits: int, seed=None) -> np.ndarray:
    r = random.Random(seed)
    return _bytes_to_bits(r.getrandbits(8 * ((n_bits + 7) // 8)).to_bytes((n_bits + 7) // 8, "big"), n_bits)


def chacha20(n_bits: int) -> np.ndarray:
    """Real ChaCha20 keystream (key+nonce from os.urandom), not just os.urandom."""
    n_bytes = (n_bits + 7) // 8
    key, nonce = os.urandom(32), os.urandom(16)
    enc = Cipher(algorithms.ChaCha20(key, nonce), mode=None).encryptor()
    return _bytes_to_bits(enc.update(b"\x00" * n_bytes), n_bits)


def os_csprng(n_bits: int) -> np.ndarray:
    return _bytes_to_bits(os.urandom((n_bits + 7) // 8), n_bits)


GENERATORS = {
    "QRNG (Aer sim)": qrng_aer,
    "Mersenne Twister": mersenne_twister,
    "ChaCha20": chacha20,
    "os.urandom": os_csprng,
}
