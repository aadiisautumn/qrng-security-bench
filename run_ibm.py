#!/usr/bin/env python3
"""Real-hardware QRNG test.
  export IBM_QUANTUM_TOKEN=...            # your API key
  python run_ibm.py --bits 100000 --qubits 1
  python run_ibm.py --fake --bits 20000   # dry run on noisy fake backend, no account needed
  python run_ibm.py --load results/ibm_xxx.npy   # re-analyse saved bits offline
"""
import argparse, json
import numpy as np
from qrngbench import stats, extractors, keys, ibm_qrng
from qrngbench.generators import qrng_aer, os_csprng


def report(label, bits):
    r = stats.run_suite(bits)
    row = " ".join(f"{t.split()[0][:6]}={'P' if r[t]['pass'] else 'F'}({r[t]['p']:.3f})" for t in stats.TESTS)
    k = keys.key_quality(bits)
    print(f"{label:24s} n={len(bits):7d} ones={r['ones_ratio']:.4f} H={r['shannon_bits_per_bit']:.4f} "
          f"ac1={r['autocorr_lag1']:+.4f} keybias={k['mean_bias']:.4f} | {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=100_000)
    ap.add_argument("--qubits", type=int, default=1)
    ap.add_argument("--backend", default=None, help="e.g. ibm_brisbane; default least busy")
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--load", default=None)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if a.load:
        bits = np.load(a.load); meta = {}
    else:
        backend, _ = ibm_qrng.get_backend(a.backend, a.fake)
        bits, meta = ibm_qrng.collect_bits(backend, a.bits, a.qubits)
        print("saved:", ibm_qrng.save(bits, meta, a.out))
        print("per-qubit ones ratio:", [round(x, 4) for x in meta["per_qubit_ones"]])

    n = len(bits)
    print("\n== Quality (fail if p<0.01) ==")
    report("IBM raw", bits)
    vn = extractors.von_neumann(bits)
    report("IBM + von Neumann", vn)
    report("IBM + SHA-256 cond.", extractors.sha256_condition(bits))
    report("Aer sim (reference)", qrng_aer(n))
    report("os.urandom (reference)", os_csprng(n))
    print("\nNote: raw hardware bits usually lean toward 0 (T1 decay / readout error); "
          "extractors fix bias but SHA-256 conditioning adds no entropy.")


if __name__ == "__main__":
    main()
