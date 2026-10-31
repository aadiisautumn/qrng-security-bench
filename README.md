# qrng-security-bench

Benchmark a Qiskit-simulated QRNG against Mersenne Twister, real ChaCha20 and `os.urandom`: NIST-style statistical tests, 256-bit key quality, bias extraction, and an MT19937 state-recovery demo showing why it's unfit for keys.

## Run
```bash
pip install -r requirements.txt
python run_benchmark.py --bits 100000 --trials 20
```

## What's inside
- `generators.py` – Aer H-gate QRNG, MT19937, ChaCha20 (real keystream), os.urandom
- `stats.py` – Monobit, Block Frequency, Runs, DFT Spectral, byte Chi-Square, Shannon/min-entropy, autocorrelation
- `keys.py` – 256-bit key bias / LSB balance / uniqueness
- `extractors.py` – Von Neumann debiasing, SHA-256 conditioning
- `mt_attack.py` – recover MT19937 state from 624 outputs, predict the rest (against your own local instance)

## Real IBM hardware
```bash
# get an API key at quantum.cloud.ibm.com (free Open plan works)
export IBM_QUANTUM_TOKEN="your_key"        # Termux/Linux/Colab: os.environ[...] in Colab
python run_ibm.py --bits 100000 --qubits 1            # least-busy real backend
python run_ibm.py --backend ibm_brisbane --qubits 5   # pick backend, more bits per shot
python run_ibm.py --fake --bits 20000                 # dry run on noisy fake backend, no account
python run_ibm.py --load results/ibm_*.npy            # re-analyse saved bits offline
```
Compares raw hardware bits, von Neumann-debiased, SHA-256-conditioned, Aer and `os.urandom` through the same suite. Raw bits + job metadata are saved in `results/` so the run is reproducible. Keep bit counts modest on the free plan (~10 min QPU/month; 1 qubit x 100k shots is only seconds).
Dry-run (`--fake`) is tested; the real-hardware path is untested against live IBM, so if the API has changed, the error will say where.

## Fixes vs. the original report
1. **ChaCha20 monobit "FAIL" (stat 12754) was a bug**, not an "OS entropy artefact" – the statistic was computed on bytes, not bits. Fixed; ChaCha20 passes.
2. **The 0/1 chi-square is just monobit z²** (same p-value, 0.0012) – counted twice. Replaced with a 256-bin byte chi-square.
3. **AerSimulator is not true randomness** – it uses a classical PRNG. "True entropy" only applies to real hardware or a physical source.
4. **10k bits is too small and one run proves little.** At alpha=0.01 you expect ~1% false fails. Now: N independent streams, report pass *rates*.
5. Key bias has an ideal non-zero value (~0.0249 for 256-bit keys), so "ideal 0.0" was wrong.
6. `os.urandom` ≠ ChaCha20 directly; both are now tested separately.

## Next (security-relevant)
- Full NIST SP 800-22 / Dieharder / PractRand on 10⁸+ bits
- Min-entropy per NIST SP 800-90B estimators
- Seed Kyber/Dilithium implementations with each source
