"""QRNG on real IBM Quantum hardware (or a noisy fake backend for dry runs).

Auth: set env var IBM_QUANTUM_TOKEN (your IBM Quantum Platform API key), optionally
IBM_QUANTUM_INSTANCE (the CRN). Or save an account once with
QiskitRuntimeService.save_account(...) and leave the env vars unset.
"""
import json, os, time
import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def get_backend(name=None, fake=False):
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        return FakeSherbrooke(), None
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = os.environ.get("IBM_QUANTUM_TOKEN")
    inst = os.environ.get("IBM_QUANTUM_INSTANCE")
    kw = {}
    if token:
        kw = {"channel": "ibm_quantum_platform", "token": token}
        if inst: kw["instance"] = inst
    svc = QiskitRuntimeService(**kw)
    backend = svc.backend(name) if name else svc.least_busy(operational=True, simulator=False)
    return backend, svc


def build_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits, name="qrng")
    qc.h(range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def collect_bits(backend, n_bits, n_qubits=1, shots_per_pub=None):
    """Return (bits uint8 array, metadata). Bits are interleaved shot-major, so
    bits[k*n_qubits + q] is qubit q of shot k."""
    from qiskit_ibm_runtime import SamplerV2 as Sampler

    shots_total = -(-n_bits // n_qubits)
    max_shots = getattr(backend, "max_shots", None) or 100_000
    per = min(shots_per_pub or max_shots, max_shots, shots_total)
    n_pubs = -(-shots_total // per)

    pm = generate_preset_pass_manager(optimization_level=1, backend=backend)
    isa = pm.run(build_circuit(n_qubits))
    sampler = Sampler(mode=backend)
    t0 = time.time()
    job = sampler.run([(isa, None, per)] * n_pubs)
    print(f"submitted job {job.job_id()} on {backend.name} ({n_pubs} PUB x {per} shots x {n_qubits} qubit)")
    res = job.result()

    rows = []
    for pub in res:
        arr = pub.data.c.get_bitstrings()           # strings, MSB = highest clbit
        rows.extend(arr)
    mat = np.array([[int(ch) for ch in s[::-1]] for s in rows], dtype=np.uint8)  # col q = clbit q
    mat = mat[:shots_total]
    bits = mat.reshape(-1)[:n_bits]
    meta = {"backend": backend.name, "job_id": job.job_id(), "shots": int(len(mat)),
            "qubits": n_qubits, "bits": int(len(bits)), "wall_s": round(time.time() - t0, 2),
            "per_qubit_ones": [float(x) for x in mat.mean(axis=0)],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return bits, meta


def save(bits, meta, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    stem = f"{outdir}/ibm_{meta['backend']}_{meta['utc'].replace(':','')}"
    np.save(stem + ".npy", bits)
    json.dump(meta, open(stem + ".json", "w"), indent=2)
    return stem
