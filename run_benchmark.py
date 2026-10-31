#!/usr/bin/env python3
"""python run_benchmark.py --bits 100000 --trials 20"""
import argparse, json, time, os
import numpy as np
from qrngbench.generators import GENERATORS
from qrngbench import stats, keys, extractors, mt_attack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=100_000)
    ap.add_argument("--trials", type=int, default=20, help="independent streams per generator")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report = {}

    print(f"== {a.trials} independent streams x {a.bits} bits per generator (fail if p<{stats.ALPHA}) ==")
    for name, gen in GENERATORS.items():
        t0 = time.perf_counter(); first = gen(a.bits); dt = time.perf_counter() - t0
        passes = {t: 0 for t in stats.TESTS}
        sh, kb = [], []
        for i in range(a.trials):
            b = first if i == 0 else gen(a.bits)
            r = stats.run_suite(b)
            for t in stats.TESTS: passes[t] += r[t]["pass"]
            sh.append(r["shannon_bits_per_bit"]); kb.append(keys.key_quality(b)["mean_bias"])
        single = stats.run_suite(first)
        report[name] = {"gen_time_s": dt, "pass_rate": {t: passes[t] / a.trials for t in passes},
                        "shannon_mean": float(np.mean(sh)), "key_bias_mean": float(np.mean(kb)),
                        "first_stream": single, "keys": keys.key_quality(first)}
        print(f"\n{name}  (gen {a.bits} bits: {dt:.4f}s)")
        for t in passes: print(f"  {t:20s} pass rate {passes[t]/a.trials:5.0%}")
        print(f"  Shannon/bit mean {np.mean(sh):.5f} | key bias mean {np.mean(kb):.5f} (ideal ~{report[name]['keys']['ideal_mean_bias']:.5f})")

    # Debiasing demo on a deliberately biased source
    rng = np.random.default_rng(0)
    biased = (rng.random(a.bits * 4) < 0.52).astype(np.uint8)
    vn = extractors.von_neumann(biased)
    print(f"\nVon Neumann on 52%-ones source: ones {biased.mean():.4f} -> {vn.mean():.4f} ({len(vn)}/{len(biased)} bits kept)")
    print("  monobit p before/after:", f"{stats.monobit(biased[:len(vn)])[1]:.2e} / {stats.monobit(vn)[1]:.3f}")
    report["von_neumann_demo"] = {"before": float(biased.mean()), "after": float(vn.mean()), "kept": int(len(vn))}

    ok, n = mt_attack.demo()
    print(f"\nMT19937 state recovery from 624 outputs -> predicted next {n} outputs correctly: {ok}")
    report["mt_state_recovery"] = ok

    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=2, default=float)
    print(f"\nSaved {a.out}/report.json")


if __name__ == "__main__":
    main()
