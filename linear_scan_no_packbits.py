"""Linear scan baseline that does NOT use packed bytes.

This is the apples-to-apples comparison baseline for QADP-NC: both run on
the same (N, d) uint8 unpacked-bit representation, so any speed difference
between this scan and QADP is purely algorithmic — no advantage from byte
packing, XOR + popcount-lookup, or SIMD-friendly byte ops.

The "real" linear_scan.py works on packed bytes (N, d/8) with
popcount-lookup and beats this version by ~8x on memory bandwidth alone.
That ~8x is an *implementation* advantage tied to the representation, not
an algorithmic one. When measuring whether QADP's pruning logic helps,
this is the baseline you should compare against.
"""
import os
import time
import argparse
import random
import numpy as np


def hamming_distances_unpacked(bits: np.ndarray, q_bits: np.ndarray) -> np.ndarray:
    """
    bits   : (N, d) uint8 — unpacked bits (each element is 0 or 1)
    q_bits : (d,)   uint8 — unpacked query bits
    returns (N,) int32 Hamming distances

    No popcount lookup, no packed-byte XOR — just elementwise compare and
    sum. The same operation QADP-NC's per-subspace check uses, just over
    all d bits at once.
    """
    return (bits != q_bits).sum(axis=1).astype(np.int32)


def linear_scan_range_unpacked(bits: np.ndarray, q_bits: np.ndarray, r: int) -> np.ndarray:
    """Return indices of rows in bits whose Hamming distance to q_bits is <= r."""
    h = hamming_distances_unpacked(bits, q_bits)
    return np.flatnonzero(h <= r).astype(np.int32)


def run_linear_scan_unpacked(
    dataset_dir: str,
    dataset: str,
    radius: int,
    query_count: int,
    seed: int = None,
) -> None:
    from dataset import read_fvecs, binarize

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    base_path  = os.path.join(dataset_dir, f'{dataset}_base.fvecs')
    query_path = os.path.join(dataset_dir, f'{dataset}_query.fvecs')

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset not found: {base_path}")

    print(f"Loading {dataset} ...")
    base    = read_fvecs(base_path)
    queries = read_fvecs(query_path)
    print(f"  base    : {base.shape}")
    print(f"  queries : {queries.shape}")

    print(f"Binarizing  (GRP seed={seed}) ...")
    base_codes  = binarize(base, seed=seed)
    query_codes = binarize(queries, seed=seed)
    D_bytes = base_codes.shape[1]
    d = D_bytes * 8
    print(f"  packed code shape : {base_codes.shape}  dtype={base_codes.dtype}")

    # Unpack everything — this is the whole point of this baseline.
    print("Unpacking bits ...")
    base_bits  = np.unpackbits(base_codes,  axis=1)   # (N, d) uint8
    query_bits = np.unpackbits(query_codes, axis=1)   # (Q, d) uint8
    print(f"  unpacked base : {base_bits.shape} ({base_bits.nbytes / 1e6:.1f} MB)")

    Q = min(query_count, len(query_bits))
    print(f"\nLinear scan (UNPACKED bits)  r={radius}, Q={Q} ...")

    results = []
    running_avg = 0.0
    for i in range(Q):
        t0 = time.perf_counter()
        res = linear_scan_range_unpacked(base_bits, query_bits[i], radius)
        qt = time.perf_counter() - t0
        running_avg += (qt - running_avg) / (i + 1)
        results.append(res)
        print(f"  q[{i:02d}]  scan={qt*1000:7.2f} ms  hits={len(res):6d}")

    total    = running_avg * Q
    avg_hits = float(np.mean([len(r) for r in results]))

    print(f"\n  total        : {total:.3f} s")
    print(f"  avg query    : {running_avg * 1000:.3f} ms  (running average)")
    print(f"  QPS          : {Q / total:.1f}")
    print(f"  avg hits     : {avg_hits:.1f}")


def main() -> None:
    from dataset import datasets, datasets_dir

    ap = argparse.ArgumentParser(
        description='Linear scan on UNPACKED bits — fair baseline for QADP-NC '
                    '(same representation, no popcount/byte-packing tricks).'
    )
    ap.add_argument('--dataset', choices=list(datasets), default='sift',
                    help='dataset to use (default: sift)')
    ap.add_argument('--radius', type=int, default=20,
                    help='Hamming radius for range query (default: 20)')
    ap.add_argument('--query_count', type=int, default=10,
                    help='number of queries to run (default: 10)')
    ap.add_argument('--seed', type=int, default=0,
                    help='RNG seed for GRP projection matrix (default: 0, matches qadp_normal.py)')
    args = ap.parse_args()

    dataset      = datasets[args.dataset]
    dataset_dir  = os.path.join(datasets_dir, dataset)
    run_linear_scan_unpacked(dataset_dir, dataset, args.radius, args.query_count, args.seed)


if __name__ == '__main__':
    main()
