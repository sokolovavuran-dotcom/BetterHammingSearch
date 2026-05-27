import os
import time
import argparse
import random
import numpy as np

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_distances(codes: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    codes : (N, D) uint8
    q     : (D,)   uint8
    returns (N,) int32 Hamming distances
    """
    xor = np.bitwise_xor(codes, q[np.newaxis, :])
    return _POPCOUNT_TABLE[xor].sum(axis=1).astype(np.int32)


def linear_scan_range(codes: np.ndarray, q: np.ndarray, r: int) -> np.ndarray:
    """Return indices of rows in codes whose Hamming distance to q is <= r."""
    d = hamming_distances(codes, q)
    return np.flatnonzero(d <= r).astype(np.int32)

def run_linear_scan(dataset_dir: str, dataset: str, radius: int, query_count: int, seed: int = None) -> None:
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
    print(f"  code shape : {base_codes.shape}  dtype={base_codes.dtype}")

    Q = min(query_count, len(query_codes))
    print(f"\nLinear scan  r={radius}, Q={Q} ...")

    results = []
    running_avg = 0.0
    for i in range(Q):
        t0 = time.perf_counter()
        res = linear_scan_range(base_codes, query_codes[i], radius)
        qt = time.perf_counter() - t0
        running_avg += (qt - running_avg) / (i + 1)
        results.append(res)

    total    = running_avg * Q
    avg_hits = float(np.mean([len(r) for r in results]))

    print(f"  total        : {total:.3f} s")
    print(f"  avg query    : {running_avg * 1000:.3f} ms  (running average)")
    print(f"  QPS          : {Q / total:.1f}")
    print(f"  avg hits     : {avg_hits:.1f}")


def main() -> None:
    from dataset import datasets, datasets_dir

    ap = argparse.ArgumentParser(description='Linear scan Hamming range query benchmark')
    ap.add_argument('--dataset', choices=list(datasets), default='sift',
                    help='dataset to use (default: sift)')
    ap.add_argument('--radius', type=int, default=16,
                    help='Hamming radius for range query (default: 16)')
    ap.add_argument('--query_count', type=int, default=100,
                    help='number of queries to run (default: 100)')
    ap.add_argument('--seed', type=int, default=None,
                    help='RNG seed for GRP projection matrix (default: random)')
    args = ap.parse_args()

    dataset      = datasets[args.dataset]
    dataset_dir = os.path.join(datasets_dir, dataset)
    run_linear_scan(dataset_dir, dataset, args.radius, args.query_count, args.seed)


if __name__ == '__main__':
    main()
