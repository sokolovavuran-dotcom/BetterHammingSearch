import os
import time
import argparse
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

def run_linear_scan(dataset_dir: str, prefix: str, radius: int, query_count: int) -> None:
    # deferred import to avoid circular dependency (dataset.py imports linear_scan_range)
    from dataset import read_fvecs, binarize

    base_path  = os.path.join(dataset_dir, f'{prefix}_base.fvecs')
    query_path = os.path.join(dataset_dir, f'{prefix}_query.fvecs')

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset not found: {base_path}")

    print(f"Loading {prefix} ...")
    base    = read_fvecs(base_path)
    queries = read_fvecs(query_path)
    print(f"  base    : {base.shape}")
    print(f"  queries : {queries.shape}")

    print("Binarizing ...")
    mean        = base.mean(axis=0)
    base_codes  = binarize(base, mean)
    query_codes = binarize(queries, mean)
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
    datasets = {
        'sift': 'sift',
        'siftsmall': 'siftsmall',
    }

    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')

    ap = argparse.ArgumentParser(description='Linear scan Hamming range query benchmark')
    ap.add_argument('--dataset', choices=list(datasets), default='sift',
                    help='dataset to use (default: sifts)')
    ap.add_argument('--radius', type=int, default=5,
                    help='Hamming radius for range query (default: 5)')
    ap.add_argument('--query_count', type=int, default=100,
                    help='number of queries to run (default: 100)')
    args = ap.parse_args()

    prefix      = datasets[args.dataset]
    dataset_dir = os.path.join(datasets_dir, prefix)
    run_linear_scan(dataset_dir, prefix, args.radius, args.query_count)


if __name__ == '__main__':
    main()
