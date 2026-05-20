import os
import time
import numpy as np
from linear_scan import linear_scan_range

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def read_fvecs(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, 1 + d)[:, 1:].view(np.float32)


def binarize(X: np.ndarray, thresholds: np.ndarray = None) -> np.ndarray:
    """Threshold each dimension; pack 8 bits per byte -> (N, D/8) uint8."""
    if thresholds is None:
        thresholds = X.mean(axis=0)
    bits = (X > thresholds).astype(np.uint8)
    return np.packbits(bits, axis=1)


def run_linear_scan(dataset_dir: str, prefix: str, radius: int):
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

    #Q = len(query_codes)
    Q = 100
    print(f"\nLinear scan  r={radius}, Q={Q} ...")
    t0 = time.perf_counter()
    results = [linear_scan_range(base_codes, query_codes[i], radius) for i in range(Q)]
    elapsed = time.perf_counter() - t0

    avg_hits = float(np.mean([len(r) for r in results]))
    qps      = Q / elapsed

    print(f"  total   : {elapsed:.3f} s")
    print(f"  QPS     : {qps:.1f}")
    print(f"  avg hits: {avg_hits:.1f}")
