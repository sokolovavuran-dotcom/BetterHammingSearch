import os
import sys
import time
import argparse
import tracemalloc
import numpy as np

from linear_scan import hamming_distances
from dataset import read_fvecs, binarize


def _seg_boundaries(n_bits: int, m: int) -> list:
    """Return (start_bit, end_bit) for each of m segments, distributing n_bits as evenly as possible."""
    base, rem = divmod(n_bits, m)
    boundaries = []
    start = 0
    for i in range(m):
        end = start + base + (1 if i < rem else 0)
        boundaries.append((start, end))
        start = end
    return boundaries


def build_index(codes: np.ndarray, r: int) -> tuple:
    """Build r+1 hash tables (one per bit-segment) for HmSearch.

    Pigeonhole guarantee: any code within Hamming distance r must match at least
    one of the r+1 segments exactly, so only exact lookups are needed at query time.
    """
    N, D = codes.shape
    n_bits = D * 8
    m = r + 1
    boundaries = _seg_boundaries(n_bits, m)
    bits = np.unpackbits(codes, axis=1)  # (N, n_bits)

    tables: list[dict] = [{} for _ in range(m)]
    for k, (s, e) in enumerate(boundaries):
        seg_bytes = np.packbits(bits[:, s:e], axis=1)
        for i in range(N):
            key = seg_bytes[i].tobytes()
            if key in tables[k]:
                tables[k][key].append(i)
            else:
                tables[k][key] = [i]

    return tables, boundaries


def hmsearch_query(
    tables: list,
    boundaries: list,
    codes: np.ndarray,
    q: np.ndarray,
    r: int,
) -> np.ndarray:
    """Return indices of rows in codes with Hamming distance <= r to q.

    For each of the r+1 segments, performs an exact hash lookup.
    Any true result is guaranteed to appear in at least one segment's bucket.
    """
    q_bits = np.unpackbits(q)
    candidates: set[int] = set()
    for k, (s, e) in enumerate(boundaries):
        key = np.packbits(q_bits[s:e]).tobytes()
        if key in tables[k]:
            candidates.update(tables[k][key])
    if not candidates:
        return np.array([], dtype=np.int32)
    cand = np.fromiter(candidates, dtype=np.int32)
    dists = hamming_distances(codes[cand], q)
    return cand[dists <= r]


def _index_size_bytes(tables: list) -> int:
    total = 0
    for t in tables:
        total += sys.getsizeof(t)
        for key, lst in t.items():
            total += sys.getsizeof(key) + sys.getsizeof(lst)
            total += len(lst) * 28
    return total


DATASETS = {
    'sift':      'sift',
    'siftsmall': 'siftsmall',
}

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')


def run_hmsearch(dataset_dir: str, prefix: str, radius: int, query_count: int) -> None:
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
    n_bits = base_codes.shape[1] * 8
    m = radius + 1
    print(f"  code shape : {base_codes.shape}  dtype={base_codes.dtype}")
    print(f"  segments   : m={m}  (~{n_bits // m} bits each)")

    # ── Build index ──────────────────────────────────────────────────────────
    print("\nBuilding HmSearch index ...")
    tracemalloc.start()
    t0 = time.perf_counter()
    tables, boundaries = build_index(base_codes, radius)
    build_time = time.perf_counter() - t0
    _, peak_build = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    idx_bytes   = _index_size_bytes(tables)
    codes_bytes = base_codes.nbytes

    print(f"  build time   : {build_time:.3f} s")
    print(f"  codes array  : {codes_bytes / 1e6:.2f} MB")
    print(f"  index size   : {idx_bytes / 1e6:.2f} MB  (deep estimate)")
    print(f"  peak (build) : {peak_build / 1e6:.2f} MB  (tracemalloc)")

    # ── Query ─────────────────────────────────────────────────────────────────
    Q = min(query_count, len(query_codes))
    print(f"\nHmSearch range query  r={radius}, m={m}, Q={Q} ...")
    tracemalloc.start()
    results = []
    running_avg = 0.0
    for i in range(Q):
        t0 = time.perf_counter()
        res = hmsearch_query(tables, boundaries, base_codes, query_codes[i], radius)
        qt = time.perf_counter() - t0
        running_avg += (qt - running_avg) / (i + 1)
        results.append(res)
    _, peak_query = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total    = running_avg * Q
    avg_hits = float(np.mean([len(r) for r in results]))

    print(f"  total        : {total:.3f} s")
    print(f"  avg query    : {running_avg * 1000:.3f} ms  (running average)")
    print(f"  QPS          : {Q / total:.1f}")
    print(f"  avg hits     : {avg_hits:.1f}")
    print(f"  peak (query) : {peak_query / 1e6:.2f} MB  (tracemalloc)")


def main() -> None:
    ap = argparse.ArgumentParser(description='HmSearch Hamming range query benchmark')
    ap.add_argument('--dataset', choices=list(DATASETS), default='siftsmall',
                    help='dataset to use (default: siftsmall)')
    ap.add_argument('--radius', type=int, default=20,
                    help='Hamming radius for range query (default: 20)')
    ap.add_argument('--query_count', type=int, default=100,
                    help='number of queries to run (default: 100)')
    args = ap.parse_args()

    prefix      = DATASETS[args.dataset]
    dataset_dir = os.path.join(DATASETS_DIR, prefix)
    run_hmsearch(dataset_dir, prefix, args.radius, args.query_count)


if __name__ == '__main__':
    main()
