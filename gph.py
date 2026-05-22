import os
import sys
import time
import argparse
import tracemalloc
import numpy as np

from linear_scan import hamming_distances
from dataset import read_fvecs, binarize


def _byte_boundaries(n_bytes: int) -> list:
    return [(i * 8, (i + 1) * 8) for i in range(n_bytes)]


def _bit_boundaries(n_bits: int, m: int) -> list:
    base, rem = divmod(n_bits, m)
    boundaries = []
    start = 0
    for i in range(m):
        end = start + base + (1 if i < rem else 0)
        boundaries.append((start, end))
        start = end
    return boundaries


def build_index(codes: np.ndarray, r: int) -> tuple:
    """Build GPH index.

    When D >= r+1 (enough bytes): use D byte-level segments with threshold = D - r.
    Any code within Hamming distance r differs in at most r bytes, so it must
    appear in at least D - r tables exactly — eliminating nearly all false positives.

    When D < r+1: fall back to r+1 bit-level segments with threshold = 1
    (equivalent to HmSearch), since byte-level pigeonhole no longer holds.
    """
    N, D = codes.shape
    n_bits = D * 8

    if D >= r + 1:
        boundaries = _byte_boundaries(D)
        threshold = D - r
    else:
        boundaries = _bit_boundaries(n_bits, r + 1)
        threshold = 1

    bits = np.unpackbits(codes, axis=1)
    tables: list[dict] = [{} for _ in range(len(boundaries))]
    for k, (s, e) in enumerate(boundaries):
        seg_bytes = np.packbits(bits[:, s:e], axis=1)
        for i in range(N):
            key = seg_bytes[i].tobytes()
            if key in tables[k]:
                tables[k][key].append(i)
            else:
                tables[k][key] = [i]

    return tables, boundaries, threshold


def gph_query(
    tables: list,
    boundaries: list,
    codes: np.ndarray,
    q: np.ndarray,
    r: int,
    threshold: int,
) -> np.ndarray:
    """Return indices of rows in codes with Hamming distance <= r to q.

    Each candidate must appear in at least `threshold` segment tables.
    Counting is done via np.unique (C-level sort) rather than a Python Counter,
    keeping the vote-accumulation step fast even with large bucket sizes.
    """
    q_bits = np.unpackbits(q)
    hits = []
    for k, (s, e) in enumerate(boundaries):
        key = np.packbits(q_bits[s:e]).tobytes()
        if key in tables[k]:
            hits.append(np.array(tables[k][key], dtype=np.int32))

    if not hits:
        return np.array([], dtype=np.int32)

    combined = np.concatenate(hits)
    unique_idx, counts = np.unique(combined, return_counts=True)
    candidates = unique_idx[counts >= threshold]

    if len(candidates) == 0:
        return np.array([], dtype=np.int32)
    dists = hamming_distances(codes[candidates], q)
    return candidates[dists <= r]


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


def run_gph(dataset_dir: str, prefix: str, radius: int, query_count: int) -> None:
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
    D = base_codes.shape[1]
    print(f"  code shape : {base_codes.shape}  dtype={base_codes.dtype}")

    # ── Build index ──────────────────────────────────────────────────────────
    print("\nBuilding GPH index ...")
    tracemalloc.start()
    t0 = time.perf_counter()
    tables, boundaries, threshold = build_index(base_codes, radius)
    build_time = time.perf_counter() - t0
    _, peak_build = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    m           = len(tables)
    idx_bytes   = _index_size_bytes(tables)
    codes_bytes = base_codes.nbytes
    mode        = 'byte' if D >= radius + 1 else 'bit'

    print(f"  mode         : {mode}-level segments")
    print(f"  segments     : m={m},  threshold={threshold}/{m}")
    print(f"  build time   : {build_time:.3f} s")
    print(f"  codes array  : {codes_bytes / 1e6:.2f} MB")
    print(f"  index size   : {idx_bytes / 1e6:.2f} MB  (deep estimate)")
    print(f"  peak (build) : {peak_build / 1e6:.2f} MB  (tracemalloc)")

    # ── Query ─────────────────────────────────────────────────────────────────
    Q = min(query_count, len(query_codes))
    print(f"\nGPH range query  r={radius}, m={m}, threshold={threshold}, Q={Q} ...")
    tracemalloc.start()
    results = []
    running_avg = 0.0
    for i in range(Q):
        t0 = time.perf_counter()
        res = gph_query(tables, boundaries, base_codes, query_codes[i], radius, threshold)
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
    ap = argparse.ArgumentParser(description='GPH Hamming range query benchmark')
    ap.add_argument('--dataset', choices=list(DATASETS), default='siftsmall',
                    help='dataset to use (default: siftsmall)')
    ap.add_argument('--radius', type=int, default=20,
                    help='Hamming radius for range query (default: 20)')
    ap.add_argument('--query_count', type=int, default=100,
                    help='number of queries to run (default: 100)')
    args = ap.parse_args()

    prefix      = DATASETS[args.dataset]
    dataset_dir = os.path.join(DATASETS_DIR, prefix)
    run_gph(dataset_dir, prefix, args.radius, args.query_count)


if __name__ == '__main__':
    main()
