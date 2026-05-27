import os
import sys
import math
import time
import argparse
import tracemalloc
import itertools
import numpy as np

from linear_scan import hamming_distances
from dataset import read_fvecs, binarize


def _build_masks(n_bits: int, max_dist: int) -> np.ndarray:
    """Precompute all XOR masks for Hamming distances 0..max_dist.

    Returns (K, n_bits//8) uint8 array where K = sum C(n_bits, d) for d in [0, max_dist].
    Applying masks ^ q_sub in one numpy broadcast gives all neighbors at once.
    """
    n_bytes = n_bits // 8
    masks = []
    for dist in range(max_dist + 1):
        for positions in itertools.combinations(range(n_bits), dist):
            mask = np.zeros(n_bytes, dtype=np.uint8)
            for p in positions:
                mask[p // 8] ^= np.uint8(1 << (p % 8))
            masks.append(mask)
    return np.array(masks, dtype=np.uint8)


def build_index(codes: np.ndarray, m: int) -> list:
    """Build m hash tables (one per segment) mapping sub-code bytes -> list of row indices."""
    N, D = codes.shape
    seg = D // m
    tables: list[dict] = [{} for _ in range(m)]
    for k in range(m):
        chunk = codes[:, k * seg:(k + 1) * seg]
        for i in range(N):
            key = chunk[i].tobytes()
            if key in tables[k]:
                tables[k][key].append(i)
            else:
                tables[k][key] = [i]
    return tables


def mih_query(
    tables: list,
    seg: int,
    codes: np.ndarray,
    q: np.ndarray,
    r: int,
    masks: np.ndarray,
) -> np.ndarray:
    """Return indices of rows in codes with Hamming distance <= r to q.

    Pigeonhole principle: any candidate within distance r must have at least one
    segment within distance floor(r/m), so we enumerate those neighbors per
    segment, union the candidate sets, then verify with exact distance.
    """
    m = len(tables)
    candidates: set[int] = set()
    for k in range(m):
        q_sub = q[k * seg:(k + 1) * seg]
        neighbors = masks ^ q_sub          # (K, seg) broadcast XOR — no Python loop
        for nb in neighbors:
            if (key := nb.tobytes()) in tables[k]:
                candidates.update(tables[k][key])
    if not candidates:
        return np.array([], dtype=np.int32)
    cand = np.fromiter(candidates, dtype=np.int32)
    dists = hamming_distances(codes[cand], q)
    return cand[dists <= r]


def _index_size_bytes(tables: list) -> int:
    """Deep-size estimate: dict overhead + keys + list objects + int items."""
    total = 0
    for t in tables:
        total += sys.getsizeof(t)
        for key, lst in t.items():
            # bytes key, list object (includes internal pointer array)
            total += sys.getsizeof(key) + sys.getsizeof(lst)
            # approximate each stored Python int at 28 bytes
            total += len(lst) * 28
    return total


DATASETS = {
    'sift':      'sift',
    'siftsmall': 'siftsmall',
}

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')


def run_mih(dataset_dir: str, prefix: str, radius: int, m: int, query_count: int) -> None:
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
    base_codes  = binarize(base)
    query_codes = binarize(queries)
    D = base_codes.shape[1]
    print(f"  code shape : {base_codes.shape}  dtype={base_codes.dtype}")

    if D % m != 0:
        valid = [i for i in range(1, D + 1) if D % i == 0]
        raise ValueError(
            f"Code length D={D} bytes is not divisible by m={m}. "
            f"Valid choices for m: {valid}"
        )

    seg   = D // m
    r_seg = radius // m
    n_neighbors = sum(math.comb(seg * 8, d) for d in range(r_seg + 1))

    print(f"\n  segments     : m={m},  seg={seg} bytes ({seg * 8} bits)")
    warn = "  (WARNING: large neighbor count, consider increasing --segments)" if n_neighbors > 10_000 else ""
    print(f"  r / r_seg    : {radius} / {r_seg}  ->  {n_neighbors:,} neighbors per segment{warn}")

    # ── Build index ──────────────────────────────────────────────────────────
    print("\nBuilding MIH index ...")
    tracemalloc.start()
    t0 = time.perf_counter()
    tables = build_index(base_codes, m)
    build_time = time.perf_counter() - t0
    _, peak_build = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    idx_bytes   = _index_size_bytes(tables)
    codes_bytes = base_codes.nbytes

    print(f"  build time   : {build_time:.3f} s")
    print(f"  codes array  : {codes_bytes / 1e6:.2f} MB")
    print(f"  index size   : {idx_bytes / 1e6:.2f} MB  (deep estimate)")
    print(f"  peak (build) : {peak_build / 1e6:.2f} MB  (tracemalloc)")

    # Precompute masks once; vectorised XOR in mih_query avoids per-neighbor Python loop
    masks = _build_masks(seg * 8, r_seg)

    # ── Query ─────────────────────────────────────────────────────────────────
    Q = min(query_count, len(query_codes))
    print(f"\nMIH range query  r={radius}, m={m}, Q={Q} ...")
    tracemalloc.start()
    results = []
    running_avg = 0.0
    for i in range(Q):
        t0 = time.perf_counter()
        res = mih_query(tables, seg, base_codes, query_codes[i], radius, masks)
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
    ap = argparse.ArgumentParser(
        description='Multi-Index Hashing Hamming range query benchmark'
    )
    ap.add_argument(
        '--dataset', choices=list(DATASETS), default='sift',
        help='dataset to use (default: sift)',
    )
    ap.add_argument(
        '--radius', type=int, default=16,
        help='Hamming radius for range query (default: 16)',
    )
    ap.add_argument(
        '--segments', type=int, default=8,
        help='number of index segments m (default: 8); must divide code length D in bytes',
    )
    ap.add_argument(
        '--query_count', type=int, default=100,
        help='number of queries to run (default: 100)',
    )
    args = ap.parse_args()

    prefix      = DATASETS[args.dataset]
    dataset_dir = os.path.join(DATASETS_DIR, prefix)
    run_mih(dataset_dir, prefix, args.radius, args.segments, args.query_count)


if __name__ == '__main__':
    main()
