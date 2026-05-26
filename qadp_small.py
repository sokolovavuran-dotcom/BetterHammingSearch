"""QADP SP-STC: Query-Aware Dimension Partitioning — Small Threshold Case.

Small threshold case applies when r < 0.05 * d (searching threshold < 5% of bit-length).
Uses the Anti-Pigeonhole Principle: each subspace has r dimensions, m = ceil(d/r) subspaces.
Dimensions are sorted by JS-divergence (descending) and assigned contiguously so that the
most distinctive dimensions are checked first, enabling early elimination.
"""
import os
import sys
import math
import time
import argparse
import tracemalloc
import numpy as np

from linear_scan import hamming_distances
from dataset import read_fvecs, binarize


# ── Helpers (same as qadp_normal.py) ──────────────────────────────────────────

def compute_sample_count(d: int, eps: float, delta: float) -> int:
    """Sample count l from Lemma 3 (Formula 2 in the paper)."""
    log_inv = math.log(1.0 / delta)
    denom = 2.0 * d * eps * eps + log_inv
    l1 = (d + 1) * log_inv / denom
    l2 = (2.0 * d * eps * eps + d * log_inv) / denom
    return max(1, min(d, math.ceil(min(l1, l2))))


def _js_div_all(data_p1: np.ndarray, q_bits: np.ndarray) -> np.ndarray:
    """Vectorised JS-divergence between data and query distribution per bit.

    data_p1 : (d,) fraction of 1s in each dimension across the dataset
    q_bits  : (d,)   uint8 query bits
    """
    p0 = 1.0 - data_p1
    p1 = data_p1
    q0 = 1.0 - q_bits.astype(np.float64)
    q1 = q_bits.astype(np.float64)

    m0 = 0.5 * (p0 + q0)
    m1 = 0.5 * (p1 + q1)

    js = np.zeros(len(data_p1), dtype=np.float64)

    def _kl_contrib(x, m, scale):
        mask = (x > 0) & (m > 0)
        js[mask] += scale * x[mask] * np.log(x[mask] / m[mask])

    _kl_contrib(p0, m0, 0.5)
    _kl_contrib(p1, m1, 0.5)
    _kl_contrib(q0, m0, 0.5)
    _kl_contrib(q1, m1, 0.5)
    return js


def qadp_small_partition(d: int, r: int, js_div: np.ndarray) -> list:
    """Partition d bit-dimensions into subspaces of size r (QADP small-threshold case).

    Sort dims by JS-divergence descending (most distinctive first) then assign
    them contiguously: subspace 0 gets the top-r dims, subspace 1 the next r, etc.
    Processing subspaces in order means we see the most filtering dims first.
    """
    sorted_dims = np.argsort(js_div)[::-1]    # descending JS
    m = math.ceil(d / r)
    subspaces = []
    for k in range(m):
        chunk = sorted_dims[k * r: min((k + 1) * r, d)]
        subspaces.append(chunk.copy())
    return subspaces


# ── Core query (Algorithm 2: SP-STC) ───────────────────────────────────────────

def sp_stc_query(
    bits: np.ndarray,
    q_bits: np.ndarray,
    r: int,
    subspaces: list,
    l: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Approximate range query — Small Threshold Case.

    bits      : (N, d) uint8 unpacked bit matrix
    q_bits    : (d,)   uint8 query bits
    r         : Hamming radius
    subspaces : list of arrays, each a set of bit-dimension indices (size ≈ r)
    l         : sampling threshold
    """
    N, d = bits.shape
    all_dims = np.concatenate(subspaces)      # ordered dim list for sampling

    Hc = np.zeros(N, dtype=np.int32)          # cumulative H in checked dims
    Bc = np.zeros(N, dtype=np.int32)          # cumulative checked dim count
    in_C = np.ones(N, dtype=bool)             # candidate set C (starts full)
    R = []

    for sub_dims in subspaces:
        active = np.where(in_C)[0]
        if len(active) == 0:
            break

        sub_size = len(sub_dims)
        h_sub = (bits[np.ix_(active, sub_dims)] != q_bits[sub_dims]).sum(axis=1).astype(np.int32)

        Hc[active] += h_sub
        Bc[active] += sub_size

        Hc_a = Hc[active]
        Bc_a = Bc[active]

        # Anti-Pigeonhole: cumulative H already exceeds r → definitely not a result
        exceed = Hc_a > r

        # Enough dimensions sampled: decide via estimated H
        not_exceed = ~exceed
        enough = not_exceed & (Bc_a >= l)
        enough_idx = active[enough]
        if enough_idx.size:
            est = d / Bc[enough_idx] * Hc[enough_idx]
            R.extend(enough_idx[est <= r].tolist())

        # Remove from C all resolved entries (eliminated or decided)
        in_C[active[exceed | enough]] = False

    # Sampling / finalisation for remaining candidates (grouped by Bc for speed).
    C_arr = np.where(in_C)[0]
    if C_arr.size:
        unique_bc = np.unique(Bc[C_arr])
        for bc_val in unique_bc:
            group = C_arr[Bc[C_arr] == bc_val]
            need = l - int(bc_val)
            if need <= 0:
                divisor = max(int(bc_val), 1)
                est = (d / divisor) * Hc[group]
                R.extend(group[est <= r].tolist())
                continue
            unchecked = all_dims[int(bc_val):]
            n_sample = min(need, unchecked.size)
            if n_sample > 0:
                sampled_dims = unchecked[rng.choice(unchecked.size, size=n_sample, replace=False)]
                h_s = (bits[np.ix_(group, sampled_dims)] != q_bits[sampled_dims]).sum(axis=1).astype(np.int32)
                Hc[group] += h_s
            est = (d / l) * Hc[group]
            R.extend(group[est <= r].tolist())

    if not R:
        return np.array([], dtype=np.int32)

    R_arr = np.unique(np.array(R, dtype=np.int32))
    exact_h = (bits[R_arr] != q_bits).sum(axis=1)
    return R_arr[exact_h <= r]


# ── Benchmark ──────────────────────────────────────────────────────────────────

DATASETS = {'sift': 'sift', 'siftsmall': 'siftsmall'}
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')


def run_qadp_small(
    dataset_dir: str,
    prefix: str,
    radius: int,
    query_count: int,
    eps: float,
    delta: float,
) -> None:
    base_path  = os.path.join(dataset_dir, f'{prefix}_base.fvecs')
    query_path = os.path.join(dataset_dir, f'{prefix}_query.fvecs')

    if not os.path.exists(base_path):
        raise FileNotFoundError(f'Dataset not found: {base_path}')

    print(f'Loading {prefix} ...')
    base    = read_fvecs(base_path)
    queries = read_fvecs(query_path)
    print(f'  base    : {base.shape}')
    print(f'  queries : {queries.shape}')

    print('Binarizing ...')
    base_codes  = binarize(base)
    query_codes = binarize(queries)
    D_bytes = base_codes.shape[1]
    d = D_bytes * 8
    print(f'  code shape : {base_codes.shape}  dtype={base_codes.dtype}  bits={d}')

    small_thresh = 0.05 * d
    if radius >= small_thresh:
        print(f'  WARNING: r={radius} >= 0.05*d={small_thresh:.1f} — '
              f'this is a normal-threshold case; consider using qadp_normal.py')

    l = compute_sample_count(d, eps, delta)
    m = math.ceil(d / radius)
    print(f'  eps={eps}, delta={delta} => l={l} samples')
    print(f'  subspaces m={m} (each ~{radius} bits, last may be smaller)')

    print('Unpacking bits ...')
    base_bits  = np.unpackbits(base_codes,  axis=1)
    query_bits = np.unpackbits(query_codes, axis=1)

    data_p1 = base_bits.mean(axis=0)

    rng = np.random.default_rng(0)

    Q = min(query_count, len(query_bits))
    print(f'\nQADP-STC range query  r={radius}, m={m}, Q={Q} ...')
    tracemalloc.start()
    results = []
    running_avg = 0.0
    for i in range(Q):
        q_bits_i  = query_bits[i]
        js_div    = _js_div_all(data_p1, q_bits_i)
        subspaces = qadp_small_partition(d, radius, js_div)
        t0 = time.perf_counter()
        res = sp_stc_query(base_bits, q_bits_i, radius, subspaces, l, rng)
        qt = time.perf_counter() - t0
        running_avg += (qt - running_avg) / (i + 1)
        results.append(res)

    _, peak_query = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total    = running_avg * Q
    avg_hits = float(np.mean([len(r) for r in results]))

    print(f'  total        : {total:.3f} s')
    print(f'  avg query    : {running_avg * 1000:.3f} ms  (running average)')
    print(f'  QPS          : {Q / total:.1f}')
    print(f'  avg hits     : {avg_hits:.1f}')
    print(f'  peak (query) : {peak_query / 1e6:.2f} MB  (tracemalloc)')


def main() -> None:
    ap = argparse.ArgumentParser(
        description='QADP Small Threshold Case Hamming range query benchmark'
    )
    ap.add_argument('--dataset', choices=list(DATASETS), default='siftsmall',
                    help='dataset to use (default: siftsmall)')
    ap.add_argument('--radius', type=int, default=4,
                    help='Hamming radius for range query (default: 4)')
    ap.add_argument('--query_count', type=int, default=100,
                    help='number of queries to run (default: 100)')
    ap.add_argument('--eps', type=float, default=0.1,
                    help='sampling error epsilon (default: 0.1)')
    ap.add_argument('--delta', type=float, default=0.1,
                    help='sampling confidence delta (default: 0.1)')
    args = ap.parse_args()

    prefix      = DATASETS[args.dataset]
    dataset_dir = os.path.join(DATASETS_DIR, prefix)
    run_qadp_small(dataset_dir, prefix, args.radius, args.query_count,
                   args.eps, args.delta)


if __name__ == '__main__':
    main()
