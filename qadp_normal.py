"""QADP SP-NC: Query-Aware Dimension Partitioning — Normal Case.

Normal case applies when r >= 0.05 * d (searching threshold >= 5% of bit-length).
Uses the basic Pigeonhole Principle with m = r subspaces of size d/r bits each.
Dimensions are sorted by JS-divergence (ascending) and interleaved across subspaces
so that skewed dimensions land in different subspaces.
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


# ── Helpers ────────────────────────────────────────────────────────────────────

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
    q_bits  : (d,) uint8 binary query vector (unpacked bits)
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


def qadp_normal_partition(d: int, r: int, js_div: np.ndarray) -> list:
    """Partition d bit-dimensions into m=r subspaces (QADP normal case).

    Sort dims by JS-divergence ascending then interleave: reshape the sorted
    list into (d//r, r) and take each column as one subspace.  This ensures
    that highly skewed dimensions (small JS) are spread across all subspaces.
    """
    m = r
    sorted_dims = np.argsort(js_div)          # ascending JS
    n_full = (d // m) * m

    # Interleaved assignment: reshape prefix into (d//m, m)
    if n_full > 0:
        parts = sorted_dims[:n_full].reshape(d // m, m)
        subspaces = [parts[:, k].tolist() for k in range(m)]
    else:
        subspaces = [[] for _ in range(m)]

    # Distribute any remaining dims round-robin
    for idx, dim in enumerate(sorted_dims[n_full:]):
        subspaces[idx % m].append(int(dim))

    return [np.array(s, dtype=np.int32) for s in subspaces]


# ── Core query (Algorithm 1: SP-NC) ────────────────────────────────────────────

def sp_nc_query(
    bits: np.ndarray,
    q_bits: np.ndarray,
    r: int,
    subspaces: list,
    l: int,
    rng: np.random.Generator,
    timers: dict = None,
    subspace_stats: list = None,
) -> np.ndarray:
    """Approximate range query — Normal Case.

    bits      : (N, d) uint8 unpacked bit matrix
    q_bits    : (d,)   uint8 query bits
    r         : Hamming radius
    subspaces : list of m arrays, each a set of bit-dimension indices
    l         : sampling threshold (number of dimensions to sample)
    timers    : optional dict — cumulative per-stage timings are added here
    subspace_stats : optional list — per-subspace |active| size is appended here
    """
    use_timers = timers is not None

    t_setup = time.perf_counter()
    N, d = bits.shape
    all_dims = np.concatenate(subspaces)      # ordered dim list for sampling

    Hc = np.zeros(N, dtype=np.int32)          # cumulative H in checked dims
    Bc = np.zeros(N, dtype=np.int32)          # cumulative checked dim count
    in_M = np.ones(N, dtype=bool)             # still in intermediate set M
    in_C = np.zeros(N, dtype=bool)            # in candidate set C
    R = []
    if use_timers:
        timers['1_setup'] += time.perf_counter() - t_setup

    for k, sub_dims in enumerate(subspaces):
        t_where = time.perf_counter()
        active = np.where(in_M)[0]
        if use_timers:
            timers['2a_where_active'] += time.perf_counter() - t_where
        if subspace_stats is not None:
            subspace_stats.append((k, int(active.size)))
        if len(active) == 0:
            break

        sub_size = len(sub_dims)

        t_fancy = time.perf_counter()
        h_sub = (bits[np.ix_(active, sub_dims)] != q_bits[sub_dims]).sum(axis=1).astype(np.int32)
        if use_timers:
            timers['2b_fancy_index'] += time.perf_counter() - t_fancy

        t_acc = time.perf_counter()
        Hc[active] += h_sub
        Bc[active] += sub_size
        Hc_a = Hc[active]
        Bc_a = Bc[active]
        if use_timers:
            timers['2c_accumulate'] += time.perf_counter() - t_acc

        t_prune = time.perf_counter()
        # Priority 1 — cumulative H already exceeds r (Anti-pigeonhole pruning)
        exceed = Hc_a > r

        # Priority 2 — enough dimensions sampled: decide via estimated H
        not_exceed = ~exceed
        enough = not_exceed & (Bc_a >= l)
        enough_idx = active[enough]
        if enough_idx.size:
            est = d / Bc[enough_idx] * Hc[enough_idx]
            R.extend(enough_idx[est <= r].tolist())

        # Priority 3 — Pigeonhole: H in this subspace <= local threshold 1
        pigeon = not_exceed & (Bc_a < l) & (h_sub <= 1)
        in_C[active[pigeon]] = True

        # Remove from M all resolved entries
        in_M[active[exceed | enough | pigeon]] = False
        if use_timers:
            timers['2d_prune_masks'] += time.perf_counter() - t_prune

    # Sampling phase for C (candidates from pigeonhole).
    t_cand = time.perf_counter()
    C_arr = np.where(in_C)[0]
    if C_arr.size:
        unique_bc = np.unique(Bc[C_arr])
        for bc_val in unique_bc:
            group = C_arr[Bc[C_arr] == bc_val]
            need = l - int(bc_val)
            if need <= 0:
                est = (d / int(bc_val)) * Hc[group]
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
    if use_timers:
        timers['3_candidate_phase'] += time.perf_counter() - t_cand

    t_exact = time.perf_counter()
    if not R:
        if use_timers:
            timers['4_exact_verify'] += time.perf_counter() - t_exact
            timers['_cand_count'] += 0
        return np.array([], dtype=np.int32)

    R_arr = np.unique(np.array(R, dtype=np.int32))
    exact_h = (bits[R_arr] != q_bits).sum(axis=1)
    result = R_arr[exact_h <= r]
    if use_timers:
        timers['4_exact_verify'] += time.perf_counter() - t_exact
        timers['_cand_count'] += int(R_arr.size)
    return result


# ── Benchmark ──────────────────────────────────────────────────────────────────

DATASETS = {'sift': 'sift', 'siftsmall': 'siftsmall'}
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')


def run_qadp_normal(
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
    if radius < small_thresh:
        print(f'  WARNING: r={radius} < 0.05*d={small_thresh:.1f} — '
              f'this is a small-threshold case; consider using qadp_small.py')

    l = compute_sample_count(d, eps, delta)
    m = radius
    print(f'  eps={eps}, delta={delta} => l={l} samples')
    print(f'  subspaces m={m} (each ~{d // m} bits)')

    # Unpack all codes to bit matrices
    print('Unpacking bits ...')
    base_bits  = np.unpackbits(base_codes,  axis=1)   # (N, d) uint8
    query_bits = np.unpackbits(query_codes, axis=1)   # (Q, d) uint8

    # Data distribution: fraction of 1s per dimension (computed once)
    data_p1 = base_bits.mean(axis=0)                  # (d,) float64

    rng = np.random.default_rng(0)

    Q = min(query_count, len(query_bits))
    print(f'\nQADP-NC range query  r={radius}, m={m}, Q={Q} ...')
    tracemalloc.start()
    results = []
    running_avg = 0.0

    # Cumulative per-stage timers (summed across all queries)
    stage_timers = {
        '0_js_div'         : 0.0,
        '0_partition'      : 0.0,
        '1_setup'          : 0.0,
        '2a_where_active'  : 0.0,
        '2b_fancy_index'   : 0.0,
        '2c_accumulate'    : 0.0,
        '2d_prune_masks'   : 0.0,
        '3_candidate_phase': 0.0,
        '4_exact_verify'   : 0.0,
        '_cand_count'      : 0,
    }
    # Per-subspace |active| sizes (across all queries) — shows pruning behavior
    all_subspace_stats = []

    for i in range(Q):
        q_bits_i = query_bits[i]

        t_js = time.perf_counter()
        js_div = _js_div_all(data_p1, q_bits_i)
        stage_timers['0_js_div'] += time.perf_counter() - t_js

        t_part = time.perf_counter()
        subspaces = qadp_normal_partition(d, m, js_div)
        stage_timers['0_partition'] += time.perf_counter() - t_part

        per_query_stats = []
        t0 = time.perf_counter()
        res = sp_nc_query(base_bits, q_bits_i, radius, subspaces, l, rng,
                          timers=stage_timers, subspace_stats=per_query_stats)
        qt = time.perf_counter() - t0
        running_avg += (qt - running_avg) / (i + 1)
        results.append(res)
        all_subspace_stats.append(per_query_stats)

        # Per-query summary line
        print(f'  q[{i:02d}]  sp_nc={qt*1000:7.2f} ms  '
              f'js={(stage_timers["0_js_div"]):.4f}s  '
              f'hits={len(res):5d}  '
              f'first/last active = {per_query_stats[0][1] if per_query_stats else 0} / '
              f'{per_query_stats[-1][1] if per_query_stats else 0}')

    _, peak_query = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total    = running_avg * Q
    avg_hits = float(np.mean([len(r) for r in results]))

    print(f'\n  total        : {total:.3f} s')
    print(f'  avg query    : {running_avg * 1000:.3f} ms  (running average)')
    print(f'  QPS          : {Q / total:.1f}')
    print(f'  avg hits     : {avg_hits:.1f}')
    print(f'  peak (query) : {peak_query / 1e6:.2f} MB  (tracemalloc)')

    # ── Stage breakdown ────────────────────────────────────────────────────────
    print(f'\n  ─── Stage breakdown (cumulative across {Q} queries) ───')
    sp_nc_total = (stage_timers['1_setup']
                   + stage_timers['2a_where_active']
                   + stage_timers['2b_fancy_index']
                   + stage_timers['2c_accumulate']
                   + stage_timers['2d_prune_masks']
                   + stage_timers['3_candidate_phase']
                   + stage_timers['4_exact_verify'])
    grand_total = sp_nc_total + stage_timers['0_js_div'] + stage_timers['0_partition']

    stage_labels = {
        '0_js_div'         : 'JS-divergence (per-query)',
        '0_partition'      : 'Subspace partition (per-query)',
        '1_setup'          : 'sp_nc_query setup (alloc Hc/Bc/in_M)',
        '2a_where_active'  : '  loop: np.where(in_M)',
        '2b_fancy_index'   : '  loop: bits[np.ix_(active, sub_dims)] != q',
        '2c_accumulate'    : '  loop: Hc/Bc scatter-add',
        '2d_prune_masks'   : '  loop: exceed/enough/pigeon masks',
        '3_candidate_phase': 'Candidate sampling phase',
        '4_exact_verify'   : 'Exact verification on R',
    }
    for key in ['0_js_div', '0_partition', '1_setup',
                '2a_where_active', '2b_fancy_index', '2c_accumulate', '2d_prune_masks',
                '3_candidate_phase', '4_exact_verify']:
        t = stage_timers[key]
        pct = 100.0 * t / grand_total if grand_total > 0 else 0
        per_query_ms = 1000.0 * t / Q
        print(f'    {stage_labels[key]:<50s} {t*1000:9.2f} ms total  '
              f'{per_query_ms:7.3f} ms/query  ({pct:5.1f}%)')
    print(f'    {"─" * 50} {"─" * 9}')
    print(f'    {"GRAND TOTAL":<50s} {grand_total*1000:9.2f} ms total  '
          f'{1000.0 * grand_total / Q:7.3f} ms/query')
    print(f'    avg candidates per query: {stage_timers["_cand_count"] / Q:.1f}')

    # Subspace pruning trace (averaged across queries)
    if all_subspace_stats:
        max_k = max(len(s) for s in all_subspace_stats)
        avg_active = []
        for k in range(max_k):
            vals = [s[k][1] for s in all_subspace_stats if k < len(s)]
            avg_active.append(np.mean(vals))
        print(f'\n  ─── Avg |active| per subspace iteration (lower = better pruning) ───')
        for k, a in enumerate(avg_active):
            bar_len = int(60 * a / avg_active[0]) if avg_active[0] > 0 else 0
            print(f'    subspace {k:2d}: {int(a):>9d}  {"█" * bar_len}')


def main() -> None:
    ap = argparse.ArgumentParser(
        description='QADP Normal Case Hamming range query benchmark'
    )
    ap.add_argument('--dataset', choices=list(DATASETS), default='siftsmall',
                    help='dataset to use (default: siftsmall)')
    ap.add_argument('--radius', type=int, default=20,
                    help='Hamming radius for range query (default: 20)')
    ap.add_argument('--query_count', type=int, default=100,
                    help='number of queries to run (default: 100)')
    ap.add_argument('--eps', type=float, default=0.1,
                    help='sampling error epsilon (default: 0.1)')
    ap.add_argument('--delta', type=float, default=0.1,
                    help='sampling confidence delta (default: 0.1)')
    args = ap.parse_args()

    prefix      = DATASETS[args.dataset]
    dataset_dir = os.path.join(DATASETS_DIR, prefix)
    run_qadp_normal(dataset_dir, prefix, args.radius, args.query_count,
                    args.eps, args.delta)


if __name__ == '__main__':
    main()
