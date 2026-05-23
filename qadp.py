import os
import time
import argparse
import tracemalloc
import numpy as np

from dataset import read_fvecs, binarize


def compute_stats(codes: np.ndarray) -> tuple:
    """Unpack bits and compute per-bit Bernoulli frequencies.

    This is the only offline step QADP requires — no hash tables or inverted indices.
    The unpacked bit matrix is kept in memory and reused across all queries.
    """
    bits = np.unpackbits(codes, axis=1)          # (N, n_bits) uint8, values in {0, 1}
    bit_freq = bits.mean(axis=0).astype(np.float64)
    return bits, bit_freq


def _js_per_bit(q_bits: np.ndarray, bit_freq: np.ndarray) -> np.ndarray:
    """JS-divergence JS(P_d ∥ Q_d) for every bit position d.

    P_d = Bernoulli(bit_freq[d])  — database marginal distribution at bit d.
    Q_d = degenerate at q_bits[d] — query value (deterministic, H(Q_d) = 0).

    JS(P ∥ Q) = H((P + Q) / 2) − (H(P) + H(Q)) / 2
              = H(M)           − H(P) / 2          since H(Q) = 0.

    High JS means the query's bit value is rarely seen in the database at that
    position → the dimension is highly informative for pruning.
    """
    eps = 1e-12
    p = np.clip(bit_freq, eps, 1.0 - eps)
    q = q_bits.astype(np.float64)                # 0.0 or 1.0
    m = (p + q) / 2.0

    def bh(x: np.ndarray) -> np.ndarray:         # binary entropy (nats)
        x = np.clip(x, eps, 1.0 - eps)
        return -x * np.log(x) - (1.0 - x) * np.log(1.0 - x)

    return bh(m) - bh(p) / 2.0


def _adaptive_partition(sort_idx: np.ndarray, js: np.ndarray, K: int) -> list:
    """Partition bit indices into K groups of equal JS-divergence mass.

    sort_idx must already be sorted by JS descending (most informative first).
    Grouping by equal JS-mass, rather than by equal width, ensures that each
    group carries the same discriminative "budget" and that the Anti-Pigeonhole
    can prune roughly equal fractions of candidates at every step.

    Per-group thresholds τᵢ are not fixed explicitly; they emerge naturally from
    the running-accumulation strategy in qadp_query (the generalized pigeonhole
    with variable τᵢ satisfying Σ τᵢ = r is implicit there).
    """
    n = len(sort_idx)
    K = min(K, n)
    sorted_js = js[sort_idx]
    total = sorted_js.sum()

    if total < 1e-15:
        # All bits equally informative; equal-size split.
        return [sort_idx[i::K].copy() for i in range(K)]

    target = total / K
    groups, start, acc = [], 0, 0.0

    for i in range(n):
        acc += sorted_js[i]
        if acc >= target * (len(groups) + 1) and len(groups) < K - 1:
            groups.append(sort_idx[start : i + 1].copy())
            start = i + 1

    groups.append(sort_idx[start:].copy())
    return groups


def qadp_query(
    all_bits: np.ndarray,
    q: np.ndarray,
    r: int,
    bit_freq: np.ndarray,
    K: int,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """QADP Hamming range query — no pre-built index, partitioning computed per query.

    Step 1 — JS-divergence ranking:
        Compute JS(P_d ∥ Q_d) for every bit d using the database marginal P_d and the
        deterministic query value Q_d. Sort dimensions by JS descending so the most
        informative ones are processed first.

    Step 2 — Adaptive K-group partition (equal JS-mass):
        Unlike equi-width partitioning (a weakness of prior methods), split the sorted
        dimensions into K groups such that each group has the same total JS-divergence.
        This places the most informative bits at the front and balances pruning power
        across groups. Per-subspace thresholds τᵢ (with Σ τᵢ = r) are implicit in the
        running accumulation of Step 4 — no explicit allocation is needed.

    Step 3 — Sampling pre-filter (optional, approximate):
        Sample `sample_size` bit positions uniformly at random. Compute the Hamming
        distance on the sample for all N codes. The expected sample distance for a true
        neighbor (d ≤ r) is d·s/n ≤ r·s/n. A 3σ slack derived from the hypergeometric
        variance provides a conservative threshold; codes whose sample distance exceeds
        it are pruned before the full scan. This step introduces a small approximate
        element (set sample_size=0 for exact mode).

    Step 4 — Anti-Pigeonhole scan with early termination:
        Process groups in JS-descending order. After each group, accumulate the group's
        Hamming distance into each code's running total and discard codes whose running
        total already exceeds r. The Anti-Pigeonhole guarantees that any non-neighbor
        (d > r) will be pruned as soon as the accumulated mismatch surpasses r — no
        need to examine all K groups. Processing the most discriminative group first
        maximises early pruning.
    """
    N, n_bits = all_bits.shape
    q_bits = np.unpackbits(q)                    # uint8, shape (n_bits,)

    # ── 1. JS-divergence ranking ──────────────────────────────────────────
    js = _js_per_bit(q_bits, bit_freq)
    sort_idx = np.argsort(js)[::-1]              # most informative first

    # ── 2. Adaptive partition ─────────────────────────────────────────────
    groups = _adaptive_partition(sort_idx, js, K)

    # ── 3. Sampling pre-filter ────────────────────────────────────────────
    if sample_size > 0:
        s = min(sample_size, n_bits)
        samp = rng.choice(n_bits, size=s, replace=False)
        dist_samp = (
            all_bits[:, samp].astype(np.int32) ^ q_bits[samp].astype(np.int32)
        ).sum(axis=1)
        # Hypergeometric std for sampling d ≤ r bits out of n_bits without replacement.
        sigma = np.sqrt(
            max(r, 1) * (n_bits - max(r, 1)) / max(n_bits - 1, 1)
            * s * (n_bits - s) / n_bits ** 2
        )
        threshold = r * s / n_bits + 3.0 * sigma
        alive = dist_samp.astype(np.float64) <= threshold
    else:
        alive = np.ones(N, dtype=bool)

    # ── 4. Anti-Pigeonhole scan ───────────────────────────────────────────
    running = np.zeros(N, dtype=np.int32)

    for g in groups:
        if not alive.any():
            break
        live = np.flatnonzero(alive)
        dist_g = (
            all_bits[live][:, g].astype(np.int32) ^ q_bits[g].astype(np.int32)
        ).sum(axis=1)
        running[live] += dist_g
        alive[live[running[live] > r]] = False

    return np.flatnonzero(alive).astype(np.int32)


DATASETS = {
    'sift':      'sift',
    'siftsmall': 'siftsmall',
}

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')


def run_qadp(
    dataset_dir: str,
    prefix: str,
    radius: int,
    K: int,
    sample_size: int,
    query_count: int,
) -> None:
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
    print(f"  code shape : {base_codes.shape}  dtype={base_codes.dtype}")
    print(f"  groups K={K},  n_bits={n_bits},  sample_size={sample_size or 'off (exact)'}")

    # ── Statistics (the only offline computation) ─────────────────────────
    print("\nComputing per-bit statistics ...")
    tracemalloc.start()
    t0 = time.perf_counter()
    all_bits, bit_freq = compute_stats(base_codes)
    stats_time = time.perf_counter() - t0
    _, peak_stats = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  stats time   : {stats_time:.3f} s  (bit-unpack + per-bit mean; no index built)")
    print(f"  codes array  : {base_codes.nbytes / 1e6:.2f} MB")
    print(f"  bits array   : {all_bits.nbytes  / 1e6:.2f} MB  (unpacked, reused per query)")
    print(f"  peak (stats) : {peak_stats        / 1e6:.2f} MB  (tracemalloc)")

    # ── Query ─────────────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    Q = min(query_count, len(query_codes))
    mode = f"sample_size={sample_size} (approx)" if sample_size > 0 else "exact"
    print(f"\nQADP range query  r={radius}, K={K}, {mode}, Q={Q} ...")
    tracemalloc.start()
    results     = []
    running_avg = 0.0
    for i in range(Q):
        t0 = time.perf_counter()
        res = qadp_query(all_bits, query_codes[i], radius, bit_freq, K, sample_size, rng)
        qt  = time.perf_counter() - t0
        running_avg += (qt - running_avg) / (i + 1)
        results.append(res)
    _, peak_query = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total    = running_avg * Q
    avg_hits = float(np.mean([len(res) for res in results]))

    print(f"  total        : {total:.3f} s")
    print(f"  avg query    : {running_avg * 1000:.3f} ms  (running average)")
    print(f"  QPS          : {Q / total:.1f}")
    print(f"  avg hits     : {avg_hits:.1f}")
    print(f"  peak (query) : {peak_query / 1e6:.2f} MB  (tracemalloc)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description='QADP Hamming range query — JS-divergence partitioning, Anti-Pigeonhole scan'
    )
    ap.add_argument('--dataset', choices=list(DATASETS), default='siftsmall',
                    help='dataset to use (default: siftsmall)')
    ap.add_argument('--radius', type=int, default=20,
                    help='Hamming radius for range query (default: 20)')
    ap.add_argument('--groups', type=int, default=16,
                    help='number of dimension groups K (default: 16)')
    ap.add_argument('--sample_size', type=int, default=0,
                    help='bits to sample for approximate pre-filter; 0 = exact (default: 0)')
    ap.add_argument('--query_count', type=int, default=100,
                    help='number of queries to run (default: 100)')
    args = ap.parse_args()

    prefix      = DATASETS[args.dataset]
    dataset_dir = os.path.join(DATASETS_DIR, prefix)
    run_qadp(dataset_dir, prefix, args.radius, args.groups, args.sample_size, args.query_count)


if __name__ == '__main__':
    main()
