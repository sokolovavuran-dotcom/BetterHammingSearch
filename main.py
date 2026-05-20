import time
import csv
import argparse
from dataclasses import dataclass
from itertools import combinations
import numpy as np


# -----------------------
# Popcount for old NumPy
# -----------------------

# Lookup table: popcount for bytes 0..255
_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_distances_u64(codes: np.ndarray, q: np.uint64) -> np.ndarray:
    """
    codes: np.ndarray dtype=uint64, shape (N,)
    q: uint64
    returns: np.ndarray int32 distances, shape (N,)

    Compatible with older NumPy (no ndarray.bit_count()).
    """
    x = np.bitwise_xor(codes, q)                  # uint64 array
    xb = x.view(np.uint8).reshape(-1, 8)          # (N, 8) bytes
    return _POPCOUNT_TABLE[xb].sum(axis=1).astype(np.int32)


def linear_scan_range(codes: np.ndarray, q: np.uint64, r: int) -> np.ndarray:
    d = hamming_distances_u64(codes, q)
    return np.flatnonzero(d <= r).astype(np.int32)


# -----------------------
# MIH (Multi-Index Hashing)
# -----------------------

def _split_plan(bits: int, m: int):
    """
    Split [0..bits) into m contiguous parts (LSB-first).
    Returns list of (shift, part_bits, mask)
    """
    base = bits // m
    rem = bits % m
    parts = []
    shift = 0
    for i in range(m):
        pb = base + (1 if i < rem else 0)
        mask = (1 << pb) - 1
        parts.append((shift, pb, mask))
        shift += pb
    return parts


def _masks_upto_t(part_bits: int, t: int):
    """
    Generate XOR masks with <= t bits set within part_bits.
    WARNING: grows as sum_{i=0..t} C(part_bits, i). Keep t small (r small).
    """
    masks = [0]
    for s in range(1, t + 1):
        for pos in combinations(range(part_bits), s):
            mm = 0
            for p in pos:
                mm |= (1 << p)
            masks.append(mm)
    return masks


@dataclass
class MIHIndex64:
    m: int
    bits: int = 64

    def __post_init__(self):
        assert self.bits == 64, "This implementation supports 64-bit codes only."
        self.parts = _split_plan(self.bits, self.m)
        self.tables = [dict() for _ in range(self.m)]
        self.codes = None
        self._mask_cache = {}  # (part_bits, t) -> masks list

    def build(self, codes: np.ndarray):
        assert codes.dtype == np.uint64 and codes.ndim == 1
        self.codes = codes
        for idx, c in enumerate(codes):
            ci = int(c)
            for ti, (shift, pb, mask) in enumerate(self.parts):
                key = (ci >> shift) & mask
                self.tables[ti].setdefault(key, []).append(idx)

    def query_range(self, q: np.uint64, r: int) -> np.ndarray:
        """
        Exact MIH-style range query:
        Probe each part within t=floor(r/m), union candidates, then verify full distance <= r.
        """
        assert self.codes is not None
        t = r // self.m
        qi = int(q)

        cand = set()
        for ti, (shift, pb, mask) in enumerate(self.parts):
            cache_key = (pb, t)
            masks = self._mask_cache.get(cache_key)
            if masks is None:
                masks = _masks_upto_t(pb, t)
                self._mask_cache[cache_key] = masks

            qkey = (qi >> shift) & mask
            table = self.tables[ti]
            for xm in masks:
                key2 = qkey ^ xm
                bucket = table.get(key2)
                if bucket:
                    cand.update(bucket)

        if not cand:
            return np.empty((0,), dtype=np.int32)

        cand_idx = np.fromiter(cand, dtype=np.int32)
        d = hamming_distances_u64(self.codes[cand_idx], q)
        return cand_idx[d <= r]


# -----------------------
# LSH (random bit sampling)
# -----------------------

@dataclass
class LSHIndex64:
    L: int         # number of tables
    k: int         # bits per key
    bits: int = 64
    seed: int = 123

    def __post_init__(self):
        assert self.bits == 64
        self.rng = np.random.default_rng(self.seed)
        self.bitpos = [self.rng.choice(self.bits, size=self.k, replace=False).tolist()
                       for _ in range(self.L)]
        self.tables = [dict() for _ in range(self.L)]
        self.codes = None

    @staticmethod
    def _key(x: int, positions) -> int:
        key = 0
        for i, p in enumerate(positions):
            key |= ((x >> p) & 1) << i
        return key

    def build(self, codes: np.ndarray):
        assert codes.dtype == np.uint64 and codes.ndim == 1
        self.codes = codes
        for idx, c in enumerate(codes):
            xi = int(c)
            for t in range(self.L):
                key = self._key(xi, self.bitpos[t])
                self.tables[t].setdefault(key, []).append(idx)

    def query_range(self, q: np.uint64, r: int) -> np.ndarray:
        """
        Approximate: union buckets with exact key match; then verify full distance <= r.
        """
        assert self.codes is not None
        qi = int(q)
        cand = set()
        for t in range(self.L):
            key = self._key(qi, self.bitpos[t])
            bucket = self.tables[t].get(key)
            if bucket:
                cand.update(bucket)

        if not cand:
            return np.empty((0,), dtype=np.int32)

        cand_idx = np.fromiter(cand, dtype=np.int32)
        d = hamming_distances_u64(self.codes[cand_idx], q)
        return cand_idx[d <= r]


# -----------------------
# Benchmarking helpers
# -----------------------

def make_data(N: int, Q: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 1 << 64, size=N, dtype=np.uint64)

    q_idx = rng.integers(0, N, size=Q)
    queries = codes[q_idx].copy()

    # perturb half queries by flipping 1 random bit
    for i in range(Q // 2):
        b = int(rng.integers(0, 64))
        queries[i] ^= (np.uint64(1) << np.uint64(b))

    return codes, queries


def avg_recall(res_list, gt_list) -> float:
    vals = []
    for res, g in zip(res_list, gt_list):
        g = np.asarray(g, dtype=np.int32)
        if g.size == 0:
            vals.append(1.0)
            continue
        gset = set(map(int, g))
        rset = set(map(int, res))
        vals.append(len(rset & gset) / len(gset))
    return float(np.mean(vals))


def benchmark(N=200_000, Q=200, r=3, m=4, L=12, k=16, seed=0):
    codes, queries = make_data(N, Q, seed=seed)

    # Ground truth (linear scan)
    t0 = time.perf_counter()
    gt = [linear_scan_range(codes, q, r) for q in queries]
    t1 = time.perf_counter()
    lin_time = t1 - t0

    # MIH
    mih = MIHIndex64(m=m)
    t0 = time.perf_counter()
    mih.build(codes)
    t1 = time.perf_counter()
    mih_build = t1 - t0

    t0 = time.perf_counter()
    mih_res = [mih.query_range(q, r) for q in queries]
    t1 = time.perf_counter()
    mih_qtime = t1 - t0

    # LSH
    lsh = LSHIndex64(L=L, k=k, seed=seed + 1)
    t0 = time.perf_counter()
    lsh.build(codes)
    t1 = time.perf_counter()
    lsh_build = t1 - t0

    t0 = time.perf_counter()
    lsh_res = [lsh.query_range(q, r) for q in queries]
    t1 = time.perf_counter()
    lsh_qtime = t1 - t0

    return {
        "N": N, "Q": Q, "r": r,
        "linear_scan_total_s": lin_time,
        "MIH_build_s": mih_build,
        "MIH_queries_total_s": mih_qtime,
        "MIH_recall_vs_linear": avg_recall(mih_res, gt),
        "LSH_build_s": lsh_build,
        "LSH_queries_total_s": lsh_qtime,
        "LSH_recall_vs_linear": avg_recall(lsh_res, gt),
        "params": {"m": m, "L": L, "k": k, "seed": seed},
    }


def print_row_header():
    print(f"{'N':>10} | {'lin_s':>10} | {'mih_build':>10} | {'mih_q':>10} | {'mih_rec':>8} | "
          f"{'lsh_build':>10} | {'lsh_q':>10} | {'lsh_rec':>8}")
    print("-" * 102)


def print_row(res: dict):
    print(f"{res['N']:>10} | "
          f"{res['linear_scan_total_s']:>10.4f} | "
          f"{res['MIH_build_s']:>10.4f} | "
          f"{res['MIH_queries_total_s']:>10.4f} | "
          f"{res['MIH_recall_vs_linear']:>8.3f} | "
          f"{res['LSH_build_s']:>10.4f} | "
          f"{res['LSH_queries_total_s']:>10.4f} | "
          f"{res['LSH_recall_vs_linear']:>8.3f}")


def save_csv(filename: str, results: list):
    with open(filename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "N","Q","r",
            "linear_scan_total_s",
            "MIH_build_s","MIH_queries_total_s","MIH_recall_vs_linear",
            "LSH_build_s","LSH_queries_total_s","LSH_recall_vs_linear",
            "m","L","k","seed"
        ])
        for res in results:
            p = res["params"]
            w.writerow([
                res["N"], res["Q"], res["r"],
                res["linear_scan_total_s"],
                res["MIH_build_s"], res["MIH_queries_total_s"], res["MIH_recall_vs_linear"],
                res["LSH_build_s"], res["LSH_queries_total_s"], res["LSH_recall_vs_linear"],
                p["m"], p["L"], p["k"], p["seed"]
            ])


# -----------------------
# CLI
# -----------------------

def parse_sizes(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def cmd_bench(args):
    res = benchmark(N=args.N, Q=args.Q, r=args.r, m=args.m, L=args.L, k=args.k, seed=args.seed)
    print_row_header()
    print_row(res)
    if args.csv:
        save_csv(args.csv, [res])
        print(f"\nSaved CSV: {args.csv}")


def cmd_sweep(args):
    sizes = parse_sizes(args.sizes)
    results = []
    print_row_header()
    for N in sizes:
        res = benchmark(N=N, Q=args.Q, r=args.r, m=args.m, L=args.L, k=args.k, seed=args.seed)
        results.append(res)
        print_row(res)

    csv_name = args.csv or "results_sweep.csv"
    save_csv(csv_name, results)
    print(f"\nSaved CSV: {csv_name}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_b = sub.add_parser("bench", help="single benchmark run")
    ap_b.add_argument("--N", type=int, default=200_000)
    ap_b.add_argument("--Q", type=int, default=200)
    ap_b.add_argument("--r", type=int, default=3)
    ap_b.add_argument("--m", type=int, default=4)
    ap_b.add_argument("--L", type=int, default=12)
    ap_b.add_argument("--k", type=int, default=16)
    ap_b.add_argument("--seed", type=int, default=0)
    ap_b.add_argument("--csv", type=str, default="", help="optional CSV path to save single result")
    ap_b.set_defaults(func=cmd_bench)

    ap_s = sub.add_parser("sweep", help="run multiple sizes and save CSV")
    ap_s.add_argument("--sizes", type=str, default="20000,50000,100000,200000,500000,1000000")
    ap_s.add_argument("--Q", type=int, default=200)
    ap_s.add_argument("--r", type=int, default=3)
    ap_s.add_argument("--m", type=int, default=4)
    ap_s.add_argument("--L", type=int, default=12)
    ap_s.add_argument("--k", type=int, default=16)
    ap_s.add_argument("--seed", type=int, default=0)
    ap_s.add_argument("--csv", type=str, default="", help="optional CSV path (default: results_sweep.csv)")
    ap_s.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
