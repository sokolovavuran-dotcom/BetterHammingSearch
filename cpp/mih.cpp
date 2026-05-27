// Multi-Index Hashing (MIH) for EXACT Hamming-range queries.
//
// Reference: Norouzi, Punjani, Fleet, "Fast Search in Hamming Space with
//            Multi-Index Hashing", CVPR 2012 / TPAMI 2014.
//
// Algorithm
// ---------
// Build: split each d-bit code into m equal substrings of d/m bits.  For
// substring i, build a sorted index mapping every possible value v in
// [0, 2^(d/m)) to the list of base-row ids whose i-th substring equals v.
//
// Query (radius r): pigeonhole — if total Hamming distance <= r then at
// least one substring has distance <= r' = floor(r/m).  For each substring i:
//   enumerate every value within Hamming distance <= r' from q's i-th substring
//   union the matching buckets into a candidate set
// then verify the candidate set with exact full Hamming distance.
//
// Layout
// ------
// Substrings must be byte-aligned: d/m must be 8 or 16 bits.  For d=128 that
// means m ∈ {8, 16}.  m=8 ⇒ 16-bit subs, K=65536 buckets/sub, fewer lookups
// at small r.  m=16 ⇒ 8-bit subs, K=256, scales better at large r.  Default
// m=8 matches the MIH paper's recommendation for d=128.
//
// Per substring k:
//   bucket_ids[k]    : (N,) int32 — row ids sorted by sub-value
//   bucket_starts[k] : (K+1,) int64 — bucket_starts[v]..bucket_starts[v+1] is
//                        the slice of bucket_ids[k] holding rows with value v.
//
// Counting-sort build is O(N + K) per substring, much faster than std::sort.
//
// Usage:
//   ./mih --dataset siftsmall --radius 8 --query_count 10 --m 8 --seed 0
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "common.hpp"

namespace fs = std::filesystem;
using common::ByteMatrix;
using common::FloatMatrix;

// ---------------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------------

struct MIHIndex {
    int m              = 0;
    int bits_per_sub   = 0;            // 8 or 16
    int bytes_per_sub  = 0;            // 1 or 2
    int64_t K          = 0;            // 1 << bits_per_sub
    std::size_t N      = 0;

    // Per-substring sorted indices.
    std::vector<std::vector<int32_t>> bucket_ids;     // m × N
    std::vector<std::vector<int64_t>> bucket_starts;  // m × (K+1)

    std::size_t memory_bytes() const {
        std::size_t total = 0;
        for (const auto& v : bucket_ids)    total += v.capacity() * sizeof(int32_t);
        for (const auto& v : bucket_starts) total += v.capacity() * sizeof(int64_t);
        return total;
    }
};

// Read substring value k from a (D,) packed code.
static inline int read_sub(const uint8_t* code, int k, int bits_per_sub) {
    if (bits_per_sub == 8)  return code[k];
    /* 16 */                return (int(code[2 * k]) << 8) | int(code[2 * k + 1]);
}

static MIHIndex build_mih(const ByteMatrix& codes, int m) {
    const std::size_t N = codes.n;
    const std::size_t D = codes.D;
    const int d = static_cast<int>(D * 8);
    if (d % m != 0)
        throw std::runtime_error("d not divisible by m");
    const int bits_per_sub = d / m;
    if (bits_per_sub != 8 && bits_per_sub != 16)
        throw std::runtime_error(
            "MIH requires byte-aligned subs: d/m must be 8 or 16.  "
            "For d=128 use m=8 or m=16.");
    const int bytes_per_sub = bits_per_sub / 8;
    const int64_t K = int64_t(1) << bits_per_sub;

    MIHIndex idx;
    idx.m             = m;
    idx.bits_per_sub  = bits_per_sub;
    idx.bytes_per_sub = bytes_per_sub;
    idx.K             = K;
    idx.N             = N;
    idx.bucket_ids.resize(m);
    idx.bucket_starts.resize(m);

    // Reusable scratch buffer.
    std::vector<int> sub_vals(N);

    for (int k = 0; k < m; ++k) {
        // 1) Extract sub-values for every row.
        for (std::size_t i = 0; i < N; ++i)
            sub_vals[i] = read_sub(codes.row(i), k, bits_per_sub);

        // 2) Counting sort: count occurrences of each sub-value.
        std::vector<int64_t> counts(K + 1, 0);
        for (std::size_t i = 0; i < N; ++i)
            counts[sub_vals[i] + 1]++;

        // 3) Prefix-sum to get bucket starts.
        std::vector<int64_t> starts(K + 1, 0);
        for (int64_t v = 0; v < K; ++v)
            starts[v + 1] = starts[v] + counts[v + 1];

        // 4) Stable placement of row ids by sub-value.
        std::vector<int32_t> order(N);
        std::vector<int64_t> offsets = starts;             // mutable copy
        for (std::size_t i = 0; i < N; ++i) {
            int v = sub_vals[i];
            order[offsets[v]++] = static_cast<int32_t>(i);
        }

        idx.bucket_ids[k]    = std::move(order);
        idx.bucket_starts[k] = std::move(starts);
    }
    return idx;
}

// ---------------------------------------------------------------------------
// XOR flip-mask enumeration via Gosper's hack (next-k-combination).
// All integers v in [0, 2^B) with popcount(v) <= r_prime.
// ---------------------------------------------------------------------------

static std::vector<uint32_t> enumerate_flip_masks(int B, int r_prime) {
    std::vector<uint32_t> masks;
    masks.push_back(0u);
    const uint32_t limit = (B >= 32) ? 0u : (1u << B);  // B is 8 or 16 here

    for (int k = 1; k <= r_prime; ++k) {
        // Smallest integer with popcount(k): (1 << k) - 1
        uint32_t v = (1u << k) - 1u;
        while ((B >= 32) || (v < limit)) {
            masks.push_back(v);
            // Gosper's hack: next bigger integer with the same popcount.
            uint32_t c = v & (~v + 1u);          // lowest set bit
            uint32_t rr = v + c;
            uint32_t nv = (((v ^ rr) >> 2) / c) | rr;
            if (nv <= v) break;                   // overflow / done
            v = nv;
            if (B < 32 && v >= limit) break;
        }
    }
    return masks;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

struct QueryStats {
    long long lookups        = 0;     // bucket lookups performed
    long long raw_hits       = 0;     // candidate entries before dedup
    long long unique_cands   = 0;     // |candidate set| after dedup
    long long hits           = 0;     // final results
};

static std::vector<int32_t>
mih_query(const MIHIndex& idx,
          const ByteMatrix& codes,
          const uint8_t* q,
          int r,
          const std::vector<uint32_t>& flip_masks,
          std::vector<uint8_t>& cand_mask,      // (N,) reused scratch
          QueryStats& stats)
{
    const int m = idx.m;
    const int bits_per_sub = idx.bits_per_sub;
    const std::size_t D = codes.D;
    const int sub_mask = (bits_per_sub >= 32) ? 0 : int((1u << bits_per_sub) - 1u);

    std::fill(cand_mask.begin(), cand_mask.end(), uint8_t(0));

    // For each substring, look up every value within Hamming distance r' of qv.
    for (int k = 0; k < m; ++k) {
        const int qv = read_sub(q, k, bits_per_sub);
        const int64_t* starts = idx.bucket_starts[k].data();
        const int32_t* ids    = idx.bucket_ids[k].data();

        for (uint32_t fm : flip_masks) {
            int v = (qv ^ int(fm)) & sub_mask;
            int64_t s = starts[v];
            int64_t e = starts[v + 1];
            stats.lookups++;
            stats.raw_hits += (e - s);
            for (int64_t i = s; i < e; ++i) {
                cand_mask[ids[i]] = 1;
            }
        }
    }

    // Collect candidates and verify with exact Hamming distance.
    std::vector<int32_t> R;
    R.reserve(64);
    long long uniq = 0;
    for (std::size_t i = 0; i < idx.N; ++i) {
        if (!cand_mask[i]) continue;
        ++uniq;
        if (common::hamming(codes.row(i), q, D) <= r) {
            R.push_back(static_cast<int32_t>(i));
        }
    }
    stats.unique_cands += uniq;
    stats.hits += static_cast<long long>(R.size());
    return R;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

static int comb(int n, int k) {
    if (k < 0 || k > n) return 0;
    long long r = 1;
    for (int i = 0; i < k; ++i) {
        r = r * (n - i) / (i + 1);
    }
    return static_cast<int>(r);
}

int main(int argc, char** argv)
{
    common::Args args(argc, argv);
    const std::string dataset     = args.get("dataset", "siftsmall");
    const int         radius      = args.get_int("radius", 8);
    const int         query_count = args.get_int("query_count", 10);
    const int         m           = args.get_int("m", 8);
    const int         seed        = args.get_int("seed", 0);

    const fs::path ddir = common::dataset_dir(dataset);
    const std::string base_path  = (ddir / (dataset + "_base.fvecs")).string();
    const std::string query_path = (ddir / (dataset + "_query.fvecs")).string();

    std::printf("Loading %s ...\n", dataset.c_str());
    common::Timer tload;
    FloatMatrix base    = common::read_fvecs(base_path);
    FloatMatrix queries = common::read_fvecs(query_path);
    std::printf("  base    : (%zu, %zu)\n", base.n, base.d);
    std::printf("  queries : (%zu, %zu)\n", queries.n, queries.d);
    std::printf("  load    : %.2f s\n", tload.s());

    std::printf("Binarizing (GRP seed=%d) ...\n", seed);
    common::Timer tbin;
    ByteMatrix base_codes  = common::binarize(base,    static_cast<uint32_t>(seed));
    ByteMatrix query_codes = common::binarize(queries, static_cast<uint32_t>(seed));
    const int d = static_cast<int>(base_codes.bits());
    std::printf("  packed base : (%zu, %zu)  d=%d\n", base_codes.n, base_codes.D, d);
    std::printf("  binarize    : %.2f s\n", tbin.s());

    const int bits_per_sub = d / m;
    const int r_prime = radius / m;
    const int neighbors_per_sub = [&] {
        int s = 0;
        for (int k = 0; k <= r_prime; ++k) s += comb(bits_per_sub, k);
        return s;
    }();

    std::printf("\nBuilding MIH index ...\n");
    std::printf("  m=%d  bits/sub=%d  K=%lld buckets/sub\n",
                m, bits_per_sub, 1LL << bits_per_sub);
    std::printf("  r=%d  r/m=%d  neighbors/sub=%d  total lookups/query=%d\n",
                radius, r_prime, neighbors_per_sub, neighbors_per_sub * m);

    common::Timer tbuild;
    MIHIndex idx;
    try {
        idx = build_mih(base_codes, m);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Build failed: %s\n", e.what());
        return 1;
    }
    double build_ms = tbuild.ms();
    std::printf("  build time  : %.2f s\n", build_ms * 1e-3);
    std::printf("  index size  : %.1f MB\n", idx.memory_bytes() / 1e6);

    std::vector<uint32_t> flip_masks =
        enumerate_flip_masks(bits_per_sub, r_prime);
    std::printf("  flip masks  : %zu\n", flip_masks.size());

    const int Q = std::min<int>(query_count, static_cast<int>(query_codes.n));
    std::printf("\nMIH range query  r=%d, m=%d, Q=%d ...\n", radius, m, Q);

    std::vector<uint8_t> cand_mask(base_codes.n);

    QueryStats stats;
    double running_avg_ms = 0.0;
    long long total_hits = 0;
    for (int i = 0; i < Q; ++i) {
        common::Timer tq;
        QueryStats s_before = stats;
        auto hits = mih_query(idx, base_codes, query_codes.row(i),
                              radius, flip_masks, cand_mask, stats);
        double qms = tq.ms();
        running_avg_ms += (qms - running_avg_ms) / (i + 1);
        total_hits += static_cast<long long>(hits.size());
        std::printf("  q[%02d]  query=%7.2f ms  lookups=%5lld  cand=%7lld  hits=%6zu\n",
                    i, qms,
                    stats.lookups      - s_before.lookups,
                    stats.unique_cands - s_before.unique_cands,
                    hits.size());
    }

    double total_ms = running_avg_ms * Q;
    double avg_hits = static_cast<double>(total_hits) / Q;

    std::printf("\n  total           : %.3f s\n",        total_ms * 1e-3);
    std::printf("  avg query       : %.3f ms\n",         running_avg_ms);
    std::printf("  QPS             : %.1f\n",            Q / (total_ms * 1e-3));
    std::printf("  avg hits        : %.1f\n",            avg_hits);
    std::printf("  avg lookups/q   : %.1f\n",            double(stats.lookups)      / Q);
    std::printf("  avg raw_hits/q  : %.1f\n",            double(stats.raw_hits)     / Q);
    std::printf("  avg unique cand : %.1f\n",            double(stats.unique_cands) / Q);

    return 0;
}
