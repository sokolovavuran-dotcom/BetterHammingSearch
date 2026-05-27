// QADP SP-NC: Query-Aware Dimension Partitioning, Normal Case.  C++ port of
// qadp_normal.py.  Operates on packed-byte codes throughout; subspaces are
// expressed as (D,) byte masks so per-subspace Hamming is popcount-of-(XOR
// AND mask) — same memory traffic as linear_scan, no unpack step.
//
// Usage:
//   ./qadp_normal --dataset siftsmall --radius 20 --query_count 10 \
//                 --eps 0.1 --delta 0.1 --seed 0
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "common.hpp"

namespace fs = std::filesystem;
using common::ByteMatrix;
using common::FloatMatrix;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int compute_sample_count(int d, double eps, double delta) {
    double log_inv = std::log(1.0 / delta);
    double denom = 2.0 * d * eps * eps + log_inv;
    double l1 = (d + 1) * log_inv / denom;
    double l2 = (2.0 * d * eps * eps + d * log_inv) / denom;
    int l = static_cast<int>(std::ceil(std::min(l1, l2)));
    return std::max(1, std::min(d, l));
}

// Fraction of 1s per bit-position across the base codes — computed once.
static std::vector<double> compute_data_p1(const ByteMatrix& codes) {
    const std::size_t N = codes.n;
    const std::size_t D = codes.D;
    const std::size_t d = D * 8;
    std::vector<double> p1(d, 0.0);
    for (std::size_t i = 0; i < N; ++i) {
        const uint8_t* row = codes.row(i);
        for (std::size_t b = 0; b < d; ++b) {
            if (row[b >> 3] & static_cast<uint8_t>(1u << (7 - (b & 7))))
                p1[b] += 1.0;
        }
    }
    for (auto& v : p1) v /= static_cast<double>(N);
    return p1;
}

// Per-bit Jensen-Shannon divergence between data and query distributions.
static std::vector<double> js_div_all(const std::vector<double>& data_p1,
                                      const std::vector<uint8_t>& q_bits)
{
    const std::size_t d = data_p1.size();
    std::vector<double> js(d, 0.0);
    auto kl_term = [](double x, double m) {
        return (x > 0.0 && m > 0.0) ? 0.5 * x * std::log(x / m) : 0.0;
    };
    for (std::size_t i = 0; i < d; ++i) {
        double p1 = data_p1[i];
        double p0 = 1.0 - p1;
        double q1 = static_cast<double>(q_bits[i]);
        double q0 = 1.0 - q1;
        double m0 = 0.5 * (p0 + q0);
        double m1 = 0.5 * (p1 + q1);
        js[i] = kl_term(p0, m0) + kl_term(p1, m1)
              + kl_term(q0, m0) + kl_term(q1, m1);
    }
    return js;
}

// Partition d dims into m=r subspaces, interleaved by ascending JS-div.
static std::vector<std::vector<int>>
qadp_normal_partition(int d, int r, const std::vector<double>& js_div)
{
    const int m = r;
    std::vector<int> order(d);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return js_div[a] < js_div[b]; });

    std::vector<std::vector<int>> subs(m);
    const int n_full = (d / m) * m;
    const int rows   = n_full / m;
    for (int i = 0; i < rows; ++i)
        for (int k = 0; k < m; ++k)
            subs[k].push_back(order[i * m + k]);
    for (int i = n_full; i < d; ++i)
        subs[(i - n_full) % m].push_back(order[i]);
    return subs;
}

// Unpack a (D,) packed code into (d,) uint8 bits, MSB-first per byte.
static std::vector<uint8_t> unpack_one(const uint8_t* code, std::size_t D)
{
    std::vector<uint8_t> bits(D * 8);
    for (std::size_t i = 0; i < D; ++i) {
        const uint8_t b = code[i];
        for (int j = 0; j < 8; ++j)
            bits[i * 8 + j] = (b >> (7 - j)) & 1u;
    }
    return bits;
}

// Convert a list of bit indices into a (D,) byte mask (MSB-first per byte).
static std::vector<uint8_t> bits_to_byte_mask(const std::vector<int>& bits, int d)
{
    const int D = d / 8;
    std::vector<uint8_t> mask(D, 0);
    for (int b : bits)
        mask[b >> 3] |= static_cast<uint8_t>(1u << (7 - (b & 7)));
    return mask;
}

// popcount(xor_full[idx] AND mask)
static inline int popcount_masked(const uint8_t* xrow, const uint8_t* mask, std::size_t D)
{
    int h = 0;
    std::size_t j = 0;
    for (; j + 8 <= D; j += 8) {
        uint64_t x, m;
        std::memcpy(&x, xrow + j, 8);
        std::memcpy(&m, mask + j, 8);
        h += std::popcount(x & m);
    }
    for (; j < D; ++j)
        h += std::popcount(static_cast<uint8_t>(xrow[j] & mask[j]));
    return h;
}

// popcount(xor_full[idx])  — full Hamming, no mask.
static inline int popcount_full(const uint8_t* xrow, std::size_t D)
{
    int h = 0;
    std::size_t j = 0;
    for (; j + 8 <= D; j += 8) {
        uint64_t x;
        std::memcpy(&x, xrow + j, 8);
        h += std::popcount(x);
    }
    for (; j < D; ++j)
        h += std::popcount(static_cast<uint8_t>(xrow[j]));
    return h;
}

// ---------------------------------------------------------------------------
// Per-query timers (cumulative across queries)
// ---------------------------------------------------------------------------

struct StageTimers {
    double js_div         = 0.0;
    double partition      = 0.0;
    double mask_pack      = 0.0;        // pack subspace masks as uint64
    double scan_loop      = 0.0;        // row-major main loop
    double candidate_phase= 0.0;
    double exact_verify   = 0.0;
    long long cand_count   = 0;          // |R| before exact verify
    long long pigeon_count = 0;          // total rows that landed in C
};

// Per-candidate state captured by the main loop, consumed by the sampling phase.
struct Cand { int32_t id; int32_t hc; int32_t bc; };

// ---------------------------------------------------------------------------
// Core query (Algorithm 1: SP-NC), row-major / cache-optimized variant.
//
// vs. the original implementation, this version:
//   * eliminates the (N*D) xor_full buffer entirely
//   * walks the base codes ONCE sequentially (~16 MB read for SIFT 1M)
//   * keeps each row's XOR in a stack-local uint64 buffer (in L1/registers)
//   * runs the priority loop INSIDE the per-row inner loop, so a pruned
//     row never touches subsequent subspace masks
//   * pre-packs masks as uint64[words_per_code] so the inner kernel is
//     just AND + popcount on machine words.
//
// All subspace masks together = m * D bytes (320 B for d=128, m=20).
// That entire mask block stays hot in L1 throughout the scan.
// ---------------------------------------------------------------------------

static std::vector<int32_t>
sp_nc_query(const ByteMatrix& codes,
            const uint8_t* q_code,
            const std::vector<uint64_t>& masks_u64,  // m * words_per_code, contiguous
            const std::vector<int>& sub_sizes,
            const std::vector<int>& all_dims,
            int d, int r, int l,
            std::mt19937& rng,
            StageTimers& timers)
{
    const std::size_t N = codes.n;
    const std::size_t D = codes.D;
    const int m = static_cast<int>(sub_sizes.size());

    // Number of uint64 words per code (round up).
    const std::size_t W = (D + 7) / 8;

    // Pre-load query into uint64 words (handles the partial tail with memcpy).
    alignas(64) uint64_t q_words[16] = {0};       // max 1024-bit codes
    {
        std::size_t j = 0;
        for (; j + 8 <= D; j += 8) std::memcpy(&q_words[j / 8], q_code + j, 8);
        if (j < D) std::memcpy(&q_words[j / 8], q_code + j, D - j);
    }

    std::vector<int32_t> R;
    R.reserve(1024);
    std::vector<Cand> C;
    C.reserve(N / 64);                              // ballpark guess

    // -----------------------------------------------------------------------
    // Main row-major scan.  One sequential pass over all N codes.
    // -----------------------------------------------------------------------
    common::Timer t_loop;

    for (std::size_t i = 0; i < N; ++i) {
        const uint8_t* row = codes.row(i);

        // XOR row with query into stack-local uint64 buffer (stays in regs/L1).
        alignas(64) uint64_t xor_buf[16] = {0};
        {
            std::size_t j = 0;
            for (; j + 8 <= D; j += 8) {
                uint64_t rw;
                std::memcpy(&rw, row + j, 8);
                xor_buf[j / 8] = rw ^ q_words[j / 8];
            }
            if (j < D) {
                uint64_t rw = 0;
                std::memcpy(&rw, row + j, D - j);
                xor_buf[j / 8] = rw ^ q_words[j / 8];
            }
        }

        int hc = 0;
        int bc = 0;
        bool decided = false;

        // Inner loop over subspaces.  AND xor_buf with mask_k, popcount, sum.
        // Three priorities checked PER SUBSPACE so a pruned row exits early.
        const uint64_t* mask_ptr = masks_u64.data();
        for (int k = 0; k < m; ++k, mask_ptr += W) {
            int h_sub = 0;
            for (std::size_t w = 0; w < W; ++w) {
                h_sub += std::popcount(xor_buf[w] & mask_ptr[w]);
            }
            hc += h_sub;
            bc += sub_sizes[k];

            // P1 — anti-pigeonhole pruning.
            if (hc > r) { decided = true; break; }
            // P2 — enough samples; commit via estimate.
            if (bc >= l) {
                double est = static_cast<double>(d) / bc * hc;
                if (est <= r) R.push_back(static_cast<int32_t>(i));
                decided = true; break;
            }
            // P3 — pigeonhole candidate; defer to sampling phase.
            if (h_sub <= 1) {
                C.push_back({static_cast<int32_t>(i), hc, bc});
                decided = true; break;
            }
        }
        // Sanity: with sub_sizes summing to d and h_sub>1 in every subspace,
        // hc would reach 2*m > r and P1 fires.  So `decided` should be true.
        (void)decided;
    }
    timers.scan_loop    += t_loop.ms();
    timers.pigeon_count += static_cast<long long>(C.size());

    // -----------------------------------------------------------------------
    // Candidate sampling phase, GROUP-BY-Bc.
    //
    // All candidates with the same Bc value share an identical random sample
    // (this matches qadp_normal.py and is variance-neutral in expectation).
    // We sort C by Bc, then process each contiguous Bc-group:
    //   1) build one random sample of (l - bc) dims                 -- once
    //   2) pack into a uint64 sample-mask                            -- once
    //   3) for each candidate in the group: popcount-of-XOR-AND-mask -- N_g times
    //
    // The per-candidate cost in the inner loop is now just one popcount call
    // over W uint64 words, no allocations.  For e.g. r=20 on sift_quarter
    // this cuts the candidate phase by ~100x compared to per-candidate.
    // -----------------------------------------------------------------------
    common::Timer t_cand;
    const int K_total = static_cast<int>(all_dims.size());
    std::vector<int> sample_pool;
    sample_pool.reserve(K_total);
    alignas(64) uint64_t sample_mask_buf[16] = {0};
    std::vector<uint8_t> sample_mask_bytes(D);

    std::sort(C.begin(), C.end(),
              [](const Cand& a, const Cand& b) { return a.bc < b.bc; });

    for (std::size_t gs = 0; gs < C.size(); ) {
        const int bc_val = C[gs].bc;
        std::size_t ge = gs;
        while (ge < C.size() && C[ge].bc == bc_val) ++ge;

        // Edge case: bc_val >= l (shouldn't happen given priority ordering).
        if (bc_val >= l) {
            const int divisor = std::max(bc_val, 1);
            for (std::size_t i = gs; i < ge; ++i) {
                const Cand& c = C[i];
                double est = static_cast<double>(d) / divisor * c.hc;
                if (est <= r) R.push_back(c.id);
            }
            gs = ge;
            continue;
        }

        const int need = l - bc_val;
        const int unchecked = K_total - bc_val;
        const int n_sample = std::min(need, unchecked);

        // One sample, one mask for the whole Bc-group.
        sample_pool.assign(all_dims.begin() + bc_val, all_dims.end());
        for (int s = 0; s < n_sample; ++s) {
            std::uniform_int_distribution<int> dist(s, unchecked - 1);
            int p = dist(rng);
            std::swap(sample_pool[s], sample_pool[p]);
        }
        std::fill(sample_mask_bytes.begin(), sample_mask_bytes.end(), uint8_t(0));
        for (int s = 0; s < n_sample; ++s) {
            int b = sample_pool[s];
            sample_mask_bytes[b >> 3] |= static_cast<uint8_t>(1u << (7 - (b & 7)));
        }
        for (std::size_t w = 0; w < W; ++w) sample_mask_buf[w] = 0;
        {
            std::size_t j = 0;
            for (; j + 8 <= D; j += 8)
                std::memcpy(&sample_mask_buf[j / 8], sample_mask_bytes.data() + j, 8);
            if (j < D)
                std::memcpy(&sample_mask_buf[j / 8], sample_mask_bytes.data() + j, D - j);
        }

        // Inner loop: popcount per candidate using the shared mask.
        for (std::size_t i = gs; i < ge; ++i) {
            const Cand& c = C[i];
            const uint8_t* row = codes.row(c.id);
            int h_s = 0;
            std::size_t j = 0;
            for (; j + 8 <= D; j += 8) {
                uint64_t rw;
                std::memcpy(&rw, row + j, 8);
                h_s += std::popcount((rw ^ q_words[j / 8]) & sample_mask_buf[j / 8]);
            }
            if (j < D) {
                uint64_t rw = 0;
                std::memcpy(&rw, row + j, D - j);
                h_s += std::popcount((rw ^ q_words[j / 8]) & sample_mask_buf[j / 8]);
            }
            const int total_h = c.hc + h_s;
            double est = static_cast<double>(d) / l * total_h;
            if (est <= r) R.push_back(c.id);
        }
        gs = ge;
    }
    timers.candidate_phase += t_cand.ms();

    // -----------------------------------------------------------------------
    // Exact verification on R (dedup + full popcount via codes[idx] ^ q).
    // -----------------------------------------------------------------------
    common::Timer t_ex;
    std::sort(R.begin(), R.end());
    R.erase(std::unique(R.begin(), R.end()), R.end());
    timers.cand_count += static_cast<long long>(R.size());

    std::vector<int32_t> result;
    result.reserve(R.size());
    for (int32_t idx : R) {
        const uint8_t* row = codes.row(idx);
        int h = 0;
        std::size_t j = 0;
        for (; j + 8 <= D; j += 8) {
            uint64_t rw;
            std::memcpy(&rw, row + j, 8);
            h += std::popcount(rw ^ q_words[j / 8]);
        }
        if (j < D) {
            uint64_t rw = 0;
            std::memcpy(&rw, row + j, D - j);
            h += std::popcount(rw ^ q_words[j / 8]);
        }
        if (h <= r) result.push_back(idx);
    }
    timers.exact_verify += t_ex.ms();
    return result;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    common::Args args(argc, argv);
    const std::string dataset     = args.get("dataset", "siftsmall");
    const int         radius      = args.get_int("radius", 20);
    const int         query_count = args.get_int("query_count", 10);
    const double      eps         = args.get_double("eps", 0.1);
    const double      delta       = args.get_double("delta", 0.1);
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
    const std::size_t D = base_codes.D;
    std::printf("  packed base : (%zu, %zu)  d=%d\n", base_codes.n, D, d);
    std::printf("  binarize    : %.2f s\n", tbin.s());

    const double small_thresh = 0.05 * d;
    if (radius < small_thresh) {
        std::printf("  WARNING: r=%d < 0.05*d=%.1f — small-threshold case\n",
                    radius, small_thresh);
    }

    const int l = compute_sample_count(d, eps, delta);
    const int m = radius;
    std::printf("  eps=%.2f delta=%.2f -> l=%d samples\n", eps, delta, l);
    std::printf("  subspaces m=%d  (each ~%d bits)\n", m, d / m);

    // Precompute data_p1 once.
    std::printf("Computing data_p1 ...\n");
    common::Timer tp1;
    std::vector<double> data_p1 = compute_data_p1(base_codes);
    std::printf("  data_p1 : %.2f s\n", tp1.s());

    std::mt19937 rng(static_cast<uint32_t>(seed));

    const int Q = std::min<int>(query_count, static_cast<int>(query_codes.n));
    std::printf("\nQADP-NC range query  r=%d, m=%d, Q=%d ...\n", radius, m, Q);

    StageTimers timers;
    // Number of uint64 words per code; mask buffer is m * W contiguous words.
    const std::size_t W = (D + 7) / 8;
    std::vector<uint64_t> masks_u64(static_cast<std::size_t>(m) * W);

    double running_avg_ms = 0.0;
    long long total_hits = 0;
    long long total_pigeons = 0;

    for (int i = 0; i < Q; ++i) {
        common::Timer t_query;

        const uint8_t* q_code = query_codes.row(i);
        std::vector<uint8_t> q_bits = unpack_one(q_code, D);

        common::Timer t_js;
        std::vector<double> js = js_div_all(data_p1, q_bits);
        timers.js_div += t_js.ms();

        common::Timer t_part;
        std::vector<std::vector<int>> subs = qadp_normal_partition(d, m, js);
        std::vector<int> all_dims;
        all_dims.reserve(d);
        for (const auto& s : subs) all_dims.insert(all_dims.end(), s.begin(), s.end());
        timers.partition += t_part.ms();

        // Pack subspace byte-masks into a single contiguous uint64 block of
        // size m*W.  Total: m*D bytes ~ 320 B for d=128, m=20.  Stays hot in L1.
        common::Timer t_mk;
        std::vector<int> sub_sizes;
        sub_sizes.reserve(m);
        std::fill(masks_u64.begin(), masks_u64.end(), uint64_t(0));
        for (int k = 0; k < m; ++k) {
            std::vector<uint8_t> mb = bits_to_byte_mask(subs[k], d);
            uint64_t* dst = masks_u64.data() + static_cast<std::size_t>(k) * W;
            std::size_t j = 0;
            for (; j + 8 <= D; j += 8) std::memcpy(&dst[j / 8], mb.data() + j, 8);
            if (j < D)                 std::memcpy(&dst[j / 8], mb.data() + j, D - j);
            sub_sizes.push_back(static_cast<int>(subs[k].size()));
        }
        timers.mask_pack += t_mk.ms();

        auto hits = sp_nc_query(base_codes, q_code, masks_u64, sub_sizes,
                                all_dims, d, radius, l, rng, timers);
        double qms = t_query.ms();
        running_avg_ms += (qms - running_avg_ms) / (i + 1);
        total_hits += static_cast<long long>(hits.size());
        total_pigeons = timers.pigeon_count;
        std::printf("  q[%02d]  query=%7.2f ms  hits=%6zu\n",
                    i, qms, hits.size());
    }

    double total_ms = running_avg_ms * Q;
    double avg_hits = static_cast<double>(total_hits) / Q;

    std::printf("\n  total        : %.3f s\n",        total_ms * 1e-3);
    std::printf("  avg query    : %.3f ms\n",         running_avg_ms);
    std::printf("  QPS          : %.1f\n",            Q / (total_ms * 1e-3));
    std::printf("  avg hits     : %.1f\n",            avg_hits);
    std::printf("  avg cand/q   : %.1f\n",            double(timers.cand_count)   / Q);
    std::printf("  avg pigeon/q : %.1f\n",            double(total_pigeons)       / Q);

    std::printf("\n  --- Stage breakdown (cumulative across %d queries) ---\n", Q);
    auto line = [&](const char* name, double tms) {
        std::printf("    %-32s %9.2f ms total  %7.3f ms/q\n",
                    name, tms, tms / Q);
    };
    line("JS-divergence",               timers.js_div);
    line("Subspace partition",          timers.partition);
    line("Pack subspace masks (u64)",   timers.mask_pack);
    line("Row-major scan (P1/P2/P3)",   timers.scan_loop);
    line("Candidate sampling",          timers.candidate_phase);
    line("Exact verification",          timers.exact_verify);

    return 0;
}
