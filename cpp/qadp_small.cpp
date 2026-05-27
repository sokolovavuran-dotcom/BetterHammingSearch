// QADP SP-STC: Query-Aware Dimension Partitioning, Small Threshold Case.
// C++ port of qadp_small.py.
//
// Differences from SP-NC (qadp_normal.cpp):
//   - Partition: dims sorted by JS-divergence DESCENDING, assigned in
//     contiguous chunks of size r.  Most-discriminative bits checked first.
//   - Number of subspaces: m = ceil(d / r)  (was r in SP-NC).
//   - Only two priorities: anti-pigeonhole (exceed) and enough-samples.
//     No pigeonhole / candidate-promotion step.  Active set after the loop
//     IS the remaining candidate set that needs sampling.
//
// Usage:
//   ./qadp_small --dataset siftsmall --radius 4 --query_count 10 \
//                --eps 0.1 --delta 0.1 --seed 0
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
// Helpers (same as qadp_normal.cpp)
// ---------------------------------------------------------------------------

static int compute_sample_count(int d, double eps, double delta) {
    double log_inv = std::log(1.0 / delta);
    double denom = 2.0 * d * eps * eps + log_inv;
    double l1 = (d + 1) * log_inv / denom;
    double l2 = (2.0 * d * eps * eps + d * log_inv) / denom;
    int l = static_cast<int>(std::ceil(std::min(l1, l2)));
    return std::max(1, std::min(d, l));
}

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

// SP-STC partition: descending JS-div, contiguous chunks of size r.
// Number of subspaces m = ceil(d / r).  Last subspace may be smaller than r.
static std::vector<std::vector<int>>
qadp_small_partition(int d, int r, const std::vector<double>& js_div)
{
    std::vector<int> order(d);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return js_div[a] > js_div[b]; });  // descending

    int m = (d + r - 1) / r;
    std::vector<std::vector<int>> subs(m);
    for (int k = 0; k < m; ++k) {
        int start = k * r;
        int end   = std::min((k + 1) * r, d);
        subs[k].assign(order.begin() + start, order.begin() + end);
    }
    return subs;
}

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

static std::vector<uint8_t> bits_to_byte_mask(const std::vector<int>& bits, int d)
{
    const int D = d / 8;
    std::vector<uint8_t> mask(D, 0);
    for (int b : bits)
        mask[b >> 3] |= static_cast<uint8_t>(1u << (7 - (b & 7)));
    return mask;
}

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
// Timers
// ---------------------------------------------------------------------------

struct StageTimers {
    double js_div          = 0.0;
    double partition       = 0.0;
    double byte_masks      = 0.0;
    double xor_full        = 0.0;
    double subspace_loop   = 0.0;
    double sampling_phase  = 0.0;
    double exact_verify    = 0.0;
    long long cand_count   = 0;     // size of R before verification, summed
    long long active_after = 0;     // size of remaining active set after loop, summed
};

// ---------------------------------------------------------------------------
// Core query (Algorithm 2: SP-STC)
// ---------------------------------------------------------------------------

static std::vector<int32_t>
sp_stc_query(const ByteMatrix& codes,
             const std::vector<uint8_t>& xor_full,           // (N*D)
             const std::vector<std::vector<uint8_t>>& sub_masks,
             const std::vector<int>& sub_sizes,
             const std::vector<int>& all_dims,
             int d, int r, int l,
             std::mt19937& rng,
             StageTimers& timers)
{
    const std::size_t N = codes.n;
    const std::size_t D = codes.D;
    const int m = static_cast<int>(sub_masks.size());

    std::vector<int32_t> Hc(N, 0);
    std::vector<int32_t> Bc(N, 0);

    // C starts as the full set — no in_M / in_C distinction in SP-STC.
    // We track candidacy via the `active` vector directly: a vector is in
    // C iff it is still in `active` at the end of the subspace loop.
    std::vector<int32_t> active(N);
    std::iota(active.begin(), active.end(), 0);

    std::vector<int32_t> R;
    R.reserve(1024);

    common::Timer t_loop;
    std::vector<int32_t> next_active;
    next_active.reserve(N);

    for (int k = 0; k < m; ++k) {
        if (active.empty()) break;
        const uint8_t* mask_k = sub_masks[k].data();
        const int sub_size = sub_sizes[k];

        next_active.clear();
        for (std::size_t a = 0; a < active.size(); ++a) {
            const int32_t idx = active[a];
            const uint8_t* xrow = xor_full.data() + static_cast<std::size_t>(idx) * D;
            const int h_sub = popcount_masked(xrow, mask_k, D);

            int32_t hc_new = Hc[idx] + h_sub;
            int32_t bc_new = Bc[idx] + sub_size;
            Hc[idx] = hc_new;
            Bc[idx] = bc_new;

            // Anti-pigeonhole pruning — definitively not a result.
            if (hc_new > r) continue;

            // Enough dims sampled → decide via estimated full Hamming.
            if (bc_new >= l) {
                double est = static_cast<double>(d) / bc_new * hc_new;
                if (est <= r) R.push_back(idx);
                continue;
            }

            // Still in C — process again next subspace.
            next_active.push_back(idx);
        }
        active.swap(next_active);
    }
    timers.subspace_loop += t_loop.ms();
    timers.active_after  += static_cast<long long>(active.size());

    // -----------------------------------------------------------------------
    // Sampling phase: for each remaining candidate, sample (l - Bc) extra
    // dims from all_dims[Bc:], compute h_s, refine estimate, decide.
    // -----------------------------------------------------------------------
    common::Timer t_smp;
    const int K_total = static_cast<int>(all_dims.size());
    std::vector<int> sample_pool;
    sample_pool.reserve(K_total);
    std::vector<uint8_t> sample_mask(D);

    for (std::size_t a = 0; a < active.size(); ++a) {
        const int32_t idx = active[a];
        const int bc_val = Bc[idx];
        if (bc_val >= l) {
            int divisor = std::max(bc_val, 1);
            double est = static_cast<double>(d) / divisor * Hc[idx];
            if (est <= r) R.push_back(idx);
            continue;
        }
        const int need = l - bc_val;
        const int unchecked = K_total - bc_val;
        const int n_sample = std::min(need, unchecked);

        sample_pool.assign(all_dims.begin() + bc_val, all_dims.end());
        for (int s = 0; s < n_sample; ++s) {
            std::uniform_int_distribution<int> dist(s, unchecked - 1);
            int p = dist(rng);
            std::swap(sample_pool[s], sample_pool[p]);
        }

        std::fill(sample_mask.begin(), sample_mask.end(), uint8_t(0));
        for (int s = 0; s < n_sample; ++s) {
            int b = sample_pool[s];
            sample_mask[b >> 3] |= static_cast<uint8_t>(1u << (7 - (b & 7)));
        }

        const uint8_t* xrow = xor_full.data() + static_cast<std::size_t>(idx) * D;
        const int h_s = popcount_masked(xrow, sample_mask.data(), D);
        const int total_h = Hc[idx] + h_s;
        double est = static_cast<double>(d) / l * total_h;
        if (est <= r) R.push_back(idx);
    }
    timers.sampling_phase += t_smp.ms();

    // -----------------------------------------------------------------------
    // Exact verification on R (dedup + full popcount).
    // -----------------------------------------------------------------------
    common::Timer t_ex;
    std::sort(R.begin(), R.end());
    R.erase(std::unique(R.begin(), R.end()), R.end());
    timers.cand_count += static_cast<long long>(R.size());

    std::vector<int32_t> result;
    result.reserve(R.size());
    for (int32_t idx : R) {
        const uint8_t* xrow = xor_full.data() + static_cast<std::size_t>(idx) * D;
        if (popcount_full(xrow, D) <= r) result.push_back(idx);
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
    const int         radius      = args.get_int("radius", 4);
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
    if (radius >= small_thresh) {
        std::printf("  WARNING: r=%d >= 0.05*d=%.1f — "
                    "this is the normal-threshold case; use qadp_normal instead\n",
                    radius, small_thresh);
    }

    const int l = compute_sample_count(d, eps, delta);
    const int m = (d + radius - 1) / radius;          // ceil(d / r)
    std::printf("  eps=%.2f delta=%.2f -> l=%d samples\n", eps, delta, l);
    std::printf("  subspaces m=%d  (each ~%d bits, last may be smaller)\n", m, radius);

    std::printf("Computing data_p1 ...\n");
    common::Timer tp1;
    std::vector<double> data_p1 = compute_data_p1(base_codes);
    std::printf("  data_p1 : %.2f s\n", tp1.s());

    std::mt19937 rng(static_cast<uint32_t>(seed));

    const int Q = std::min<int>(query_count, static_cast<int>(query_codes.n));
    std::printf("\nQADP-STC range query  r=%d, m=%d, Q=%d ...\n", radius, m, Q);

    StageTimers timers;
    std::vector<uint8_t> xor_full(base_codes.n * D);
    double running_avg_ms = 0.0;
    long long total_hits = 0;

    for (int i = 0; i < Q; ++i) {
        common::Timer t_query;

        const uint8_t* q_code = query_codes.row(i);
        std::vector<uint8_t> q_bits = unpack_one(q_code, D);

        common::Timer t_js;
        std::vector<double> js = js_div_all(data_p1, q_bits);
        timers.js_div += t_js.ms();

        common::Timer t_part;
        std::vector<std::vector<int>> subs = qadp_small_partition(d, radius, js);
        std::vector<int> all_dims;
        all_dims.reserve(d);
        for (const auto& s : subs) all_dims.insert(all_dims.end(), s.begin(), s.end());
        timers.partition += t_part.ms();

        common::Timer t_mk;
        std::vector<std::vector<uint8_t>> sub_masks;
        std::vector<int> sub_sizes;
        sub_masks.reserve(m);
        sub_sizes.reserve(m);
        for (const auto& s : subs) {
            sub_masks.push_back(bits_to_byte_mask(s, d));
            sub_sizes.push_back(static_cast<int>(s.size()));
        }
        timers.byte_masks += t_mk.ms();

        common::Timer t_xor;
        for (std::size_t r_i = 0; r_i < base_codes.n; ++r_i) {
            const uint8_t* row = base_codes.row(r_i);
            uint8_t* xrow = xor_full.data() + r_i * D;
            for (std::size_t j = 0; j < D; ++j) xrow[j] = row[j] ^ q_code[j];
        }
        timers.xor_full += t_xor.ms();

        auto hits = sp_stc_query(base_codes, xor_full, sub_masks, sub_sizes,
                                 all_dims, d, radius, l, rng, timers);
        double qms = t_query.ms();
        running_avg_ms += (qms - running_avg_ms) / (i + 1);
        total_hits += static_cast<long long>(hits.size());
        std::printf("  q[%02d]  query=%7.2f ms  hits=%6zu\n",
                    i, qms, hits.size());
    }

    double total_ms = running_avg_ms * Q;
    double avg_hits = static_cast<double>(total_hits) / Q;

    std::printf("\n  total        : %.3f s\n",        total_ms * 1e-3);
    std::printf("  avg query    : %.3f ms\n",         running_avg_ms);
    std::printf("  QPS          : %.1f\n",            Q / (total_ms * 1e-3));
    std::printf("  avg hits     : %.1f\n",            avg_hits);
    std::printf("  avg cand/q   : %.1f\n",            double(timers.cand_count) / Q);
    std::printf("  avg active_after_loop/q : %.1f\n", double(timers.active_after) / Q);

    std::printf("\n  --- Stage breakdown (cumulative across %d queries) ---\n", Q);
    auto line = [&](const char* name, double tms) {
        std::printf("    %-32s %9.2f ms total  %7.3f ms/q\n",
                    name, tms, tms / Q);
    };
    line("JS-divergence",            timers.js_div);
    line("Subspace partition",       timers.partition);
    line("Build byte masks",         timers.byte_masks);
    line("Precompute xor_full",      timers.xor_full);
    line("Subspace loop (P1/P2)",    timers.subspace_loop);
    line("Sampling phase",           timers.sampling_phase);
    line("Exact verification",       timers.exact_verify);

    return 0;
}
