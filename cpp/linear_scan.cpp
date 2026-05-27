// Linear-scan Hamming range query, C++ version.
//
// Mirrors linear_scan.py: load fvecs, binarize with GRP, run Q queries,
// report per-query and average timing.
//
// Usage:
//   ./linear_scan --dataset siftsmall --radius 20 --query_count 10 --seed 0
#include <cstdio>
#include <iostream>
#include <vector>

#include "common.hpp"

namespace fs = std::filesystem;
using common::ByteMatrix;
using common::FloatMatrix;

// Return indices i with hamming(codes[i], q) <= r.
static std::vector<int32_t> linear_scan_range(const ByteMatrix& codes,
                                              const uint8_t* q,
                                              int r)
{
    std::vector<int32_t> hits;
    hits.reserve(64);
    const std::size_t N = codes.n;
    const std::size_t D = codes.D;
    for (std::size_t i = 0; i < N; ++i) {
        if (common::hamming(codes.row(i), q, D) <= r) {
            hits.push_back(static_cast<int32_t>(i));
        }
    }
    return hits;
}

int main(int argc, char** argv)
{
    common::Args args(argc, argv);
    const std::string dataset     = args.get("dataset", "siftsmall");
    const int         radius      = args.get_int("radius", 20);
    const int         query_count = args.get_int("query_count", 10);
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
    std::printf("  packed base    : (%zu, %zu) bytes  bits=%zu\n",
                base_codes.n, base_codes.D, base_codes.bits());
    std::printf("  binarize       : %.2f s\n", tbin.s());

    const int Q = std::min<int>(query_count, static_cast<int>(query_codes.n));
    std::printf("\nLinear scan  r=%d, Q=%d ...\n", radius, Q);

    double running_avg_ms = 0.0;
    long long total_hits = 0;
    for (int i = 0; i < Q; ++i) {
        common::Timer tq;
        auto hits = linear_scan_range(base_codes, query_codes.row(i), radius);
        double qms = tq.ms();
        running_avg_ms += (qms - running_avg_ms) / (i + 1);
        total_hits += static_cast<long long>(hits.size());
        std::printf("  q[%02d]  scan=%7.2f ms  hits=%6zu\n", i, qms, hits.size());
    }

    double total_ms = running_avg_ms * Q;
    double avg_hits = static_cast<double>(total_hits) / Q;

    std::printf("\n  total        : %.3f s\n",        total_ms * 1e-3);
    std::printf("  avg query    : %.3f ms\n",         running_avg_ms);
    std::printf("  QPS          : %.1f\n",            Q / (total_ms * 1e-3));
    std::printf("  avg hits     : %.1f\n",            avg_hits);
    return 0;
}
