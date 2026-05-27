// Shared utilities for the C++ benchmark binaries.
//
// - read_fvecs : load Texmex .fvecs files
// - binarize   : Gaussian Random Projection -> packed binary codes
// - hamming    : popcount-of-XOR over packed bytes (uint64-chunked)
// - Timer      : high-resolution wall-clock helper
//
// All operations work on packed uint8 codes of shape (N, D) where D = d/8.
#pragma once

#include <bit>            // std::popcount
#include <chrono>
#include <cstdint>
#include <cstring>        // std::memcpy
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace common {

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

class Timer {
public:
    Timer() : start_(Clock::now()) {}
    void reset()              { start_ = Clock::now(); }
    double ms() const {
        auto now = Clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    double s() const { return ms() * 1e-3; }
private:
    Clock::time_point start_;
};

// ---------------------------------------------------------------------------
// Float matrix (row-major, owns its data)
// ---------------------------------------------------------------------------

struct FloatMatrix {
    std::size_t n = 0;
    std::size_t d = 0;
    std::vector<float> data;

    const float* row(std::size_t i) const { return data.data() + i * d; }
    float*       row(std::size_t i)       { return data.data() + i * d; }
};

// Read a Texmex .fvecs file.  Each record: int32 dim, float32 * dim.
inline FloatMatrix read_fvecs(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);

    f.seekg(0, std::ios::end);
    std::size_t file_bytes = static_cast<std::size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    int32_t d;
    f.read(reinterpret_cast<char*>(&d), sizeof(d));
    if (!f) throw std::runtime_error("Empty/short fvecs file: " + path);

    std::size_t rec_bytes = 4ULL * (1 + d);
    if (file_bytes % rec_bytes != 0)
        throw std::runtime_error("fvecs file size not divisible by record size");
    std::size_t n = file_bytes / rec_bytes;

    FloatMatrix m;
    m.n = n;
    m.d = d;
    m.data.resize(n * d);

    f.seekg(0, std::ios::beg);
    for (std::size_t i = 0; i < n; ++i) {
        int32_t dim_check;
        f.read(reinterpret_cast<char*>(&dim_check), 4);
        if (dim_check != d) throw std::runtime_error("Inconsistent dim in fvecs");
        f.read(reinterpret_cast<char*>(m.data.data() + i * d), sizeof(float) * d);
    }
    return m;
}

// ---------------------------------------------------------------------------
// Packed binary code matrix (row-major)
// ---------------------------------------------------------------------------

struct ByteMatrix {
    std::size_t n = 0;
    std::size_t D = 0;             // bytes per code = d / 8
    std::vector<uint8_t> data;

    const uint8_t* row(std::size_t i) const { return data.data() + i * D; }
    uint8_t*       row(std::size_t i)       { return data.data() + i * D; }
    std::size_t    bits() const { return D * 8; }
};

// ---------------------------------------------------------------------------
// Gaussian Random Projection binarization
// Bit b of code i is set iff (X[i] . R[:, b]) > 0.
// Packs bits in MSB-first byte order to match numpy.packbits's default.
//
// NOTE: This uses std::mt19937 — it produces a DIFFERENT projection matrix
// than Python's numpy.random.default_rng(seed).standard_normal(...), so C++
// hit counts will not match the Python ones bit-for-bit. The Hamming space
// produced is, however, statistically equivalent (i.i.d. standard normals).
// ---------------------------------------------------------------------------

inline ByteMatrix binarize(const FloatMatrix& X, uint32_t seed = 0) {
    const std::size_t n = X.n;
    const std::size_t d = X.d;
    const std::size_t n_bits = d;             // bit-length matches input dim
    if (n_bits % 8 != 0)
        throw std::runtime_error("binarize: d must be a multiple of 8");
    const std::size_t D = n_bits / 8;

    // Build (d, n_bits) projection matrix.
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> R(d * n_bits);
    for (std::size_t i = 0; i < R.size(); ++i) R[i] = dist(gen);

    ByteMatrix out;
    out.n = n;
    out.D = D;
    out.data.assign(n * D, 0u);

    // For each row, compute the projection and pack each sign bit.
    std::vector<float> proj(n_bits);
    for (std::size_t i = 0; i < n; ++i) {
        const float* x = X.row(i);
        // proj = x @ R   — naive triple loop; n is the outer.
        // For SIFT 1M / d=128, this is ~16 G mul-adds; -march=native + -O3
        // will SIMD-vectorize the inner loop to ~3-5 seconds.
        std::fill(proj.begin(), proj.end(), 0.0f);
        for (std::size_t j = 0; j < d; ++j) {
            const float xj = x[j];
            const float* Rj = R.data() + j * n_bits;
            for (std::size_t b = 0; b < n_bits; ++b) {
                proj[b] += xj * Rj[b];
            }
        }
        uint8_t* code = out.row(i);
        for (std::size_t b = 0; b < n_bits; ++b) {
            if (proj[b] > 0.0f)
                code[b >> 3] |= static_cast<uint8_t>(1u << (7 - (b & 7)));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Hamming distance via popcount of XOR, chunked into uint64 words.
// ---------------------------------------------------------------------------

inline int hamming(const uint8_t* a, const uint8_t* b, std::size_t D) {
    int total = 0;
    std::size_t i = 0;
    for (; i + 8 <= D; i += 8) {
        uint64_t x, y;
        std::memcpy(&x, a + i, 8);
        std::memcpy(&y, b + i, 8);
        total += std::popcount(x ^ y);
    }
    for (; i < D; ++i) {
        total += std::popcount(static_cast<uint8_t>(a[i] ^ b[i]));
    }
    return total;
}

// ---------------------------------------------------------------------------
// Argument parser (tiny — enough for our --flag value pairs).
// ---------------------------------------------------------------------------

class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i + 1 < argc; i += 2) {
            std::string k = argv[i];
            std::string v = argv[i + 1];
            if (k.starts_with("--")) k = k.substr(2);
            kv_[k] = v;
        }
    }
    std::string get(const std::string& k, const std::string& def) const {
        auto it = kv_.find(k);
        return it == kv_.end() ? def : it->second;
    }
    int    get_int   (const std::string& k, int def)    const { return std::stoi  (get(k, std::to_string(def))); }
    double get_double(const std::string& k, double def) const { return std::stod  (get(k, std::to_string(def))); }
private:
    std::unordered_map<std::string, std::string> kv_;
};

// ---------------------------------------------------------------------------
// Resolve dataset directory: ../datasets/<name>/  (relative to cpp/)
// ---------------------------------------------------------------------------

inline std::filesystem::path dataset_dir(const std::string& name) {
    namespace fs = std::filesystem;
    fs::path here = fs::current_path();
    // Try ../datasets first (when running from cpp/), then ../../datasets (just in case).
    for (const fs::path candidate : { here / ".." / "datasets" / name,
                                      here / ".." / ".." / "datasets" / name }) {
        if (fs::exists(candidate / (name + "_base.fvecs"))) return candidate;
    }
    throw std::runtime_error("Cannot find dataset directory for: " + name);
}

}  // namespace common
