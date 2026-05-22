# Benchmark Results

**Dataset:** SIFT (1,000,000 base vectors, 128-dim, binarized to 16 bytes / 128 bits)  
**Queries:** 100

---

## Summary Table — radius 5

Avg hits per query: **1.7**

| Algorithm    | Segments       | Prework time | Index size (est.) | Peak mem (build) | Avg query time | QPS    | Peak mem (query) |
|--------------|----------------|-------------:|------------------:|-----------------:|---------------:|-------:|-----------------:|
| Linear Scan  | —              | —            | —                 | —                | 53.339 ms      |   18.7 | —                |
| MIH          | m=8, r_seg=0   | 11.332 s     | 319.93 MB         | 351.87 MB        | 4.957 ms       |  201.7 | 6.44 MB          |
| HmSearch     | m=6, ~21 bits  | 11.873 s     | 332.92 MB         | 487.87 MB        | 0.830 ms       | 1204.4 | 1.53 MB          |

MIH speedup vs linear scan: **10.8×**  
HmSearch speedup vs linear scan: **64.3×**

---

## Summary Table — radius 20

Avg hits per query: **2540.6**

| Algorithm    | Segments       | Prework time | Index size (est.) | Peak mem (build) | Avg query time | QPS  | Peak mem (query) |
|--------------|----------------|-------------:|------------------:|-----------------:|---------------:|-----:|-----------------:|
| Linear Scan  | —              | —            | —                 | —                | 53.473 ms      | 18.7 | —                |
| MIH          | m=8, r_seg=2   | 11.383 s     | 319.93 MB         | 351.87 MB        | 77.415 ms      | 12.9 | 40.13 MB         |
| MIH          | m=16, r_seg=1  | 13.921 s     | 584.57 MB         | 648.30 MB        | 333.515 ms     |  3.0 | 82.59 MB         |
| HmSearch     | m=21, ~6 bits  | 21.550 s     | 765.95 MB         | 978.72 MB        | 197.080 ms     |  5.1 | 75.27 MB         |

At r=20 all index-based methods are **slower** than linear scan (see explanation below).

---

## Analysis

### Why r=5 favors index methods

At r=5 there are only 1.7 true results per query on average — a very sparse regime.

- **MIH (r_seg=0):** each of 8 segments does a single exact hash lookup in a 16-bit (2-byte) table. Buckets hold ~15 entries on average (1M / 65536), yielding ~120 candidates to verify via numpy — far less than scanning 1M.
- **HmSearch (m=6, ~21-bit segments):** 6 exact lookups in tables with 2^21 possible keys. Buckets average < 1 entry — almost no false positives reach verification, making query time dominated purely by the 6 dict lookups.

### Why r=20 hurts index methods

At r=20 there are 2540 true results per query (0.25% of the database) — a high-recall, high-hit regime.

- **MIH (m=8, r_seg=2):** 8 × 137 = 1096 Python-level hash lookups generate a large candidate set. Python loop overhead + verification of thousands of candidates exceeds numpy's fully-vectorized linear pass.
- **MIH (m=16, r_seg=1):** fewer lookups (144), but single-byte buckets average 3900 entries, flooding the candidate set (hundreds of thousands) — much worse.
- **HmSearch:** 21 exact lookups, but each bucket is large enough at this radius to generate many candidates, and 21 Python-level iterations add overhead.
- **Linear Scan:** a single numpy pass over 16 MB saturates memory bandwidth with no Python overhead — unbeatable in this regime.

Index-based methods show their full advantage in **compiled implementations** (C++, Cython, Numba) where Python loop overhead is eliminated, or at **billion-scale** datasets where vectorized scan itself becomes the bottleneck.

---

## Raw Output — radius 5

### Linear Scan

```
Linear scan  r=5, Q=100 ...
  total        : 5.334 s
  avg query    : 53.339 ms  (running average)
  QPS          : 18.7
  avg hits     : 1.7
```

### MIH (m=8, seg=2 bytes / 16 bits, r_seg=0, 1 neighbor/segment)

```
  segments     : m=8,  seg=2 bytes (16 bits)
  r / r_seg    : 5 / 0  ->  1 neighbors per segment

Building MIH index ...
  build time   : 11.332 s
  codes array  : 16.00 MB
  index size   : 319.93 MB  (deep estimate)
  peak (build) : 351.87 MB  (tracemalloc)

MIH range query  r=5, m=8, Q=100 ...
  total        : 0.496 s
  avg query    : 4.957 ms  (running average)
  QPS          : 201.7
  avg hits     : 1.7
  peak (query) : 6.44 MB  (tracemalloc)
```

### HmSearch (m=6, ~21 bits each)

```
  segments   : m=6  (~21 bits each)

Building HmSearch index ...
  build time   : 11.873 s
  codes array  : 16.00 MB
  index size   : 332.92 MB  (deep estimate)
  peak (build) : 487.87 MB  (tracemalloc)

HmSearch range query  r=5, m=6, Q=100 ...
  total        : 0.083 s
  avg query    : 0.830 ms  (running average)
  QPS          : 1204.4
  avg hits     : 1.7
  peak (query) : 1.53 MB  (tracemalloc)
```

---

## Raw Output — radius 20

### Linear Scan

```
Linear scan  r=20, Q=100 ...
  total        : 5.347 s
  avg query    : 53.473 ms  (running average)
  QPS          : 18.7
  avg hits     : 2540.6
```

### MIH (m=8, seg=2 bytes / 16 bits, r_seg=2, 137 neighbors/segment)

```
  segments     : m=8,  seg=2 bytes (16 bits)
  r / r_seg    : 20 / 2  ->  137 neighbors per segment

Building MIH index ...
  build time   : 11.383 s
  codes array  : 16.00 MB
  index size   : 319.93 MB  (deep estimate)
  peak (build) : 351.87 MB  (tracemalloc)

MIH range query  r=20, m=8, Q=100 ...
  total        : 7.741 s
  avg query    : 77.415 ms  (running average)
  QPS          : 12.9
  avg hits     : 2540.6
  peak (query) : 40.13 MB  (tracemalloc)
```

### MIH (m=16, seg=1 byte / 8 bits, r_seg=1, 9 neighbors/segment)

```
  segments     : m=16,  seg=1 bytes (8 bits)
  r / r_seg    : 20 / 1  ->  9 neighbors per segment

Building MIH index ...
  build time   : 13.921 s
  codes array  : 16.00 MB
  index size   : 584.57 MB  (deep estimate)
  peak (build) : 648.30 MB  (tracemalloc)

MIH range query  r=20, m=16, Q=100 ...
  total        : 33.352 s
  avg query    : 333.515 ms  (running average)
  QPS          : 3.0
  avg hits     : 2540.6
  peak (query) : 82.59 MB  (tracemalloc)
```

### HmSearch (m=21, ~6 bits each)

```
  segments   : m=21  (~6 bits each)

Building HmSearch index ...
  build time   : 21.550 s
  codes array  : 16.00 MB
  index size   : 765.95 MB  (deep estimate)
  peak (build) : 978.72 MB  (tracemalloc)

HmSearch range query  r=20, m=21, Q=100 ...
  total        : 19.708 s
  avg query    : 197.080 ms  (running average)
  QPS          : 5.1
  avg hits     : 2540.6
  peak (query) : 75.27 MB  (tracemalloc)
```
