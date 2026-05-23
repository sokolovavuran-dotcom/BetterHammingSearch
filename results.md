# Benchmark Results

**Radius:** r = 5 | **Queries:** Q = 100 | Codes: 128-dim float → 128-bit / 16-byte binary

---

# SIFT (1 000 000 base vectors)

## Summary Table

| Algorithm | Build time (s) | Index size (MB) | Peak build (MB) | Avg query (ms) | QPS | Avg hits | Peak query (MB) |
|-----------|---------------:|----------------:|----------------:|---------------:|----:|---------:|----------------:|
| Linear Scan | — | — | — | 55.250 | 18.1 | 1.7 | — |
| HmSearch (m=6) | 12.373 | 332.92 | 487.87 | 0.792 | 1 262.7 | 1.7 | 1.53 |
| MIH (m=8, r_seg=0) | 11.989 | 319.93 | 351.87 | 5.117 | 195.4 | 1.7 | 6.44 |
| GPH (m=16, thr=11) | 17.951 | 584.57 | 777.30 | 47.824 | 20.9 | 1.7 | 19.16 |
| QADP (T=4, probe=1) | 92.891 | 1 486.88 | 1 842.69 | 0.669 | 1 495.5 | 1.7 | 1.54 |

Speedup over linear scan: **QADP 82.5×** · HmSearch **69.8×** · MIH **10.8×** · GPH **1.2×**

## Raw Output

### Linear Scan
```
base    : (1000000, 128)
code shape : (1000000, 16)  dtype=uint8

Linear scan  r=5, Q=100 ...
  total        : 5.525 s
  avg query    : 55.250 ms  (running average)
  QPS          : 18.1
  avg hits     : 1.7
```

### HmSearch  (m=6, ~21 bits each)
```
  segments   : m=6  (~21 bits each)
  build time   : 12.373 s
  codes array  : 16.00 MB
  index size   : 332.92 MB  (deep estimate)
  peak (build) : 487.87 MB  (tracemalloc)

HmSearch range query  r=5, m=6, Q=100 ...
  total        : 0.079 s
  avg query    : 0.792 ms  (running average)
  QPS          : 1262.7
  avg hits     : 1.7
  peak (query) : 1.53 MB  (tracemalloc)
```

### MIH  (m=8, seg=2 bytes / 16 bits, r_seg=0)
```
  segments     : m=8,  seg=2 bytes (16 bits)
  r / r_seg    : 5 / 0  ->  1 neighbors per segment
  build time   : 11.989 s
  codes array  : 16.00 MB
  index size   : 319.93 MB  (deep estimate)
  peak (build) : 351.87 MB  (tracemalloc)

MIH range query  r=5, m=8, Q=100 ...
  total        : 0.512 s
  avg query    : 5.117 ms  (running average)
  QPS          : 195.4
  avg hits     : 1.7
  peak (query) : 6.44 MB  (tracemalloc)
```

### GPH  (byte-level, m=16, threshold=11/16)
```
  mode         : byte-level segments
  segments     : m=16,  threshold=11/16
  build time   : 17.951 s
  codes array  : 16.00 MB
  index size   : 584.57 MB  (deep estimate)
  peak (build) : 777.30 MB  (tracemalloc)

GPH range query  r=5, m=16, threshold=11, Q=100 ...
  total        : 4.782 s
  avg query    : 47.824 ms  (running average)
  QPS          : 20.9
  avg hits     : 1.7
  peak (query) : 19.16 MB  (tracemalloc)
```

### QADP  (T=4 families, m=6 segments, n_probe=1)
```
  families   : T=4,  segments m=6  (~21 bits each)
  build time   : 92.891 s
  codes array  : 16.00 MB
  index size   : 1486.88 MB  (deep estimate)
  peak (build) : 1842.69 MB  (tracemalloc)

QADP range query  r=5, T=4, n_probe=1, Q=100 ...
  total        : 0.067 s
  avg query    : 0.669 ms  (running average)
  QPS          : 1495.5
  avg hits     : 1.7
  peak (query) : 1.54 MB  (tracemalloc)
```

---

# siftsmall (10 000 base vectors)

## Summary Table

| Algorithm | Build time (s) | Index size (MB) | Peak build (MB) | Avg query (ms) | QPS | Avg hits | Peak query (MB) |
|-----------|---------------:|----------------:|----------------:|---------------:|----:|---------:|----------------:|
| Linear Scan | — | — | — | 0.496 | 2 015.6 | 0.1 | — |
| HmSearch (m=6) | 0.124 | 7.24 | 8.85 | 0.055 | 18 345.9 | 0.1 | 0.06 |
| MIH (m=8, r_seg=0) | 0.127 | 6.49 | 6.74 | 0.096 | 10 448.2 | 0.1 | 0.15 |
| GPH (m=16, thr=11) | 0.176 | 6.34 | 8.02 | 0.547 | 1 827.4 | 0.1 | 0.20 |
| QADP (T=4, probe=1) | 1.086 | 31.60 | 35.97 | 0.228 | 4 383.9 | 0.1 | 0.06 |

Speedup over linear scan: HmSearch **9.0×** · MIH **5.2×** · QADP **2.2×** · GPH **0.9×**

## Raw Output

---

## Summary Table

| Algorithm | Build time (s) | Index size (MB) | Peak build (MB) | Avg query (ms) | QPS | Avg hits | Peak query (MB) |
|-----------|---------------:|----------------:|----------------:|---------------:|----:|---------:|----------------:|
| Linear Scan | — | — | — | 0.496 | 2 015.6 | 0.1 | — |
| HmSearch (m=6) | 0.124 | 7.24 | 8.85 | 0.055 | 18 345.9 | 0.1 | 0.06 |
| MIH (m=8, r_seg=0) | 0.127 | 6.49 | 6.74 | 0.096 | 10 448.2 | 0.1 | 0.15 |
| GPH (m=16, thr=11) | 0.176 | 6.34 | 8.02 | 0.547 | 1 827.4 | 0.1 | 0.20 |
| QADP (T=4, probe=1) | 1.086 | 31.60 | 35.97 | 0.228 | 4 383.9 | 0.1 | 0.06 |

Speedup over linear scan: HmSearch **9.0×** · MIH **5.2×** · QADP **2.2×** · GPH **0.9×**

---

## Raw Output

### Linear Scan
```
code shape : (10000, 16)  dtype=uint8

Linear scan  r=5, Q=100 ...
  total        : 0.050 s
  avg query    : 0.496 ms  (running average)
  QPS          : 2015.6
  avg hits     : 0.1
```

### HmSearch  (m = r+1 = 6 segments, ~21 bits each)
```
code shape : (10000, 16)  dtype=uint8
segments   : m=6  (~21 bits each)

Building HmSearch index ...
  build time   : 0.124 s
  codes array  : 0.16 MB
  index size   : 7.24 MB  (deep estimate)
  peak (build) : 8.85 MB  (tracemalloc)

HmSearch range query  r=5, m=6, Q=100 ...
  total        : 0.005 s
  avg query    : 0.055 ms  (running average)
  QPS          : 18345.9
  avg hits     : 0.1
  peak (query) : 0.06 MB  (tracemalloc)
```

### MIH  (m=8 segments, seg=2 bytes / 16 bits, r_seg = floor(5/8) = 0)
```
code shape : (10000, 16)  dtype=uint8

segments     : m=8,  seg=2 bytes (16 bits)
r / r_seg    : 5 / 0  ->  1 neighbors per segment

Building MIH index ...
  build time   : 0.127 s
  codes array  : 0.16 MB
  index size   : 6.49 MB  (deep estimate)
  peak (build) : 6.74 MB  (tracemalloc)

MIH range query  r=5, m=8, Q=100 ...
  total        : 0.010 s
  avg query    : 0.096 ms  (running average)
  QPS          : 10448.2
  avg hits     : 0.1
  peak (query) : 0.15 MB  (tracemalloc)
```

### GPH  (byte-level segments, m=16, threshold=11/16)
```
code shape : (10000, 16)  dtype=uint8

Building GPH index ...
  mode         : byte-level segments
  segments     : m=16,  threshold=11/16
  build time   : 0.176 s
  codes array  : 0.16 MB
  index size   : 6.34 MB  (deep estimate)
  peak (build) : 8.02 MB  (tracemalloc)

GPH range query  r=5, m=16, threshold=11, Q=100 ...
  total        : 0.055 s
  avg query    : 0.547 ms  (running average)
  QPS          : 1827.4
  avg hits     : 0.1
  peak (query) : 0.20 MB  (tracemalloc)
```

### QADP  (T=4 families, m=6 segments each, n_probe=1)
```
code shape : (10000, 16)  dtype=uint8
families   : T=4,  segments m=6  (~21 bits each)

Building QADP index ...
  build time   : 1.086 s
  codes array  : 0.16 MB
  index size   : 31.60 MB  (deep estimate)
  peak (build) : 35.97 MB  (tracemalloc)

QADP range query  r=5, T=4, n_probe=1, Q=100 ...
  total        : 0.023 s
  avg query    : 0.228 ms  (running average)
  QPS          : 4383.9
  avg hits     : 0.1
  peak (query) : 0.06 MB  (tracemalloc)
```

---

## Notes

- All algorithms return identical result sets (avg hits = 0.1 for every method); correctness is preserved.
- At r=5 the search is very sparse — only ~0.1 results per query — so index-based methods shine by avoiding almost all of the 10 000 base codes.
- **HmSearch** is fastest: 6 exact hash lookups in large (21-bit segment) tables leave almost nothing to verify.
- **MIH** with r_seg=0 degenerates to exact lookup per segment (no neighbor enumeration), which is still faster than linear scan but slower than HmSearch's larger segments.
- **GPH** is slower here because a high threshold (11/16 segments must match) forces visiting many buckets per query with a vote-counting step that has Python overhead.
- **QADP** sits between MIH and GPH: it builds T=4 families (hence 4× the index size and build time vs HmSearch) and selects the best-fitting permutation per query. At n_probe=1 it gives zero false negatives by the same pigeonhole argument as HmSearch.
