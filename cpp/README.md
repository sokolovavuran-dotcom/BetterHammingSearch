# C++ implementations

Native implementations of the Hamming-range-query algorithms studied in
this diploma, so we can compare algorithms instead of comparing languages.

## Why C++?

The Python versions are vectorized differently for each algorithm:

- `linear_scan` is one big numpy XOR + popcount — runs entirely in C.
- `qadp_normal`, `mih`, `hmsearch` have irregular control flow
  (subspace loops, dict lookups, set unions) — they run mostly in
  Python and pay heavy interpreter overhead per element.

That makes `linear_scan` look ~10× faster than the index-based
methods, but the comparison measures **languages**, not algorithms.
In C++ all algorithms run at hardware speed, and the algorithmic
trade-offs become visible.

## Requirements

- C++20 compiler:
  - **Windows (MSYS2)**: `pacman -S mingw-w64-x86_64-gcc make`
  - **Linux**: `apt install build-essential` (g++ >= 10)
  - **macOS**: `xcode-select --install`
- GNU make

## Build

```
cd cpp
make
```

Produces `linear_scan(.exe)` and `qadp_normal(.exe)` in this folder.

## Run

Each binary reads `../../datasets/<name>/<name>_base.fvecs` and
`../../datasets/<name>/<name>_query.fvecs` (same layout as the Python
scripts).

```
./linear_scan  --dataset siftsmall --radius 20 --query_count 10
./qadp_normal  --dataset sift      --radius 20 --query_count 10 --eps 0.1 --delta 0.1
```

Available datasets: any subfolder under `../datasets/` that follows
the `<name>_base.fvecs` / `<name>_query.fvecs` convention, e.g.
`sift`, `siftsmall`, `sift_half`, `sift_quarter`.

## A note on cross-language comparability

The C++ `binarize` uses `std::mt19937` to draw the Gaussian random
projection matrix; Python's `binarize` uses numpy's PCG64. Same seed
will produce **different** projection matrices, and therefore different
binary codes — so C++ hit counts will not match Python hit counts on
the same dataset.

This is fine for **within-C++** algorithm comparisons (which is the
point of this folder). For cross-language validation, both sides would
need to use the same RNG or share a pre-binarized code file.
