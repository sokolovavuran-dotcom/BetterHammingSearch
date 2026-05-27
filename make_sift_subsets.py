"""Create sift_half (500k) and sift_quarter (250k) subsets of SIFT 1M.

Reads ../datasets/sift/sift_base.fvecs and writes:
  ../datasets/sift_half/sift_half_base.fvecs       (first 500k vectors)
  ../datasets/sift_half/sift_half_query.fvecs      (copy of original queries)
  ../datasets/sift_quarter/sift_quarter_base.fvecs (first 250k vectors)
  ../datasets/sift_quarter/sift_quarter_query.fvecs

Uses the first N vectors (deterministic). The SIFT base file isn't sorted
by anything semantic, so first-N is fine for scaling experiments. If you
want a randomized subset, pass --shuffle (uses seed=0 for reproducibility).
"""
import os
import argparse
import numpy as np

from dataset import read_fvecs, datasets_dir


def write_fvecs(path: str, X: np.ndarray) -> None:
    """Write an (N, d) float32 array in fvecs format.

    Layout: per row, int32 dim followed by d float32 values. Since both
    int32 and float32 are 4 bytes, we build a single (N, 1+d) int32 buffer
    and reinterpret the float bytes — one contiguous tofile() write.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    N, d = X.shape
    buf = np.empty((N, 1 + d), dtype=np.int32)
    buf[:, 0] = d
    buf[:, 1:] = X.view(np.int32)  # reinterpret float32 bits as int32 bits
    buf.tofile(path)


def make_subset(src_dir: str, dst_dir: str, src_prefix: str, dst_prefix: str,
                n_keep: int, shuffle: bool, seed: int) -> None:
    src_base  = os.path.join(src_dir, f'{src_prefix}_base.fvecs')
    src_query = os.path.join(src_dir, f'{src_prefix}_query.fvecs')
    dst_base  = os.path.join(dst_dir, f'{dst_prefix}_base.fvecs')
    dst_query = os.path.join(dst_dir, f'{dst_prefix}_query.fvecs')

    os.makedirs(dst_dir, exist_ok=True)

    print(f'\n[{dst_prefix}] reading {src_base} ...')
    base = read_fvecs(src_base)
    N_src, d = base.shape
    print(f'  source base: {base.shape} ({base.nbytes / 1e6:.1f} MB)')

    if n_keep > N_src:
        raise ValueError(f'n_keep={n_keep} > source size {N_src}')

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N_src)[:n_keep]
        subset = base[idx]
        print(f'  randomized subset (seed={seed}): kept {n_keep} of {N_src}')
    else:
        subset = base[:n_keep]
        print(f'  first-N subset: kept {n_keep} of {N_src}')

    print(f'  writing {dst_base} ...')
    write_fvecs(dst_base, subset)
    print(f'  -> {os.path.getsize(dst_base) / 1e6:.1f} MB')

    # Queries are copied unchanged so result-set comparisons across sizes
    # use the same query vectors.
    print(f'  copying queries -> {dst_query} ...')
    queries = read_fvecs(src_query)
    write_fvecs(dst_query, queries)
    print(f'  queries shape: {queries.shape}')


def main() -> None:
    ap = argparse.ArgumentParser(description='Build sift_half and sift_quarter subsets')
    ap.add_argument('--shuffle', action='store_true',
                    help='Use a randomized subset instead of first-N (seed=0)')
    ap.add_argument('--seed', type=int, default=0,
                    help='Seed for --shuffle (default: 0)')
    args = ap.parse_args()

    sift_dir = os.path.join(datasets_dir, 'sift')
    if not os.path.exists(sift_dir):
        raise FileNotFoundError(f'SIFT source not found at {sift_dir}')

    # Half
    make_subset(
        src_dir=sift_dir,
        dst_dir=os.path.join(datasets_dir, 'sift_half'),
        src_prefix='sift', dst_prefix='sift_half',
        n_keep=500_000,
        shuffle=args.shuffle, seed=args.seed,
    )

    # Quarter
    make_subset(
        src_dir=sift_dir,
        dst_dir=os.path.join(datasets_dir, 'sift_quarter'),
        src_prefix='sift', dst_prefix='sift_quarter',
        n_keep=250_000,
        shuffle=args.shuffle, seed=args.seed,
    )

    print('\nDone. Remember to register the new datasets in dataset.py / qadp_normal.py.')


if __name__ == '__main__':
    main()
