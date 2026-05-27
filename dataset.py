import os
import numpy as np

datasets = {
    'sift': 'sift',
    'siftsmall': 'siftsmall',
    'sift_half': 'sift_half',         # first 500k of sift1M (see make_sift_subsets.py)
    'sift_quarter': 'sift_quarter',   # first 250k of sift1M
}

datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'datasets')


def read_fvecs(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, 1 + d)[:, 1:].view(np.float32)


def binarize(X: np.ndarray, seed: int = 0) -> np.ndarray:
    """Gaussian Random Projection binarization.

    Projects X onto n_bits random hyperplanes drawn from N(0,1) and thresholds
    at 0. The projection matrix is generated deterministically from (D, n_bits, seed)
    so calling binarize on base and queries with the same arguments yields codes
    in the same Hamming space.

    X      : (N, D) float array
    seed   : RNG seed for the projection matrix
    returns: (N, n_bits//8) uint8 packed binary codes
    """
    N, D = X.shape
    n_bits = D
    R = np.random.default_rng(seed).standard_normal((D, n_bits)).astype(np.float32)
    bits = (X @ R > 0).astype(np.uint8)
    return np.packbits(bits, axis=1)
