import numpy as np

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_distances(codes: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    codes : (N, D) uint8
    q     : (D,)   uint8
    returns (N,) int32 Hamming distances
    """
    xor = np.bitwise_xor(codes, q[np.newaxis, :])
    return _POPCOUNT_TABLE[xor].sum(axis=1).astype(np.int32)


def linear_scan_range(codes: np.ndarray, q: np.ndarray, r: int) -> np.ndarray:
    """Return indices of rows in codes whose Hamming distance to q is <= r."""
    d = hamming_distances(codes, q)
    return np.flatnonzero(d <= r).astype(np.int32)
