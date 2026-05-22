import os
import time
import numpy as np
from linear_scan import linear_scan_range

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def read_fvecs(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        data = np.fromfile(f, dtype=np.int32)
    d = data[0]
    return data.reshape(-1, 1 + d)[:, 1:].view(np.float32)


def binarize(X: np.ndarray, thresholds: np.ndarray = None) -> np.ndarray:
    """Threshold each dimension; pack 8 bits per byte -> (N, D/8) uint8."""
    if thresholds is None:
        thresholds = X.mean(axis=0)
    bits = (X > thresholds).astype(np.uint8)
    return np.packbits(bits, axis=1)
