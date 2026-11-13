"""A small generalized Procrustes analysis, so the package has no dependencies beyond NumPy.

For a fuller, tested GPA (missing data, semilandmark sliding, QC) see
`gmm-core <https://github.com/gbravomorante/gmm-core>`_.
"""

from __future__ import annotations

import numpy as np


def _rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The proper rotation that best maps centred ``source`` onto centred ``target``."""
    u, _, vt = np.linalg.svd(source.T @ target)
    d = np.sign(np.linalg.det(u @ vt))
    return u @ np.diag([1.0] * (source.shape[1] - 1) + [d]) @ vt


def gpa(shapes: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Align (n, k, d) configurations by generalized Procrustes analysis.

    Returns the aligned shapes, scaled back to the sample's mean centroid size so they stay in
    the original units, and each shape's squared Procrustes distance to the consensus (on the
    unit-size scale, where the original study's 0.05 outlier threshold applies).
    """
    x = np.asarray(shapes, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"expected an (n, k, d) array; got {x.shape}")
    x = x - x.mean(axis=1, keepdims=True)
    sizes = np.sqrt((x**2).sum(axis=(1, 2)))
    x = x / sizes[:, None, None]
    mean = x[0]
    for _ in range(max_iter):
        x = np.array([s @ _rotation(s, mean) for s in x])
        new = x.mean(axis=0)
        new /= np.sqrt((new**2).sum())
        x = np.array([s @ _rotation(s, new) for s in x])
        done = np.sqrt(((new - mean) ** 2).sum()) < tol
        mean = new
        if done:
            break
    d2 = ((x - mean) ** 2).sum(axis=(1, 2))
    return x * sizes.mean(), d2
