"""Bandpass filtering of partial-warp bending energy for 3D surface semilandmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _tps_kernel(r2: np.ndarray) -> np.ndarray:
    """The 2-D thin-plate spline kernel U(r) = r^2 log r^2, with U(0) = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = r2 * np.log(r2)
    out[r2 == 0] = 0.0
    return out


def bending_energy_matrix(points_2d: np.ndarray) -> np.ndarray:
    """Bookstein's bending-energy matrix for a 2-D configuration of k points.

    The upper-left k x k block of the inverse of the thin-plate spline system matrix
    ``[[K, P], [P^T, 0]]``. It is symmetric and positive semi-definite, and its null space
    is the three-dimensional space of affine functions, which cost no bending energy.
    """
    pts = np.asarray(points_2d, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"expected a (k, 2) configuration; got {pts.shape}")
    k = len(pts)
    if k < 4:
        raise ValueError("at least 4 points are needed for a non-affine deformation")
    diff = pts[:, None, :] - pts[None, :, :]
    K = _tps_kernel((diff**2).sum(axis=2))
    P = np.column_stack([np.ones(k), pts])
    L = np.zeros((k + 3, k + 3))
    L[:k, :k], L[:k, k:], L[k:, :k] = K, P, P.T
    Linv = np.linalg.inv(L)
    B = Linv[:k, :k]
    return (B + B.T) / 2.0


@dataclass
class PartialWarps:
    """The partial-warp basis of a reference surface.

    ``eigenvalues[j]`` is the specific bending energy of partial warp ``j + 1`` (ascending:
    partial warp 1 is the largest-scale bend), and ``vectors[:, j]`` its pattern over the k points.
    ``normal`` is the direction in which deformations are measured, perpendicular to the
    reference's best-fitting plane.
    """

    eigenvalues: np.ndarray
    vectors: np.ndarray
    origin: np.ndarray
    plane_axes: np.ndarray
    normal: np.ndarray

    @property
    def n_warps(self) -> int:
        return len(self.eigenvalues)


def partial_warps(reference: np.ndarray) -> PartialWarps:
    """Partial warps of a (k, 3) reference surface, such as a sample's mean shape.

    The reference is projected onto its best-fitting plane; the bending-energy matrix of that
    2-D configuration is diagonalised; the three affine (zero-energy) directions are dropped and
    the rest are sorted from the largest geometric scale to the smallest.
    """
    ref = np.asarray(reference, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError(f"expected a (k, 3) reference; got {ref.shape}")
    origin = ref.mean(axis=0)
    _, _, vt = np.linalg.svd(ref - origin, full_matrices=False)
    plane_axes, normal = vt[:2], vt[2]
    B = bending_energy_matrix((ref - origin) @ plane_axes.T)
    values, vectors = np.linalg.eigh(B)
    keep = np.argsort(values)[3:]  # the three smallest are the affine null space
    return PartialWarps(values[keep], vectors[:, keep], origin, plane_axes, normal)


def partial_warp_amplitudes(shapes: np.ndarray, warps: PartialWarps, reference: np.ndarray) -> np.ndarray:
    """Amplitude of every partial warp for every shape, as an (n, n_warps) array.

    Each shape's departure from the reference is measured perpendicular to the reference's
    best-fitting plane and projected onto the partial warps. Shapes must already be aligned to
    the reference (for example by generalized Procrustes analysis) and in the same units.
    """
    shapes = np.asarray(shapes, dtype=float)
    if shapes.ndim == 2:
        shapes = shapes[None]
    if shapes.shape[1:] != np.shape(reference):
        raise ValueError("shapes and reference must have the same (k, 3) shape")
    heights = (shapes - np.asarray(reference)) @ warps.normal  # (n, k)
    return heights @ warps.vectors


def band_energy(amplitudes: np.ndarray, warps: PartialWarps, first: int = 1, last: int | None = None) -> np.ndarray:
    """Bending energy restricted to partial warps ``first`` .. ``last`` (1-based, inclusive).

    With the full band this is the total bending energy of each deformation; a band starting at
    partial warp 1 is a lowpass filter.
    """
    last = warps.n_warps if last is None else last
    if not 1 <= first <= last <= warps.n_warps:
        raise ValueError(f"band must satisfy 1 <= first <= last <= {warps.n_warps}")
    a = np.atleast_2d(amplitudes)[:, first - 1 : last]
    return (a**2 * warps.eigenvalues[first - 1 : last]).sum(axis=1)


def band_correlations(amplitudes: np.ndarray, warps: PartialWarps, target: np.ndarray, max_warp: int = 15) -> np.ndarray:
    """Pearson correlation of band energy with ``target`` for every band within 1..max_warp.

    Returns a (max_warp, max_warp) array ``r`` with ``r[first-1, last-1]`` filled for
    ``first <= last`` and NaN elsewhere: the surface the original method searches for its band.
    """
    target = np.asarray(target, dtype=float)
    max_warp = min(max_warp, warps.n_warps)
    out = np.full((max_warp, max_warp), np.nan)
    for first in range(1, max_warp + 1):
        for last in range(first, max_warp + 1):
            energy = band_energy(amplitudes, warps, first, last)
            out[first - 1, last - 1] = np.corrcoef(energy, target)[0, 1]
    return out


def best_band(correlations: np.ndarray, sign: int = -1) -> tuple[int, int, float]:
    """The band with the strongest correlation of the requested sign, as ``(first, last, r)``.

    ``sign=-1`` (the default) looks for the most negative correlation, as energy falling with
    age; ``sign=1`` for the most positive.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be -1 or 1")
    r = np.asarray(correlations, dtype=float)
    i, j = np.unravel_index(np.nanargmax(sign * r), r.shape)
    return int(i) + 1, int(j) + 1, float(r[i, j])
