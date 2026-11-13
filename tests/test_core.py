import numpy as np
import pytest

from pwbandpass import (
    band_correlations,
    band_energy,
    bending_energy_matrix,
    best_band,
    gpa,
    partial_warp_amplitudes,
    partial_warps,
)
from pwbandpass.synthetic import simulate, template


@pytest.fixture(scope="module")
def grid():
    return template()


def test_bending_energy_matrix_is_symmetric_and_psd(grid):
    B = bending_energy_matrix(grid[:, :2])
    assert np.allclose(B, B.T)
    assert np.linalg.eigvalsh(B).min() > -1e-10


def test_affine_functions_cost_no_energy(grid):
    pts = grid[:, :2]
    B = bending_energy_matrix(pts)
    for f in (np.ones(len(pts)), pts[:, 0], pts[:, 1], 2 - 3 * pts[:, 0] + 0.5 * pts[:, 1]):
        assert f @ B @ f == pytest.approx(0.0, abs=1e-9)


def test_bending_energy_matrix_rejects_bad_input():
    with pytest.raises(ValueError):
        bending_energy_matrix(np.zeros((5, 3)))
    with pytest.raises(ValueError):
        bending_energy_matrix(np.random.default_rng(0).normal(size=(3, 2)))


def test_partial_warps_are_orthonormal_and_ordered(grid):
    pw = partial_warps(grid)
    assert pw.n_warps == len(grid) - 3
    assert np.all(np.diff(pw.eigenvalues) >= 0)
    assert np.all(pw.eigenvalues > 0)
    assert np.allclose(pw.vectors.T @ pw.vectors, np.eye(pw.n_warps), atol=1e-8)


def test_partial_warps_do_not_depend_on_position_or_orientation(grid):
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    moved = grid @ q.T + rng.normal(size=3)
    assert np.allclose(partial_warps(grid).eigenvalues, partial_warps(moved).eigenvalues)


def test_the_full_band_equals_the_total_bending_energy(grid):
    pw = partial_warps(grid)
    rng = np.random.default_rng(1)
    heights = rng.normal(size=(4, len(grid)))
    shapes = grid + heights[:, :, None] * pw.normal
    a = partial_warp_amplitudes(shapes, pw, grid)
    B = bending_energy_matrix((grid - pw.origin) @ pw.plane_axes.T)
    direct = np.einsum("ni,ij,nj->n", heights, B, heights)
    assert np.allclose(band_energy(a, pw), direct)
    split = band_energy(a, pw, 1, 7) + band_energy(a, pw, 8)
    assert np.allclose(split, direct)


def test_a_single_partial_warp_lands_in_its_own_band(grid):
    pw = partial_warps(grid)
    shape = grid + 2.0 * np.outer(pw.vectors[:, 4], pw.normal)  # partial warp 5
    a = partial_warp_amplitudes(shape, pw, grid)
    assert band_energy(a, pw, 5, 5)[0] == pytest.approx(4.0 * pw.eigenvalues[4])
    assert band_energy(a, pw, 1, 4)[0] == pytest.approx(0.0, abs=1e-12)
    assert band_energy(a, pw, 6)[0] == pytest.approx(0.0, abs=1e-12)


def test_band_limits_are_validated(grid):
    pw = partial_warps(grid)
    a = np.zeros((1, pw.n_warps))
    for first, last in [(0, 3), (5, 4), (1, pw.n_warps + 1)]:
        with pytest.raises(ValueError):
            band_energy(a, pw, first, last)


def test_band_correlations_fill_the_upper_triangle(grid):
    pw = partial_warps(grid)
    rng = np.random.default_rng(2)
    a = rng.normal(size=(30, pw.n_warps))
    r = band_correlations(a, pw, rng.normal(size=30), max_warp=6)
    assert r.shape == (6, 6)
    assert np.all(np.isnan(r[np.tril_indices(6, -1)]))
    assert np.all(np.isfinite(r[np.triu_indices(6)]))


def test_best_band():
    r = np.array([[-0.2, -0.6], [np.nan, 0.4]])
    assert best_band(r) == (1, 2, -0.6)
    assert best_band(r, sign=1) == (2, 2, 0.4)
    with pytest.raises(ValueError):
        best_band(r, sign=0)


def test_gpa_removes_position_rotation_and_scale(grid):
    rng = np.random.default_rng(4)
    shapes = []
    for _ in range(5):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q[:, 0] *= np.sign(np.linalg.det(q))
        shapes.append(rng.uniform(0.5, 2) * grid @ q.T + rng.normal(size=3))
    aligned, d2 = gpa(np.array(shapes))
    assert np.allclose(d2, 0.0, atol=1e-12)
    assert np.allclose(aligned, aligned[0], atol=1e-8)


def test_the_lowpass_band_recovers_the_simulated_age_signal():
    rng = np.random.default_rng(1)
    ages = rng.uniform(15, 82, 220)
    shapes, _ = gpa(simulate(ages, seed=2))
    ref = shapes.mean(axis=0)
    pw = partial_warps(ref)
    a = partial_warp_amplitudes(shapes, pw, ref)
    young = ages < 50
    lowpass = np.corrcoef(band_energy(a[young], pw, 1, 7), ages[young])[0, 1]
    full = np.corrcoef(band_energy(a[young], pw), ages[young])[0, 1]
    assert lowpass < -0.5
    assert lowpass < full - 0.2
