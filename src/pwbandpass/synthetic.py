"""Synthetic articular surfaces with a known, scale-dependent age signal.

Used for the example and the tests. Each surface carries 2 fixed landmarks and a 10 x 10 patch
of surface semilandmarks, like the protocol of the original study. Three components:

* **broad transverse ridges** whose height falls steadily with age up to 50, then stops
  changing (the young, billowed surface flattening out);
* **fine-scale texture** with no relation to age at any time;
* **degenerative change after 50**: rim lipping and pitting whose amount grows with age but
  varies a lot between individuals.

The age signal therefore lives in the largest geometric scales and only below 50, which is the
situation the bandpass method was designed for. The numbers are illustrative, not anatomical.
"""

from __future__ import annotations

import numpy as np


def template(n_side: int = 10, length: float = 30.0, width: float = 12.0) -> np.ndarray:
    """2 fixed landmarks (superior, inferior) followed by an n_side x n_side patch, flat, in mm."""
    u = np.linspace(-0.85, 0.85, n_side) * length / 2
    v = np.linspace(-0.85, 0.85, n_side) * width / 2
    uu, vv = np.meshgrid(u, v, indexing="ij")
    taper = np.sqrt(1 - (uu / (length / 2)) ** 2)  # the patch narrows towards both ends
    patch = np.column_stack([uu.ravel(), (vv * taper).ravel(), np.zeros(uu.size)])
    fixed = np.array([[length / 2, 0.0, 0.0], [-length / 2, 0.0, 0.0]])
    return np.vstack([fixed, patch])


def simulate(ages: np.ndarray, seed: int = 0) -> np.ndarray:
    """Simulate one (k, 3) surface per age, randomly rotated and translated, in mm."""
    rng = np.random.default_rng(seed)
    base = template()
    u, v = base[:, 0], base[:, 1]
    length = np.ptp(u)
    rim = np.clip(np.abs(v) / np.abs(v).max(), 0, 1) ** 3
    shapes = []
    for age in np.asarray(ages, dtype=float):
        youth = np.clip((50.0 - age) / 35.0, 0.0, 1.0)  # 1 at 15 years, 0 from 50 on
        phase = rng.uniform(0, 2 * np.pi)
        ridges = (0.15 + 0.85 * youth) * rng.normal(1.0, 0.25) * 0.9 * np.sin(2 * np.pi * u / (length / 2.5) + phase)
        texture = rng.normal(0, 0.18, len(u))
        decay = max(age - 50.0, 0.0) / 30.0 * rng.gamma(1.5, 1.0)
        degenerative = decay * (0.8 * rim + rng.normal(0, 0.35, len(u)))
        h = ridges + texture + degenerative
        h[:2] = 0.0  # fixed landmarks sit on the rim
        shape = base + np.column_stack([np.zeros_like(h), np.zeros_like(h), h])
        shape *= rng.normal(1.0, 0.06)  # size variation
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        shapes.append(shape @ q.T + rng.normal(0, 20, 3))
    return np.array(shapes)
