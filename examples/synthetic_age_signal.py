"""Find the age band on synthetic surfaces and draw docs/img/bandpass.png.

The simulated surfaces carry an age signal only in their broad shape and only before 50, under
fine-scale texture and post-50 degeneration (see ``pwbandpass.synthetic``). The total bending
energy hides it; the lowpass band brings it back.

    pip install -e ".[examples]"
    python examples/synthetic_age_signal.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from pwbandpass import (
    band_correlations,
    band_energy,
    best_band,
    gpa,
    partial_warp_amplitudes,
    partial_warps,
)
from pwbandpass.synthetic import simulate

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, RED, MID = "#2a78d6", "#e34948", "#f0efec"
OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "bandpass.png"


def main() -> None:
    rng = np.random.default_rng(1)
    ages = rng.uniform(15, 82, 220)
    shapes, d2 = gpa(simulate(ages, seed=2))
    keep = d2 < 0.05  # the original study's outlier cut
    shapes, ages = shapes[keep], ages[keep]
    reference = shapes.mean(axis=0)
    warps = partial_warps(reference)
    amplitudes = partial_warp_amplitudes(shapes, warps, reference)

    young = ages < 50
    r = band_correlations(amplitudes[young], warps, ages[young], max_warp=15)
    first, last, r_best = best_band(r)
    lowpass = band_energy(amplitudes, warps, 1, 7)
    total = band_energy(amplitudes, warps)
    r_low = np.corrcoef(lowpass[young], ages[young])[0, 1]
    r_total = np.corrcoef(total[young], ages[young])[0, 1]
    print(f"n = {len(ages)} ({young.sum()} under 50), {warps.n_warps} partial warps")
    print(f"under 50: total energy r = {r_total:+.2f}; PW 1-7 r = {r_low:+.2f}; "
          f"best band PW {first}-{last} r = {r_best:+.2f}")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.45})

    cmap = LinearSegmentedColormap.from_list("div", [BLUE, MID, RED])
    n = r.shape[0]
    im = ax1.imshow(r, cmap=cmap, vmin=-0.8, vmax=0.8, origin="lower",
                    extent=(0.5, n + 0.5, 0.5, n + 0.5))
    ax1.plot(last, first, marker="o", ms=9, mfc="none", mec=INK, mew=1.5)
    ax1.annotate(f"PW {first}–{last}\nr = {r_best:.2f}", (last, first), xytext=(8, 22),
                 textcoords="offset points", color=INK, fontsize=9)
    ax1.set(xlabel="last partial warp in band", ylabel="first partial warp in band",
            title="Correlation of band energy with age (under 50)", facecolor=SURFACE)
    ax1.set_xticks(range(1, n + 1, 2))
    ax1.set_yticks(range(1, n + 1, 2))
    ax1.title.set_color(INK)
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED)
    cb.set_label("Pearson r", color=INK2)

    ax2.set_facecolor(SURFACE)
    ax2.grid(axis="y", color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax2.axvspan(50, ages.max() + 2, color=MID, lw=0)
    ax2.axvline(50, color=AXIS, lw=1, ls="--")
    y = np.log(lowpass)
    ax2.scatter(ages, y, s=22, color=BLUE, alpha=0.8, edgecolor=SURFACE, lw=0.6)
    edges = np.arange(15, 85, 5)
    mids = (edges[:-1] + edges[1:]) / 2
    med = [np.median(y[(ages >= a) & (ages < b)]) for a, b in zip(edges[:-1], edges[1:])]
    ax2.plot(mids, med, color=INK, lw=2, label="5-year median")
    ax2.legend(loc="lower left", frameon=False, labelcolor=INK2, fontsize=9)
    ax2.text(50.8, y.max(), "degenerative change", color=INK2, fontsize=9, va="top")
    ax2.set(xlabel="age (years)", ylabel="ln bending energy, PW 1–7",
            title=f"Lowpass energy falls with age to 50 (r = {r_low:.2f}; total: {r_total:.2f})",
            xlim=(13, ages.max() + 2))
    ax2.title.set_color(INK)
    for side in ("top", "right", "left"):
        ax2.spines[side].set_visible(False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(OUT.parents[2])}")


if __name__ == "__main__":
    main()
