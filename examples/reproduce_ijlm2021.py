"""Re-run the 2021 analysis on the published landmark data (OSF, https://osf.io/867md/).

Downloads ``Landmarks.csv`` and ``Pubis_info.csv`` into ``data/osf/`` (checksummed; not part of
this repository), then, for each side of the male sample: GPA, the 0.05 squared-Procrustes-distance
outlier cut, partial warps of the mean, and the correlation of band energy with age under 50.

    python examples/reproduce_ijlm2021.py

Layout of the OSF files (worked out against the original thesis spreadsheets):

* ``Landmarks.csv`` holds 657 blocks of 102 points (2 fixed landmarks, 100 surface
  semilandmarks). Blocks 0-235 are the female surfaces of the wider thesis sample (with one
  duplicated block at index 100); blocks 236-656 are the 421 male surfaces, left sides first.
* ``Pubis_info.csv`` lists the 421 male surfaces in block order as ``<M><L|R><number>a<age>``,
  with one exception: the left surface of individual 177 is listed among the right sides as
  ``MR177a38``. It is moved back here.
"""

from __future__ import annotations

import csv
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

from pwbandpass import band_correlations, best_band, gpa, partial_warp_amplitudes, partial_warps

FILES = {
    "Landmarks.csv": ("https://osf.io/download/sn52q/",
                      "a0bd515182367a9da1d5223ffba4b4e2d9580351921922824ecf7dd0b65dc2d7"),
    "Pubis_info.csv": ("https://osf.io/download/tdbqr/",
                       "3af481fdbe3e5a277cafaf26fd0735b3a57402c77f369a2c1eebf8fabcae7faa"),
}
DATA = Path(__file__).resolve().parents[1] / "data" / "osf"
FIRST_MALE_BLOCK = 236
PUBLISHED = {"L": "-0.51 / -0.521", "R": "-0.544 / -0.648"}


def fetch() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (url, sha) in FILES.items():
        path = DATA / name
        if not path.exists():
            print(f"downloading {name} from OSF")
            urllib.request.urlretrieve(url, path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"{name} does not match the published file; delete it and retry")


def load() -> tuple[np.ndarray, list[str], np.ndarray]:
    blocks = np.loadtxt(DATA / "Landmarks.csv", delimiter=",").reshape(-1, 102, 3)
    ids = [row[0] for row in csv.reader(open(DATA / "Pubis_info.csv")) if row]
    ids.remove("MR177a38")
    ids.insert(ids.index("ML181a35"), "ML177a38")
    males = blocks[FIRST_MALE_BLOCK:]
    if len(males) != len(ids):
        raise RuntimeError("unexpected file layout")
    ages = np.array([float(i.split("a")[-1]) for i in ids])
    return males, ids, ages


def main() -> None:
    fetch()
    shapes, ids, ages = load()
    side = np.array([i[1] for i in ids])
    print(f"{'side':<6}{'n':>5}{'kept':>6}{'<50':>6}{'PW1-7 r':>10}{'best band':>12}{'r':>8}   published PW1-7 r")
    for s in "LR":
        x, age = shapes[side == s], ages[side == s]
        aligned, d2 = gpa(x)
        keep = d2 < 0.05
        aligned, age_k = aligned[keep], age[keep]
        reference = aligned.mean(axis=0)
        warps = partial_warps(reference)
        amplitudes = partial_warp_amplitudes(aligned, warps, reference)
        young = age_k < 50
        r = band_correlations(amplitudes[young], warps, age_k[young], max_warp=15)
        first, last, r_best = best_band(r)
        print(f"{s:<6}{len(x):>5}{keep.sum():>6}{young.sum():>6}{r[0, 6]:>10.3f}"
              f"{f'PW {first}-{last}':>12}{r_best:>8.3f}   {PUBLISHED[s]}")


if __name__ == "__main__":
    main()
