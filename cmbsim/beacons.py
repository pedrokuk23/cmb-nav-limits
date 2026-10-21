"""Sensors for DSN tracking gaps: beacons (asteroids), Sun–Earth–probe geometry and solar plasma.

Integrated into the simulator: "ast" events, DSN cutoff by SEP, and plasma noise on the Doppler.

Model references:
- Asteroid-based optical navigation in cruise: Riedel et al. 2000 (Deep Space 1 AutoNav); Ely et al. 2022
  (J. Astronaut. Sci. 69:385, Table 1: 10 µrad/pixel camera, 0.25 px centroid; Eqs. 41-44: beacon
  ephemeris error added to the noise); Broschart et al. 2019 (asteroid selection).
- Solar plasma on the Doppler (X band): Iess et al. / Zannoni, ISSFD 2012 (Cassini empirical model).
- Command moratorium at conjunction: < 2-3° (MAVEN 2015, Curiosity 2019).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import physics as ph

MU_SUN = ph.MU_SUN


@lru_cache(maxsize=4)
def belt(seed: int = 2026, n: int = 3200, h_min: float = 3.0, h_max: float = 11.0, slope: float = 0.45):
    """SYNTHETIC population of bright main-belt asteroids (mean Keplerian elements).

    Cumulative count N(<H) ∝ 10^(slope·H) between h_min and h_max, with n objects: the defaults give N(<9) ≈ 400
    and N(<11) ≈ 3200, the order of magnitude of the real counts [est.; ~200 objects with D > 100 km].
    The elements lie in the typical ranges of the belt. They are not real objects: they provide the density and
    geometry of beacons that a navigation camera would see (tens visible down to magnitude ~10-11)."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    # inversion of N(<H) ∝ 10^(slope·H) on [h_min, h_max]
    a0, a1 = 10 ** (slope * h_min), 10 ** (slope * h_max)
    H = np.log10(a0 + u * (a1 - a0)) / slope
    a = rng.uniform(2.15, 3.3, n) * ph.AU_KM
    e = rng.uniform(0.0, 0.25, n)
    inc = np.deg2rad(np.abs(rng.normal(0, 9.0, n)))
    Om, w, M0 = (np.deg2rad(rng.uniform(0, 360, n)) for _ in range(3))
    return dict(H=H, a=a, e=e, i=inc, Om=Om, w=w, M0=M0)


def belt_positions(jd, jd_ref, B):
    """Heliocentric positions (J2000 ecliptic, km) of all beacons at date jd (two-body problem)."""
    a, e = B["a"], B["e"]
    n_mean = np.sqrt(MU_SUN / a**3)
    M = B["M0"] + n_mean * (jd - jd_ref) * ph.DAY
    E = M.copy()
    for _ in range(20):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    x_ = a * (np.cos(E) - e)
    y_ = a * np.sqrt(1 - e**2) * np.sin(E)
    cO, sO, cw, sw, ci, si = np.cos(B["Om"]), np.sin(B["Om"]), np.cos(B["w"]), np.sin(B["w"]), np.cos(B["i"]), np.sin(B["i"])
    X = (cO * cw - sO * sw * ci) * x_ + (-cO * sw - sO * cw * ci) * y_
    Y = (sO * cw + cO * sw * ci) * x_ + (-sO * sw + cO * cw * ci) * y_
    Z = (sw * si) * x_ + (cw * si) * y_
    return np.column_stack([X, Y, Z])


def apparent_mag(H, r_ast, r_sc):
    """Apparent magnitude seen from the probe: H + 5·log10(r·Δ) + phase (0.035 mag/° linear, H-G approximation)."""
    d = r_ast - r_sc
    Delta = np.linalg.norm(d, axis=1) / ph.AU_KM
    r = np.linalg.norm(r_ast, axis=1) / ph.AU_KM
    cos_ph = np.sum(r_ast * d, axis=1) / (np.linalg.norm(r_ast, axis=1) * np.linalg.norm(d, axis=1))
    phase = np.degrees(np.arccos(np.clip(cos_ph, -1, 1)))
    return H + 5 * np.log10(r * Delta) + 0.035 * phase


def visible_beacons(r_sc, jd, jd_ref, B, mag_lim=10.0, sun_excl_deg=25.0, k=3, min_sep_deg=20.0):
    """Indices of the k brightest visible beacons (magnitude ≤ limit, away from the Sun), with a minimum
    angular separation between them (geometry). Also returns the nominal positions of all beacons."""
    P = belt_positions(jd, jd_ref, B)
    m = apparent_mag(B["H"], P, r_sc)
    u = (P - r_sc) / np.linalg.norm(P - r_sc, axis=1, keepdims=True)
    sun_dir = -r_sc / np.linalg.norm(r_sc)
    elong = np.degrees(np.arccos(np.clip(u @ sun_dir, -1, 1)))
    cand = np.where((m <= mag_lim) & (elong >= sun_excl_deg))[0]
    cand = cand[np.argsort(m[cand])]
    sel = []
    for j in cand:
        if all(np.degrees(np.arccos(np.clip(u[j] @ u[s], -1, 1))) >= min_sep_deg for s in sel):
            sel.append(int(j))
        if len(sel) >= k:
            break
    return sel, P, m


def sep_angle_deg(r_sc, r_earth):
    """Sun–Earth–probe angle (SEP) seen from Earth [deg]."""
    to_sun = -r_earth
    to_sc = r_sc - r_earth
    c = to_sun @ to_sc / (np.linalg.norm(to_sun) * np.linalg.norm(to_sc))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def plasma_allan_x(sep_deg):
    """Allan deviation (60 s) of the X-band Doppler due to solar plasma, Cassini empirical model
    (Iess et al., ISSFD 2012, eq. 2), piecewise in the Sun-Earth-probe angle."""
    s = np.sin(np.deg2rad(np.clip(sep_deg, 0.5, 180.0)))
    if sep_deg <= 90:
        return 1.76e-14 * s**-1.98 + 6.25e-14 * s**0.06
    if sep_deg <= 170:
        return (1.76e-14 + 6.25e-14) * s**1.05
    return 1.27e-14


def plasma_rate_sigma_mmps(sep_deg):
    """Two-way range-rate error due to plasma, over 60 s: σ_v ≈ c·σ_y/2 [mm/s]."""
    return ph.C_KMS * 1e6 * plasma_allan_x(sep_deg) / 2
