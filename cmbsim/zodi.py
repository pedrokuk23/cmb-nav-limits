"""Zodiacal emission (interplanetary dust) seen by the probe, computed with ZodiPy.

ZodiPy (San et al. 2022, A&A 666, A107; listed on NASA's LAMBDA under "Zodiacal Light Software")
implements the COBE/DIRBE interplanetary dust model (Kelsall et al. 1998) with the Planck
extension (Planck 2013 XIV / 2018) for 100-857 GHz, and computes the emission for an observer at
ANY position in the Solar System. Below 100 GHz we use ZodiPy's own extrapolation.

This matters for navigation because:
- the anti-Sun sensor looks at the ecliptic plane, where the zodiacal emission is 1 to 20 µK
  between 70 and 150 GHz;
- it CHANGES as the probe moves away from the Sun (Earth -> Mars): it is not a fixed bias.

The onboard model subtracts the ZodiPy prediction with a relative amplitude error (default 10%,
the order of the uncertainty of the Planck emissivities); what remains is ε·Z(n̂, r(t)).

Implementation: HEALPix maps (nside 32, ~1.8°) every 5 days along the nominal trajectory,
interpolated in time; cached on disk (~/.cache/cmbsim).
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import physics as ph

try:
    import healpy as hp
    import zodipy
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    HAVE = True
except Exception:                                 # pragma: no cover
    HAVE = False

NSIDE = 32
CACHE = Path.home() / ".cache" / "cmbsim"
H, KB, TCMB = 6.62607015e-34, 1.380649e-23, 2.7255


def available():
    return HAVE


def mjysr_to_uKcmb(nu_ghz):
    """1 MJy/sr in thermodynamic µK at frequency ν."""
    nu = nu_ghz * 1e9
    x = H * nu / (KB * TCMB)
    dBdT = 2 * KB * (nu / 2.99792458e8) ** 2 * x**2 * np.exp(x) / np.expm1(x) ** 2
    return 1e-20 / dBdT * 1e6


def nominal_positions(jd0, tof_days, step_days=5.0):
    """Days and heliocentric positions [km] of the nominal trajectory (Lambert + planets), every step_days."""
    from .simulator import Config, reference_transfer          # late import (avoids a cycle)
    dyn = ph.Dynamics(jd0, planets=("venus", "jupiter", "saturn"))   # starts from Earth's center
    days = np.arange(0.0, tof_days + step_days, step_days)
    y, m_, d_ = _jd_to_ymd(jd0)
    src = ph.EPHEM["source"]
    try:                                      # does not swap the global ephemeris of the running simulation
        _, r0, v0, _ = reference_transfer(Config(departure=(y, m_, d_), tof_days=tof_days, departure_model="center",
                                                 retarget=False, ephem=src))
    finally:
        ph.set_ephemeris(src)
    dyn.tabulate(0.0, (tof_days + step_days) * ph.DAY)
    x = np.concatenate([r0, v0])
    pos = [r0]
    for k in range(1, len(days)):
        x = dyn.propagate(x, days[k - 1] * ph.DAY, step_days * ph.DAY, max_step=86400.0)
        pos.append(x[:3])
    return days, np.array(pos)


@lru_cache(maxsize=8)
def table_positions(nu_ghz, jd0, days_t, pos_t):
    """(days, maps) of the zodiacal emission seen from given positions (tuples; km, rounded by the caller)."""
    days, pos = np.array(days_t), np.array(pos_t).reshape(-1, 3)
    key = hashlib.md5((f"v2|{nu_ghz:.3f}|{jd0:.5f}|{NSIDE}|" + ",".join(f"{v:.0f}" for v in pos.ravel())
                       + "|" + ",".join(f"{v:.2f}" for v in days)).encode()).hexdigest()[:16]
    f = CACHE / f"zodi_p_{key}.npz"
    if f.exists():
        d = np.load(f)
        return d["days"], fill_nan(d["maps"])
    maps = _evaluate(nu_ghz, jd0, days, pos)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, days=days, maps=maps)
    return days, fill_nan(maps)


def fill_nan(maps):
    """ZodiPy returns NaN in some pixels (division by zero along the line of sight for certain observer
    positions). Each NaN is replaced by the mean of its finite neighbours; without this the zodiacal bias
    becomes NaN and the filter diverges (seen, e.g., with electric propulsion and a precessing spin axis)."""
    maps = np.array(maps, float, copy=True)
    for m in np.atleast_2d(maps):
        for _ in range(10):
            bad = np.flatnonzero(~np.isfinite(m))
            if not bad.size:
                break
            nb = hp.get_all_neighbours(NSIDE, bad)
            vals = np.where(nb >= 0, m[np.clip(nb, 0, None)], np.nan)
            m[bad] = np.nanmean(np.where(np.isfinite(vals), vals, np.nan), axis=0)
        m[~np.isfinite(m)] = np.nanmedian(m)
    return maps


def _evaluate(nu_ghz, jd0, days, pos):
    band = nu_ghz * (1 + 0.2 * np.linspace(-0.5, 0.5, 5))
    model = zodipy.Model(x=band * u.GHz, weights=np.ones(len(band)), name="planck18",
                         extrapolate=bool(band.min() < 100.0 or band.max() > 857.0))
    th, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    maps = []
    for t, r in zip(days, pos):
        # mean ecliptic of J2000 (the same as the rest of the code; the true ecliptic differs by 14″)
        c = SkyCoord(np.degrees(phi) * u.deg, (90 - np.degrees(th)) * u.deg, frame="barycentricmeanecliptic",
                     obstime=Time(jd0 + t, format="jd"))
        I = model.evaluate(c, obspos=(np.asarray(r) / ph.AU_KM) * u.AU).to_value(u.MJy / u.sr)
        maps.append(I * mjysr_to_uKcmb(nu_ghz))
    return np.array(maps)


@lru_cache(maxsize=8)
def table(nu_ghz, jd0, tof_days, step_days=5.0):
    """(days, maps[N, npix] in µK_CMB) of the zodiacal emission along the nominal trajectory."""
    key = hashlib.md5(f"v2|{ph.EPHEM['source']}|{nu_ghz:.3f}|{jd0:.5f}|{tof_days:.2f}|{step_days}|{NSIDE}".encode()).hexdigest()[:16]
    f = CACHE / f"zodi_{key}.npz"
    if f.exists():
        d = np.load(f)
        return d["days"], fill_nan(d["maps"])
    days, pos = nominal_positions(jd0, tof_days, step_days)
    maps = _evaluate(nu_ghz, jd0, days, pos)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, days=days, maps=maps)
    return days, fill_nan(maps)


def _jd_to_ymd(jd):
    t = Time(jd, format="jd").datetime
    return t.year, t.month, t.day


class Zodi:
    """Zodiacal emission (µK -> K) in an ecliptic direction on day t, interpolated in time."""

    def __init__(self, nu_ghz, jd0, tof_days, fwhm_deg=0.5, bands=None, weights=None, positions=None):
        """bands/weights: combination of N bands (component separation); the zodiacal term enters with the
        same weights. positions: (days, positions [km]) of the actual trajectory; None = nominal trajectory."""
        if positions is None:
            tab = lambda b: table(float(b), float(jd0), float(tof_days))
        else:
            dd, pp = positions
            key = (tuple(float(x) for x in dd), tuple(float(x) for x in (np.round(np.asarray(pp) / 1e6) * 1e6).ravel()))
            tab = lambda b: table_positions(float(b), float(jd0), *key)
        self.days, self.maps = tab(nu_ghz)
        self.nu = nu_ghz
        if bands is not None and weights is not None and len(bands) > 1:
            self.maps = sum(float(w) * tab(b)[1] for b, w in zip(bands, weights))

    def value(self, dirs_ecl, day):
        d = np.atleast_2d(dirs_ecl)
        th = np.arccos(np.clip(d[:, 2], -1, 1))
        phi = np.arctan2(d[:, 1], d[:, 0])
        k = int(np.clip(np.searchsorted(self.days, day) - 1, 0, len(self.days) - 2))
        w = float(np.clip((day - self.days[k]) / (self.days[k + 1] - self.days[k]), 0.0, 1.0))   # no extrapolation
        a = hp.get_interp_val(self.maps[k], th, phi)
        b = hp.get_interp_val(self.maps[k + 1], th, phi)
        return np.atleast_1d((1 - w) * a + w * b) * 1e-6
