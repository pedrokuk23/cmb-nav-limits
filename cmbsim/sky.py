"""CMB anisotropies (T_F) as a FIXED pattern on the sky, not as white noise.

The temperature field is Gaussian and isotropic, and the covariance between two directions
separated by an angle θ is

    C(θ) = Σ_ℓ (2ℓ+1)/(4π) · C_ℓ · B_ℓ² · P_ℓ(cos θ),    ℓ ≥ 2

where B_ℓ is the sensor's Gaussian beam. Since the sensors stare at a few fixed
directions, it suffices to draw the field in those directions from this covariance (the result is
exact, with no need to generate a full HEALPix map).

Spectrum: D_ℓ = ℓ(ℓ+1)C_ℓ/2π of the Planck 2018 ΛCDM computed with CAMB (Lewis, Challinor & Lasenby 2000, ApJ 538, 473; version 2: Lewis 2026,
arXiv:2607.14854), stored in data/cl_tt_camb.txt. Without that file, an approximate
table is used (RMS at 0.5° only 2% below CAMB).
"""
from pathlib import Path

import numpy as np

# ℓ, D_ℓ [µK²]  (approximation of the Planck 2018 ΛCDM TT spectrum)
_DL = np.array([
    [2, 1000], [5, 850], [10, 800], [20, 820], [30, 900], [50, 1300], [80, 2000],
    [100, 2500], [150, 4300], [200, 5600], [220, 5750], [260, 5000], [300, 3500],
    [350, 2300], [410, 1800], [470, 2100], [540, 2600], [600, 2200], [680, 1900],
    [750, 2200], [810, 2500], [900, 1700], [1000, 1150], [1100, 1100], [1200, 1000],
    [1400, 600], [1600, 400], [1800, 250], [2000, 150], [2500, 50], [3000, 15],
])


_CAMB = Path(__file__).with_name("data") / "cl_tt_camb.txt"
if _CAMB.exists():
    # CAMB 2 spectrum; the table above is kept as a fallback (differs by < 3% in RMS)
    _DL = np.loadtxt(_CAMB)
    CL_SOURCE = "CAMB 2, Planck 2018"
else:                                                    # pragma: no cover
    CL_SOURCE = "approximate table"


def dl(ell):
    return np.interp(ell, _DL[:, 0], _DL[:, 1])


def cl_table(lmax):
    """C_ℓ [K²] for ℓ = 0..lmax (zeros for ℓ < 2), for generating sky realizations."""
    ell = np.arange(lmax + 1)
    out = np.zeros(lmax + 1)
    out[2:] = cl_K2(ell[2:])
    return out


def cl_K2(ell):
    ell = np.asarray(ell, float)
    return 2 * np.pi * dl(ell) / (ell * (ell + 1)) * 1e-12


def beam_bl(ell, fwhm_deg):
    sig = np.deg2rad(fwhm_deg) / np.sqrt(8 * np.log(2))
    return np.exp(-0.5 * ell * (ell + 1) * sig**2)


def _lmax(fwhm_deg):
    return int(min(3000, max(50, 6 * 180.0 / max(fwhm_deg, 1e-3))))


def correlation(cos_theta, fwhm_deg, lmin=2):
    """C(θ) [K²] for a vector of cos θ."""
    ell = np.arange(lmin, _lmax(fwhm_deg) + 1)
    w = (2 * ell + 1) / (4 * np.pi) * cl_K2(ell) * beam_bl(ell, fwhm_deg) ** 2
    ct = np.atleast_1d(np.asarray(cos_theta, float))
    # Legendre recurrence vectorized over cos θ: P_{l+1} = ((2l+1) x P_l - l P_{l-1}) / (l+1)
    p_prev, p_cur = np.ones_like(ct), ct.copy()          # P0, P1
    out = np.zeros_like(ct)
    for l in range(1, ell[-1]):
        p_prev, p_cur = p_cur, ((2 * l + 1) * ct * p_cur - l * p_prev) / (l + 1)
        if l + 1 >= lmin:
            out += w[l + 1 - lmin] * p_cur
    return out


def anisotropy_rms(fwhm_deg):
    """RMS of T_F seen by a sensor with beam width fwhm [K]."""
    return float(np.sqrt(correlation(1.0, fwhm_deg)[0]))


def covariance(n_dirs, fwhm_deg):
    n = np.asarray(n_dirs, float)
    cos = np.clip(n @ n.T, -1, 1)
    iu = np.triu_indices(len(n))
    vals = correlation(cos[iu], fwhm_deg)
    C = np.zeros_like(cos)
    C[iu] = vals
    return C + np.triu(C, 1).T


def sample_anisotropy(n_dirs, fwhm_deg, rng):
    """Draws T_F [K] in the given directions (one sky realization)."""
    C = covariance(n_dirs, fwhm_deg)
    return rng.multivariate_normal(np.zeros(len(C)), C, method="eigh")


def planck_residual_rms(fwhm_deg, map_depth_uK_arcmin=42.0, foreground_uK=1.0):
    """RMS of the error left after subtracting the Planck map [K].

    map_depth: map noise in µK·arcmin (Planck 143 GHz: goal of 0.7 µK·deg ≈ 42 µK·arcmin).
    foreground: foreground/systematics residual at degree scales [µK] (assumption).
    The white map noise seen by a Gaussian beam of solid angle Ω = 1.133·FWHM²
    is w/√(2Ω) (the variance of smoothed white noise is w²/(4πσ²) = w²/(2Ω); previously w/√Ω,
    conservative by √2).
    """
    fwhm_arcmin = fwhm_deg * 60.0
    white = map_depth_uK_arcmin / np.sqrt(2 * 1.133 * fwhm_arcmin**2)
    return np.hypot(white, foreground_uK) * 1e-6
