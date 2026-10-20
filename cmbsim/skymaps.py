"""Real sky: measured CMB + Galactic foregrounds that depend on frequency and direction.

Data (data/sky_n64.npz):
- CMB: the cleaned, Wiener-filtered WMAP 3-year map of Tegmark, de Oliveira-Costa & Hamilton
  (2003, PRD 68, 123523), HEALPix nside 128, Galactic coordinates, µK (thermodynamic).
- Foregrounds (default): PySM3, large-scale part of d11 (GNILC dust) and s6 (synchrotron), plus
  f1 and a1; alternative: PySM v1 templates (Thorne et al. 2017, MNRAS 469, 2821), nside 64, µK_RJ, with the
  nominal PySM model (s1, f1, a1, d1):
    synchrotron  T_RJ(ν) = A_s(408 MHz)·(ν/0.408)^β_s(n̂)                 β_s varies over the sky
    free-free    T_RJ(ν) = A_ff(30 GHz)·(ν/30)^-2.14
    AME          T_RJ(ν) = A·(ν₀/ν)²·j(ν·30/ν_p)/j(ν₀·30/ν_p)             spdust2, 2 components
    dust         T_RJ(ν) = A_d(545 GHz)·(ν/545)^(β_d-2)·B_ν(T_d)/B_545(T_d)  β_d and T_d vary
  converted to thermodynamic (CMB) µK and integrated over a top-hat band.

What the sensor sees as a fixed bias in a direction n̂:
- with nothing subtracted: T_CMB(n̂) + Σ foregrounds(ν, n̂) (smoothed by the beam);
- subtracting the sky model (CMB map + foreground model): what remains is
    (CMB map error) + (true foreground − foreground model).
  The onboard model uses CONSTANT spectral indices (Planck 2018 IV: β_s = −3.1, β_d = 1.53,
  T_d = 19.6 K) and amplitudes with a relative error (default 3%). Variations of β over the sky are the effect
  that Tegmark (1998, ApJ 502, 1) showed to be important even for Δβ ≈ 0.1.
- option with N frequencies per sensor: linear combination with Σw = 1 that nulls the chosen nominal
  spectra (dust; synchrotron and dust; all) with minimum noise (Dodelson 1997), or minimum-variance
  ILC; it costs noise (each band gets 1/N of the detectors: factor √N·‖w‖) and leaves the spectral
  error (β and T_d varying over the sky).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from . import physics as ph

try:
    import healpy as hp
    HAVE_HEALPY = True
except ImportError:                                      # pragma: no cover
    HAVE_HEALPY = False

DATA = Path(__file__).with_name("data") / "sky_n64.npz"
H, KB, TCMB = 6.62607015e-34, 1.380649e-23, 2.7255
ECL_TO_GAL = ph.GAL_TO_EQ.T @ ph.EQ_TO_ECL.T            # n_gal = ECL_TO_GAL @ n_ecl
# Planck 2018 IV (β, T_d); AME peak: the onboard model uses 19 GHz, close to the mean of the PySM template (18.95 GHz;
# a 30 GHz peak would leave a 13 µK AME residual at 30 GHz)
NOMINAL = dict(beta_s=-3.1, beta_d=1.53, T_d=19.6, beta_ff=-2.14, nup_ame=19.0)


def spectrum_table(freqs, mask_deg=30.0, model="pysm3", co_on=True):
    """RMS (|b| > mask) of each foreground and of the residual after the nominal model, in µK_CMB, per frequency."""
    ns = 64
    th, _ = hp.pix2ang(ns, np.arange(12 * ns * ns))
    hi = np.abs(90 - np.degrees(th)) > mask_deg
    rows = []
    for nu in freqs:
        tot, comps = foregrounds_cmb(nu, "true", model=model, co_on=co_on)
        mod, _ = foregrounds_cmb(nu, "nominal", model=model, co_on=co_on)
        row = {"nu": nu, **{k: float(np.std(v[hi])) for k, v in comps.items()},
               "total": float(np.std(tot[hi])), "model residual": float(np.std((tot - mod)[hi]))}
        rows.append(row)
    return rows


def available():
    return HAVE_HEALPY and DATA.exists()


@lru_cache(maxsize=1)
def _data():
    d = np.load(DATA)
    return {k: d[k].astype(np.float64) for k in d.files}


def rj_to_cmb(nu_ghz):
    """Conversion factor K_RJ -> K_CMB (thermodynamic) at frequency ν."""
    x = H * nu_ghz * 1e9 / (KB * TCMB)
    return (np.expm1(x)) ** 2 / (x**2 * np.exp(x))


def _planck_ratio(nu, nu0, T):
    """B_ν(T)/B_ν0(T)."""
    a = H * 1e9 / KB
    return (nu / nu0) ** 3 * np.expm1(a * nu0 / T) / np.expm1(a * nu / T)


def _ame_scaling(nu, nu0, nup, d):
    J = lambda x: np.interp(x, d["ame_emis_nu"], d["ame_emis_j"], left=0.0, right=0.0)
    return (nu0 / nu) ** 2 * J(nu * 30.0 / nup) / np.maximum(J(nu0 * 30.0 / nup), 1e-300)


FG_MODELS = {"pysm3": "PySM3 (d11 + s6 large scale, f1, a1)", "pysm1": "PySM v1 (d1, s1, f1, a1)"}


def has_pysm3():
    return "p3_dust_A353" in _data()


def components_rj(nu, params="true", amp_err=None, model="pysm3"):
    """Foregrounds in µK_RJ at frequency ν (GHz), nside 64. params: 'true' (β, T maps) or 'nominal'.

    model 'pysm3': GNILC dust (d11, 353 GHz, GNILC β_d and T_d) and s6 synchrotron (23 GHz, new β_s),
    large-scale part only (ℓ ≤ 160); free-free and AME as in PySM3 (f1, a1 = v1 templates).
    model 'pysm1': nominal PySM v1 model (s1, d1, f1, a1)."""
    d = _data()
    n = NOMINAL
    true = params == "true"
    e = amp_err or {}
    p3 = model == "pysm3" and has_pysm3()
    nup = d["ame1_nup"] if true else n["nup_ame"]
    if p3:
        bs = d["p3_synch_beta"] if true else n["beta_s"]
        bd = d["p3_dust_beta"] if true else n["beta_d"]
        Td = d["p3_dust_T"] if true else n["T_d"]
        synch = d["p3_synch_A23"] * (nu / 23.0) ** bs
        dust = d["p3_dust_A353"] * (nu / 353.0) ** (bd - 2) * _planck_ratio(nu, 353.0, Td)
    else:
        bs = d["synch_beta"] if true else n["beta_s"]
        bd = d["dust_beta"] if true else n["beta_d"]
        Td = d["dust_T"] if true else n["T_d"]
        synch = d["synch_A"] * (nu / 0.408) ** bs
        dust = d["dust_A"] * (nu / 545.0) ** (bd - 2) * _planck_ratio(nu, 545.0, Td)
    out = {
        "synchrotron": synch * (1 + e.get("s", 0.0)),
        "free-free": d["ff_A"] * (nu / 30.0) ** n["beta_ff"] * (1 + e.get("ff", 0.0)),
        "AME": (d["ame1_A"] * _ame_scaling(nu, 22.8, nup, d) +
                d["ame2_A"] * _ame_scaling(nu, 41.0, 33.35, d)) * (1 + e.get("ame", 0.0)),
        "dust": dust * (1 + e.get("d", 0.0)),
    }
    return out


def band(nu_c, frac=0.2, n=7):
    return nu_c * (1 + frac * np.linspace(-0.5, 0.5, n))


# CO lines (Planck 2015 X, HFI Type 1 maps, templates of the PySM3 co1 model): K_CMB as seen in the
# Planck channels. In a top-hat band of width Δν containing the line, the mean signal is
# T_Planck × Δν_Planck/Δν (the line flux is fixed). HFI effective widths ≈ 33% (100 GHz) and 29% (217, 353).
CO_LINES = {"co10": (115.271, 33.0), "co21": (230.538, 63.0), "co32": (345.796, 102.0)}
CO_DATA = Path(__file__).with_name("data") / "co_planck_n64.npz"


def co_available():
    return CO_DATA.exists()


@lru_cache(maxsize=1)
def _co():
    d = np.load(CO_DATA)
    return {k: d[k].astype(np.float64) for k in d.files}


def co_band(nu_c, frac=0.2, co_on=True):
    """CO map [µK_CMB] in the top-hat band centered on nu_c (zero if no line falls in the band)."""
    out = 0.0
    if not (co_on and co_available()):
        return out
    lo, hi = nu_c * (1 - frac / 2), nu_c * (1 + frac / 2)
    for k, (nu_l, dnu_p) in CO_LINES.items():
        if lo <= nu_l <= hi:
            out = out + _co()[k] * dnu_p / (frac * nu_c)
    return out


def foregrounds_cmb(nu_c, params="true", amp_err=None, frac=0.2, model="pysm3", co_on=True):
    """Sum of the foregrounds in µK_CMB, averaged over the top-hat band (nside 64)."""
    tot = 0.0
    comps = {}
    nus = band(nu_c, frac)
    for nu in nus:
        c = components_rj(nu, params, amp_err, model)
        f = rj_to_cmb(nu)
        for k, v in c.items():
            comps[k] = comps.get(k, 0.0) + v * f / len(nus)
    co_ = co_band(nu_c, frac, co_on)
    if np.ndim(co_):
        comps["CO"] = co_ * (1 + (amp_err or {}).get("co", 0.0))
    for v in comps.values():
        tot = tot + v
    return tot, comps


def dust_ratio(nu1, nu2, frac=0.2):
    """Nominal ratio dust(ν2)/dust(ν1) in CMB units (for the 2-frequency combination)."""
    d1 = np.mean([(nu / 545.0) ** (NOMINAL["beta_d"] - 2) * _planck_ratio(nu, 545.0, NOMINAL["T_d"]) * rj_to_cmb(nu)
                  for nu in band(nu1, frac)])
    d2 = np.mean([(nu / 545.0) ** (NOMINAL["beta_d"] - 2) * _planck_ratio(nu, 545.0, NOMINAL["T_d"]) * rj_to_cmb(nu)
                  for nu in band(nu2, frac)])
    return d2 / d1


def two_freq_weights(nu1, nu2):
    """Weights (w1, w2) with w1 + w2 = 1 (preserves the CMB) and w1·d1 + w2·d2 = 0 (cancels the nominal dust)."""
    r = dust_ratio(nu1, nu2)
    w2 = -1.0 / (r - 1.0)
    return 1.0 - w2, w2


# ---------------------------------------------------------------------------------------------
# Component separation with N frequencies (Dodelson 1997, ApJ 482, 577; Hu & Dodelson 2002 §5.1)
# ---------------------------------------------------------------------------------------------
SEP_METHODS = {
    "dust": "nulls dust",
    "sync+dust": "nulls synchrotron and dust",
    "all": "nulls synchrotron, free-free, dust and the 2 AME components",
    "ilc": "minimum variance (model foregrounds + noise), nulls nothing",
}
NULL_ORDER = {"dust": ("dust",), "sync+dust": ("dust", "synchrotron"),
              "all": ("dust", "synchrotron", "free-free", "AME", "AME2")}


def sed_cmb(comp, nu_c, frac=0.2):
    """NOMINAL spectrum of a foreground (arbitrary unit amplitude), in CMB units, averaged over the band.
    This is what the onboard computer knows: constant indices from Planck 2018 IV."""
    n = NOMINAL
    out = 0.0
    nus = band(nu_c, frac)
    for nu in nus:
        if comp == "synchrotron":
            s = (nu / 23.0) ** n["beta_s"]
        elif comp == "free-free":
            s = (nu / 30.0) ** n["beta_ff"]
        elif comp == "dust":
            s = (nu / 353.0) ** (n["beta_d"] - 2) * _planck_ratio(nu, 353.0, n["T_d"])
        elif comp == "AME":
            d = _data()
            s = _ame_scaling(nu, 22.8, n["nup_ame"], d)
        elif comp == "AME2":                       # 2nd AME component (peak fixed at 33.35 GHz)
            d = _data()
            s = _ame_scaling(nu, 41.0, 33.35, d)
        else:
            raise ValueError(comp)
        out += s * rj_to_cmb(nu) / len(nus)
    return float(out)


def nfreq_weights(freqs, method="dust", noise_rel=None, fg_cov=None):
    """Weights w (Σw = 1, preserves the CMB) for combining N bands.

    Constrained methods (Dodelson 1997): w = N⁻¹A(AᵀN⁻¹A)⁻¹e₁, A = [1, f_c1, f_c2, ...] with the nominal
    spectra of the components to be nulled; the remaining degrees of freedom minimize the noise. If there are
    more components than bands − 1, the first ones in the order are nulled (dust, synchrotron, free-free, AME, AME2).
    'ilc': minimizes wᵀ(C_fg + N)w subject only to Σw = 1, where C_fg is the band-band covariance of the
    foreground-model maps (requires fg_cov).
    noise_rel: relative noise variance per band (default: equal in all).
    Returns (w, nulled_components)."""
    f = [float(x) for x in freqs]
    n = len(f)
    if n == 1:
        return np.array([1.0]), ()
    Nn = np.diag(np.ones(n) if noise_rel is None else np.asarray(noise_rel, float))
    if method == "ilc":
        if fg_cov is None:
            raise ValueError("ilc requires fg_cov")
        C = np.asarray(fg_cov, float) + Nn
        Ci1 = np.linalg.solve(C, np.ones(n))
        return Ci1 / Ci1.sum(), ()
    comps = NULL_ORDER[method][: n - 1]
    A = np.column_stack([np.ones(n)] + [[sed_cmb(c, x) for x in f] for c in comps])
    # normalize columns (numerical conditioning; does not change the solution)
    A[:, 1:] /= np.abs(A[:, 1:]).max(0)
    NiA = np.linalg.solve(Nn, A)
    e1 = np.zeros(A.shape[1]); e1[0] = 1.0
    w = NiA @ np.linalg.solve(A.T @ NiA, e1)
    return w, comps


def combo_noise_factor(w):
    """Noise factor of the combination: each band has 1/N of the detectors (σ_i² = N·σ²), hence √N·‖w‖."""
    w = np.asarray(w, float)
    return float(np.sqrt(len(w)) * np.linalg.norm(w)) if len(w) > 1 else 1.0


def fg_model_cov(freqs, model="pysm3", mask_deg=20.0, fwhm_deg=1.0, co_on=True):
    """Band-band covariance of the NOMINAL foreground maps (K², |b| > mask), for the ILC."""
    ns = 64
    th, _ = hp.pix2ang(ns, np.arange(12 * ns * ns))
    hi = np.abs(90 - np.degrees(th)) > mask_deg
    ms = []
    for nu in freqs:
        m, _ = foregrounds_cmb(float(nu), "nominal", model=model, co_on=co_on)
        m = hp.smoothing(m, fwhm=np.deg2rad(fwhm_deg), lmax=3 * ns - 1, iter=1)
        ms.append(m[hi] * 1e-6)
    M = np.array(ms)
    return M @ M.T / M.shape[1]


# ---------------------------------------------------------------------------------------------
# Extragalactic (radio) point sources, statistical model
# ---------------------------------------------------------------------------------------------
# Differential counts n(S) = A·S^-2.5 (Euclidean) at 100 GHz, with S^2.5·n(S) ≈ 15 Jy^1.5/sr between ~0.01 and
# 10 Jy: the order of the radio source counts of de Zotti et al. (2005, A&A 431, 893) and of the Planck catalogue
# (PCCS2, Planck 2015 XXVI) at 100–143 GHz, dominated by flat-spectrum sources. Spectrum S ∝ ν^α with
# α = −0.5 ± 0.3. Isotropic positions. The onboard model knows the sources above the catalogue limit
# (PCCS2 complete at ~0.3 Jy at 100–143 GHz) with a flux error (blazar variability, ~10–30%).
PS_A_EUCL = 15.0
PS_SMIN, PS_SMAX = 0.01, 30.0
C_LIGHT = 2.99792458e8


def jy_to_uKcmb_pixel(nu_ghz, nside):
    """1 Jy spread over one HEALPix pixel → µK_CMB."""
    om = 4 * np.pi / (12 * nside**2)
    nu = nu_ghz * 1e9
    t_rj = 1e-26 * C_LIGHT**2 / (2 * KB * nu**2 * om)
    return t_rj * rj_to_cmb(nu_ghz) * 1e6


@lru_cache(maxsize=8)
def point_sources(bands, weights, seed, cut_jy, err_pct, nside=128):
    """(true, onboard model) point-source maps combined with the band weights, µK_CMB, at the given nside."""
    rng = np.random.default_rng(seed + 31415)
    a = PS_A_EUCL
    n_tot = rng.poisson(4 * np.pi * a / 1.5 * (PS_SMIN**-1.5 - PS_SMAX**-1.5))
    u = rng.random(n_tot)                                        # S by inversion of the cumulative counts
    S100 = (PS_SMIN**-1.5 - u * (PS_SMIN**-1.5 - PS_SMAX**-1.5)) ** (-1 / 1.5)
    alpha = rng.normal(-0.5, 0.3, n_tot)
    z = rng.uniform(-1, 1, n_tot); ph_ = rng.uniform(0, 2 * np.pi, n_tot)
    pix = hp.ang2pix(nside, np.arccos(z), ph_)
    known = S100 > cut_jy
    err = 1 + rng.normal(0, err_pct / 100, n_tot)
    tru = np.zeros(12 * nside**2)
    mod = np.zeros(12 * nside**2)
    for nu, w in zip(bands, weights):
        S = S100 * (nu / 100.0) ** alpha * jy_to_uKcmb_pixel(nu, nside) * w
        np.add.at(tru, pix, S)
        np.add.at(mod, pix[known], (S * err)[known])
    return tru, mod


def ps_count(cut_jy):
    """Expected number of sources above cut_jy (100 GHz) over the full sky."""
    return 4 * np.pi * PS_A_EUCL / 1.5 * (cut_jy**-1.5 - PS_SMAX**-1.5)


def small_scale_fg(fg64, ns, seed, slope=-2.4, ell0=150, bcut=30.0):
    """Small scales of the foregrounds (ℓ > 160), missing from the large-scale templates: Gaussian field with
    C_ℓ ∝ ℓ^-2.4 (typical dust slope; Planck 2018 XI), amplitude fitted to the map's own spectrum
    at ℓ = 80–150 off the plane (|b| > 30°, pseudo-C_ℓ corrected by f_sky) and modulated by the local amplitude
    (normalized off the plane), as in PySM3."""
    n64 = hp.npix2nside(len(fg64))
    th, _ = hp.pix2ang(n64, np.arange(len(fg64)))
    hi = np.abs(90 - np.degrees(th)) > bcut
    m = np.where(hi, fg64 - np.mean(fg64[hi]), 0.0)
    cl = hp.anafast(m, lmax=ell0) / hi.mean()
    ell = np.arange(len(cl))
    A = float(np.mean(cl[80:ell0 + 1] * ell[80:ell0 + 1] ** (-slope)))
    L = 3 * ns - 1
    el = np.arange(L + 1).astype(float)
    cls = np.zeros(L + 1)
    cls[2:] = A * el[2:] ** slope
    cls *= 1 / (1 + np.exp(-(el - 170) / 8.0))                      # only above the template scale
    np.random.seed(seed + 5151)
    g = hp.synfast(cls, ns, lmax=L, pixwin=False)
    amp = hp.ud_grade(np.abs(hp.smoothing(fg64, fwhm=np.deg2rad(3.0), lmax=191, iter=1)), ns)
    thn, _ = hp.pix2ang(ns, np.arange(12 * ns * ns))
    hin = np.abs(90 - np.degrees(thn)) > bcut
    return g * amp / np.sqrt(np.mean(amp[hin] ** 2))

class Sky:
    """Maps of the bias each sensor sees (already smoothed by the beam), in K, Galactic coordinates."""

    def __init__(self, nu_ghz, fwhm_deg, subtract=True, two_freq=None, amp_err_pct=3.0,
                 map_depth_uK_arcmin=42.0, seed=0, fg_model="pysm3", cmb_sky="wmap", weights=None,
                 ps_on=False, ps_cut_jy=0.3, ps_err_pct=20.0, co_on=True, co_err_pct=10.0, nside=128):
        """two_freq: one frequency (2 bands, nulls dust) or a tuple of extra bands; weights: precomputed weights
        of all bands (ν, *extras) (default: null the dust)."""
        rng = np.random.default_rng(seed + 4242)
        d = _data()
        self.nu, self.fwhm, self.subtract, self.two_freq = nu_ghz, fwhm_deg, subtract, two_freq
        self.fg_model, self.cmb_sky = fg_model, cmb_sky
        ns = int(nside)                              # 128 (default) or 256 (high resolution: beam ≥ 0.25°)
        self.nside = ns
        beam_min = 0.5 if ns <= 128 else 0.25
        up = lambda m: hp.ud_grade(m, ns)
        e = {k: rng.normal(0, amp_err_pct / 100) for k in ("s", "ff", "ame", "d")}
        # CO: amplitude error of the onboard model (separate generator); co_err_pct ≥ 100 = CO not subtracted onboard
        e["co"] = -1.0 if co_err_pct >= 100 else np.random.default_rng(seed + 7177).normal(0, co_err_pct / 100)
        co_on = bool(co_on)                    # passed explicitly (not taken from a global)
        fg_true, self.comps = foregrounds_cmb(nu_ghz, "true", model=fg_model, co_on=co_on)
        fg_mod, _ = foregrounds_cmb(nu_ghz, "nominal", e, model=fg_model, co_on=co_on)
        extra = () if not two_freq else (tuple(two_freq) if np.ndim(two_freq) else (two_freq,))
        self.bands = (nu_ghz,) + tuple(float(x) for x in extra)
        if extra:
            w = np.asarray(weights, float) if weights is not None else nfreq_weights(self.bands, "dust")[0]
            self.w = tuple(float(x) for x in w)
            fg_true = w[0] * fg_true
            fg_mod = w[0] * fg_mod
            for wi, nu_i in zip(w[1:], extra):
                ft_i, _ = foregrounds_cmb(nu_i, "true", model=fg_model, co_on=co_on)
                fm_i, _ = foregrounds_cmb(nu_i, "nominal", e, model=fg_model, co_on=co_on)
                fg_true = fg_true + wi * ft_i
                fg_mod = fg_mod + wi * fm_i
        else:
            self.w = (1.0, 0.0)
        if cmb_sky == "random":
            # Gaussian realization with the CAMB C_ℓ (one per seed: the Monte Carlo varies the sky)
            from . import sky as _sky
            np.random.seed(seed + 9001)
            cmb = hp.synfast(_sky.cl_table(3 * ns - 1) * 1e12, ns, lmax=3 * ns - 1, pixwin=False)
        else:
            cmb = d["cmb_uK"]
            if ns != 128:
                # the (filtered, nside 128) WMAP map gets the small scales of a Gaussian realization with the
                # CAMB C_ℓ above ℓ = 300; the onboard model knows the full map (Planck map), so this only
                # appears in the unsubtracted sky and in the kinematic effects
                from . import sky as _sky
                cl_ = _sky.cl_table(3 * ns - 1) * 1e12
                ell_ = np.arange(len(cl_))
                cl_ *= 1 / (1 + np.exp(-(ell_ - 320) / 15.0))
                np.random.seed(seed + 9002)
                cmb = hp.ud_grade(cmb, ns) + hp.synfast(cl_, ns, lmax=3 * ns - 1, pixwin=False)
        if ns > 128:
            fg_true_hi = up(fg_true) + small_scale_fg(fg_true, ns, seed)
        else:
            fg_true_hi = up(fg_true)
        if ps_on:
            wts = self.w[:len(self.bands)] if extra else (1.0,)
            ps_t, ps_m = point_sources(tuple(self.bands), tuple(float(x) for x in wts), int(seed), float(ps_cut_jy),
                                       float(ps_err_pct), ns)
        else:
            ps_t = ps_m = np.zeros(12 * ns**2)
        self.ps_on = ps_on
        sig_ = np.deg2rad(max(fwhm_deg, beam_min))
        # full sky (CMB + true foregrounds), used for the kinematic effects
        self.full = hp.smoothing(cmb + fg_true_hi + ps_t, fwhm=sig_, lmax=3 * ns - 1, iter=1) * 1e-6
        if subtract:
            # error of the subtracted CMB map: white map noise (depth in µK·arcmin)
            pix_arcmin = np.sqrt(4 * np.pi / (12 * ns**2)) * 180 / np.pi * 60
            err = rng.normal(0, map_depth_uK_arcmin / pix_arcmin, 12 * ns**2)
            resid = err + (fg_true_hi - up(fg_mod)) + (ps_t - ps_m)
            self.model_level = up(np.abs(fg_mod))
        else:
            resid = cmb + fg_true_hi + ps_t
            self.model_level = up(np.abs(fg_mod))
        sig = np.deg2rad(max(fwhm_deg, beam_min))      # pixel of ~0.46° (nside 128) or ~0.23° (256)
        self.map = hp.smoothing(resid, fwhm=sig, lmax=3 * ns - 1, iter=1) * 1e-6       # K
        self.fg_resid = hp.smoothing(fg_true_hi - up(fg_mod), fwhm=sig, lmax=3 * ns - 1, iter=1) * 1e-6
        self.level = hp.smoothing(self.model_level, fwhm=sig, lmax=3 * ns - 1, iter=1) * 1e-6
        # filter prior calibration: |foreground residual| ≈ k·(model level) + floor
        if subtract:
            lv, rs = self.level, self.fg_resid
            self.k_prior = float(np.sqrt(np.mean(rs**2) / np.mean(lv**2)))
            self.floor = float(np.std(hp.smoothing(err + (ps_t - ps_m), fwhm=sig, lmax=3 * ns - 1, iter=1)) * 1e-6)
        else:
            self.k_prior, self.floor = 1.0, float(np.std(hp.smoothing(cmb, fwhm=sig, lmax=3 * ns - 1, iter=1)) * 1e-6)
        self.ps_resid = hp.smoothing(ps_t - ps_m if subtract else ps_t, fwhm=sig, lmax=3 * ns - 1, iter=1) * 1e-6
        self.noise_factor = combo_noise_factor(self.w) if extra else 1.0

    def kinematic(self, dirs_ecl, beta_vec):
        """Effect of the probe velocity (β = v/c, ecliptic) on the sky: aberration and Doppler
        modulation of the anisotropies, T(n̂) -> T(n̂')·(1 + β·n̂), with n̂' = n̂ − β + (β·n̂)n̂ (1st order).
        Returns the difference relative to the sky without this effect [K]."""
        n = np.atleast_2d(dirs_ecl)
        b = np.asarray(beta_vec, float)
        bn = n @ b
        src = n - b[None] + bn[:, None] * n
        src /= np.linalg.norm(src, axis=1)[:, None]
        th0, ph0 = self._gal(n)
        th1, ph1 = self._gal(src)
        t0 = hp.get_interp_val(self.full, th0, ph0)
        t1 = hp.get_interp_val(self.full, th1, ph1)
        return np.atleast_1d(t1 * (1 + bn) - t0)

    def _gal(self, dirs_ecl):
        g = np.atleast_2d(dirs_ecl) @ ECL_TO_GAL.T
        th = np.arccos(np.clip(g[:, 2], -1, 1))
        phi = np.arctan2(g[:, 1], g[:, 0])
        return th, phi

    def value(self, dirs_ecl):
        th, phi = self._gal(dirs_ecl)
        return np.atleast_1d(np.asarray(hp.get_interp_val(self.map, th, phi), float))

    def prior_sigma(self, dirs_ecl):
        th, phi = self._gal(dirs_ecl)
        lv = np.atleast_1d(np.asarray(hp.get_interp_val(self.level, th, phi), float))
        return np.hypot(self.k_prior * lv if self.subtract else lv, self.floor)

    def fg_level(self, dirs_ecl):
        th, phi = self._gal(dirs_ecl)
        return np.atleast_1d(np.asarray(hp.get_interp_val(self.level, th, phi), float))

    def gal_lonlat_deg(self, dirs_ecl):
        th, phi = self._gal(dirs_ecl)
        return np.degrees(phi), 90.0 - np.degrees(th)

    def gal_lat_deg(self, dirs_ecl):
        th, _ = self._gal(dirs_ecl)
        return 90.0 - np.degrees(th)


@lru_cache(maxsize=16)
def get_sky(nu, fwhm, subtract, two_freq, amp_err_pct, seed, fg_model="pysm3", cmb_sky="wmap", weights=None,
            ps_on=False, ps_cut_jy=0.3, ps_err_pct=20.0, co_on=True, co_err_pct=10.0, nside=128):
    return Sky(nu, fwhm, subtract, two_freq, amp_err_pct, seed=seed, fg_model=fg_model, cmb_sky=cmb_sky,
               weights=weights, ps_on=ps_on, ps_cut_jy=ps_cut_jy, ps_err_pct=ps_err_pct, co_on=co_on,
               co_err_pct=co_err_pct, nside=nside)
