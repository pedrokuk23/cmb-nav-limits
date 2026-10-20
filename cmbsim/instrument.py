"""Instrument model: radiometer noise and accumulation of readings of the same sky point.

Radiometer equation (Eq. 3.13): when looking at the same point for τ seconds, the white noise
falls as NET/√τ. A systematic floor remains (gain/offset stability, residual 1/f
noise) that does not fall with τ:

    σ_T(τ)² = NET² / τ + σ_sys²

NET in K·√s, per SENSOR (one sensor = a set of detectors looking in the same direction).

Technology classes (equivalent NET derived from the published map sensitivities):
NET = w · √(T_obs / Ω_sky), with w the map depth and T_obs the observing time.
- Planck LFI 70 GHz: goal of 3.0 µK·deg in ~14 months (2 surveys) -> ~90 µK√s for the channel
  (12 radiometers) -> ~310 µK√s per radiometer.
- Planck HFI 143 GHz: goal of 0.7 µK·deg -> ~21 µK√s for the channel (12 bolometers) ->
  ~73 µK√s per bolometer.
- LiteBIRD class: 2.2 µK·arcmin (polarization) in 3 years -> ~1.8 µK√s (pol.) ≈ 1.2 µK√s
  in temperature, summing the whole focal plane (~4500 detectors).
- Dedicated navigation sensor (assumption): 100 detectors of ~50 µK√s each -> 5 µK√s.
- Ideal: photon-limited detectors (~30 µK√s each) in an array of 10⁴ detectors
  -> ~0.3 µK√s.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Tech:
    key: str
    name: str
    net_uK_rts: float          # NET per sensor [µK·√s]
    fwhm_deg: float            # beam [deg]
    sis_nK: float = 0.0        # systematic floor [nK]
    fs_hz: float = 1.0         # detector sampling rate [Hz]
    tau_default_s: float = 7200.0   # default filter update interval (window) [s]
    fs_source: str = ""         # source of the sampling rate
    freq_ghz: float = 90.0     # band center frequency
    f_knee_hz: float = 0.05    # 1/f noise knee
    alpha_1f: float = 1.0      # 1/f noise slope, S ∝ f^-α
    knee_source: str = ""       # source of the knee
    bands_ghz: tuple = ()      # design bands (family presets; the 1st = freq_ghz). Empty = single band
    mass_kg: tuple = ()        # instrument + cryogenics mass (min, max) [kg]
    power_w: tuple = ()        # power (min, max) [W]
    family: str = ""          # technology family (A–F)
    net_comb_uK_rts: tuple = ()  # combined NET of the design (min, max), already including component separation

    def flicker_floor(self):
        """1/f noise floor for absolute measurements with a staring sensor [K]: Allan deviation of flicker
        noise, σ = NET·√(4 ln2 · f_knee) (independent of τ; valid for α = 1)."""
        return self.net * np.sqrt(4 * np.log(2) * self.f_knee_hz)

    @property
    def net(self):
        return self.net_uK_rts * 1e-6

    def sigma_T(self, tau_s):
        tau_s = np.asarray(tau_s, float)
        return np.sqrt(self.net**2 / tau_s + (self.sis_nK * 1e-9) ** 2)

    def sigma_sample(self):
        """Noise of ONE sample [K] = NET·√fs (the noise per √s does not depend on the sampling)."""
        return self.net * np.sqrt(self.fs_hz)

    def tau_for_sigma(self, sigma_T):
        """Integration time to reach σ_T [K]; inf if below the systematic floor."""
        floor = self.sis_nK * 1e-9
        if sigma_T <= floor:
            return np.inf
        return self.net**2 / (sigma_T**2 - floor**2)


DAY = 86400.0
# 1/f knees: LFI "few tens of mHz at 70 GHz" (Zacchei et al. 2011, A&A 536, A5); HFI 143 GHz in flight
# ~100-230 mHz with α ≈ 1 (Planck early results IV 2011, A&A 536, A4); LiteBIRD: requirement "common
# 1/f knee ≤ 20 mHz" (LiteBIRD Collaboration 2023, PTEP 042F01, Table 8). HFI 143: 91-230 mHz, median ~170.
PRESETS = {
    "lfi70": Tech("lfi70", "Planck LFI 70 GHz (1 radiometer)", 310.0, 0.23, 0.0, 78.769, 2 * DAY,
                  "Planck 2015 II, Table 12", 70.0, 0.03, 1.0, "Zacchei et al. 2011 (A&A 536, A5)"),
    "hfi143": Tech("hfi143", "Planck HFI 143 GHz (1 bolometer)", 73.0, 0.12, 0.0, 180.3737, 2 * DAY,
                   "Planck 2015 VII (TOI at 180.3737 Hz)", 143.0, 0.15, 1.0, "Planck early results IV (2011)"),
    "hfi143_channel": Tech("hfi143_channel", "Planck HFI 143 GHz (channel, 12 bolometers)", 21.0, 0.12, 0.0,
                         180.3737, DAY, "Planck 2015 VII (TOI at 180.3737 Hz)", 143.0, 0.15, 1.0,
                         "Planck early results IV (2011)"),
    "dedicated": Tech("dedicated", "Dedicated sensor (100 detectors of 50 µK√s)", 5.0, 0.5, 0.0, 10.0, 7200.0,
                     "assumption", 90.0, 0.05, 1.0, "assumption (between LiteBIRD and Planck)"),
    "litebird": Tech("litebird", "LiteBIRD class (full focal plane)", 1.2, 0.5, 0.0, 19.0, 1800.0,
                     "LiteBIRD, ~19 Hz after decimation (arXiv:2209.12492)", 140.0, 0.02, 1.0,
                     "LiteBIRD 2023 (PTEP), requirement ≤ 20 mHz"),
    "ideal": Tech("ideal", "Ideal (10⁴ photon-limited detectors)", 0.3, 0.5, 0.0, 100.0, 1800.0,
                  "assumption", 90.0, 0.001, 1.0, "assumption"),
}

# Design families of the CMB sensor. The tabulated combined NET already includes a ×2 penalty
# for component separation; since the simulator applies its own penalty (combo_noise_factor, ~1.5–3)
# when the extra bands are enabled, the preset NET is the center of the range ÷ 2 (NET of the detector set
# in a single band). Mass and power: instrument + cryogenics, uncertainty of ×2.
_SC = "CMB sensor design-family study"
FAMILY_PRESETS = {
    "warm_2b": Tech("warm_2b", "A · Warm sky-sky (DMR/TEMPEST), 70+90 GHz", 1000.0, 5.0, 0.0, 10.0, 7200.0,
                     _SC, 90.0, 0.005, 1.0, "Dicke/sky-sky, mHz", (90.0, 70.0), (8.0, 15.0), (30.0, 60.0), "A",
                     (1500.0, 2500.0)),
    "hemt90_3b": Tech("hemt90_3b", "B · Passive HEMT 50–90 K (WMAP-lite), 40/70/90 GHz", 150.0, 2.0, 0.0, 10.0,
                      7200.0, _SC, 70.0, 0.01, 1.0, "WMAP: < 10 mHz", (70.0, 40.0, 90.0), (30.0, 60.0), (40.0, 80.0),
                      "B", (200.0, 400.0)),
    "hemt20_3b": Tech("hemt20_3b", "C · 20 K pseudo-correlation HEMT (LFI-lite), 44/70/90 GHz", 30.0, 1.0, 0.0,
                      78.769, 7200.0, _SC, 70.0, 0.04, 1.0, "LFI: 20–60 mHz", (70.0, 44.0, 90.0), (50.0, 90.0),
                      (150.0, 300.0), "C", (40.0, 80.0)),
    "tes_3b": Tech("tes_3b", "D · Sub-K trichroic TES (MFT-lite), 90/150/220 GHz", 2.25, 0.5, 0.0, 19.0, 1800.0,
                   _SC, 150.0, 0.15, 1.0, "TES: 0.1–0.2 Hz", (150.0, 90.0, 220.0), (150.0, 300.0), (300.0, 600.0),
                   "D", (3.0, 6.0)),
    "multimode_2b": Tech("multimode_2b", "E · Multimode bolometers (SWIPE/PIXIE), 140/220 GHz", 3.75, 3.0, 0.0,
                         19.0, 1800.0, _SC, 140.0, 0.1, 1.0, "~0.1 Hz", (140.0, 220.0), (100.0, 200.0),
                         (250.0, 500.0), "E", (5.0, 10.0)),
    "mkid_3b": Tech("mkid_3b", "F · Compact MKID, 90/150/220 GHz", 2.75, 0.5, 0.0, 19.0, 1800.0, _SC, 150.0, 0.5,
                    1.0, "TLS: 0.1–1 Hz", (150.0, 90.0, 220.0), (130.0, 270.0), (280.0, 550.0), "F", (3.0, 8.0)),
}
PRESETS.update(FAMILY_PRESETS)


def family_config(key):
    """Config fields for using a family preset with its bands (onboard component separation)."""
    t = PRESETS[key]
    extra = tuple(b for b in t.bands_ghz[1:]) if t.bands_ghz else ()
    return dict(tech=key, freq_ghz=None, freq2_ghz=None, freqs_extra_ghz=extra)


class Flicker:
    """Independent 1/f noise per sensor, S(f) = 2·NET²·(f_k/f)^α (one-sided, on top of the white noise).

    Approximated by a sum of Ornstein-Uhlenbeck processes with cutoff frequencies spaced
    logarithmically (2 per decade), a standard technique for generating 1/f noise. Components faster than
    the window become extra white noise (variance of the mean over the window); the slow ones become a drift
    that the window does not remove. With spin, demodulation at the spin frequency only sees S(f_spin)."""

    def __init__(self, net_K, f_knee, alpha, tau_s, n, rng, f_lo=1e-8, f_hi=50.0, per_decade=2):
        self.rng, self.n, self.tau = rng, n, tau_s
        fc = np.logspace(np.log10(f_lo), np.log10(f_hi), int(per_decade * np.log10(f_hi / f_lo)) + 1)
        tc = 1 / (2 * np.pi * fc)
        # weights: σ_j² ∝ f_c^(1-α); normalized to S(f) = 2 NET² (f_k/f)^α at the middle of the interval
        w = fc ** (1 - alpha)
        f0 = np.sqrt(f_lo * f_hi) if f_hi > 1e-4 else f_knee
        S0 = np.sum(4 * w * tc / (1 + (2 * np.pi * f0 * tc) ** 2))
        var = w * 2 * net_K**2 * (f_knee / f0) ** alpha / S0
        slow = tc > tau_s / 2
        # variance of the mean, over a window τ, of an OU process with correlation time t_c
        r = tau_s / tc
        vwin = 2 * var * (r - 1 + np.exp(-r)) / r**2
        self.white_var = float(np.sum(vwin[~slow]))           # enters as extra white noise
        self.tc, self.var = tc[slow], var[slow]
        self.x = rng.normal(0, 1, (n, slow.sum())) * np.sqrt(self.var)
        self.t = None
        self.f_knee, self.alpha, self.net = f_knee, alpha, net_K

    def step(self, t):
        """Slow drift of each sensor at time t (K)."""
        if self.t is not None:
            phi = np.exp(-(t - self.t) / self.tc)
            self.x = phi * self.x + np.sqrt(1 - phi**2) * np.sqrt(self.var) * self.rng.normal(0, 1, self.x.shape)
        self.t = t
        return self.x.sum(1)

    def increment_var(self, dt):
        """Variance of the drift increment over dt (for the filter process noise) [K²]."""
        return float(np.sum(2 * self.var * (1 - np.exp(-dt / self.tc))))

    def drift_rms(self):
        return float(np.sqrt(np.sum(self.var)))

    def spin_factor(self, f_spin):
        """Factor on the noise variance of the harmonics demodulated at the spin frequency."""
        return 1.0 + (self.f_knee / f_spin) ** self.alpha
