"""Demonstrator simulator: Earth-Mars cruise with CMB navigation (+ optional CeleNav).

How it works (summary):
1. Default trajectory: Lambert Earth -> Mars (2026 window), propagated with the Sun + Venus, Earth,
   Mars, Jupiter and Saturn + solar radiation pressure. Trajectory correction maneuvers (TCM) with unknown
   execution error. The "truth" contains what the filter does not know (default): an error in the solar
   radiation pressure coefficient, unmodeled acceleration (Gauss-Markov) and small attitude impulses.
2. Four CMB sensors in a tetrahedron, fixed in inertial space: they ALWAYS look at the same
   4 sky points and accumulate readings (at the instrument sampling rate) over a window of τ seconds. Every τ
   one measurement is produced (window average) with noise NET/√τ.
3. The sky anisotropies (T_F) are a fixed bias per sensor (residual after subtracting the Planck
   map, or the full T_F if subtraction is off). When the sensors are re-pointed
   (solar angle constraint), the biases change.
4. UKF with 14 states: position, velocity, T0, 4 sky biases and the solar velocity error
   (+8 gain states when the filter estimates the gain - default: constant calibration and
   Gauss-Markov drift of each sensor).
5. Optional CeleNav: angles between Venus/Mars/Earth and stars, 0.5'' error.
6. Real sky (default when healpy and the maps are available): WMAP CMB (Tegmark et al. 2003) + PySM
   foregrounds at the sensor frequency; residual = map error + (true foreground − model with
   constant spectral indices). The pointing rotates the tetrahedron about the anti-Sun axis to
   avoid the galactic plane. Option of 2 frequencies per sensor (on-board component separation).
7. 1/f noise per sensor (knee per technology): in staring mode it becomes drift + extra noise per
   window; with spin, the demodulated harmonics only see S(f_spin) and the DC channels (ring mean)
   are discarded, as in the per-ring offset removal (destriping) of spinning missions.
8. Attitude (optional, on by default): star tracker model (QUEST/ESOQ2/q/TRIAD over the
   GSC6 catalog). Attitude error = sensor noise averaged over the window + thermal drift +
   fixed misalignment; equivalent to measuring the dipole with v + δθ × v (1″ ≈ 1.8 m/s).
"""
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import attitude as at
from . import beacons as bc
from . import physics as ph
from . import skymaps
from . import zodi as zodimod
from . import sky
from .instrument import PRESETS, Flicker, Tech
from .ukf import UKF, EKF, GPF

STARS = np.array([ph.radec_to_ecl(101.287, -16.716),     # Sirius
                  ph.radec_to_ecl(95.988, -52.696),      # Canopus
                  ph.radec_to_ecl(213.915, 19.182)])     # Arcturus
CELE_BODIES = ("venus", "mars", "earth")


@dataclass
class Config:
    # trajectory
    departure: tuple = (2026, 10, 31)
    tof_days: float = 295.0
    start_day: float = 5.0          # cruise start (outside the Earth sphere of influence)
    end_day: float | None = None    # default: tof - 5 days
    # instrument
    tech: str = "dedicated"
    freq_ghz: float | None = None    # sensor frequency (None = instrument default)
    freq2_ghz: float | None = None   # 2nd frequency per sensor (on-board component separation)
    freqs_extra_ghz: tuple = ()      # extra bands per sensor (N frequencies; added to freq2, if any)
    sep_method: str = "dust"       # 'dust' | 'sync+dust' | 'all' | 'ilc' (skymaps.SEP_METHODS)
    sky_real: bool = True            # real sky (maps) instead of the statistical sky
    fg_amp_err_pct: float = 3.0      # amplitude error of the on-board foreground model
    sky_nside: int = 128             # sky map resolution: 128 (beam ≥ 0.5°) or 256 (≥ 0.25°, small scales)
    co_on: bool = True               # CO lines (Planck maps; only in bands containing 115/230/345 GHz)
    co_err_pct: float = 10.0         # error of the on-board CO model (≥ 100 = not subtracted)
    ps_on: bool = True               # extragalactic point sources (statistical number-count model)
    ps_cut_jy: float = 0.3           # on-board catalog limit (brighter sources are subtracted)
    ps_err_pct: float = 20.0         # flux error of catalog sources (variability)
    gal_avoid: bool = True           # rotates the tetrahedron to avoid the galactic plane
    one_over_f: bool = True          # detector 1/f noise
    f_knee_hz: float | None = None   # 1/f knee (None = instrument default)
    fg_model: str = "pysm3"          # 'pysm3' (d11+s6 large scale) or 'pysm1'
    cmb_sky: str = "wmap"            # 'wmap' (Tegmark map) or 'random' (CAMB realization from the seed)
    zodi_on: bool = True             # zodiacal emission (ZodiPy) along the trajectory
    zodi_real_traj: bool = True      # zodiacal seen from the true trajectory (recomputed if > 10⁶ km from nominal)
    zodi_err_pct: float = 10.0       # error of the zodiacal model subtracted on board
    zodi_estimate: bool = False      # the filter estimates the zodiacal model amplitude error (1 state) instead of
                                     # treating it as a random walk in the sky biases
    kinematic: bool = True           # aberration and Doppler modulation of the anisotropies by the probe velocity
    kin_correct: bool = True         # the on-board software corrects the kinematic effect with the estimated velocity
    receiver: str = "differential"    # 'differential' (reference load ≈ T0), 'total' or 'sky-sky' (paired horns, COBE DMR/WMAP)
    gain_cal_ppm: float = 10.0       # residual gain calibration error of each sensor (1σ, after commissioning)
    gain_drift_ppm: float = 10.0     # gain drift (1σ, Gauss-Markov) - requirement assumption
    gain_tau_h: float = 240.0        # correlation time of the gain drift
    temp_linearized: bool = True     # the detector measures intensity: output in linearized temperature (2nd order)
    lin_correct: bool = True         # the on-board model uses the linearized temperature (else, thermodynamic)
    band_frac: float = 0.25          # fractional bandwidth of the detector
    band_err_pct: float = 1.0        # error (1σ) of the band center known on board [%]
    obs_analysis: bool = True        # observability analysis (Fisher information, once per day)
    filter_type: str = "ukf"         # 'ukf' | 'ekf' (numerical Jacobians) | 'pf' (Gaussian particle, UKF proposal)
    pf_particles: int = 1000         # particles per update in 'pf'
    allow_fallback: bool = False     # if a requested component cannot run (missing package or data file), raise an
                                     # error (default) or, if True, fall back to the reduced model with a warning
    # electric propulsion (continuous thrust along the velocity, in arcs)
    ep_on: bool = False
    ep_accel_mms2: float = 0.1       # nominal thrust acceleration [mm/s²] (e.g., 0.2 N on 2000 kg)
    ep_arcs: tuple = ((20.0, 120.0), (180.0, 260.0))   # thrust arcs [days]
    ep_mag_err_pct: float = 1.0      # thrust magnitude error (1σ, Gauss-Markov)
    ep_point_err_deg: float = 0.5    # thrust pointing error (1σ per axis, Gauss-Markov)
    ep_tau_h: float = 24.0           # correlation time of the thrust errors
    ep_mag_bias_pct: float = 1.0     # thrust magnitude bias, constant within each arc (1σ)
    ep_point_bias_deg: float = 0.2   # pointing bias, constant within each arc (1σ per axis)
    ep_estimate: bool = True         # the filter estimates the thrust bias (empirical acceleration, 3 states)
    gain_estimate: bool = True       # the UKF estimates each sensor gain (8 extra states); otherwise only tolerates it
    net_uK_rts: float | None = None  # overrides the preset NET
    fwhm_deg: float | None = None
    tau_s: float | None = None      # accumulation window = interval between measurements (None = instrument default)
    fs_hz: float | None = None      # detector sampling rate (None = instrument default)
    subtract_planck: bool = True
    repoint_days: float = 30.0      # 0 = never; >0 = periodic; -1 = automatic (only when the Sun gets close)
    sun_excl_deg: float = 45.0      # channel discarded if the sensor points closer than this to the Sun
    spin: bool = False              # axial spin about the anti-Sun axis
    spin_rate_dps: float = 6.0      # spin rate [deg/s] (6 °/s = 1 rpm)
    spin_tilt_deg: float = 0.0      # tilt of the spin axis relative to anti-Sun (0 = anti-Sun axis)
    prec_n: int = 4                 # number of axis orientations on the cone (changes every window; stepped precession)
    destripe: bool = False          # spin + 1/f: estimates each sensor offset (destriping) instead of discarding a0
    destripe_model: str = "exact"   # 'exact': each slow 1/f component becomes one state per sensor; 'simple': 1 Gauss-Markov
    alpha_1f: float | None = None   # 1/f slope (None = instrument default; FAST measured ~0.65)
    cal_period_h: float = 0.0       # periodic offset calibration (internal load / noise diode); 0 = none
    cal_sigma_nK: float = 10.0      # noise of each offset measurement during calibration [nK]
    f_min_1f_hz: float = 1e-8       # below this the 1/f is flat (Poutanen et al. 2006 use 10⁻⁴ Hz for the LFI)
    one_f_common: float = 0.0       # fraction of the 1/f variance common to all sensors (amplifier, thermal)
    intrinsic_dipole_uK: float = 0.0  # intrinsic (non-kinematic) CMB dipole, random direction
    vssb_prior_kms: float = 0.08    # filter prior for the solar velocity error (per axis)
    # filter and scenario
    dsn_calibration: bool = True
    dsn_pass_days: float = 0.0       # DSN passes during cruise: interval [days] (0 = initial calibration only)
    dsn_pass_h: float = 8.0          # duration of each pass [h] (one range and range-rate measurement per hour)
    dsn_range_m: float = 3.0         # range error (1σ)
    dsn_rate_mmps: float = 0.1       # range-rate error (1σ, 60 s integration)
    dsn_ddor: bool = False           # ΔDOR at each pass: plane-of-sky position (two angles)
    ddor_nrad: float = 2.0           # ΔDOR error (1σ per axis)
    opnav_days: float = 0.0          # optical navigation on approach: images of Mars in the last N days (0 = off)
    opnav_urad: float = 5.0          # error of the Mars center direction in each image (1σ per axis)
    opnav_period_h: float = 6.0      # interval between images
    xnav_on: bool = False            # X-ray pulsar navigation (XNAV): range along the pulsar direction
    xnav_sigma_km: float = 20.0      # error of each measurement (1σ), incl. photon integration (SEXTANT/NICER: ~5–10 km)
    xnav_period_h: float = 24.0      # interval between measurements (one pulsar at a time, in rotation)
    dsn_loss_pct: float = 0.0        # probability that a pass is cancelled [%]
    cmb_loss_pct: float = 0.0        # data loss: fraction of CMB windows lost [%]
    # tracking gaps and solar conjunction (use case 1)
    dsn_gaps: tuple = ()             # intervals without DSN or ΔDOR [(start_day, end_day), ...]
    dsn_sep_min_deg: float = 0.0     # cuts DSN and ΔDOR when the Sun–Earth–probe angle (SEP) is below this (0 = no cut)
    dsn_plasma: bool = False         # adds solar plasma (Cassini model) to the range-rate error
    # anomaly: unknown impulse (safe mode, stuck thruster); time known from telemetry
    anomaly_days: tuple = ()         # anomaly days
    anomaly_dv_mps: float = 0.3      # ΔV of each anomaly (1σ per axis) [m/s]
    # accelerometer (navigation IMU): measures the ΔV of TCMs, desaturations and anomalies
    imu_on: bool = False
    imu_dv_err_mmps: float = 1.0     # error of the measured ΔV (1σ per axis) [mm/s]
    imu_scale_pct: float = 0.2       # scale factor error (1σ, % of |ΔV|)
    # solar Doppler: heliocentric radial velocity from the Fraunhofer lines
    sdop_on: bool = False
    sdop_sigma_mps: float = 2.0      # noise of each measurement (1σ) [m/s]
    sdop_period_h: float = 0.5       # interval between measurements
    sdop_bias_mps: float = 3.0       # Gauss-Markov bias (1σ): convective blueshift, activity, potential [m/s]
    sdop_tau_d: float = 13.0         # bias correlation time (~half a solar rotation) [days]
    # optical navigation by asteroids during cruise (DS1 AutoNav)
    ast_on: bool = False
    ast_period_d: float = 2.0        # interval between imaging sessions [days]
    ast_k: int = 3                   # beacons per session
    ast_urad: float = 2.5            # direction error of each beacon (1σ per axis) [µrad]
    ast_eph_km: float = 30.0         # ephemeris error of each beacon (1σ per axis, constant) [km]
    ast_mag_lim: float = 10.5        # camera limiting magnitude
    ast_sun_excl_deg: float = 25.0   # camera solar exclusion
    # XNAV: error model
    xnav_model: str = "fixed"         # 'fixed' (xnav_sigma_km) | 'ely' (σ = S/√(A·h), Table 2 of Ely 2022, "best 4")
    xnav_area_cm2: float = 129.0     # effective detector area in the 'ely' model [cm²]
    xnav_int_h: float = 24.0         # integration time of each measurement in the 'ely' model [h]
    # apparent diameter of the Sun: heliocentric distance (weak anchor)
    sund_on: bool = False
    sund_sigma_km: float = 1e4       # error of |r| (1σ) [km]
    sund_period_h: float = 24.0
    # fast signal modulation (chopper/Dicke: sky × reference load, or sky × sky): the detector 1/f
    # is only seen at the modulation frequency, at the cost of a noise factor (Dicke 50 %: ×2; pseudo-correlation: ×√2)
    chop_hz: float = 0.0             # modulation frequency [Hz] (0 = none)
    chop_penalty: float = 2.0        # NET factor due to the modulation
    # reference load (dipole/monopole ratio method, receiver='total' + chopper): the load temperature
    # drifts (Gauss-Markov per sensor) and adds an offset to the DC channels; the filter estimates it (4 states)
    load_drift_nK: float = 0.0       # reference load drift (1σ, per sensor) [nK]; 0 = none
    load_tau_h: float = 24.0         # correlation time of the load drift [h]
    load_estimate: bool = True       # the filter estimates the load offset (4 Gauss-Markov states)
    departure_model: str = "hyperbola"  # 'hyperbola' (perigee at dep_alt_km, Earth and Moon) or 'center' (legacy)
    dep_alt_km: float = 300.0        # perigee altitude of the departure hyperbola
    retarget: bool = True            # arrival targeting: true periapsis at Mars with the full dynamics
    rp_alt_km: float = 300.0         # target periapsis altitude [km]
    ephem: str = "de421"             # ephemerides: 'de421' (JPL) or 'kepler' (Standish mean elements)
    maneuvers_days: tuple = (10.0, 60.0, 150.0, 250.0)
    maneuver_err_mps: float = 1.0
    tcm_guided: bool = False         # TCMs computed by the navigation to steer the probe to the target at Mars
    injection_err_mps: float = 0.0   # injection error at departure (1σ per axis; truth misses the target, the 1st TCM corrects)
    tcm_deadband_sigma: float = 0.0  # dead band of guided TCMs: executed only if ΔV > k·σ of the knowledge (0 = always)
    tcm_prop_err_pct: float = 1.0    # proportional execution error of guided TCMs (1σ per axis, % of ΔV)
    sigma_acel: float = 1e-12       # km/s², unmodeled accelerations (1σ; 1e-12 km/s² = 1 nm/s²)
    accel_tau_h: float = 24.0       # correlation time of the unmodeled acceleration
    sigma_acel_true: float | None = None  # unmodeled acceleration of the truth (None = the same the filter assumes)
    dyn_mismatch: bool = True       # the truth has model errors unknown to the filter
    srp_err_pct: float = 5.0        # error of the radiation pressure coefficient C_R·A/m (1σ)
    srp_estimate: bool = True       # the filter estimates the radiation pressure error (1 state)
    desat_days: float = 7.0         # interval between attitude impulses (momentum unloading); 0 = none
    desat_dv_mmps: float = 1.0      # residual ΔV of each impulse (1σ per axis) [mm/s]
    celenav: bool = False
    celenav_interval_s: float = 6 * 3600.0
    celenav_err_arcsec: float = 0.5
    cmb_on: bool = True
    # attitude (star tracker model)
    att_on: bool = True
    st_method: str = "quest"
    st_heads: int = 1
    st_frame_hz: float = 5.0
    st_centroid_px: float = 0.25
    st_exposure_s: float = 0.02      # star tracker exposure (20 ms: with 0.1 s the smear at 6 °/s degrades attitude)
    att_misalign_arcsec: float = 3.0   # fixed misalignment star tracker x CMB sensors (1σ/axis)
    att_drift_arcsec: float = 1.0      # thermoelastic drift (1σ/axis)
    att_drift_tau_h: float = 24.0      # correlation time of the drift
    att_est_drift: bool = True         # also estimate the drift (False: fixed misalignment only; drift in the biases)
    sky_bias_walk_nK: float = 0.0      # extra random walk in the sky bias states (nK/√day; 0 = off)
    sky_bias_walk_ep_only: bool = False  # extra walk only during thrust arcs
    align_cal_h: float = 0.0           # alignment calibration with compact sources in the CMB itself (period, h; 0 = off)
    align_cal_arcsec: float = 2.0      # error of each alignment calibration (1σ per axis)
    att_in_filter: bool = True         # the filter knows the statistics of the attitude error
    st_catalog: str = "gsc6"         # 'gsc6' (default) | 'hipparcos' (V ≤ 7.5)
    st_mag_limit: float = 6.0        # star tracker limiting magnitude (without smear)
    st_ident: str = "tracking"       # 'lost' | 'tracking' | 'ins' (attitude.IDENT_MODES)
    st_excl: bool = True             # exclusion by the Sun, Earth/Moon and planets in the star tracker field of view
    st_sun_excl_deg: float = 45.0
    st_earth_excl_deg: float = 25.0
    st_coast_max_arcsec: float = 30.0  # star gap: uses the gyroscope while the propagated error ≤ this
    st_boresight_deg: float = 90.0     # angle of the star tracker boresight to the spin axis
    gyro: str = "none"               # gyroscopes: 'none', 'mems', 'fog', 'hrg' (gyro-stellar filter)
    att_estimate: bool = True          # the filter estimates misalignment (3) and thermal drift (3) star tracker × CMB
    seed: int = 1

    def instrument(self) -> Tech:
        from dataclasses import replace
        base = PRESETS[self.tech]
        return replace(base,
                       net_uK_rts=self.net_uK_rts if self.net_uK_rts else base.net_uK_rts,
                       fwhm_deg=self.fwhm_deg if self.fwhm_deg else base.fwhm_deg,
                       fs_hz=self.fs_hz if self.fs_hz else base.fs_hz)

    def freq(self) -> float:
        return float(self.freq_ghz) if self.freq_ghz else float(PRESETS[self.tech].freq_ghz)

    def f_knee(self) -> float:
        return float(self.f_knee_hz) if self.f_knee_hz is not None else float(PRESETS[self.tech].f_knee_hz)

    def tau(self) -> float:
        return float(self.tau_s) if self.tau_s else float(PRESETS[self.tech].tau_default_s)

    def extra_bands(self) -> tuple:
        """Bands besides the main one (freq2 + freqs_extra), without repetitions or the main band."""
        nu = self.freq()
        out = []
        for b in ((self.freq2_ghz,) if self.freq2_ghz else ()) + tuple(self.freqs_extra_ghz or ()):
            b = float(b)
            if abs(b - nu) > 1e-9 and all(abs(b - o) > 1e-9 for o in out):
                out.append(b)
        return tuple(out)


@dataclass
class Results:
    cfg: Config
    t: np.ndarray                    # update epochs [s since departure]
    kind: np.ndarray                 # 'cmb' | 'cel' | 'man' | 'rep'
    x_true: np.ndarray
    x_est: np.ndarray
    P_diag: np.ndarray
    x_dyn: np.ndarray                # dynamics only (no measurements)
    cmb_t: np.ndarray                # CMB measurement epochs
    cmb_T: np.ndarray                # CMB measurements (N, m): 4 temperatures or spin channels
    cmb_snap_err: np.ndarray         # "instantaneous" velocity error (no filter) [m/s] (N,3)
    sun_angle_min: np.ndarray        # smallest sensor-Sun angle in each measurement [deg]
    traj_t: np.ndarray
    traj_r: np.ndarray
    planets: dict
    bias_true: np.ndarray = None     # true bias of the 4 bias states at each update
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- trajectory
def reference_transfer(cfg: Config):
    """Initial state (t = 0) of the reference trajectory: (jd0, r0, v0, info).
    'center': Earth center with the heliocentric Lambert velocity (or the targeted one).
    'hyperbola': perigee of the departure hyperbola (altitude dep_alt_km) with the Lambert v∞ (or the targeted one)."""
    ph.set_ephemeris(cfg.ephem)
    jd0 = ph.julian_date(*cfg.departure)
    hyp = cfg.departure_model == "hyperbola" and ph.default_planets()[1] == "earth_c"
    rE, vE = ph.planet_state("earth_c" if hyp else "earth", jd0)
    rM, vM = ph.planet_state("mars", jd0 + cfg.tof_days)
    v1, v2 = ph.lambert(rE, rM, cfg.tof_days * ph.DAY)
    info = dict(c3=float(np.sum((v1 - vE) ** 2)), vinf_arr=float(np.linalg.norm(v2 - vM)), targeted=False,
                departure="hyperbola" if hyp else "center")
    if cfg.retarget:
        ep_key = (float(cfg.ep_accel_mms2), tuple(tuple(a) for a in cfg.ep_arcs)) if cfg.ep_on else None
        v1t, tinfo = targeted_departure(tuple(cfg.departure), float(cfg.tof_days), float(cfg.start_day),
                                        float(cfg.rp_alt_km), cfg.ephem, ep_key, tuple(v1),
                                        float(cfg.dep_alt_km) if hyp else None)
        if v1t is not None:
            v1 = np.array(v1t)
            info.update(tinfo, targeted=True, c3=float(np.sum((v1 - vE) ** 2)))
    if hyp:
        r0, v0 = departure_state(jd0, v1 - vE, cfg.dep_alt_km)
        return jd0, r0, v0, info
    return jd0, rE, v1, info


R_EARTH_KM = 6378.137


def departure_state(jd0, vinf_vec, alt_km):
    """Perigee of the departure hyperbola (heliocentric) for a given v∞: outgoing asymptote along v∞,
    plane containing the ecliptic pole × v∞; perigee at alt_km altitude, at epoch jd0."""
    mu = ph.MU_PLANETS["earth_c"]
    rp = R_EARTH_KM + alt_km
    vinf = float(np.linalg.norm(vinf_vec))
    S = np.asarray(vinf_vec, float) / vinf
    hh = ph.unit(np.cross(S, [0, 0, 1.0]))
    e = 1 + rp * vinf**2 / mu
    nu = np.arccos(-1 / e)
    p = np.cos(nu) * S - np.sin(nu) * np.cross(hh, S)
    q = np.cross(hh, p)
    rE, vE = ph.planet_state("earth_c", jd0)
    return rE + rp * p, vE + np.sqrt(vinf**2 + 2 * mu / rp) * q


def departure_leg(jd0, X, t1):
    """Propagates states (n, 6) from perigee (t = 0) to t1 with the Sun, Earth, Moon and planets (DOP853)."""
    from scipy.integrate import solve_ivp
    dyn = ph.Dynamics(jd0)
    X = np.atleast_2d(X)
    n = len(X)
    sol = solve_ivp(lambda t, y: dyn.deriv(t, y.reshape(n, 6)).ravel(), (0.0, t1), X.ravel(), method="DOP853",
                    rtol=1e-11, atol=1e-9)
    return sol.y[:, -1].reshape(n, 6)


R_MARS_KM = 3389.5
XNAV_REF = dict(area_m2=0.2, t_h=24.0, sigma_km=20.0)


def xnav_sigma_from(area_m2, t_int_h):
    """XNAV error per measurement scaled from the reference case (NICER-class detector, ~0.2 m², 1 day → 20 km,
    demonstrator assumption): in the photon-limited regime, σ ∝ 1/√(A·t)."""
    return XNAV_REF["sigma_km"] * np.sqrt(XNAV_REF["area_m2"] * XNAV_REF["t_h"] / (area_m2 * t_int_h))
# Millisecond pulsars used in XNAV (SEXTANT/NICER, Mitchell et al. 2018): RA, Dec J2000 [deg]
XNAV_PULSARS = {"J0437-4715": (69.316, -47.252), "B1821-24": (276.133, -24.870),
                "B1937+21": (294.911, 21.583), "J0030+0451": (7.614, 4.861)}
_ra = np.deg2rad([v[0] for v in XNAV_PULSARS.values()])
_de = np.deg2rad([v[1] for v in XNAV_PULSARS.values()])
XNAV_DIRS = np.column_stack([np.cos(_ra) * np.cos(_de), np.sin(_ra) * np.cos(_de), np.sin(_de)]) @ ph.EQ_TO_ECL.T
# "Best 4" of Ely et al. 2022 (J. Astronaut. Sci. 69:385, Table 2): TOA σ in range with 129 cm² and 30 min
# = 87, 97, 5 and 8 km. Scaling law (eq. 72): σ = S/√(A·h), A in cm² and h in hours -> S = σ·√(129·0.5).
XNAV_ELY = {"J0437-4715": ((69.316, -47.252), 87.0), "B0833-45": ((128.836, -45.176), 97.0),
            "B1821-24": ((276.133, -24.870), 5.0), "B1937+21": ((294.911, 21.583), 8.0)}
XNAV_ELY_S = np.array([s_ * np.sqrt(129.0 * 0.5) for _, s_ in XNAV_ELY.values()])     # km·√(cm²·h)
_ra2 = np.deg2rad([v[0][0] for v in XNAV_ELY.values()])
_de2 = np.deg2rad([v[0][1] for v in XNAV_ELY.values()])
XNAV_ELY_DIRS = np.column_stack([np.cos(_ra2) * np.cos(_de2), np.sin(_ra2) * np.cos(_de2), np.sin(_de2)]) @ ph.EQ_TO_ECL.T


def xnav_ely_sigma(area_cm2, t_int_h):
    """σ [km] of each "best 4" pulsar for the given detector and integration time (Ely 2022, eq. 72)."""
    return XNAV_ELY_S / np.sqrt(max(area_cm2, 1e-6) * max(t_int_h, 1e-6))


def sep_profile(G, jd0, t0, t1, step=ph.DAY):
    """Sun–Earth–probe angle (SEP) [deg] along the true trajectory, every `step` seconds."""
    ts = np.arange(t0, t1, step)
    X = G(ts)
    sep = np.array([bc.sep_angle_deg(x_[:3], ph.planet_position("earth", jd0 + t_ / ph.DAY)) for t_, x_ in zip(ts, X)])
    return ts, sep


def gap_intervals(days, mask):
    """Contiguous intervals [start_day, end_day] where mask is true."""
    out, a = [], None
    for d_, m_ in zip(days, mask):
        if m_ and a is None:
            a = d_
        if not m_ and a is not None:
            out.append((float(a), float(d_)))
            a = None
    if a is not None:
        out.append((float(a), float(days[-1])))
    return out


TARGET_LEAD_DAYS = 5.0


def mars_bplane(jd, x):
    """Mars B-plane from the osculating elements of the Mars-relative hyperbola (interplanetary navigation
    standard): B·T, B·R [km], time to periapsis [s], v∞ [km/s] and periapsis radius [km].
    T̂ = Ŝ × ecliptic pole (normalized), R̂ = Ŝ × T̂."""
    mu = ph.MU_PLANETS["mars"]
    rM, vM = ph.planet_state("mars", jd)
    r, v = x[:3] - rM, x[3:6] - vM
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    evec = np.cross(v, h) / mu - r / rn
    e = np.linalg.norm(evec)
    eps = vn**2 / 2 - mu / rn
    vinf = np.sqrt(max(2 * eps, 1e-12))
    eh, hh = evec / e, h / hn
    S = eh / e + np.sqrt(max(e**2 - 1, 0.0)) / e * np.cross(hh, eh)       # incoming asymptote direction
    B = (hn / vinf) * np.cross(S, hh)
    T = ph.unit(np.cross(S, [0, 0, 1.0]))
    Rv = np.cross(S, T)
    a = -mu / (2 * eps)
    coshF = (1 - rn / a) / e                                             # hyperbolic anomaly
    F = np.arccosh(max(coshF, 1.0)) * (-1.0 if r @ v < 0 else 1.0)
    tp = -(e * np.sinh(F) - F) * np.sqrt(-a**3 / mu)                    # time to periapsis (>0 before)
    rp = -a * (e - 1)
    return np.array([B @ T, B @ Rv, tp]), dict(vinf=float(vinf), rp=float(rp), e=float(e))


def make_flyer(jd0, tof_days, ep_key):
    """Nominal propagator (Sun, planets, SRP, relativity and nominal thrust in arcs) up to the targeting point."""
    dyn = ph.Dynamics(jd0)
    dyn.tabulate(0.0, tof_days * ph.DAY)
    t_c = (tof_days - TARGET_LEAD_DAYS) * ph.DAY
    arcs_ = []
    if ep_key is not None:
        A_, arcs_ = ep_key[0] * 1e-6, [(a * ph.DAY, b * ph.DAY) for a, b in ep_key[1]]

        def thr(t, xx):
            if not any(a <= t < b for a, b in arcs_):
                return 0.0
            v_ = xx[..., 3:6]
            return A_ * v_ / np.linalg.norm(v_, axis=-1, keepdims=True)
        dyn.thrust = thr

    def fly(X, t0):
        cuts = sorted({c for ab in arcs_ for c in ab if t0 < c < t_c})
        t_ = t0
        for c in cuts + [t_c]:
            X = dyn.propagate(X, t_, c - t_, max_step=6 * 3600.0)
            t_ = c
        return X
    return fly, t_c


def b_target(rp_km, vinf):
    """|B| of a hyperbola with periapsis rp and hyperbolic excess velocity v∞ (Mars)."""
    mu = ph.MU_PLANETS["mars"]
    return rp_km * np.sqrt(1 + 2 * mu / (rp_km * vinf**2))


def guided_dv(fly, t_c, jd_c, x, t0, rp_km, iters=3, P6=None):
    """ΔV [km/s] that takes state x (at epoch t0) to the target (periapsis rp_km: B·T = B(rp, v∞), B·R = 0,
    arrival on the planned day). With P6, also returns the ΔV covariance due to the knowledge error
    (linear mapping: ΔV = −J_v⁻¹·J_x·δx)."""
    dv_tot = np.zeros(3)
    J = None
    for _ in range(iters):
        h = 1e-6
        X = np.tile(x, (4, 1))
        X[:, 3:6] += dv_tot
        X[1:, 3:6] += np.eye(3) * h
        Y = fly(X, t0)
        G = []
        for y in Y:
            bp, inf = mars_bplane(jd_c, y)
            G.append([bp[0], bp[1], (bp[2] - TARGET_LEAD_DAYS * ph.DAY) * inf["vinf"]])
        G = np.array(G)
        vinf0 = mars_bplane(jd_c, Y[0])[1]["vinf"]
        res = G[0] - np.array([b_target(rp_km, vinf0), 0.0, 0.0])
        J = (G[1:] - G[0]).T / h
        step = np.linalg.solve(J, -res)
        dv_tot = dv_tot + step
        if np.linalg.norm(step) < 1e-8:
            break
    if P6 is None:
        return dv_tot
    # sensitivity to position (1 km) to map the knowledge error into the ΔV
    X = np.tile(x, (4, 1))
    X[:, 3:6] += dv_tot
    X[1:, :3] += np.eye(3) * 1.0
    Y = fly(X, t0)
    G = []
    for y in Y:
        bp, inf = mars_bplane(jd_c, y)
        G.append([bp[0], bp[1], (bp[2] - TARGET_LEAD_DAYS * ph.DAY) * inf["vinf"]])
    G = np.array(G)
    Jr = (G[1:] - G[0]).T / 1.0
    M = -np.linalg.solve(J, np.hstack([Jr, J]))                 # δΔV = M·δx
    return dv_tot, M @ P6 @ M.T


@lru_cache(maxsize=8)
def targeted_departure(departure, tof_days, start_day, rp_alt_km, ephem, ep_key, v1_guess, dep_alt_km=None):
    """Departure velocity that, with the full dynamics (Sun + planets, and the nominal thrust if any),
    delivers the probe to a periapsis of altitude rp_alt_km at Mars on day tof_days (B·T = target B, B·R = 0).
    Differential correction (Newton with finite differences) starting from the Lambert solution; the target is evaluated
    TARGET_LEAD_DAYS before arrival, with the Mars-relative osculating elements. Cached on disk."""
    import hashlib
    from pathlib import Path
    ph.set_ephemeris(ephem)
    key = hashlib.md5(repr((departure, tof_days, start_day, rp_alt_km, ephem, ep_key, dep_alt_km)).encode()).hexdigest()[:16]
    cache = Path.home() / ".cache" / "cmbsim" / f"alvo3_{key}.npz"
    if cache.exists():
        d = np.load(cache)
        return tuple(d["v1"]), dict(B_target_km=float(d["Bt"]), miss_km=float(d["miss"]), dv_target_mps=float(d["dv"]))
    jd0 = ph.julian_date(*departure)
    r0, vE0 = ph.planet_state("earth_c" if dep_alt_km is not None else "earth", jd0)
    t_s, t_c = start_day * ph.DAY, (tof_days - TARGET_LEAD_DAYS) * ph.DAY
    leg1 = ph.Dynamics(jd0, planets=("venus", "jupiter", "saturn"))
    dyn = ph.Dynamics(jd0)
    dyn.tabulate(0.0, tof_days * ph.DAY)
    if ep_key is not None:
        A_, arcs_ = ep_key[0] * 1e-6, [(a * ph.DAY, b * ph.DAY) for a, b in ep_key[1]]

        def thr(t, xx):
            if not any(a <= t < b for a, b in arcs_):
                return 0.0
            v_ = xx[..., 3:6]
            return A_ * v_ / np.linalg.norm(v_, axis=-1, keepdims=True)
        dyn.thrust = thr
        cuts = sorted({c for ab in arcs_ for c in ab if t_s < c < t_c})
    else:
        cuts = []
    jd_c = jd0 + t_c / ph.DAY

    def fly(V):
        if dep_alt_km is not None:                    # perigee of the departure hyperbola, Earth and Moon
            X = np.array([np.concatenate(departure_state(jd0, v_ - vE0, dep_alt_km)) for v_ in V])
            X = departure_leg(jd0, X, t_s)
        else:
            X = np.column_stack([np.tile(r0, (len(V), 1)), V])
            X = leg1.propagate(X, 0.0, t_s, max_step=600.0)
        t_ = t_s
        for c in cuts + [t_c]:                       # the steps do not cross the start/end of thrust arcs
            X = dyn.propagate(X, t_, c - t_, max_step=6 * 3600.0)
            t_ = c
        return X

    v = np.array(v1_guess, float)
    rp = R_MARS_KM + rp_alt_km
    mu = ph.MU_PLANETS["mars"]
    Bt = None
    for it in range(30):
        dv = 1e-6
        V = np.vstack([v, v + np.eye(3) * dv])
        X = fly(V)
        G = []
        for x_ in X:
            bp, inf = mars_bplane(jd_c, x_)
            G.append(np.array([bp[0], bp[1], (bp[2] - TARGET_LEAD_DAYS * ph.DAY) * inf["vinf"]]))
        G = np.array(G)
        Bt = b_target(rp, mars_bplane(jd_c, X[0])[1]["vinf"])
        res = G[0] - np.array([Bt, 0.0, 0.0])
        J = (G[1:] - G[0]).T / dv
        step = np.linalg.solve(J, -res)
        # limited step (the 1st guess may miss Mars by millions of km)
        lim = 0.05
        if np.linalg.norm(step) > lim:
            step *= lim / np.linalg.norm(step)
        v = v + step
        if np.linalg.norm(res) < 1.0 and np.linalg.norm(step) < 1e-9:
            break
    miss = float(np.linalg.norm(res))
    if not np.isfinite(miss) or miss > 1e3:
        return None, {}
    dv_t = float(np.linalg.norm(v - np.array(v1_guess)) * 1e3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, v1=v, Bt=Bt, miss=miss, dv=dv_t)
    return tuple(v), dict(B_target_km=float(Bt), miss_km=miss, dv_target_mps=dv_t)


def truth_grid(dyn, x0, t0, t1, dt, maneuvers, accel=None, seg=None, acc_t0=None):
    """Truth: DOP853 with piecewise dense output (impulses), sampled on a grid.

    accel: unmodeled acceleration sampled every `seg` seconds, (N, 3) in km/s² (spline-interpolated)."""
    from scipy.integrate import solve_ivp
    from scipy.interpolate import CubicSpline
    n = int(np.ceil((t1 - t0) / dt))
    t = t0 + dt * np.arange(n + 1)
    X = np.empty((n + 1, 6))
    imp = {}
    for tm, dv in maneuvers:
        imp[tm] = imp.get(tm, np.zeros(3)) + dv
    cuts = sorted(set([t0, t[-1]] + [c for c in imp if t0 < c < t[-1]]))
    a0_ = t0 if acc_t0 is None else acc_t0
    acc_fn = CubicSpline(a0_ + seg * np.arange(len(accel)), accel) if accel is not None else None

    def rhs(tt, xx):
        if acc_fn is not None:
            dyn.a_extra = acc_fn(tt)
        return dyn.deriv(tt, xx)
    x = np.asarray(x0, float).copy()
    for a, b in zip(cuts[:-1], cuts[1:]):
        x = x.copy(); x[3:] += imp.get(a, 0.0)
        sol = solve_ivp(rhs, (a, b), x, method="DOP853", rtol=1e-10, atol=1e-6, dense_output=True)
        m = (t >= a) & (t <= b)
        X[m] = sol.sol(t[m]).T
        x = sol.y[:, -1]
    dyn.a_extra = None
    return t, X, None


class Grid:
    """Fast linear interpolation on a uniform grid."""

    def __init__(self, t, X):
        self.t0, self.dt, self.X = t[0], t[1] - t[0], X

    def __call__(self, tq):
        tq = np.atleast_1d(np.asarray(tq, float))
        u = (tq - self.t0) / self.dt
        i = np.clip(u.astype(int), 0, len(self.X) - 2)
        w = (u - i)[:, None]
        return self.X[i] * (1 - w) + self.X[i + 1] * w


# ----------------------------------------------------------------------------- geometry
RING_M = 24                          # phases used to compute the ring harmonics


class Stare:
    """4 staring sensors (tetrahedron): each measurement is the window average in 4 fixed directions."""
    spin = False
    labels = ["Sensor 1 (axis)", "Sensor 2", "Sensor 3", "Sensor 4"]

    def __init__(self, axis, sigma_T, Cb, roll=0.0):
        self.axis = ph.unit(axis)
        self.roll = roll
        self.n = ph.tetra_dirs(axis, roll)
        self.R = np.eye(4) * sigma_T**2
        self.B = np.eye(4)
        self.Cb = Cb
        self.C0 = at.mount(self.axis, roll)
        self.dc = np.arange(4)                 # channels that carry the 1/f drift (sensor of each channel)
        self.dc_sensor = np.arange(4)
        self.ch_sensor = np.arange(4)          # sensor of each channel (for the gain)
        self.dc_bias, self.dc_bias_ch = np.arange(4), np.arange(4)

    lin_true = lin_filt = None          # linearized temperature (truth / on-board model)

    def model(self, v, T0, truth=False):
        """v: (S,3) velocity relative to the CMB; T0: (S,). Returns (S,4)."""
        T = T0[:, None] * ph.dipole_factor(v[:, None, :], self.n[None])
        f = self.lin_true if truth else self.lin_filt
        return f(T) if f is not None else T

    def model_att(self, v, T0, dI, db, truth=False):
        """Model with attitude error: dI (S,3) in the inertial frame, db (S,3) in the body frame [rad]."""
        dth = dI + db @ self.C0.T
        return self.model(v + np.cross(dth, v), T0, truth)

    def sky_dirs(self):
        return self.n

    def channel_sun_angles(self, sun):
        return np.degrees(np.arccos(np.clip(self.n @ sun, -1, 1)))


class Spin:
    """Axial spin about the anti-Sun axis. Sensor 1 lies on the axis (constant signal); the
    sensors 2-4 (at 109.5° from the axis) scan the same sky ring. Each window is demodulated at the
    spin frequency: mean (a0) and 1st harmonic (a1, b1) of the ring, per sensor."""
    spin = True
    labels = ["Sensor 1 (axis)", "Ring: mean a0", "Ring: harmonic a1", "Ring: harmonic b1"]

    def __init__(self, axis, sigma_T, Cb, roll=0.0):
        self.axis = ph.unit(axis)
        self.roll = 0.0
        self.dc = np.array([0, 1, 4, 7])       # axis + ring mean of each sensor (DC channels)
        self.dc_sensor = np.array([0, 1, 2, 3])
        self.ch_sensor = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        self.dc_bias, self.dc_bias_ch = np.array([0]), np.array([0])   # gain as a bias only in the axis sensor
        e1, e2 = ph.perpendicular_basis(self.axis)
        th = np.arccos(-1 / 3)
        self.psi = 2 * np.pi * np.arange(RING_M) / RING_M
        self.ring = (np.cos(th) * self.axis[None] +
                     np.sin(th) * (np.cos(self.psi)[:, None] * e1 + np.sin(self.psi)[:, None] * e2))
        # noise: a0 ~ σ², a1 and b1 ~ 2σ² (sine fit over an integer number of turns)
        self.R = np.diag([sigma_T**2] + [sigma_T**2, 2 * sigma_T**2, 2 * sigma_T**2] * 3)
        self.B = np.zeros((10, 4))
        self.B[0, 0] = 1
        for j in range(3):
            self.B[1 + 3 * j:4 + 3 * j, 1:4] = np.eye(3)
        self.Cb = Cb
        self.Cs = np.array([at.mount(self.axis, p) for p in self.psi])     # body->inertial per phase

    lin_true = lin_filt = None          # linearized temperature (truth / on-board model)

    def _lin(self, T, truth):
        f = self.lin_true if truth else self.lin_filt
        return f(T) if f is not None else T

    def model_att(self, v, T0, dI, db, truth=False):
        """Attitude error with spin: the body-frame error (db) rotates with the spacecraft."""
        dth = dI[:, None, :] + np.einsum("mij,sj->smi", self.Cs, db)        # (S, M, 3)
        vk = v[:, None, :] + np.cross(dth, v[:, None, :])
        D = self._lin(T0[:, None] * ph.dipole_factor(vk, self.ring[None]), truth)            # (S, M)
        h = np.stack([D.mean(1), 2 * (D * np.cos(self.psi)).mean(1), 2 * (D * np.sin(self.psi)).mean(1)], 1)
        ax = self._lin(T0[:, None] * ph.dipole_factor(vk, self.axis[None, None]), truth).mean(1)
        return np.hstack([ax[:, None], h, h, h])

    def ring_harmonics(self, v, T0, truth=False):
        D = self._lin(T0[:, None] * ph.dipole_factor(v[:, None, :], self.ring[None]), truth)        # (S, M)
        return np.stack([D.mean(1), 2 * (D * np.cos(self.psi)).mean(1),
                         2 * (D * np.sin(self.psi)).mean(1)], 1)

    def model(self, v, T0, truth=False):
        ax = self._lin(T0 * ph.dipole_factor(v, self.axis), truth)
        h = self.ring_harmonics(v, T0, truth)
        return np.hstack([ax[:, None], h, h, h])

    def sky_dirs(self):
        return np.vstack([self.axis[None], self.ring])

    def channel_sun_angles(self, sun):
        ax = np.degrees(np.arccos(np.clip(self.axis @ sun, -1, 1)))
        rg = np.degrees(np.arccos(np.clip(self.ring @ sun, -1, 1))).min()
        return np.array([ax] + [rg] * 9)


@lru_cache(maxsize=32)
def sky_bias_cov(spin, fwhm_deg, subtract):
    """Covariance (4x4) of the sky biases for the chosen geometry (rotation invariant)."""
    th = np.arccos(-1 / 3)
    if subtract:
        s = sky.planck_residual_rms(fwhm_deg)
        if not spin:
            return np.eye(4) * s**2
        nb = 2 * np.pi * np.sin(th) / np.deg2rad(fwhm_deg)       # independent beams on the ring
        return np.diag([s**2, s**2 / nb, 2 * s**2 / nb, 2 * s**2 / nb])
    axis = np.array([0.0, 0.0, 1.0])
    if not spin:
        return sky.covariance(ph.tetra_dirs(axis), fwhm_deg)
    K = 72
    psi = 2 * np.pi * np.arange(K) / K
    ring = np.column_stack([np.sin(th) * np.cos(psi), np.sin(th) * np.sin(psi), np.full(K, np.cos(th))])
    C = sky.covariance(np.vstack([axis, ring]), fwhm_deg)
    W = np.zeros((4, K + 1))
    W[0, 0] = 1
    W[1, 1:] = 1 / K
    W[2, 1:] = 2 * np.cos(psi) / K
    W[3, 1:] = 2 * np.sin(psi) / K
    return W @ C @ W.T


def num_jacobian(geom, v, T0):
    """Numerical Jacobian of the measurement model with respect to [v(3), T0]."""
    x0 = np.append(v, T0)
    steps = np.array([1e-3, 1e-3, 1e-3, 1e-7])
    X = np.tile(x0, (9, 1))
    for i in range(4):
        X[1 + 2 * i, i] += steps[i]
        X[2 + 2 * i, i] -= steps[i]
    Z = geom.model(X[:, :3], X[:, 3])
    J = np.column_stack([(Z[1 + 2 * i] - Z[2 + 2 * i]) / (2 * steps[i]) for i in range(4)])
    return Z[0], J


def snapshot_velocity(z, geom, v_guess, iters=3):
    """Weighted least squares for [v, T0] from ONE measurement (no filter)."""
    x = np.append(v_guess, ph.T0_CMB)
    Wd = 1 / np.diag(geom.R)
    for _ in range(iters):
        zm, J = num_jacobian(geom, x[:3], x[3])
        A = J.T @ (J * Wd[:, None])
        x = x + np.linalg.solve(A, J.T @ ((z - zm) * Wd))
    return x[:3]


def att_jacobians(geom, v, T0, h=2e-6):
    """∂z/∂δθ (inertial) and ∂z/∂δθ (body), (m,3) each, by central differences."""
    E = np.vstack([np.eye(3), -np.eye(3)]) * h
    Z0 = np.zeros((6, 3))
    VV = np.tile(v, (12, 1))
    TT = np.full(12, T0)
    Z = geom.model_att(VV, TT, np.vstack([E, Z0]), np.vstack([Z0, E]))
    JI = (Z[0:3] - Z[3:6]).T / (2 * h)
    Jb = (Z[6:9] - Z[9:12]).T / (2 * h)
    return JI, Jb


def snapshot_gain(geom, v):
    """Least-squares gain of 1 measurement: δv [km/s] = G δz."""
    _, J = num_jacobian(geom, v, ph.T0_CMB)
    W = np.diag(1 / np.diag(geom.R))
    return (np.linalg.inv(J.T @ W @ J) @ J.T @ W)[:3]


def snapshot_budget(geom, v):
    """Velocity σ of 1 measurement (noise) and velocity bias caused by the sky [m/s]."""
    _, J = num_jacobian(geom, v, ph.T0_CMB)
    W = np.diag(1 / np.diag(geom.R))
    A = np.linalg.inv(J.T @ W @ J)
    cov_noise = A[:3, :3]
    G = (A @ J.T @ W)[:3]
    cov_bias = G @ geom.B @ geom.Cb @ geom.B.T @ G.T
    return float(np.sqrt(np.trace(cov_noise)) * 1e3), float(np.sqrt(np.trace(cov_bias)) * 1e3)


# ----------------------------------------------------------------------------- simulation
def sky_two(cfg):
    """'two_freq' argument of skymaps.get_sky: None, one extra band or a tuple of extra bands (hashable)."""
    eb = cfg.extra_bands()
    return None if not eb else (eb[0] if len(eb) == 1 else eb)


def band_weights(cfg):
    """Bands (ν, *extras) and on-board combination weights; returns (bands, weights, nulled components)."""
    bands = (cfg.freq(),) + cfg.extra_bands()
    if len(bands) == 1:
        return bands, np.array([1.0]), ()
    method = cfg.sep_method
    if method == "ilc":
        if not skymaps.available():
            method = "dust"
        else:
            tech = cfg.instrument()
            fg = skymaps.fg_model_cov(bands, cfg.fg_model, fwhm_deg=max(float(tech.fwhm_deg), 0.5), co_on=bool(cfg.co_on))
            s2 = len(bands) * float(tech.sigma_T(cfg.tau())) ** 2      # white noise per band in one window
            w, null = skymaps.nfreq_weights(bands, "ilc", noise_rel=np.full(len(bands), s2), fg_cov=fg)
            return bands, w, null
    w, null = skymaps.nfreq_weights(bands, method)
    return bands, w, null



def vnc_basis(r, v):
    """VNC basis (columns: v̂, n̂ = (r×v)/|r×v|, ĉ = v̂×n̂) for states (..., 3)."""
    vh = v / np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.cross(r, v)
    nh = n / np.linalg.norm(n, axis=-1, keepdims=True)
    ch = np.cross(vh, nh)
    return np.stack([vh, nh, ch], axis=-1)


def bplane_metrics(jd, x_true, x_est, P6, rp_target=None, t_left_s=None):
    """Knowledge error at Mars arrival, in the osculating B-plane (estimate − truth), and filter σ
    (numerical Jacobian); also the dispersion of the truth relative to the target (without guided correction
    maneuvers) and the Mars position difference between the Standish elements and DE421."""
    bt, it = mars_bplane(jd, x_true)
    be, _ = mars_bplane(jd, x_est)
    B_target = b_target(rp_target, it["vinf"]) if rp_target is not None else None
    J = np.zeros((3, 6))
    for k in range(6):
        h = 1.0 if k < 3 else 1e-6
        xp = x_true.copy(); xp[k] += h
        J[:, k] = (mars_bplane(jd, xp)[0] - bt) / h
    d = be - bt
    Pb = J @ P6 @ J.T
    src = ph.EPHEM["source"]
    ph.set_ephemeris("kepler"); rK = ph.planet_position("mars", jd)
    ph.set_ephemeris("de421"); rD = ph.planet_position("mars", jd)
    ph.set_ephemeris(src)
    rM, _ = ph.planet_state("mars", jd)
    out = dict(dBT_km=float(d[0]), dBR_km=float(d[1]), dt_s=float(d[2]), sig_BT_km=float(np.sqrt(Pb[0, 0])),
               sig_BR_km=float(np.sqrt(Pb[1, 1])), sig_t_s=float(np.sqrt(Pb[2, 2])), vinf_kms=it["vinf"],
               rp_true_km=it["rp"], dist_km=float(np.linalg.norm(x_true[:3] - rM)),
               BT_true_km=float(bt[0]), BR_true_km=float(bt[1]), B_target_km=B_target,
               mars_ephem_err_km=float(np.linalg.norm(rK - rD)) if ph.de421_available() else None)
    if B_target is not None and t_left_s is not None:
        out["deliv_BT_km"], out["deliv_BR_km"] = float(bt[0] - B_target), float(bt[1])
        out["deliv_t_s"] = float(bt[2] - t_left_s)
    return out


def check_components(cfg: Config):
    """Components requested in cfg that cannot run in this environment (missing package or data file)."""
    missing = []
    if cfg.ephem == "de421" and not ph.de421_available():
        missing.append("ephem='de421' needs cmbsim/data/de421_2020_2040.npz (would fall back to Kepler elements)")
    if cfg.cmb_on and cfg.sky_real and not skymaps.available():
        missing.append("sky_real=True needs healpy and cmbsim/data/sky_n64.npz (would fall back to the statistical sky)")
    if cfg.cmb_on and cfg.sky_real and cfg.co_on and not skymaps.co_available():
        missing.append("co_on=True needs cmbsim/data/co_planck_n64.npz (CO lines would be omitted)")
    if cfg.cmb_on and cfg.zodi_on and not zodimod.available():
        missing.append("zodi_on=True needs zodipy, healpy and astropy (zodiacal emission would be omitted)")
    if cfg.cmb_on and cfg.sep_method == "ilc" and len(cfg.extra_bands()) > 0 and not skymaps.available():
        missing.append("sep_method='ilc' needs healpy and the sky maps (would fall back to 'dust')")
    return missing


def run(cfg: Config, progress=None) -> Results:
    missing = check_components(cfg)
    if missing:
        msg = "Requested components unavailable:\n  - " + "\n  - ".join(missing)
        if not cfg.allow_fallback:
            raise RuntimeError(msg + "\nInstall requirements.txt, or pass allow_fallback=True to run the reduced model.")
        import warnings
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    rng = np.random.default_rng(cfg.seed)
    tech = cfg.instrument()
    jd0, r0, v0, traj_info = reference_transfer(cfg)
    dyn = ph.Dynamics(jd0)
    dyn.tabulate(0.0, cfg.tof_days * ph.DAY)
    t_start = cfg.start_day * ph.DAY
    t_end = (cfg.end_day if cfg.end_day else cfg.tof_days - 5.0) * ph.DAY

    # true state at cruise start (two-body up to t_start, then full dynamics)
    x = np.concatenate([r0, v0])
    if cfg.injection_err_mps > 0:                    # injection dispersion (the DSN calibrates the state on day 5)
        x[3:6] += np.random.default_rng(cfg.seed + 2719).normal(0, cfg.injection_err_mps * 1e-3, 3)
    if traj_info.get("departure") == "hyperbola":
        # departure: from the perigee of the hyperbola (300 km) with the Sun, Earth, Moon and planets to cruise start
        x = departure_leg(jd0, x, t_start)[0]
    else:
        # (up to t_start without Earth and Mars: the state starts at the Earth center; the departure is not modeled)
        x = ph.Dynamics(jd0, planets=("venus", "jupiter", "saturn")).propagate(x, 0.0, t_start, max_step=600.0)
    dtg = 600.0
    man_true = []
    for d in cfg.maneuvers_days:
        tm = round(d * ph.DAY / dtg) * dtg
        if t_start < tm < t_end:
            man_true.append((tm, rng.normal(0, cfg.maneuver_err_mps * 1e-3, 3)))
    # dynamic model errors present only in the truth (own generator: does not change the other draws)
    rng_d = np.random.default_rng(cfg.seed + 8887)
    srp_true = rng_d.normal(0, cfg.srp_err_pct / 100) if cfg.dyn_mismatch else 0.0
    dyn_true = ph.Dynamics(jd0, srp_scale=1.0 + srp_true)
    dyn_true.tabulate(0.0, cfg.tof_days * ph.DAY)
    des_true = []
    if cfg.dyn_mismatch and cfg.desat_days > 0:
        for td in np.arange(t_start + cfg.desat_days * ph.DAY, t_end, cfg.desat_days * ph.DAY):
            td = round(td / dtg) * dtg
            if all(abs(td - tm) > ph.DAY for tm, _ in man_true):
                des_true.append((td, rng_d.normal(0, cfg.desat_dv_mmps * 1e-6, 3)))
    # anomalies with unknown ΔV (own generator): the truth receives the impulse; the filter only knows the time
    rng_an = np.random.default_rng(cfg.seed + 7717)
    ano_true = []
    for d in cfg.anomaly_days or ():
        ta = round(float(d) * ph.DAY / dtg) * dtg
        if t_start < ta < t_end:
            ano_true.append((ta, rng_an.normal(0, cfg.anomaly_dv_mps * 1e-3, 3)))
    imp_true = des_true + ano_true             # small impulses of the truth (desaturations and anomalies)
    imp_map = {}
    for t_, dv_ in imp_true:
        imp_map[t_] = imp_map.get(t_, np.zeros(3)) + dv_
    rng_imu = np.random.default_rng(cfg.seed + 7723)

    def imu_meas(dv):
        """ΔV measured by the IMU: true + error (floor + scale factor); also returns the σ per axis [km/s]."""
        s_ = float(np.hypot(cfg.imu_dv_err_mmps * 1e-6, cfg.imu_scale_pct / 100 * np.linalg.norm(dv)))
        return dv + rng_imu.normal(0, s_, 3), s_
    acc_seg = acc_true = None
    sig_at = cfg.sigma_acel if cfg.sigma_acel_true is None else cfg.sigma_acel_true
    if cfg.dyn_mismatch and sig_at > 0:
        tau_a = max(cfg.accel_tau_h, 0.1) * 3600.0
        acc_seg = float(np.clip(round(tau_a / 4 / dtg) * dtg, dtg, 6 * 3600.0))
        nseg = int(np.ceil((t_end - t_start) / acc_seg)) + 1
        phi_a = np.exp(-acc_seg / tau_a)
        acc_true = np.empty((nseg, 3))
        rng_acc = np.random.default_rng(cfg.seed + 8893)       # own generator (independent of the desaturations)
        acc_true[0] = rng_acc.normal(0, sig_at, 3)
        for k_ in range(1, nseg):
            acc_true[k_] = phi_a * acc_true[k_ - 1] + np.sqrt(1 - phi_a**2) * rng_acc.normal(0, sig_at, 3)
    # electric propulsion: the filter knows the nominal profile; the truth has magnitude and pointing errors
    ep_A = cfg.ep_accel_mms2 * 1e-6 if cfg.ep_on else 0.0            # km/s²
    ep_arcs = [(a_ * ph.DAY, b_ * ph.DAY) for a_, b_ in cfg.ep_arcs] if cfg.ep_on else []
    ep_rec = None
    if ep_A > 0:
        from scipy.interpolate import CubicSpline
        rng_e = np.random.default_rng(cfg.seed + 9931)
        tau_e = max(cfg.ep_tau_h, 0.1) * 3600.0
        seg_e = float(np.clip(round(tau_e / 4 / dtg) * dtg, dtg, 6 * 3600.0))
        ne = int(np.ceil((t_end - t_start) / seg_e)) + 2
        phi_e = np.exp(-seg_e / tau_e)
        sig_e = np.array([cfg.ep_mag_err_pct / 100] + [np.deg2rad(cfg.ep_point_err_deg)] * 3)
        E_ = np.empty((ne, 4))
        E_[0] = rng_e.normal(0, sig_e)
        for k_ in range(1, ne):
            E_[k_] = phi_e * E_[k_ - 1] + np.sqrt(1 - phi_e**2) * rng_e.normal(0, sig_e)
        ep_err = CubicSpline(t_start + seg_e * np.arange(ne), E_)
        ep_bias = [np.concatenate([[rng_e.normal(0, cfg.ep_mag_bias_pct / 100)],
                                   rng_e.normal(0, np.deg2rad(cfg.ep_point_bias_deg), 3)]) for _ in ep_arcs]

        def ep_arc_of(t):
            for k_, (a_, b_) in enumerate(ep_arcs):
                if a_ <= t < b_:
                    return k_
            return None

        def ep_on_at(t):
            return any(a_ <= t < b_ for a_, b_ in ep_arcs)

        def thrust_nom(t, xx):
            # the filter switches thrust on/off at the arc start/end events (steps do not cross the boundary)
            if not hold.get("ep_flag", False):
                return 0.0
            v_ = xx[..., 3:6]
            return ep_A * v_ / np.linalg.norm(v_, axis=-1, keepdims=True)

        def thrust_true(t, xx):
            k_ = ep_arc_of(t)
            if k_ is None:
                return 0.0
            e_ = ep_err(t) + ep_bias[k_]
            # pointing error in the thrust frame (VNC: velocity, orbit normal, co-normal),
            # like the misalignment of a body-fixed thruster (previously applied in the inertial frame)
            Bv = vnc_basis(xx[..., :3], xx[..., 3:6])
            u_ = Bv[..., 0] + e_[1] * Bv[..., 1] + e_[2] * Bv[..., 2]
            u_ = u_ / np.linalg.norm(u_, axis=-1, keepdims=True)
            return ep_A * (1 + e_[0]) * u_
        dyn.thrust = thrust_nom
        dyn_true.thrust = thrust_true
        te_ = np.arange(t_start, t_end, 3600.0)
        ep_rec = dict(t_days=te_ / ph.DAY, err=ep_err(te_), on=np.array([ep_on_at(t_) for t_ in te_]),
                      A_mms2=cfg.ep_accel_mms2, arcs=list(cfg.ep_arcs), bias=[b_.tolist() for b_ in ep_bias])
    cut_imp = [(c_, np.zeros(3)) for ab in ep_arcs for c_ in ab if t_start < c_ < t_end]
    tg, Xg, _ = truth_grid(dyn_true, x, t_start, t_end, dtg, man_true + imp_true + cut_imp, acc_true, acc_seg)
    G = Grid(tg, Xg)
    if progress:
        progress(0.1, "true trajectory propagated")

    # fixed truths
    vssb_nom = ph.v_ssb()
    vssb_true = ph.v_ssb_random(rng)
    T0_true = ph.T0_CMB + rng.normal(0, ph.T0_CMB_SIG)
    tau = cfg.tau()
    nu = cfg.freq()
    sky = None
    extra_b = cfg.extra_bands()
    bands, sep_w, sep_null = band_weights(cfg)
    if cfg.cmb_on and cfg.sky_real and skymaps.available():
        sky = skymaps.get_sky(nu, float(tech.fwhm_deg), bool(cfg.subtract_planck), sky_two(cfg),
                              float(cfg.fg_amp_err_pct), int(cfg.seed), cfg.fg_model, cfg.cmb_sky,
                              tuple(float(x) for x in sep_w) if extra_b else None,
                              bool(cfg.ps_on), float(cfg.ps_cut_jy), float(cfg.ps_err_pct), bool(cfg.co_on),
                              float(cfg.co_err_pct), int(cfg.sky_nside))
    zd = None
    if cfg.cmb_on and cfg.zodi_on and zodimod.available():
        if progress:
            progress(0.02, "zodiacal emission along the trajectory (ZodiPy; first time ~15 s per band, then cached)")
        zpos, zdev = None, 0.0
        if cfg.zodi_real_traj:
            # true trajectory: the zodiacal is recomputed only if it deviates from the nominal one (> 1 million km),
            # e.g., with electric propulsion; positions rounded to 1 million km to reuse the cache
            zdays, zp_nom = zodimod.nominal_positions(jd0, cfg.tof_days)
            ts_ = np.clip(zdays * ph.DAY, tg[0], tg[-1])
            zp_true = np.where(((zdays * ph.DAY >= tg[0]) & (zdays * ph.DAY <= tg[-1]))[:, None], G(ts_)[:, :3], zp_nom)
            zdev = float(np.max(np.linalg.norm(zp_true - zp_nom, axis=1)))
            if zdev > 1e6:
                zpos = (zdays, zp_true)
        zd = zodimod.Zodi(nu, jd0, cfg.tof_days, bands=bands, weights=sep_w, positions=zpos)
        zd.traj_info = dict(real=zpos is not None, dev_km=zdev)
    rng_x = np.random.default_rng(cfg.seed + 5557)            # draws for the new effects (do not change the old ones)
    eps_z = rng_x.normal(0, cfg.zodi_err_pct / 100)            # amplitude error of the on-board zodiacal model
    est_zodi = bool(zd is not None and cfg.zodi_estimate)
    g0 = rng_x.normal(0, cfg.gain_cal_ppm * 1e-6, 4)            # gain calibration error per sensor
    sig_gd = cfg.gain_drift_ppm * 1e-6
    tau_g = max(cfg.gain_tau_h, 1e-3) * 3600.0
    gd = rng_x.normal(0, sig_gd, 4)
    T_ref = 0.0 if cfg.receiver == "total" else ph.T0_CMB
    load_on = bool(cfg.cmb_on and cfg.load_drift_nK > 0)
    sig_ld = cfg.load_drift_nK * 1e-9
    tau_ld = max(cfg.load_tau_h, 1e-3) * 3600.0
    rng_ld = np.random.default_rng(cfg.seed + 9091)
    sky_sky = cfg.receiver == "sky-sky"
    # intrinsic dipole: fixed sky pattern, indistinguishable from a solar velocity error (Quartin & Ferreira 2020)
    rng_i = np.random.default_rng(cfg.seed + 4441)
    d_int = ph.unit(rng_i.normal(0, 1, 3))
    v_int = d_int * cfg.intrinsic_dipole_uK * 1e-6 / ph.T0_CMB * ph.C_KMS     # equivalent velocity [km/s]
    est_gain = bool(cfg.gain_estimate and cfg.cmb_on)
    # linearized temperature: truth in band intensity; on board with the band center known to ±band_err
    lin_T = lin_F = None
    band_delta = 0.0
    if cfg.cmb_on and cfg.temp_linearized:
        band_delta = float(np.random.default_rng(cfg.seed + 6661).normal(0, cfg.band_err_pct / 100))

        def mk_lin(scale):
            fs_ = [(float(w_), ph.linearizer(float(b_) * scale, cfg.band_frac)) for b_, w_ in zip(bands, sep_w)]
            return lambda T: sum(w_ * f_(T) for w_, f_ in fs_)
        lin_T = mk_lin(1.0)
        lin_F = mk_lin(1.0 + band_delta) if cfg.lin_correct else None
    noise_factor = skymaps.combo_noise_factor(sep_w)
    # chopper: noise becomes white with the 1/f density at the modulation frequency, S(f_m) = 2NET²(1 + (f_k/f_m)^α),
    # times the modulation penalty; the slow detector drift vanishes (the load drift remains, not modeled)
    chop_on = bool(cfg.cmb_on and cfg.chop_hz > 0)
    mod_factor = 1.0
    if chop_on:
        alpha_c = float(cfg.alpha_1f) if cfg.alpha_1f is not None else tech.alpha_1f
        k_ = (cfg.f_knee() / cfg.chop_hz) ** alpha_c if (cfg.one_over_f and cfg.f_knee() > 0) else 0.0
        mod_factor = float(cfg.chop_penalty * np.sqrt(1.0 + k_))
    sigma_T = float(tech.sigma_T(tau)) * noise_factor * mod_factor
    Cb = sky_bias_cov(cfg.spin, tech.fwhm_deg, cfg.subtract_planck)
    Lb = np.linalg.cholesky(Cb + np.eye(4) * 1e-30)
    Geo = Spin if cfg.spin else Stare
    flick = None
    f_spin = cfg.spin_rate_dps / 360.0
    if cfg.cmb_on and cfg.one_over_f and cfg.f_knee() > 0 and not chop_on:
        alpha_ = float(cfg.alpha_1f) if cfg.alpha_1f is not None else tech.alpha_1f
        flick = Flicker(tech.net * noise_factor, cfg.f_knee(), alpha_, tau, 4,
                        np.random.default_rng(cfg.seed + 3331), f_lo=float(cfg.f_min_1f_hz))
    flick_c = None
    cfrac = float(np.clip(cfg.one_f_common, 0.0, 1.0))
    if flick is not None and cfrac > 0:
        flick_c = Flicker(tech.net * noise_factor, cfg.f_knee(), alpha_, tau, 1,
                          np.random.default_rng(cfg.seed + 3332), f_lo=float(cfg.f_min_1f_hz))
    Ccm = (1 - cfrac) * np.eye(4) + cfrac * np.ones((4, 4))     # 1/f correlation between sensors

    destripe_on = bool(cfg.spin and cfg.destripe and flick is not None)

    def apply_flicker(g):
        """1/f noise in R: DC channels get the extra white noise; with spin, a1/b1 only see S(f_spin)."""
        if flick is None:
            g.use = np.ones(len(g.R), bool)
            return g
        R = np.diag(g.R).copy()
        g.use = np.ones(len(R), bool)
        if cfg.spin:
            harm = np.setdiff1d(np.arange(len(R)), g.dc)
            R[harm] *= flick.spin_factor(f_spin)
            if not destripe_on:
                g.use[g.dc[1:]] = False            # ring means: unknown offset, discarded
        Rf = np.diag(R)
        # fast part of the 1/f (extra white noise per window) in the slow channels, with the common fraction
        Rf[np.ix_(g.dc, g.dc)] += flick.white_var * Ccm[np.ix_(g.dc_sensor, g.dc_sensor)]
        g.R = Rf
        return g

    def ring_harmonics_of(vals, psi):
        return np.array([vals.mean(), 2 * (vals * np.cos(psi)).mean(), 2 * (vals * np.sin(psi)).mean()])

    def sky_bias(g):
        """True bias and prior covariance from the maps (real sky)."""
        if not cfg.spin:
            return sky.value(g.n), np.diag(sky.prior_sigma(g.n) ** 2)
        K = 72
        psi = 2 * np.pi * np.arange(K) / K
        e1, e2 = ph.perpendicular_basis(g.axis)
        th = np.arccos(-1 / 3)
        ring = np.cos(th) * g.axis[None] + np.sin(th) * (np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2)
        v = sky.value(ring)
        a0, a1, b1 = ring_harmonics_of(v, psi)
        b_ax = sky.value(g.axis[None])[0]
        s2 = sky.prior_sigma(ring) ** 2
        nb = 2 * np.pi * np.sin(th) / np.deg2rad(max(tech.fwhm_deg, 0.5))
        Cb_ = np.diag([sky.prior_sigma(g.axis[None])[0] ** 2, s2.mean() / nb, 2 * s2.mean() / nb, 2 * s2.mean() / nb])
        return np.array([b_ax, a0, a1, b1]), Cb_

    att = None
    if cfg.cmb_on and cfg.att_on:
        st_ = at.StarTracker(centroid_px=cfg.st_centroid_px, frame_hz=cfg.st_frame_hz, heads=int(cfg.st_heads),
                             method=cfg.st_method, exposure_s=cfg.st_exposure_s,
                             boresight_deg=float(cfg.st_boresight_deg), catalog=cfg.st_catalog,
                             mag_limit=float(cfg.st_mag_limit) if cfg.st_catalog == "hipparcos" else min(6.0, float(cfg.st_mag_limit)),
                             ident=cfg.st_ident, excl_on=bool(cfg.st_excl), sun_excl_deg=float(cfg.st_sun_excl_deg),
                             earth_excl_deg=float(cfg.st_earth_excl_deg))
        # separate generator: enabling attitude does not change the other draws (fair comparison)
        att = at.AttitudeModel(st_, tau, np.random.default_rng(cfg.seed + 7919), cfg.att_misalign_arcsec,
                               cfg.att_drift_arcsec, cfg.att_drift_tau_h, cfg.spin, cfg.spin_rate_dps, gyro=cfg.gyro,
                               coast_max_arcsec=float(cfg.st_coast_max_arcsec))

    point_log = []
    tilt_on = bool(cfg.spin and cfg.spin_tilt_deg > 0 and cfg.prec_n >= 1)
    NPREC = int(cfg.prec_n) if tilt_on else 1
    PT = {}

    rng_bias = np.random.default_rng(cfg.seed + 1717)       # statistical-sky biases per pointing

    def new_pointing(t):
        xt = G([t])[0]
        roll = 0.0
        if sky is not None and not cfg.spin and cfg.gal_avoid:
            # free roll about the anti-Sun axis: choose the one that keeps the sensors farthest
            # from the foregrounds (by the on-board model), preserving the tetrahedron geometry
            rolls = np.deg2rad(np.arange(0, 120, 5))
            cost = [np.sum(sky.prior_sigma(ph.tetra_dirs(xt[:3], r_)) ** 2) for r_ in rolls]
            roll = float(rolls[int(np.argmin(cost))])
        if tilt_on:
            # spin axis tilted by α relative to anti-Sun, in N orientations on the cone (stepped
            # precession, as in WMAP/Planck): the harmonics then also measure the radial component
            a_ = ph.unit(xt[:3])
            e1_, e2_ = ph.perpendicular_basis(a_)
            al_ = np.deg2rad(cfg.spin_tilt_deg)
            gl_, bl_ = [], []
            for k_ in range(NPREC):
                ph_ = 2 * np.pi * k_ / NPREC
                ax_ = np.cos(al_) * a_ + np.sin(al_) * (np.cos(ph_) * e1_ + np.sin(ph_) * e2_)
                g_, b0_ = build_pointing(t, xt, ax_, 0.0)
                g_.k = k_
                gl_.append(g_); bl_.append(b0_)
            PT.update(glist=gl_, blist=bl_, kc=0)
            return gl_[0], bl_[0]
        g, b_ = build_pointing(t, xt, xt[:3], roll)
        g.k = 0
        PT.update(glist=[g], blist=[b_], kc=0)
        return g, b_

    def build_pointing(t, xt, axis_, roll):
        g = apply_flicker(Geo(axis_, sigma_T, Cb, roll))
        g.lin_true, g.lin_filt = lin_T, lin_F
        if sky is not None:
            b_, g.Cb = sky_bias(g)
            dirs_ = g.sky_dirs() if not cfg.spin else g.axis[None]
            lon_, lat_ = sky.gal_lonlat_deg(dirs_)
            point_log.append(dict(day=t / ph.DAY, roll=np.degrees(roll), lat=lat_.tolist(), lon=lon_.tolist(),
                                  axis=g.axis.tolist(), bias_uK=(b_ * 1e6).tolist(),
                                  sigma_uK=(np.sqrt(np.diag(g.Cb)) * 1e6).tolist()))
        else:
            b_ = Lb @ rng_bias.normal(0, 1, 4)
        # extra prior: zodiacal model (relative error) and gain calibration, in the bias space
        extra = np.zeros(4)
        if zd is not None and not est_zodi:
            extra += (cfg.zodi_err_pct / 100 * slow_dirs_values(g, lambda d_: zd.value(d_, t / ph.DAY))) ** 2
        lev = lever(g, xt[3:] + vssb_nom, ph.T0_CMB)
        if not est_gain:
            extra[g.dc_bias] += (cfg.gain_cal_ppm * 1e-6 * lev[g.dc_bias_ch]) ** 2
        g.Cb = g.Cb + np.diag(extra)
        if att is not None:
            att.set_pointing(axis_, roll)            # actual spin axis (with precession, that of the current orientation)
            g.Bp = np.linalg.pinv(g.B)
        return g, b_

    def slow_dirs_values(g, fn):
        """Values of a sky field (function of directions) projected onto the space of the 4 biases."""
        if not cfg.spin:
            return fn(g.n)
        K = 72
        psi = 2 * np.pi * np.arange(K) / K
        e1, e2 = ph.perpendicular_basis(g.axis)
        th = np.arccos(-1 / 3)
        ring = np.cos(th) * g.axis[None] + np.sin(th) * (np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2)
        vr = fn(ring)
        return np.concatenate([fn(g.axis[None])[:1], ring_harmonics_of(vr, psi)])

    def lever(g, v, T0_):
        """Gain lever arm in each channel: (T − T_ref) in the DC channels, the harmonic itself in the others."""
        zm = g.model(np.atleast_2d(v), np.array([T0_]))[0]
        L = zm.copy()
        L[g.dc] -= T_ref
        return L

    geom, bias_true = new_pointing(t_start)
    sig_v_snap, bias_v = snapshot_budget(geom, Xg[0, 3:] + vssb_nom)
    k_geo = sig_v_snap / (sigma_T * 1e6)
    att_info = {}
    if att is not None:
        v_ini = Xg[0, 3:] + vssb_nom
        JI, Jb = att_jacobians(geom, v_ini, ph.T0_CMB)
        Gs = snapshot_gain(geom, v_ini)
        cw = att.cov_white if att.cov_white is not None else np.full((3, 3), np.nan)
        cwb = geom.C0.T @ cw @ geom.C0 if not cfg.spin else cw
        pf_body = at.frame_cov(att.fields[0][2], att.sig_frame) if att.fields else np.full((3, 3), np.nan)
        att_info = dict(
            method=at.ALGORITHMS[cfg.st_method], heads=int(cfg.st_heads), n_stars=att.n_stars,
            avail=att.avail, frames=att.n_frames, smear_px=float(att.st.smear_px(cfg.spin_rate_dps)) if cfg.spin else 0.0,
            sig_frame_as=float(att.sig_frame / ph.ARCSEC),
            frame_sig_body_as=np.sqrt(np.diag(pf_body)) / ph.ARCSEC,
            win_sig_as=np.sqrt(np.diag(cwb)) / ph.ARCSEC,
            budget_white=float(np.sqrt(np.trace(Gs @ JI @ cw @ JI.T @ Gs.T)) * 1e3),
            budget_mis=float(np.sqrt(np.trace(Gs @ Jb @ Jb.T @ Gs.T)) * att.sig_mis * 1e3),
            budget_drift=float(np.sqrt(np.trace(Gs @ Jb @ Jb.T @ Gs.T)) * att.sig_lf * 1e3),
            mag_lim=float(att.mag_lim))

    def att_bias_cov(g, v):
        """Extra covariance of the bias states caused by misalignment + drift."""
        _, Jb_ = att_jacobians(g, v, ph.T0_CMB)
        M = g.Bp @ Jb_
        var_ = att.sig_lf**2 if est_att else att.var_body       # with attitude states, only the drift stays in the bias
        return M @ M.T * var_, M @ M.T * att.q_drift

    # ---- events ------------------------------------------------------------------------
    ev = []
    if cfg.cmb_on:
        ev += [(t, "cmb") for t in np.arange(t_start + tau / 2, t_end, tau)]
    if cfg.celenav:
        ev += [(t, "cel") for t in np.arange(t_start + cfg.celenav_interval_s, t_end,
                                             cfg.celenav_interval_s)]
    ev += [(tm, "man") for tm, _ in man_true]
    ev += [(td, "des") for td, _ in des_true]
    if cfg.cal_period_h > 0 and destripe_on:      # periodic offset calibration (internal load / diode)
        ev += [(t_, "cal") for t_ in np.arange(t_start + cfg.cal_period_h * 3600.0 + 7.0, t_end,
                                               cfg.cal_period_h * 3600.0)]
    ev += [(c_, "ep") for ab in ep_arcs for c_ in ab if t_start < c_ < t_end]
    if cfg.align_cal_h > 0 and cfg.cmb_on and cfg.att_on and cfg.att_estimate:
        # alignment calibration: a compact source (planet, radio source) crosses the CMB beams
        ev += [(t_, "alc") for t_ in np.arange(t_start + cfg.align_cal_h * 3600.0 + 11.0, t_end, cfg.align_cal_h * 3600.0)]
    # Sun–Earth–probe angle along the trajectory (once per day) and effective DSN gaps
    sep_t, sep_deg = sep_profile(G, jd0, t_start, t_end)
    gaps_d = [(float(a_), float(b_)) for a_, b_ in (cfg.dsn_gaps or ())]
    if cfg.dsn_sep_min_deg > 0:
        gaps_d += gap_intervals(sep_t / ph.DAY, sep_deg < cfg.dsn_sep_min_deg)

    def in_gap(t_):
        return any(a_ * ph.DAY <= t_ < b_ * ph.DAY for a_, b_ in gaps_d)
    n_dsn_gap = 0
    if cfg.dsn_pass_days and cfg.dsn_pass_days > 0:
        rng_dsn = np.random.default_rng(cfg.seed + 3343)        # (formerly + 3331, the same seed as the 1/f)
        for t0_ in np.arange(t_start + cfg.dsn_pass_days * ph.DAY, t_end, cfg.dsn_pass_days * ph.DAY):
            if rng_dsn.random() * 100 < cfg.dsn_loss_pct:
                continue                                       # pass cancelled
            if in_gap(t0_ + 1800.0 * max(1, int(round(cfg.dsn_pass_h)))):
                n_dsn_gap += 1                                 # tracking gap (conjunction, ground failure...)
                continue
            ev += [(t0_ + 3600.0 * (k_ + 0.5), "dsn") for k_ in range(max(1, int(round(cfg.dsn_pass_h))))]
            if cfg.dsn_ddor:
                ev += [(t0_ + 1800.0 * max(1, int(round(cfg.dsn_pass_h))) + 61.0, "ddor")]
    if cfg.xnav_on:
        ev += [(t_, "xnav") for t_ in np.arange(t_start + 17.0, t_end, cfg.xnav_period_h * 3600.0)]
    if cfg.opnav_days > 0:
        t_o0 = max(t_start, cfg.tof_days * ph.DAY - cfg.opnav_days * ph.DAY)
        ev += [(t_, "opn") for t_ in np.arange(t_o0 + 13.0, t_end, cfg.opnav_period_h * 3600.0)]
    ev += [(ta, "anom") for ta, _ in ano_true]
    if cfg.sdop_on:
        ev += [(t_, "sdop") for t_ in np.arange(t_start + 29.0, t_end, cfg.sdop_period_h * 3600.0)]
    if cfg.ast_on:
        ev += [(t_, "ast") for t_ in np.arange(t_start + 23.0, t_end, cfg.ast_period_d * ph.DAY)]
    if cfg.sund_on:
        ev += [(t_, "sund") for t_ in np.arange(t_start + 31.0, t_end, cfg.sund_period_h * 3600.0)]
    if cfg.repoint_days and cfg.repoint_days > 0:
        ev += [(t, "rep") for t in np.arange(t_start + cfg.repoint_days * ph.DAY, t_end,
                                             cfg.repoint_days * ph.DAY)]
    ev = [e_ for e_ in ev if e_[0] < t_end - 1.0]    # nothing after the end (truth would be extrapolated)
    ev.append((t_end - 1.0, "end"))                 # last step: the filter reaches the end (arrival evaluation)
    ev.sort(key=lambda e: (e[0], e[1] not in ("man", "des", "anom")))

    # ---- filter -------------------------------------------------------------------------
    est_srp = bool(cfg.srp_estimate)
    est_ep = bool(ep_A > 0 and cfg.ep_estimate)
    nx = 14 + (8 if est_gain else 0) + (1 if est_srp else 0)
    IC, ID = slice(14, 18), slice(18, 22)  # gain: calibration (constant) and drift (Gauss-Markov), per sensor
    IS = nx - 1                            # relative error of the radiation pressure
    if est_ep:
        nx += 3
    IE = slice(nx - 3, nx)                 # thrust bias (empirical acceleration, km/s², VNC components), per arc
    IHB0 = nx                              # sky biases (axis, a0, a1, b1) of orientations 2..N on the cone
    nx += 4 * (NPREC - 1)
    # destriping in the filter: the 1/f offset of each sensor becomes a Gauss-Markov state (common to all
    # axis orientations), with the slow-drift variance and the correlation time that reproduces the
    # increment over one window
    # 'exact': the offset of each sensor is the sum of the same Ornstein-Uhlenbeck components as the truth
    # (multiscale 1/f), one per state; 'simple': a single equivalent Gauss-Markov
    NK = 1e-9                                          # unit of the offset states: nK
    if destripe_on and cfg.destripe_model == "exact":
        c_var, c_tau = flick.var.copy(), flick.tc.copy()
    elif destripe_on:
        c_var = np.array([flick.drift_rms() ** 2])
        c_tau = np.array([2 * flick.drift_rms() ** 2 * tau / max(flick.increment_var(tau), 1e-40)])
    NBAND = len(c_var) if destripe_on else 0
    if destripe_on:
        IO = nx + np.arange(4 * NBAND)                 # index 4·c + s: component c, sensor s
        nx += 4 * NBAND
        o_var = np.repeat(c_var, 4) / NK**2           # states in nK (avoids P spanning 24 orders of magnitude)
        o_tau = np.repeat(c_tau, 4)
        Ccm_all = np.kron(np.eye(NBAND), Ccm)
    else:
        IO = None
    # reference load offset, per sensor (Gauss-Markov, in nK)
    est_load = bool(load_on and cfg.load_estimate)
    if est_load:
        ILd = nx + np.arange(4)
        nx += 4
    if est_zodi:
        IZ = nx
        nx += 1
    # solar Doppler bias (Gauss-Markov, km/s): before the attitude states, which come last
    est_sdop = bool(cfg.sdop_on)
    sig_sdb = cfg.sdop_bias_mps * 1e-3
    tau_sdb = max(cfg.sdop_tau_d, 1e-3) * ph.DAY
    if est_sdop:
        ISD = nx
        nx += 1
    # body-frame attitude error (star tracker × CMB sensors): constant misalignment + Gauss-Markov
    # thermal drift, as states (in ″, for numerical conditioning)
    est_att = bool(att is not None and cfg.att_estimate)
    est_drift = bool(est_att and cfg.att_est_drift)
    if est_att:
        IAm = nx + np.arange(3)
        IAd = nx + 3 + np.arange(3)
        nx += 6

    def bidx(k):
        return np.arange(7, 11) if k == 0 else IHB0 + 4 * (k - 1) + np.arange(4)
    sig_epb = ep_A * np.hypot(cfg.ep_mag_bias_pct / 100, np.deg2rad(cfg.ep_point_bias_deg)) if ep_A > 0 else 0.0
    s_r, s_v = (1.0, 1e-4) if cfg.dsn_calibration else (1e4, 1.0)
    x_init = np.concatenate([Xg[0, :3] + rng.normal(0, s_r, 3), Xg[0, 3:] + rng.normal(0, s_v, 3),
                             [ph.T0_CMB], np.zeros(4), np.zeros(3)])
    P0 = np.diag([s_r**2] * 3 + [s_v**2] * 3 + [ph.T0_CMB_SIG**2] + [0.0] * 4 + [cfg.vssb_prior_kms**2] * 3)
    if cfg.vssb_prior_kms <= 0.1:                 # Planck prior (anisotropic: ~110 m/s along the dipole)
        P0[11:14, 11:14] = ph.v_ssb_cov()
    P0[7:11, 7:11] = geom.Cb
    if flick is not None and not destripe_on:      # 1/f drift already present at the 1st pointing
        if cfg.spin:
            P0[7, 7] += flick.drift_rms() ** 2
        else:
            P0[7:11, 7:11] += flick.drift_rms() ** 2 * Ccm
    if est_gain:
        x_init = np.concatenate([x_init, np.zeros(8)])
        P0 = np.pad(P0, (0, 8))
        P0[IC, IC] = np.eye(4) * (cfg.gain_cal_ppm * 1e-6) ** 2
        P0[ID, ID] = np.eye(4) * sig_gd**2
    if est_srp:
        x_init = np.concatenate([x_init, [0.0]])
        P0 = np.pad(P0, (0, 1))
        P0[IS, IS] = (cfg.srp_err_pct / 100) ** 2
    if est_ep:
        x_init = np.concatenate([x_init, np.zeros(3)])
        P0 = np.pad(P0, (0, 3))
        P0[IE, IE] = np.eye(3) * sig_epb**2
    if NPREC > 1:
        x_init = np.concatenate([x_init, np.zeros(4 * (NPREC - 1))])
        P0 = np.pad(P0, (0, 4 * (NPREC - 1)))
        for k_ in range(1, NPREC):
            bk_ = bidx(k_)
            P0[np.ix_(bk_, bk_)] = PT["glist"][k_].Cb
    if destripe_on:
        x_init = np.concatenate([x_init, np.zeros(4 * NBAND)])
        P0 = np.pad(P0, (0, 4 * NBAND))
        P0[np.ix_(IO, IO)] = np.outer(np.sqrt(o_var), np.sqrt(o_var)) * Ccm_all + np.eye(4 * NBAND) * 1e-12
    if est_load:
        x_init = np.concatenate([x_init, np.zeros(4)])
        P0 = np.pad(P0, (0, 4))
        P0[ILd, ILd] = (sig_ld / NK) ** 2
    if est_zodi:
        x_init = np.concatenate([x_init, [0.0]])
        P0 = np.pad(P0, (0, 1))
        P0[IZ, IZ] = (cfg.zodi_err_pct / 100) ** 2
    if est_sdop:
        x_init = np.concatenate([x_init, [0.0]])
        P0 = np.pad(P0, (0, 1))
        P0[ISD, ISD] = max(sig_sdb, 1e-9) ** 2
    if est_att:
        x_init = np.concatenate([x_init, np.zeros(6)])
        P0 = np.pad(P0, (0, 6))
        P0[IAm, IAm] = max(cfg.att_misalign_arcsec, 1e-3) ** 2
        P0[IAd, IAd] = max(cfg.att_drift_arcsec, 1e-3) ** 2 if est_drift else 1e-12
    # ---- observability analysis (Gu et al. 2018; Fisher/Cramér-Rao) ------------------------
    st_labels = (["r_x", "r_y", "r_z", "v_x", "v_y", "v_z", "T0", "bias 1", "bias 2", "bias 3", "bias 4",
                  "δv_Sun x", "δv_Sun y", "δv_Sun z"]
                 + ([f"gain cal {j+1}" for j in range(4)] + [f"gain drift {j+1}" for j in range(4)] if est_gain else [])
                 + (["rad. pressure"] if est_srp else [])
                 + (["thrust x", "thrust y", "thrust z"] if est_ep else [])
                 + [f"bias {h_} orient. {k_ + 1}" for k_ in range(1, NPREC) for h_ in ("axis", "a0", "a1", "b1")]
                 + ([f"1/f offset s{j + 1} comp. {b_ + 1}" for b_ in range(NBAND) for j in range(4)] if destripe_on else [])
                 + ([f"load s{j + 1}" for j in range(4)] if est_load else [])
                 + (["zodiacal scale"] if est_zodi else [])
                 + (["solar Doppler bias"] if est_sdop else [])
                 + (["misalign. x", "misalign. y", "misalign. z", "att. drift x", "att. drift y", "att. drift z"]
                    if est_att else []))
    # reference scale: the filter prior, but position and velocity WITHOUT the DSN (10⁴ km, 1 km/s), to
    # measure what the measurements themselves determine
    S0 = np.sqrt(np.clip(np.diag(P0), 0, None))
    S0[:3], S0[3:6] = 1e4, 1.0
    obs = dict(J=np.zeros((nx, nx)), Phi=np.eye(nx), last_day=-1, days=[], delta=[], ratio=[],
               on=bool(cfg.obs_analysis))

    def obs_step(tk, meas):
        """Adds the information of 1 day of measurements, mapped to the initial state (no process noise).

        meas: list of (h(X) -> Z, R, number of measurements in the day). Jacobians by central differences around
        the current estimate, in coordinates normalized by the prior deviation (x̃ = x / σ0)."""
        live = S0 > 0
        n = int(live.sum())
        idx = np.where(live)[0]
        eps = 1e-3
        E = np.zeros((n, nx))
        E[np.arange(n), idx] = eps * S0[idx]
        Xp, Xm = f.x + E, f.x - E
        Jd = np.zeros((nx, nx))
        for hfun, R_, cnt in meas:
            Ht = np.zeros((len(R_), nx))
            Ht[:, idx] = ((hfun(Xp) - hfun(Xm)) / (2 * eps)).T
            Jd += cnt * Ht.T @ np.linalg.solve(R_, Ht)
        obs["J"] += obs["Phi"].T @ Jd @ obs["Phi"]
        keep = hold["xdyn"].copy()                         # fx modifies the "dynamics only" state: save and restore
        Fp, Fm = fx(Xp, tk, ph.DAY), fx(Xm, tk, ph.DAY)
        hold["xdyn"] = keep
        Pd = np.eye(nx)
        Pd[np.ix_(np.arange(nx), idx)] = ((Fp - Fm) / (2 * eps)).T / np.where(live, S0, 1.0)[:, None]
        obs["Phi"] = Pd @ obs["Phi"]
        Jl = obs["J"][np.ix_(idx, idx)]
        rl = np.sqrt(np.clip(np.diag(np.linalg.inv(np.eye(n) + Jl)), 0, None))
        inf_ = rl < 0.99                                    # states the measurements say something about
        w = np.linalg.eigvalsh(Jl[np.ix_(inf_, inf_)]) if inf_.sum() > 1 else np.array([0.0, 1.0])
        obs["days"].append(tk / ph.DAY)
        obs["delta"].append(float(np.sqrt(max(w[0], 0.0) / max(w[-1], 1e-300))))
        r_ = np.ones(nx)
        r_[idx] = rl
        obs["ratio"].append(r_)
    q_bias = np.zeros((4, 4))
    if att is not None and cfg.att_in_filter and not est_drift:
        Pb_att, q_bias = att_bias_cov(geom, Xg[0, 3:] + vssb_nom)
        P0[7:11, 7:11] += Pb_att
        for k_ in range(1, NPREC):
            bk_ = bidx(k_)
            P0[np.ix_(bk_, bk_)] += Pb_att

    def Qfun(dt):
        Q = np.zeros((nx, nx))
        # unmodeled acceleration: continuous white noise with the density of a Gauss-Markov (S = 2σ²τ)
        q = 2 * cfg.sigma_acel**2 * max(cfg.accel_tau_h, 0.1) * 3600.0
        Q[:3, :3] = np.eye(3) * q * dt**3 / 3
        Q[3:6, 3:6] = np.eye(3) * q * dt
        Q[:3, 3:6] = Q[3:6, :3] = np.eye(3) * q * dt**2 / 2
        walk_on = not cfg.sky_bias_walk_ep_only or hold.get("ep_flag", False)
        qb_ = hold["q_bias"] + np.eye(4) * (cfg.sky_bias_walk_nK * 1e-9 * walk_on) ** 2 / ph.DAY
        Q[7:11, 7:11] = qb_ * dt                      # attitude drift (and generic walk) seen as a slow bias
        for k_ in range(1, NPREC):
            bk_ = bidx(k_)
            Q[np.ix_(bk_, bk_)] = qb_ * dt
        if flick is not None:                        # 1/f drift of the sensors (DC channels)
            qd = flick.increment_var(dt)
            if destripe_on:                          # destriping: sensor offsets (Gauss-Markov)
                sq_ = np.sqrt(o_var * (1 - np.exp(-2 * dt / o_tau)))
                Q[np.ix_(IO, IO)] += np.outer(sq_, sq_) * Ccm_all
            elif cfg.spin:
                for k_ in range(NPREC):              # only the axis sensor (ring means discarded)
                    Q[bidx(k_)[0], bidx(k_)[0]] += qd
            else:
                Q[7:11, 7:11] += Ccm * qd
        if est_gain:                                 # gain drift: 1st-order Gauss-Markov (same model as the truth)
            Q[ID, ID] = np.eye(4) * sig_gd**2 * (1 - np.exp(-2 * dt / tau_g))
        if est_load:                                 # reference load drift (same model as the truth)
            Q[ILd, ILd] = (sig_ld / NK) ** 2 * (1 - np.exp(-2 * dt / tau_ld))
        elif load_on and not cfg.spin:               # not estimated: process noise in the biases (approximation)
            Q[7:11, 7:11] += np.eye(4) * 2 * sig_ld**2 / tau_ld * dt
        if est_sdop:                                 # solar Doppler bias (Gauss-Markov)
            Q[ISD, ISD] = max(sig_sdb, 1e-9) ** 2 * (1 - np.exp(-2 * dt / tau_sdb))
        if est_att:                                  # thermal drift of the alignment (same model as the truth)
            Q[IAd, IAd] = cfg.att_drift_arcsec**2 * (1 - np.exp(-2 * dt / att.tc)) if est_drift else 0.0
        if ep_A > 0:                                 # thrust error (known statistics) during the arcs
            t_mid = hold.get("t_pred", 0.0) + dt / 2
            if any(a_ <= t_mid < b_ for a_, b_ in ep_arcs):
                u_ = ph.unit(f.x[3:6])
                te = max(cfg.ep_tau_h, 0.1) * 3600.0
                s_par = ep_A * cfg.ep_mag_err_pct / 100
                s_perp = ep_A * np.deg2rad(cfg.ep_point_err_deg)
                Sa = 2 * te * (s_par**2 * np.outer(u_, u_) + s_perp**2 * (np.eye(3) - np.outer(u_, u_)))
                Q[:3, :3] += Sa * dt**3 / 3
                Q[3:6, 3:6] += Sa * dt
                Q[:3, 3:6] += Sa * dt**2 / 2
                Q[3:6, :3] += Sa * dt**2 / 2
        return Q

    hold = {"q_bias": q_bias, "ep_flag": bool(ep_A > 0 and any(a_ <= t_start < b_ for a_, b_ in ep_arcs))}

    def fx(Xs, tk, dt):
        # propagates the sigma points and, in the same call, the "dynamics only" state
        Xs = Xs.copy()
        if est_srp:
            X7 = np.hstack([Xs[:, :6], Xs[:, IS:IS + 1]])
            st = dyn.propagate(np.vstack([X7, np.append(hold["xdyn"], 0.0)]), tk, dt, max_step=3600.0)[:, :6]
        else:
            st = dyn.propagate(np.vstack([Xs[:, :6], hold["xdyn"]]), tk, dt, max_step=3600.0)
        Xs[:, :6] = st[:-1]
        hold["xdyn"] = st[-1]
        if destripe_on:
            Xs[:, IO] *= np.exp(-dt / o_tau)
        if est_sdop:
            Xs[:, ISD] *= np.exp(-dt / tau_sdb)
        if est_load:
            Xs[:, ILd] *= np.exp(-dt / tau_ld)
        if est_drift:                                # without drift estimation the state stays frozen (P0 ~1e-12):
            Xs[:, IAd] *= np.exp(-dt / att.tc)        # decaying without Q drove P to ~1e-54 and the UKF to NaN
        if est_ep and hold.get("ep_flag", False):     # thrust bias: constant acceleration over the step, in VNC
            a_e = np.einsum("nij,nj->ni", vnc_basis(Xs[:, :3], Xs[:, 3:6]), Xs[:, IE])
            Xs[:, :3] += 0.5 * a_e * dt**2
            Xs[:, 3:6] += a_e * dt
        if est_gain:
            Xs[:, ID] *= np.exp(-dt / tau_g)
        return Xs

    def grav_grad(r, P_, srp_):
        """∂a/∂r of gravity (Sun with radiation pressure + point-mass planets)."""
        rn = np.linalg.norm(r)
        u = r / rn
        mu_eff = ph.MU_SUN - dyn.k_srp * (1.0 + srp_)
        Gm = -mu_eff / rn**3 * (np.eye(3) - 3 * np.outer(u, u))
        for mu_p, p_ in zip(dyn.mu, P_):
            d_ = r - p_
            dn = np.linalg.norm(d_)
            ud = d_ / dn
            Gm = Gm - mu_p / dn**3 * (np.eye(3) - 3 * np.outer(ud, ud))
        return Gm

    def fjac(x, tk, dt):
        # EKF with analytic Jacobian: mean propagated in substeps of ≤ 1 h; (r, v) transition matrix from the
        # gravity gradient (3rd-order Taylor series per substep) and sensitivity to the SRP scale
        n_ = max(1, int(np.ceil(abs(dt) / 3600.0)))
        h_ = dt / n_
        F = np.eye(nx)
        Phi, dPs = np.eye(6), np.zeros(6)
        xm, xd = x.copy(), hold["xdyn"].copy()
        I3 = np.eye(3)
        for k_ in range(n_):
            t_ = tk + k_ * h_
            srp_ = xm[IS] if est_srp else 0.0
            Gm = grav_grad(xm[:3], dyn.positions(t_ + h_ / 2), srp_)
            A = np.zeros((6, 6)); A[:3, 3:] = I3; A[3:, :3] = Gm
            Ah = A * h_
            Phik = np.eye(6) + Ah + Ah @ Ah / 2 + Ah @ Ah @ Ah / 6
            a_s = dyn.k_srp * xm[:3] / np.linalg.norm(xm[:3]) ** 3
            if est_srp:
                st = dyn.propagate(np.vstack([np.append(xm[:6], srp_), np.append(xd, 0.0)]), t_, h_, max_step=3600.0)[:, :6]
            else:
                st = dyn.propagate(np.vstack([xm[:6], xd]), t_, h_, max_step=3600.0)
            xm[:6], xd = st[0], st[1]
            Phi = Phik @ Phi
            dPs = Phik @ dPs + np.concatenate([0.5 * h_**2 * a_s, h_ * a_s])
        hold["xdyn"] = xd
        F[:6, :6] = Phi
        if est_srp:
            F[:6, IS] = dPs
        if destripe_on:
            ph_o = np.exp(-dt / o_tau); xm[IO] *= ph_o; F[IO, IO] = ph_o
        if est_sdop:
            ph_s = np.exp(-dt / tau_sdb); xm[ISD] *= ph_s; F[ISD, ISD] = ph_s
        if est_drift:
            ph_a = np.exp(-dt / att.tc); xm[IAd] *= ph_a; F[IAd, IAd] = ph_a
        if est_ep and hold.get("ep_flag", False):
            Bm = vnc_basis(xm[:3], xm[3:6])
            a_e = Bm @ xm[IE]
            xm[:3] += 0.5 * a_e * dt**2
            xm[3:6] += a_e * dt
            F[:3, IE] = 0.5 * dt**2 * Bm
            F[3:6, IE] = dt * Bm
        if est_gain:
            ph_g = np.exp(-dt / tau_g); xm[ID] *= ph_g
            idg = np.arange(nx)[ID]
            F[idg, idg] = ph_g
        return xm, F

    def h_cmb(Xs, tk):
        v = Xs[:, 3:6] + vssb_nom + Xs[:, 11:14] + hold.get("vsun", 0.0)
        if est_att:
            db_ = (Xs[:, IAm] + Xs[:, IAd]) * ph.ARCSEC
            m = geom.model_att(v, Xs[:, 6], np.zeros_like(db_), db_)
        else:
            m = geom.model(v, Xs[:, 6])
        if est_gain:                                 # z = T_ref + (1 + g)(T − T_ref) in DC channels; × (1 + g) in harmonics
            L = m.copy()
            L[:, geom.dc] -= T_ref
            m = m + (Xs[:, IC] + Xs[:, ID])[:, geom.ch_sensor] * L
        if destripe_on:
            m = m.copy()
            off_ = Xs[:, IO].reshape(len(Xs), NBAND, 4).sum(1) * NK
            m[:, geom.dc] += off_[:, geom.dc_sensor]
        if est_load:
            m = m.copy()
            m[:, geom.dc] += Xs[:, ILd][:, geom.dc_sensor] * NK
        if est_zodi and hold.get("zb_geom") is geom:
            m = m - Xs[:, IZ][:, None] * (geom.B @ hold["zb"])[None]
        return m + Xs[:, bidx(geom.k)] @ geom.B.T

    def h_cel(Xs, tk):
        jd = jd0 + tk / ph.DAY
        rb = np.array([ph.planet_position(b, jd) for b in CELE_BODIES])
        los = ph.unit(rb[None, :, :] - Xs[:, None, :3])
        return np.arccos(np.clip(np.sum(los * STARS[None], axis=-1), -1, 1))

    # R consistent with the truth error (previously R = (0.001°)² = (3.6″)², with the truth at 0.5″)
    R_cel = np.eye(3) * (cfg.celenav_err_arcsec * ph.ARCSEC) ** 2
    rng_cel = np.random.default_rng(cfg.seed + 6007)
    if cfg.filter_type == "ekf":
        f = EKF(fx, h_cmb, x_init, P0, np.zeros((nx, nx)), geom.R)
    elif cfg.filter_type == "ekf_an":
        f = EKF(fx, h_cmb, x_init, P0, np.zeros((nx, nx)), geom.R, fjac=fjac)
    elif cfg.filter_type == "pf":
        f = GPF(fx, h_cmb, x_init, P0, np.zeros((nx, nx)), geom.R, n_particles=cfg.pf_particles,
                seed=cfg.seed, alpha=1.0)
    else:
        f = UKF(fx, h_cmb, x_init, P0, np.zeros((nx, nx)), geom.R, alpha=1.0)
    xdyn = x_init[:6].copy()
    hold["xdyn"] = xdyn
    # filter timer (on-board cost: prediction and update)
    import time as _time
    ftime = {"pred": 0.0, "upd": 0.0, "n_pred": 0, "n_upd": 0}

    def _timed(fn, key):
        def w(*a, **k):
            t0_ = _time.perf_counter()
            out_ = fn(*a, **k)
            ftime[key] += _time.perf_counter() - t0_
            ftime["n_" + key] += 1
            return out_
        return w
    f.predict = _timed(f.predict, "pred")
    f.update = _timed(f.update, "upd")

    out_t, out_k, out_xt, out_xe, out_P, out_xd, out_b = [], [], [], [], [], [], []
    cmb_t, cmb_T, snap_err, sun_ang = [], [], [], []
    v_guess = Xg[0, 3:] + vssb_nom
    sub = np.linspace(-0.5, 0.5, 21)
    t_prev = t_start

    def record(tk, kind):
        xt = G([tk])[0]
        out_t.append(tk); out_k.append(kind); out_xt.append(xt)
        out_xe.append(f.x.copy()); out_P.append(np.diag(f.P).copy()); out_xd.append(xdyn.copy())
        out_b.append(np.asarray(PT["blist"][0]).copy())

    def reset_bias():
        allb = np.unique(np.concatenate([bidx(k_) for k_ in range(NPREC)]))
        f.x[allb] = 0.0
        f.P[allb, :] = 0.0; f.P[:, allb] = 0.0
        Pb_ = np.zeros((4, 4))
        if att is not None and cfg.att_in_filter and not est_drift:
            Pb_, hold["q_bias"] = att_bias_cov(geom, f.x[3:6] + vssb_nom + f.x[11:14])
        for k_ in range(NPREC):
            bk_ = bidx(k_)
            f.P[np.ix_(bk_, bk_)] = PT["glist"][k_].Cb + Pb_
            if flick is not None and not destripe_on:  # the 1/f drift persists in the sensor after re-pointing
                if cfg.spin:
                    f.P[bk_[0], bk_[0]] += flick.drift_rms() ** 2
                else:
                    f.P[np.ix_(bk_, bk_)] += flick.drift_rms() ** 2 * Ccm

    n_auto_rep, n_excl, n_att_lost = 0, 0, 0
    n_cal, n_alc, n_dsn, n_cmb_lost = 0, 0, 0, 0
    n_ang = {"ddor": 0, "opn": 0, "xnav": 0, "ast": 0, "sund": 0}
    n_imu = 0
    sdop_rec, ast_rec = [], []
    rng_sd = np.random.default_rng(cfg.seed + 7727)             # solar Doppler and solar diameter
    rng_ast = np.random.default_rng(cfg.seed + 7741)
    xnav_sig_ely = xnav_ely_sigma(cfg.xnav_area_cm2, cfg.xnav_int_h)
    B_ast = ast_eph = None
    if cfg.ast_on:
        B_ast = bc.belt()
        ast_eph = rng_ast.normal(0, cfg.ast_eph_km, (len(B_ast["H"]), 3))   # fixed ephemeris error of each beacon
    guide, tcm_rec = None, []
    rng_tcm = np.random.default_rng(cfg.seed + 6113)
    if cfg.tcm_guided and traj_info.get("targeted"):
        ep_key_ = (float(cfg.ep_accel_mms2), tuple(tuple(a) for a in cfg.ep_arcs)) if cfg.ep_on else None
        fly_, t_c_ = make_flyer(jd0, cfg.tof_days, ep_key_)
        guide = dict(fly=fly_, t_c=t_c_, jd_c=jd0 + t_c_ / ph.DAY, rp=R_MARS_KM + cfg.rp_alt_km)
    dsn_sig = np.array([cfg.dsn_range_m * 1e-3, cfg.dsn_rate_mmps * 1e-6])      # km, km/s
    rng_loss = np.random.default_rng(cfg.seed + 4447)
    rng_cal = np.random.default_rng(cfg.seed + 2221)
    # one generator per measurement source (enabling one measurement does not change the draws of the others)
    rng_dsnm = np.random.default_rng(cfg.seed + 2237)
    rng_xn = np.random.default_rng(cfg.seed + 2243)
    rng_ang = np.random.default_rng(cfg.seed + 2251)

    def rng_win(tk):
        """CMB noise indexed by window: the same draw in the same window, with or without other measurements."""
        return np.random.default_rng([int(cfg.seed), 9907, int(round((tk - t_start) / max(tau, 1.0)))])
    drift_rec, zodi_rec, kin_rec, gain_rec = [], [], [], []
    load_rec = []
    gain_est_rec, gain_sig_rec = [], []
    lin_rec = []
    att_rec = {k_: [] for k_ in ("t", "white", "drift", "mis", "dv_white", "dv_drift", "dv_mis", "dv_total",
                                "avail_t", "sig_white")}
    rep_days = [t / ph.DAY for t, k in ev if k == "rep"]
    record(t_start, "init")
    nev = len(ev)
    for i, (tk, kind) in enumerate(ev):
        dt = tk - t_prev
        if dt > 0:
            hold["t_pred"] = t_prev
            f.Q = Qfun(dt)
            f.predict(t_prev, dt)
            xdyn = hold["xdyn"]
        t_prev = tk
        if kind == "man" and guide is not None and tk < guide["t_c"] - ph.DAY:
            # guided TCM: computed from the estimate; the truth receives the command + execution error
            dv_cmd, C_dv = guided_dv(guide["fly"], guide["t_c"], guide["jd_c"], f.x[:6].copy(), tk, guide["rp"],
                                     P6=f.P[:6, :6].copy())
            # dead band: not executed if the computed ΔV is consistent with the knowledge error itself
            if cfg.tcm_deadband_sigma > 0:
                m2 = float(dv_cmd @ np.linalg.solve(C_dv + np.eye(3) * 1e-18, dv_cmd))
                if m2 < cfg.tcm_deadband_sigma**2:
                    # TCM cancelled: the truth does not receive the execution error drawn for it
                    k0 = int(round((tk - tg[0]) / dtg))
                    x_new = Xg[k0].copy()
                    x_new[3:6] -= sum((d_ for t_, d_ in man_true if abs(t_ - tk) < 1.0), np.zeros(3))
                    fut = [(t_, d_) for t_, d_ in man_true + imp_true + cut_imp if t_ > tk]
                    _, Xn, _ = truth_grid(dyn_true, x_new, tg[k0], t_end, dtg, fut, acc_true, acc_seg, acc_t0=t_start)
                    Xg[k0:k0 + len(Xn)] = Xn[:len(Xg) - k0]
                    tcm_rec.append((tk / ph.DAY, 0.0))
                    record(tk, kind)
                    continue
            e_prop = rng_tcm.normal(0, cfg.tcm_prop_err_pct / 100 * np.linalg.norm(dv_cmd), 3)
            k0 = int(round((tk - tg[0]) / dtg))
            x_new = Xg[k0].copy()                    # already includes the absolute error drawn in man_true
            x_new[3:6] += dv_cmd + e_prop
            fut = [(t_, d_) for t_, d_ in man_true + imp_true + cut_imp if t_ > tk]
            _, Xn, _ = truth_grid(dyn_true, x_new, tg[k0], t_end, dtg, fut, acc_true, acc_seg, acc_t0=t_start)
            Xg[k0:k0 + len(Xn)] = Xn[:len(Xg) - k0]
            if cfg.imu_on:                           # IMU: the filter applies the measured ΔV (command + errors + IMU error)
                dv_exec = dv_cmd + e_prop + sum((d_ for t_, d_ in man_true if abs(t_ - tk) < 1.0), np.zeros(3))
                dv_m, s_imu = imu_meas(dv_exec)
                f.x[3:6] += dv_m
                f.P[3:6, 3:6] += np.eye(3) * s_imu**2
                n_imu += 1
            else:
                f.x[3:6] += dv_cmd
                f.P[3:6, 3:6] += np.eye(3) * ((cfg.maneuver_err_mps * 1e-3) ** 2
                                              + (cfg.tcm_prop_err_pct / 100 * np.linalg.norm(dv_cmd)) ** 2)
            tcm_rec.append((tk / ph.DAY, float(np.linalg.norm(dv_cmd) * 1e3)))
        elif kind == "man" and cfg.imu_on:
            # planned maneuver measured by the IMU: the filter applies the measured execution error
            dv_m, s_imu = imu_meas(sum((d_ for t_, d_ in man_true if abs(t_ - tk) < 1.0), np.zeros(3)))
            f.x[3:6] += dv_m
            f.P[3:6, 3:6] += np.eye(3) * s_imu**2
            n_imu += 1
        elif kind == "man":
            # planned maneuver: the filter only knows the statistics of the execution error
            f.P[3:6, 3:6] += np.eye(3) * (cfg.maneuver_err_mps * 1e-3) ** 2
        elif kind == "ep":
            hold["ep_flag"] = any(a_ <= tk + 1.0 < b_ for a_, b_ in ep_arcs)
            if est_ep and hold["ep_flag"]:             # new arc: new bias (prior)
                f.x[IE] = 0.0
                f.P[IE, :] = 0.0; f.P[:, IE] = 0.0
                f.P[IE, IE] = np.eye(3) * sig_epb**2
        elif kind == "cal":
            # each sensor looks at the internal load of known temperature: measures its own 1/f offset
            drift_c = flick.step(tk)
            if flick_c is not None:
                drift_c = np.sqrt(1 - cfrac) * drift_c + np.sqrt(cfrac) * flick_c.step(tk)[0]
            z_c = drift_c + rng_cal.normal(0, cfg.cal_sigma_nK * 1e-9, 4)
            f.update(z_c, tk, R=np.eye(4) * (cfg.cal_sigma_nK * 1e-9) ** 2,
                     hx=lambda X, t: X[:, IO].reshape(len(X), NBAND, 4).sum(1) * NK)
            n_cal += 1
        elif kind == "alc":
            # measures the star tracker × CMB beam alignment (misalignment + current drift), in ″
            z_a = (att.mis_b + att.lf_b) / ph.ARCSEC + rng_cal.normal(0, cfg.align_cal_arcsec, 3)
            f.update(z_a, tk, R=np.eye(3) * cfg.align_cal_arcsec**2, hx=lambda X, t: X[:, IAm] + X[:, IAd])
            n_alc += 1
        elif kind in ("des", "anom") and cfg.imu_on:
            # IMU on at the event: the filter applies the measured ΔV and adds only the IMU error
            dv_m, s_imu = imu_meas(imp_map.get(tk, np.zeros(3)))
            f.x[3:6] += dv_m
            f.P[3:6, 3:6] += np.eye(3) * s_imu**2
            n_imu += 1
        elif kind == "des":
            # attitude impulse: known time, residual ΔV only through its statistics
            f.P[3:6, 3:6] += np.eye(3) * (cfg.desat_dv_mmps * 1e-6) ** 2
        elif kind == "anom":
            # anomaly: time known from telemetry, unknown ΔV (statistics only)
            f.P[3:6, 3:6] += np.eye(3) * (cfg.anomaly_dv_mps * 1e-3) ** 2
        elif kind == "sdop":
            # solar Doppler: v_r = r̂·v (heliocentric) + Gauss-Markov bias + noise
            xt = G([tk])[0]
            if hold.get("sdt") is not None:
                ph_ = np.exp(-(tk - hold["sdt"]) / tau_sdb)
                hold["sdb"] = ph_ * hold["sdb"] + np.sqrt(1 - ph_**2) * rng_sd.normal(0, sig_sdb)
            else:
                hold["sdb"] = rng_sd.normal(0, sig_sdb)
            hold["sdt"] = tk
            z_s = np.array([ph.unit(xt[:3]) @ xt[3:6] + hold["sdb"] + rng_sd.normal(0, cfg.sdop_sigma_mps * 1e-3)])
            f.update(z_s, tk, R=np.eye(1) * (cfg.sdop_sigma_mps * 1e-3) ** 2,
                     hx=lambda X, t: (np.sum(X[:, :3] * X[:, 3:6], axis=1) / np.linalg.norm(X[:, :3], axis=1)
                                      + X[:, ISD])[:, None])
            sdop_rec.append((tk / ph.DAY, hold["sdb"] * 1e3, f.x[ISD] * 1e3, np.sqrt(f.P[ISD, ISD]) * 1e3))
        elif kind == "ast":
            # optical navigation by asteroids: direction of k beacons against the star background
            xt = G([tk])[0]
            jd_ = jd0 + tk / ph.DAY
            sel, Pn, mags = bc.visible_beacons(xt[:3], jd_, jd0, B_ast, cfg.ast_mag_lim, cfg.ast_sun_excl_deg,
                                               int(cfg.ast_k))
            sig_a = cfg.ast_urad * 1e-6
            for j_ in sel:
                pt_ = Pn[j_] + ast_eph[j_]                  # true beacon position (ephemeris with error)
                u0 = ph.unit(pt_ - xt[:3])
                e1_, e2_ = ph.perpendicular_basis(u0)
                z_a = np.array([u0 @ e1_, u0 @ e2_]) + rng_ast.normal(0, sig_a, 2)
                dist_ = np.linalg.norm(Pn[j_] - f.x[:3])
                s_eff = np.hypot(sig_a, cfg.ast_eph_km / dist_)   # R inflated by the ephemeris error (Ely 2022, eq. 41)

                def h_ast(X, t, pb_=Pn[j_], e1_=e1_, e2_=e2_):
                    u = pb_ - X[:, :3]
                    u = u / np.linalg.norm(u, axis=1, keepdims=True)
                    return np.column_stack([u @ e1_, u @ e2_])
                f.update(z_a, tk, R=np.eye(2) * s_eff**2, hx=h_ast)
                n_ang["ast"] += 1
            ast_rec.append((tk / ph.DAY, len(sel), float(np.min(mags[sel])) if sel else np.nan))
        elif kind == "sund":
            # apparent diameter of the Sun: heliocentric distance |r|
            xt = G([tk])[0]
            z_r = np.array([np.linalg.norm(xt[:3])]) + rng_sd.normal(0, cfg.sund_sigma_km, 1)
            f.update(z_r, tk, R=np.eye(1) * cfg.sund_sigma_km**2,
                     hx=lambda X, t: np.linalg.norm(X[:, :3], axis=1)[:, None])
            n_ang["sund"] += 1
        elif kind == "rep":
            geom, bias_true = new_pointing(tk)
            reset_bias()
        elif kind == "cmb" and cfg.cmb_loss_pct > 0 and rng_loss.random() * 100 < cfg.cmb_loss_pct:
            n_cmb_lost += 1                              # lost window (telemetry, on-board events)
        elif kind == "cmb":
            xt = G([tk])[0]
            sun = ph.unit(-xt[:3])
            vsun_k = ph.sun_bary_velocity(jd0 + tk / ph.DAY)      # Sun → barycenter (truth and filter)
            hold["vsun"] = vsun_k
            if NPREC > 1:                            # stepped precession: next orientation on the cone
                PT["kc"] = (PT["kc"] + 1) % NPREC
                geom, bias_true = PT["glist"][PT["kc"]], PT["blist"][PT["kc"]]
            ang = geom.channel_sun_angles(sun)
            ang_all = min(g_.channel_sun_angles(sun).min() for g_ in PT["glist"])
            if cfg.repoint_days < 0 and ang_all < cfg.sun_excl_deg + 5.0:
                # automatic re-pointing: only when the Sun approaches a sensor
                geom, bias_true = new_pointing(tk)
                reset_bias()
                n_auto_rep += 1
                rep_days.append(tk / ph.DAY)
                ang = geom.channel_sun_angles(sun)
            ok = ang >= cfg.sun_excl_deg
            xs = G(tk + sub * tau)
            R_k = geom.R
            if att is None:
                z_avg = geom.model(xs[:, 3:] + vssb_true + vsun_k, np.full(len(xs), T0_true), truth=True).mean(0)
            else:
                smp = att.sample(tk, xs[0, :3], jd0 + tk / ph.DAY)
                if smp is None:                      # not enough stars: no attitude, no measurement
                    n_att_lost += 1
                    record(tk, kind)
                    continue
                w_i, lf_b, mis_b, cov_w = smp
                ns = len(xs)
                z_avg = geom.model_att(xs[:, 3:] + vssb_true + vsun_k, np.full(ns, T0_true), np.tile(w_i, (ns, 1)),
                                       np.tile(lf_b + mis_b, (ns, 1)), truth=True).mean(0)
                if hold.get("J_geom") is not geom or tk - hold["J_t"] > 0.5 * ph.DAY:
                    # Jacobians change slowly (direction of v): recomputed twice per day or on re-pointing
                    v_f = f.x[3:6] + vssb_nom + f.x[11:14]
                    hold["J"] = att_jacobians(geom, v_f, f.x[6]) + (snapshot_gain(geom, v_f),)
                    hold["J_geom"], hold["J_t"] = geom, tk
                JI, Jb, Gs = hold["J"]
                if cfg.att_in_filter:
                    R_k = geom.R + JI @ cov_w @ JI.T
                    if not est_drift:
                        hold["q_bias"] = (geom.Bp @ Jb) @ (geom.Bp @ Jb).T * att.q_drift
                att_rec["t"].append(tk)
                att_rec["avail_t"].append(att.avail)
                att_rec["sig_white"].append(float(np.sqrt(np.trace(cov_w) / 3) / ph.ARCSEC))
                att_rec["white"].append(w_i / ph.ARCSEC)
                att_rec["drift"].append(lf_b / ph.ARCSEC)
                att_rec["mis"].append(mis_b / ph.ARCSEC)
                for nm_, dz_ in (("dv_white", JI @ w_i), ("dv_drift", Jb @ lf_b), ("dv_mis", Jb @ mis_b),
                                 ("dv_total", JI @ w_i + Jb @ (lf_b + mis_b))):
                    att_rec[nm_].append(float(np.linalg.norm(Gs @ dz_) * 1e3))
            day_ = tk / ph.DAY
            v_true_sc = xt[3:]
            if lin_T is not None:                  # linearization effect and what remains after the on-board model
                vt_ = np.atleast_2d(xt[3:] + vssb_true + vsun_k)
                T0a_ = np.array([T0_true])
                zl_ = geom.model(vt_, T0a_, truth=True)[0]
                zf_ = geom.model(vt_, T0a_)[0]
                lf_keep = geom.lin_filt
                geom.lin_filt = None
                zt_ = geom.model(vt_, T0a_)[0]
                geom.lin_filt = lf_keep
                lin_rec.append(np.stack([zl_ - zt_, zl_ - zf_]) * 1e6)
            extra_b = np.zeros(4)
            qinc = np.zeros(4)
            if zd is not None:                     # zodiacal: on-board model with relative error eps_z
                zb = slow_dirs_values(geom, lambda d_: zd.value(d_, day_))
                extra_b += -eps_z * zb
                if hold.get("zb_geom") is geom and not est_zodi:
                    qinc += (cfg.zodi_err_pct / 100 * (zb - hold["zb"])) ** 2
                hold["zb"], hold["zb_geom"] = zb, geom
                zodi_rec.append(zb * 1e6)
            if sky is not None and cfg.kinematic:  # aberration + Doppler modulation by the probe velocity
                kin_t = slow_dirs_values(geom, lambda d_: sky.kinematic(d_, v_true_sc / ph.C_KMS))
                kin_e = slow_dirs_values(geom, lambda d_: sky.kinematic(d_, f.x[3:6] / ph.C_KMS)) \
                    if cfg.kin_correct else 0.0
                extra_b += kin_t - kin_e
                kin_rec.append(np.atleast_1d(kin_t) * 1e6)
            z_avg = z_avg + geom.B @ extra_b
            if cfg.intrinsic_dipole_uK > 0:        # intrinsic dipole: fixed pattern on the sky (no aberration)
                T0a_ = np.array([T0_true])
                z_avg = z_avg + (geom.model(v_int[None], T0a_, truth=True)
                                 - geom.model(np.zeros((1, 3)), T0a_, truth=True))[0]
            # gain of each sensor: z = T_ref + (1 + g)·(z − T_ref) in the DC channels; × (1 + g) in the harmonics
            if hold.get("tg") is not None:
                ph_ = np.exp(-(tk - hold["tg"]) / tau_g)
                gd = ph_ * hold["gd"] + np.sqrt(1 - ph_**2) * rng_x.normal(0, sig_gd, 4)
            else:
                gd = hold.get("gd", rng_x.normal(0, sig_gd, 4))
            hold["gd"], hold["tg"] = gd, tk
            gch = (g0 + gd)[geom.ch_sensor]
            Lt = z_avg.copy()
            Lt[geom.dc] -= T_ref
            z_avg = z_avg + gch * Lt
            gain_rec.append((g0 + gd) * 1e6)
            # the filter: gain drift as process noise (DC channels) and extra noise (harmonics)
            Lf = lever(geom, f.x[3:6] + vssb_nom + f.x[11:14], f.x[6])
            if hold.get("tgf") is not None and not est_gain:
                dtg_ = tk - hold["tgf"]
                qinc[geom.dc_bias] += 2 * sig_gd**2 / tau_g * dtg_ * Lf[geom.dc_bias_ch] ** 2
            hold["tgf"] = tk
            bk_ = bidx(geom.k)
            f.P[np.ix_(bk_, bk_)] += np.diag(qinc)
            harm = np.setdiff1d(np.arange(len(Lf)), geom.dc_bias_ch)
            if harm.size and not est_gain:
                R_k = R_k + np.diag(np.where(np.isin(np.arange(len(Lf)), harm),
                                             ((cfg.gain_cal_ppm * 1e-6) ** 2 + sig_gd**2) * Lf**2, 0.0))
            if cfrac > 0:                          # noise correlated between sensors (common 1/f)
                z = z_avg + geom.B @ bias_true + np.linalg.cholesky(geom.R + np.eye(len(z_avg)) * 1e-40) @ rng_win(tk).normal(0, 1, len(z_avg))
            else:
                z = z_avg + geom.B @ bias_true + rng_win(tk).normal(0, 1, len(z_avg)) * np.sqrt(np.diag(geom.R))
            if load_on:                            # reference load drift (Gauss-Markov per sensor)
                if hold.get("tld") is not None:
                    ph_l = np.exp(-(tk - hold["tld"]) / tau_ld)
                    hold["ld"] = ph_l * hold["ld"] + np.sqrt(1 - ph_l**2) * rng_ld.normal(0, sig_ld, 4)
                else:
                    hold["ld"] = rng_ld.normal(0, sig_ld, 4)
                hold["tld"] = tk
                z[geom.dc] += hold["ld"][geom.dc_sensor]
                load_rec.append(hold["ld"] * 1e9)
            if flick is not None:
                drift = flick.step(tk)
                if flick_c is not None:
                    drift = np.sqrt(1 - cfrac) * drift + np.sqrt(cfrac) * flick_c.step(tk)[0]
                z[geom.dc] += drift[geom.dc_sensor]
                drift_rec.append(drift.copy())
            n_excl += int((~ok).sum())
            ok_sun = ok.copy()                 # logging: only the solar exclusion becomes NaN
            ok = ok & geom.use
            Tm = np.eye(len(z))
            if sky_sky:                            # paired horns: DC channels become differences with sensor 1
                d0 = geom.dc[0]
                for c_ in geom.dc[1:]:
                    Tm[c_, d0] = -1.0
                z = Tm @ z
                R_k = Tm @ np.asarray(R_k) @ Tm.T
                ok[geom.dc[1:]] &= ok[d0]
                ok[d0] = False                     # the absolute channel does not exist in this receiver

            def h_eff(X, t, Tm=Tm):
                return h_cmb(X, t) @ Tm.T
            if obs["on"] and ok.any() and int(tk // ph.DAY) != obs["last_day"]:
                obs["last_day"] = int(tk // ph.DAY)
                meas_ = [(lambda X, tk=tk, ok=ok: h_eff(X, tk)[:, ok], R_k[np.ix_(ok, ok)], ph.DAY / tau)]
                if cfg.celenav:
                    meas_.append((lambda X, tk=tk: h_cel(X, tk), R_cel, ph.DAY / cfg.celenav_interval_s))
                obs_step(tk, meas_)
            if ok.any():
                f.update(z[ok], tk, R=R_k[np.ix_(ok, ok)],
                         hx=(lambda X, t, ok=ok: h_eff(X, t)[:, ok]))
            if ok.all():
                v_snap = snapshot_velocity(z, geom, v_guess)
                v_guess = v_snap
                snap_err.append((v_snap - vssb_nom - xt[3:]) * 1e3)
            else:
                snap_err.append(np.full(3, np.nan))
            if est_gain:
                gain_est_rec.append((f.x[IC] + f.x[ID]) * 1e6)
                Pg = f.P[IC, IC] + f.P[ID, ID] + f.P[IC, ID] + f.P[ID, IC]
                gain_sig_rec.append(np.sqrt(np.diag(Pg)) * 1e6)
            zz = z.copy(); zz[~ok_sun] = np.nan
            cmb_t.append(tk); cmb_T.append(zz)
            sun_ang.append(ang.min())
        elif kind == "dsn":
            # DSN tracking: range and range rate relative to the Earth center
            xt = G([tk])[0]
            rE_, vE_ = ph.planet_state("earth", jd0 + tk / ph.DAY)
            d_ = xt[:3] - rE_
            rho = np.linalg.norm(d_)
            sig_d = dsn_sig
            if cfg.dsn_plasma:                       # solar plasma in the Doppler (Cassini model, X band, 60 s)
                sep_k = bc.sep_angle_deg(xt[:3], rE_)
                sig_d = np.array([dsn_sig[0], np.hypot(dsn_sig[1], bc.plasma_rate_sigma_mmps(sep_k) * 1e-6)])
            z_d = np.array([rho, d_ @ (xt[3:6] - vE_) / rho]) + rng_dsnm.normal(0, 1, 2) * sig_d

            def h_dsn(X, t, rE_=rE_, vE_=vE_):
                dd = X[:, :3] - rE_
                rr = np.linalg.norm(dd, axis=1)
                return np.column_stack([rr, np.sum(dd * (X[:, 3:6] - vE_), axis=1) / rr])
            f.update(z_d, tk, R=np.diag(sig_d**2), hx=h_dsn)
            n_dsn += 1
        elif kind == "xnav":
            # XNAV: the pulse time of arrival, compared with the prediction at the barycenter, measures n̂·r (km);
            # assumes the phase ambiguity resolved and the on-board clock calibrated
            if cfg.xnav_model == "ely":              # best 4 of Ely 2022, σ = S/√(A·h) per pulsar
                j_p = n_ang["xnav"] % len(XNAV_ELY_DIRS)
                n_p, s_x = XNAV_ELY_DIRS[j_p], float(xnav_sig_ely[j_p])
            else:
                n_p, s_x = XNAV_DIRS[n_ang["xnav"] % len(XNAV_DIRS)], cfg.xnav_sigma_km
            xt = G([tk])[0]
            z_x = np.array([n_p @ xt[:3]]) + rng_xn.normal(0, s_x, 1)
            f.update(z_x, tk, R=np.eye(1) * s_x**2, hx=lambda X, t, n_p=n_p: (X[:, :3] @ n_p)[:, None])
            n_ang["xnav"] += 1
        elif kind in ("ddor", "opn"):
            # ΔDOR: probe direction seen from Earth; optical navigation: Mars center direction seen from the probe
            xt = G([tk])[0]
            body = "earth" if kind == "ddor" else "mars"
            rb_ = ph.planet_position(body, jd0 + tk / ph.DAY)
            sgn = 1.0 if kind == "ddor" else -1.0
            u0 = ph.unit(sgn * (xt[:3] - rb_))
            e1_, e2_ = ph.perpendicular_basis(u0)
            sig_a = cfg.ddor_nrad * 1e-9 if kind == "ddor" else cfg.opnav_urad * 1e-6
            z_a = np.array([u0 @ e1_, u0 @ e2_]) + rng_ang.normal(0, sig_a, 2)

            def h_ang(X, t, rb_=rb_, sgn=sgn, e1_=e1_, e2_=e2_):
                u = sgn * (X[:, :3] - rb_)
                u = u / np.linalg.norm(u, axis=1, keepdims=True)
                return np.column_stack([u @ e1_, u @ e2_])
            f.update(z_a, tk, R=np.eye(2) * sig_a**2, hx=h_ang)
            n_ang[kind] += 1
        elif kind == "cel":
            xt = G([tk])[0]
            jd = jd0 + tk / ph.DAY
            z = []
            for b, s in zip(CELE_BODIES, STARS):
                los = ph.unit(ph.planet_position(b, jd) - xt[:3])
                e1, e2 = ph.perpendicular_basis(los)
                # 2D Gaussian line-of-sight error, σ = celenav_err per axis (the measured angle errs by σ)
                g2 = rng_cel.normal(0, cfg.celenav_err_arcsec * ph.ARCSEC, 2)
                los_m = ph.unit(los + g2[0] * e1 + g2[1] * e2)
                z.append(np.arccos(np.clip(los_m @ s, -1, 1)))
            if obs["on"] and not cfg.cmb_on and int(tk // ph.DAY) != obs["last_day"]:
                obs["last_day"] = int(tk // ph.DAY)
                obs_step(tk, [(lambda X, tk=tk: h_cel(X, tk), R_cel, ph.DAY / cfg.celenav_interval_s)])
            f.update(np.array(z), tk, R=R_cel, hx=h_cel)
        record(tk, kind)
        if progress and i % 100 == 0:
            progress(0.1 + 0.9 * i / max(nev, 1), f"filter: {i}/{nev} events")

    tk_end = t_prev
    obs_info = None
    if obs["on"] and obs["days"]:
        rat = obs["ratio"][-1]
        idx = np.where((S0 > 0) & (rat < 0.99))[0]         # states the measurements say something about
        Pt = np.linalg.inv(np.eye(len(idx)) + obs["J"][np.ix_(idx, idx)])
        w, V = np.linalg.eigh(Pt)
        weak = []
        for k_ in range(len(w) - 1, max(len(w) - 5, -1), -1):   # least observable combinations (largest remaining σ)
            vec = np.zeros(nx); vec[idx] = V[:, k_]
            weak.append(dict(ratio=float(np.sqrt(max(w[k_], 0.0))), vec=vec))
        no_info = [st_labels[i] for i in range(nx) if S0[i] > 0 and rat[i] >= 0.99]
        obs_info = dict(labels=st_labels, days=np.array(obs["days"]), delta=np.array(obs["delta"]),
                        ratio=np.array(obs["ratio"]), weak=weak, sigma0=S0, no_info=no_info)
    planets = {p: ph.planet_position(p, jd0 + np.linspace(0, 700, 700)) for p in ("earth", "mars")}
    info = dict(traj_info, tau=tau, sigma_T=sigma_T, k_geo=k_geo,
                sigma_v_snapshot=sig_v_snap, bias_cov=Cb, bias_v=bias_v, tech=tech, jd0=jd0,
                labels=Geo.labels, spin=cfg.spin, T0_true=T0_true,
                dvssb_true=vssb_true - vssb_nom,
                vssb_err=float(np.linalg.norm(vssb_true - vssb_nom) * 1e3),
                maneuvers=[(tm / ph.DAY, float(np.linalg.norm(dv) * 1e3)) for tm, dv in man_true
                           if not any(abs(tm / ph.DAY - d_) < 1e-6 and m_ == 0.0 for d_, m_ in tcm_rec)],  # without cancelled TCMs
                obs=obs_info, ep=ep_rec, n_cal=n_cal, n_alc=n_alc,
                n_dsn=n_dsn, n_cmb_lost=n_cmb_lost, tcm=tcm_rec,
                conj=dict(days=sep_t / ph.DAY, deg=sep_deg, gaps=gaps_d, n_dsn_gap=n_dsn_gap,
                         min_deg=float(sep_deg.min()) if len(sep_deg) else np.nan),
                anomalies=[(ta / ph.DAY, float(np.linalg.norm(dv) * 1e3)) for ta, dv in ano_true],
                n_imu=n_imu, n_ast=n_ang["ast"], n_sund=n_ang["sund"],
                ast=(np.array(ast_rec) if ast_rec else None),
                sdop=(np.array(sdop_rec) if sdop_rec else None),
                xnav_sigma_km=(xnav_sig_ely.tolist() if cfg.xnav_model == "ely" else [cfg.xnav_sigma_km]),
                tcm_dv_total_mps=float(sum(d_ for _, d_ in tcm_rec)), n_ddor=n_ang["ddor"], n_opnav=n_ang["opn"], n_xnav=n_ang["xnav"], ephem=ph.EPHEM["source"],
                bplane=bplane_metrics(jd0 + tk_end / ph.DAY, G([tk_end])[0], f.x[:6].copy(), f.P[:6, :6].copy(),
                                     (R_MARS_KM + cfg.rp_alt_km) if traj_info.get("targeted") else None,
                                     cfg.tof_days * ph.DAY - tk_end),
                filt=dict(type=cfg.filter_type, time=dict(ftime), n_state=int(nx), ess=(np.array(f.ess) if cfg.filter_type == "pf" else None)),
                intrinsic=(dict(uK=cfg.intrinsic_dipole_uK, dir=d_int.tolist(), v_kms=float(np.linalg.norm(v_int)))
                           if cfg.intrinsic_dipole_uK > 0 else None),
                lin=(dict(correct=bool(cfg.lin_correct), band_frac=cfg.band_frac, band_delta=band_delta,
                          boost=float(ph.lin_boost(nu)), uK=np.array(lin_rec)) if lin_T is not None else None),
                dyn=dict(mismatch=bool(cfg.dyn_mismatch), srp_true=float(srp_true), srp_idx=(IS if est_srp else None),
                         acc_nms=(acc_true * 1e12 if acc_true is not None else None), acc_seg_s=acc_seg,
                         desat=[(td / ph.DAY, float(np.linalg.norm(dv) * 1e6)) for td, dv in des_true],
                         srp_acc_nms=float(dyn.k_srp_nom / ph.AU_KM**2 * 1e12)),
                n_cmb=len(cmb_t), n_repoint=len(rep_days), repoint_days=rep_days,
                n_excluded=n_excl, auto_repoints=n_auto_rep, n_att_lost=n_att_lost,
                freq_ghz=nu, freq2_ghz=cfg.freq2_ghz, noise_factor=float(noise_factor),
                chop=(dict(hz=float(cfg.chop_hz), factor=mod_factor) if chop_on else None),
                load=(dict(sig_nK=float(cfg.load_drift_nK), tau_h=float(cfg.load_tau_h), estimated=est_load,
                           rms_nK=float(np.sqrt(np.mean(np.square(load_rec)))) if load_rec else 0.0) if load_on else None),
                sep=dict(bands=list(bands), w=[float(x) for x in sep_w], null=list(sep_null), method=cfg.sep_method),
                sky=(dict(real=True, k_prior=sky.k_prior, floor_uK=sky.floor * 1e6, w=sky.w,
                          pointings=point_log) if sky is not None else dict(real=False)),
                zodi=(dict(eps=float(eps_z), z_uK=np.array(zodi_rec), **zd.traj_info) if zd is not None else None),
                kin=(np.array(kin_rec) if kin_rec else None),
                gain=dict(g0_ppm=(g0 * 1e6).tolist(), g_ppm=np.array(gain_rec), receiver=cfg.receiver,
                          T_ref=T_ref, estimated=est_gain,
                          est_ppm=np.array(gain_est_rec) if est_gain else None,
                          sig_ppm=np.array(gain_sig_rec) if est_gain else None),
                flicker=(dict(f_knee=flick.f_knee, white_extra_uK=np.sqrt(flick.white_var) * 1e6,
                              white_uK=float(tech.sigma_T(tau)) * noise_factor * 1e6,
                              drift_rms_uK=flick.drift_rms() * 1e6, spin_factor=flick.spin_factor(f_spin) if cfg.spin else None,
                              drift=np.array(drift_rec) * 1e6) if flick is not None else None),
                att=dict(att_info, n_coast=att.n_coast, n_reduced=att.n_reduced, catalog=cfg.st_catalog,
                         ident=cfg.st_ident, **{k_: np.array(v_) for k_, v_ in att_rec.items()}) if att is not None else None)
    info["components"] = dict(ephem=ph.EPHEM["source"], sky_real=sky is not None, zodi=zd is not None,
                              fallbacks=missing)
    if isinstance(f, GPF):
        info["pf"] = dict(ess_frac=np.array(f.ess), n_dim=np.array(f.n_dim), n_fallback=f.n_fallback,
                          n_particles=f.N)
    if progress:
        progress(1.0, "done")
    return Results(cfg, np.array(out_t), np.array(out_k), np.array(out_xt), np.array(out_xe),
                   np.array(out_P), np.array(out_xd), np.array(cmb_t), (np.array(cmb_T).reshape(len(cmb_t), -1) if cmb_t
                                       else np.zeros((0, 10 if cfg.spin else 4))),
                   np.array(snap_err).reshape(-1, 3) if snap_err else np.zeros((0, 3)), np.array(sun_ang), tg, Xg[:, :3], planets, np.array(out_b), info)


# ----------------------------------------------------------------------------- statistics
def segment_stats(res: Results, recover_mps: float = 1.0, period_days=None):
    """Filter and error statistics in each segment between maneuvers (TCM).

    period_days=(d0, d1) restricts the analysis to that interval (segments outside it are omitted and
    segments crossing it are clipped). Returns a list of dictionaries (one row per segment).
    """
    DAY = ph.DAY
    t = res.t
    starts = [t[0]] + [d * DAY for d, _ in res.info["maneuvers"]]
    ends = starts[1:] + [t[-1] + 1.0]
    p0, p1 = ((period_days[0] * DAY, period_days[1] * DAY) if period_days else (-np.inf, np.inf))
    ev = (res.x_est[:, 3:6] - res.x_true[:, 3:]) * 1e3
    evn = np.linalg.norm(ev, axis=1)
    sv = np.sqrt(res.P_diag[:, 3:6]) * 1e3
    svn = np.sqrt((sv**2).sum(1))
    er = np.linalg.norm(res.x_est[:, :3] - res.x_true[:, :3], axis=1)
    ed = np.linalg.norm(res.x_dyn[:, 3:] - res.x_true[:, 3:], axis=1) * 1e3
    nees = np.sum(ev**2 / np.maximum(sv, 1e-12) ** 2, axis=1)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    rows = []
    for k, (a, b) in enumerate(zip(starts, ends)):
        m = (t >= max(a, p0)) & (t < min(b, p1 + 1.0))
        if m.sum() < 2:
            continue
        tm = t[m]
        inside3 = np.all(np.abs(ev[m]) <= 3 * sv[m], axis=1).mean() * 100
        rec = np.where(evn[(t >= a) & (t < b)] < recover_mps)[0]
        t_rec = (t[(t >= a) & (t < b)][rec[0]] - a) / 3600 if len(rec) else np.nan
        man = res.info["maneuvers"][k - 1][1] if k > 0 else np.nan
        rows.append({
            "Segment": "start → TCM 1" if k == 0 else (f"TCM {k} → TCM {k+1}" if k < len(starts) - 1
                                                        else f"TCM {k} → end"),
            "Start [day]": round(tm[0] / DAY, 1),
            "End [day]": round(tm[-1] / DAY, 1),
            "TCM execution error [m/s]": round(man, 2) if np.isfinite(man) else np.nan,
            "CMB measurements": int(np.sum(res.kind[m] == "cmb")),
            "CeleNav measurements": int(np.sum(res.kind[m] == "cel")),
            "v error RMS [m/s]": round(rms(evn[m]), 3),
            "v error mean [m/s]": round(float(evn[m].mean()), 3),
            "v error max [m/s]": round(float(evn[m].max()), 3),
            "v error end [m/s]": round(float(evn[m][-1]), 3),
            "1σ filter RMS [m/s]": round(rms(svn[m]), 3),
            "1σ filter end [m/s]": round(float(svn[m][-1]), 3),
            "% within ±3σ": round(float(inside3), 1),
            "NEES mean": round(float(nees[m].mean()), 2),
            f"Time to error < {recover_mps:g} m/s [h]": round(float(t_rec), 1) if np.isfinite(t_rec) else np.nan,
            "r error RMS [km]": round(rms(er[m]), 1),
            "r error end [km]": round(float(er[m][-1]), 1),
            "Dynamics only: v error RMS [m/s]": round(rms(ed[m]), 3),
        })
    return rows


def gap_stats(res: Results, gaps=None):
    """Metrics of each tracking gap: velocity and position error at the end of the gap and the maximum
    within it, the filter 1σ, the consistency (fraction within ±3σ, per axis) and the same error with dynamics
    only. gaps: [(start_day, end_day)] (default: the effective DSN gaps of the simulation)."""
    DAY = ph.DAY
    gaps = list(gaps) if gaps is not None else list((res.info.get("conj") or {}).get("gaps", []))
    d = res.t / DAY
    ev = (res.x_est[:, 3:6] - res.x_true[:, 3:]) * 1e3
    sv = np.sqrt(res.P_diag[:, 3:6]) * 1e3
    er = res.x_est[:, :3] - res.x_true[:, :3]
    sr = np.sqrt(res.P_diag[:, :3])
    ed = np.linalg.norm(res.x_dyn[:, 3:] - res.x_true[:, 3:], axis=1) * 1e3
    rows = []
    for a, b in gaps:
        m = (d >= a) & (d < b)
        pre = np.where(d < b)[0]
        if not len(pre):
            continue
        i = pre[-1]
        mm = m if m.any() else (np.arange(len(d)) == i)
        inside = np.all(np.abs(np.hstack([ev[mm], er[mm]])) <= 3 * np.hstack([sv[mm], sr[mm]]), axis=1).mean() * 100
        rows.append({
            "Gap [days]": f"{a:g}–{b:g}",
            "Duration [d]": round(float(b - a), 1),
            "v error end [m/s]": round(float(np.linalg.norm(ev[i])), 4),
            "1σ v end [m/s]": round(float(np.linalg.norm(sv[i])), 4),
            "v error max [m/s]": round(float(np.linalg.norm(ev[mm], axis=1).max()), 4),
            "r error end [km]": round(float(np.linalg.norm(er[i])), 2),
            "1σ r end [km]": round(float(np.linalg.norm(sr[i])), 2),
            "r error max [km]": round(float(np.linalg.norm(er[mm], axis=1).max()), 2),
            "% within ±3σ": round(float(inside), 1),
            "Dynamics only: v error end [m/s]": round(float(ed[i]), 4),
        })
    return rows
