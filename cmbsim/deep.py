"""Navigation in distant deep space (use case 3: interstellar precursor, ~100 AU).

The main simulator is Earth–Mars (DE421 trimmed to 2020–2040, Mars targeting, zodiacal emission). This module is
a lean, LINEARIZED model for any heliocentric distance:

- dynamics: Sun (point mass) + non-gravitational acceleration in two parts, a slow one (Gauss-Markov bias,
  "RTG thermal recoil", like the Pioneer anomaly) and a fast one (1-day GM); attitude impulses
  (antenna precession) and ΔV anomalies;
- Earth on a circular 1 AU orbit in the ecliptic (arbitrary phase); geocentric measurements, no stations, no
  Earth rotation and no light time (conservative for the DSN: the diurnal signature would give angle information);
- measurements: DSN (two-way range and Doppler per hour of pass, optional ΔDOR), ABSTRACT CMB (3D velocity
  with white noise and a Gauss-Markov bias representing the systematics), XNAV (Ely 2022, "best 4") and IMU at
  the ΔV events;
- filter: EKF with the same stochastic model as the truth.

The CMB here has no explicit sky, 1/f, attitude or gain: it is summarized by (σ_day, σ_bias, τ_bias). These
numbers are CALIBRATED against the full simulator in the case of the 60-day gap at 1.5 AU (`calibration_case`),
reproducing the three levels of the 1/f study (realistic, requirements, ideal).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import physics as ph

MU = ph.MU_SUN
AU = ph.AU_KM
DAY = ph.DAY
YEAR = 365.25 * DAY

# Abstract CMB levels (white-noise σ_day [m/s], bias σ [m/s], bias τ [days]), matched with calibration_case()
# (60 d gap, 0.3 m/s anomaly) to the full simulator: the first levels to studies/one_over_f_study.py, the *_sim
# levels to the full-simulator CMB levels (studies/deep_space_calibration_fit.py reproduces their calibration).
CMB_LEVELS = {
    "realistic": (8.0, 3.0, 5.0),      # family D with all systematics: no improvement over dynamics only
    "requirements": (0.6, 0.15, 5.0),   # NET 0.3 + chopper + 0.1″ + 1 ppm: ~0.30 × dynamics only
    "ideal": (0.9, 0.0, 5.0),         # NET 1 without attitude, gain or 1/f: ~0.38 × dynamics only
    # levels from the analysis of the physical limits of CMB navigation
    "today": (1.5, 1.0, 5.0),          # best today: NET 1, chopper, 1″, 10 ppm: ~0.77 × dynamics only (1/f study)
    "metrology": (0.05, 0.02, 5.0),  # hypothetical: étendue ~0.01 m²sr at the photon limit, co-alignment ~10 mas,
                                      # gain from the dipole/monopole ratio with a load stable to ~30 nK
    "limit": (0.017, 0.002, 5.0),    # practical physical limit: étendue 0.1 m²sr, ~1 mas, ~3 nK
    # the same two levels calibrated against the full simulator (studies/ratio_study.py): ratio method
    # modeled (total power, drifting load, tilted axis), estimated zodiacal scale, real sky; the 4-sensor
    # instrument and the nuisance parameters raise the noise per day relative to the analytic estimates
    "requirements_sim": (1.1, 0.45, 5.0),  # 60 d gap: ~0.26 m/s (full simulator, 6 seeds: 0.262)
    "ratio_sim": (1.75, 0.7, 5.0),       # 60 d gap: ~0.33 m/s (full simulator: 0.333)
    "metrology_sim": (0.30, 0.12, 5.0),  # 60 d gap: 0.090 m/s (full simulator: 0.0895)
    "limit_sim": (0.12, 0.018, 5.0),     # 60 d gap: 0.027 m/s (full simulator: 0.0266)
}


@dataclass
class DeepConfig:
    # trajectory: initial position at r0 AU, radial hyperbolic velocity (precursor: ~7 AU/yr ≈ 33 km/s)
    r0_au: float = 100.0
    v_kms: float = 33.0
    lon_deg: float = 255.0           # direction (ecliptic): ~heliosphere nose (λ ≈ 255°, β ≈ 5°)
    lat_deg: float = 5.0
    earth_phase_deg: float = 0.0     # Earth's phase at the start
    days: float = 365.0
    dt_h: float = 1.0                # propagation step
    # dynamics (truth = filter model)
    acc_slow_nms: float = 1.0        # slow acceleration (RTG, Pioneer ~0.9 nm/s²), 1σ per axis [nm/s²]
    acc_slow_tau_d: float = 1000.0
    acc_fast_nms: float = 0.1        # fast acceleration, 1σ per axis [nm/s²]
    acc_fast_tau_d: float = 1.0
    att_days: float = 7.0            # attitude maneuvers (precession to point the antenna); 0 = none
    att_dv_mmps: float = 1.0         # residual ΔV of each one (1σ per axis)
    anomaly_days: tuple = ()
    anomaly_dv_mps: float = 0.3
    imu_on: bool = False
    imu_dv_err_mmps: float = 1.0
    imu_scale_pct: float = 0.2
    # initial knowledge (DSN calibration before the start)
    p0_km: float = 10.0
    v0_mmps: float = 1.0
    # DSN
    dsn_pass_days: float = 30.0      # 0 = no DSN
    dsn_pass_h: float = 8.0
    dsn_range_m: float = 10.0
    dsn_rate_mmps: float = 0.1
    dsn_ddor_nrad: float = 0.0       # 0 = no ΔDOR
    dsn_until_day: float | None = None   # DSN only up to this day (None = always)
    dsn_gaps: tuple = ()
    # abstract CMB
    cmb_on: bool = False
    cmb_level: str = "realistic"      # key of CMB_LEVELS or 'custom'
    cmb_sig_day_mps: float = 3.0     # used if cmb_level == 'custom'
    cmb_bias_mps: float = 2.0
    cmb_bias_tau_d: float = 5.0
    cmb_period_h: float = 2.0
    # XNAV (Ely 2022, best 4)
    xnav_on: bool = False
    xnav_area_cm2: float = 129.0
    xnav_int_h: float = 24.0
    seed: int = 1

    def cmb_params(self):
        if self.cmb_level == "custom":
            return self.cmb_sig_day_mps, self.cmb_bias_mps, self.cmb_bias_tau_d
        return CMB_LEVELS[self.cmb_level]


@dataclass
class DeepResults:
    cfg: DeepConfig
    t: np.ndarray
    x_true: np.ndarray
    x_est: np.ndarray
    P: np.ndarray                      # covariance of r, v (N, 6, 6)
    los: np.ndarray                    # Earth→probe direction at each record
    info: dict = field(default_factory=dict)


def earth_state(t, phase_deg=0.0):
    """Earth on a circular 1 AU orbit in the ecliptic."""
    w = 2 * np.pi / YEAR
    th = np.deg2rad(phase_deg) + w * t
    r = AU * np.array([np.cos(th), np.sin(th), 0.0])
    v = AU * w * np.array([-np.sin(th), np.cos(th), 0.0])
    return r, v


def _grav(r):
    return -MU * r / np.linalg.norm(r) ** 3


def _ggrad(r):
    rn = np.linalg.norm(r)
    u = r / rn
    return -MU / rn**3 * (np.eye(3) - 3 * np.outer(u, u))


def _rk4(x, a_extra, dt):
    """RK4 step of (r, v) with the Sun and an extra acceleration held constant over the step."""
    def f(y):
        return np.concatenate([y[3:], _grav(y[:3]) + a_extra])
    k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def run_deep(cfg: DeepConfig, x0=None) -> DeepResults:
    rng = np.random.default_rng(cfg.seed)
    rng_meas = np.random.default_rng(cfg.seed + 101)
    dt = cfg.dt_h * 3600.0
    n_steps = int(round(cfg.days * DAY / dt))
    u0 = ph.unit(np.array([np.cos(np.deg2rad(cfg.lat_deg)) * np.cos(np.deg2rad(cfg.lon_deg)),
                           np.cos(np.deg2rad(cfg.lat_deg)) * np.sin(np.deg2rad(cfg.lon_deg)),
                           np.sin(np.deg2rad(cfg.lat_deg))]))
    x_t = np.concatenate([cfg.r0_au * AU * u0, cfg.v_kms * u0]) if x0 is None else np.asarray(x0, float).copy()

    # stochastic processes (km/s²)
    s_sl, t_sl = cfg.acc_slow_nms * 1e-12, cfg.acc_slow_tau_d * DAY
    s_fa, t_fa = cfg.acc_fast_nms * 1e-12, cfg.acc_fast_tau_d * DAY
    sig_c, sig_b, tau_b = cfg.cmb_params()
    sig_b_k, tau_b_s = sig_b * 1e-3, max(tau_b, 1e-3) * DAY
    a_sl = rng.normal(0, s_sl, 3)
    a_fa = rng.normal(0, s_fa, 3)
    b_c = rng.normal(0, sig_b_k, 3)

    # filter state: r, v, a_slow (3), CMB bias (3, only if enabled)
    nb = 3 if cfg.cmb_on else 0
    nx = 9 + nb
    IA, IB = slice(6, 9), slice(9, 9 + nb)
    x_e = np.zeros(nx)
    x_e[:6] = x_t + np.concatenate([rng.normal(0, cfg.p0_km, 3), rng.normal(0, cfg.v0_mmps * 1e-6, 3)])
    P = np.zeros((nx, nx))
    P[:3, :3] = np.eye(3) * cfg.p0_km**2
    P[3:6, 3:6] = np.eye(3) * (cfg.v0_mmps * 1e-6) ** 2
    P[IA, IA] = np.eye(3) * s_sl**2
    if nb:
        P[IB, IB] = np.eye(3) * max(sig_b_k, 1e-12) ** 2
    q_fast = 2 * s_fa**2 * t_fa                               # equivalent white-noise density

    # ΔV events
    imp = {}
    if cfg.att_days > 0:
        for d in np.arange(cfg.att_days, cfg.days, cfg.att_days):
            k = int(round(d * DAY / dt))
            imp[k] = ("att", rng.normal(0, cfg.att_dv_mmps * 1e-6, 3))
    for d in cfg.anomaly_days or ():
        k = int(round(float(d) * DAY / dt))
        imp[k] = ("anom", rng.normal(0, cfg.anomaly_dv_mps * 1e-3, 3))

    # measurement schedule per step
    def dsn_on(d):
        if cfg.dsn_pass_days <= 0:
            return False
        if cfg.dsn_until_day is not None and d > cfg.dsn_until_day:
            return False
        if any(a <= d < b for a, b in (cfg.dsn_gaps or ())):
            return False
        ph_ = d % cfg.dsn_pass_days
        return ph_ < cfg.dsn_pass_h / 24.0
    k_cmb = max(1, int(round(cfg.cmb_period_h / cfg.dt_h)))
    k_xn = max(1, int(round(24.0 / cfg.dt_h)))
    if cfg.xnav_on:
        from .simulator import XNAV_ELY_DIRS, xnav_ely_sigma
        xn_dirs, xn_sig = XNAV_ELY_DIRS, xnav_ely_sigma(cfg.xnav_area_cm2, cfg.xnav_int_h)
    n_dsn = n_ddor = n_cmb = n_xn = n_imu = 0
    out_t, out_xt, out_xe, out_P, out_los = [], [], [], [], []
    I3 = np.eye(3)

    def update(z, h, H, R):
        nonlocal x_e, P
        S = H @ P @ H.T + R
        K = np.linalg.solve(S, H @ P).T
        x_e = x_e + K @ (z - h)
        IKH = np.eye(nx) - K @ H
        P = IKH @ P @ IKH.T + K @ R @ K.T

    def record(k):
        rE, _ = earth_state(k * dt, cfg.earth_phase_deg)
        out_t.append(k * dt); out_xt.append(x_t.copy()); out_xe.append(x_e[:6].copy())
        out_P.append(P[:6, :6].copy()); out_los.append(ph.unit(x_t[:3] - rE))

    record(0)
    last_pass = -1
    for k in range(1, n_steps + 1):
        # --- truth
        x_t = _rk4(x_t, a_sl + a_fa, dt)
        ph_s, ph_f = np.exp(-dt / t_sl), np.exp(-dt / t_fa)
        a_sl = ph_s * a_sl + np.sqrt(1 - ph_s**2) * rng.normal(0, s_sl, 3)
        a_fa = ph_f * a_fa + np.sqrt(1 - ph_f**2) * rng.normal(0, s_fa, 3)
        if nb or sig_b > 0:
            ph_b = np.exp(-dt / tau_b_s)
            b_c = ph_b * b_c + np.sqrt(1 - ph_b**2) * rng.normal(0, sig_b_k, 3)
        # --- filter prediction
        A = np.zeros((nx, nx))
        A[:3, 3:6] = I3
        A[3:6, :3] = _ggrad(x_e[:3])
        A[3:6, IA] = I3
        A[IA, IA] = -I3 / t_sl
        if nb:
            A[IB, IB] = -I3 / tau_b_s
        F = np.eye(nx) + A * dt + A @ A * dt**2 / 2
        x_e[:6] = _rk4(x_e[:6], x_e[IA], dt)
        x_e[IA] *= ph_s
        if nb:
            x_e[IB] *= np.exp(-dt / tau_b_s)
        Q = np.zeros((nx, nx))
        Q[:3, :3] = I3 * q_fast * dt**3 / 3
        Q[:3, 3:6] = Q[3:6, :3] = I3 * q_fast * dt**2 / 2
        Q[3:6, 3:6] = I3 * q_fast * dt
        Q[IA, IA] = I3 * s_sl**2 * (1 - ph_s**2)
        if nb:
            Q[IB, IB] = I3 * max(sig_b_k, 1e-12) ** 2 * (1 - np.exp(-2 * dt / tau_b_s))
        P = F @ P @ F.T + Q
        # --- impulses (truth) and what the filter knows
        if k in imp:
            kind, dv = imp[k]
            x_t[3:6] += dv
            sig_ev = cfg.att_dv_mmps * 1e-6 if kind == "att" else cfg.anomaly_dv_mps * 1e-3
            if cfg.imu_on:
                s_ = float(np.hypot(cfg.imu_dv_err_mmps * 1e-6, cfg.imu_scale_pct / 100 * np.linalg.norm(dv)))
                x_e[3:6] += dv + rng_meas.normal(0, s_, 3)
                P[3:6, 3:6] += I3 * s_**2
                n_imu += 1
            else:
                P[3:6, 3:6] += I3 * sig_ev**2
        t = k * dt
        d = t / DAY
        # --- DSN: range and Doppler (1 pair per hour of pass), ΔDOR in the middle of the pass
        if dsn_on(d):
            rE, vE = earth_state(t, cfg.earth_phase_deg)
            for xx, use in ((x_t, True), (x_e, False)):
                dd = xx[:3] - rE
                rho = np.linalg.norm(dd)
                u = dd / rho
                hv = np.array([rho, u @ (xx[3:6] - vE)])
                if use:
                    z = hv + rng_meas.normal(0, 1, 2) * np.array([cfg.dsn_range_m * 1e-3, cfg.dsn_rate_mmps * 1e-6])
                    u_t = u
                else:
                    H = np.zeros((2, nx))
                    H[0, :3] = u
                    vr = xx[3:6] - vE
                    H[1, :3] = (vr - u * (u @ vr)) / rho
                    H[1, 3:6] = u
                    update(z, hv, H, np.diag([(cfg.dsn_range_m * 1e-3) ** 2, (cfg.dsn_rate_mmps * 1e-6) ** 2]))
                    n_dsn += 1
            pass_id = int(d // cfg.dsn_pass_days)
            if cfg.dsn_ddor_nrad > 0 and pass_id != last_pass:
                last_pass = pass_id
                e1, e2 = ph.perpendicular_basis(u_t)
                s_a = cfg.dsn_ddor_nrad * 1e-9
                z = np.array([u_t @ e1, u_t @ e2]) + rng_meas.normal(0, s_a, 2)
                dd = x_e[:3] - rE
                rho = np.linalg.norm(dd)
                u = dd / rho
                Ju = (I3 - np.outer(u, u)) / rho
                H = np.zeros((2, nx))
                H[0, :3], H[1, :3] = e1 @ Ju, e2 @ Ju
                update(z, np.array([u @ e1, u @ e2]), H, I3[:2, :2] * s_a**2)
                n_ddor += 1
        # --- abstract CMB: heliocentric velocity + bias + white noise
        if cfg.cmb_on and k % k_cmb == 0:
            s_m = sig_c * 1e-3 * np.sqrt(24.0 / cfg.cmb_period_h)
            z = x_t[3:6] + b_c + rng_meas.normal(0, s_m, 3)
            H = np.zeros((3, nx)); H[:, 3:6] = I3; H[:, IB] = I3
            update(z, x_e[3:6] + x_e[IB], H, I3 * s_m**2)
            n_cmb += 1
        # --- XNAV: one pulsar per day, in rotation
        if cfg.xnav_on and k % k_xn == 0:
            j = n_xn % len(xn_dirs)
            n_p, s_x = xn_dirs[j], float(xn_sig[j])
            z = np.array([n_p @ x_t[:3] + rng_meas.normal(0, s_x)])
            H = np.zeros((1, nx)); H[0, :3] = n_p
            update(z, np.array([n_p @ x_e[:3]]), H, np.eye(1) * s_x**2)
            n_xn += 1
        if k % int(round(24.0 / cfg.dt_h)) == 0 or k == n_steps:
            record(k)
    info = dict(n_dsn=n_dsn, n_ddor=n_ddor, n_cmb=n_cmb, n_xnav=n_xn, n_imu=n_imu,
                cmb=dict(zip(("sig_day", "bias", "tau_d"), cfg.cmb_params())) if cfg.cmb_on else None,
                anomalies=[(k * dt / DAY, float(np.linalg.norm(v_) * 1e3)) for k, (kk, v_) in imp.items() if kk == "anom"])
    return DeepResults(cfg, np.array(out_t), np.array(out_xt), np.array(out_xe), np.array(out_P), np.array(out_los),
                       info)


def deep_metrics(res: DeepResults, i=-1):
    """Errors and 1σ at record i, split into the Earth→probe direction (DSN radial) and the plane of sky."""
    u = res.los[i]
    T = np.eye(3) - np.outer(u, u)
    e_r = res.x_est[i, :3] - res.x_true[i, :3]
    e_v = (res.x_est[i, 3:6] - res.x_true[i, 3:6]) * 1e3
    Pr, Pv = res.P[i, :3, :3], res.P[i, 3:, 3:] * 1e6
    return {
        "r error line of sight [km]": float(abs(u @ e_r)),
        "r error plane of sky [km]": float(np.linalg.norm(T @ e_r)),
        "1σ r plane of sky [km]": float(np.sqrt(np.trace(T @ Pr @ T))),
        "v error line of sight [m/s]": float(abs(u @ e_v)),
        "v error plane of sky [m/s]": float(np.linalg.norm(T @ e_v)),
        "1σ v line of sight [m/s]": float(np.sqrt(u @ Pv @ u)),
        "1σ v plane of sky [m/s]": float(np.sqrt(np.trace(T @ Pv @ T))),
        "v error total [m/s]": float(np.linalg.norm(e_v)),
        "r error total [km]": float(np.linalg.norm(e_r)),
    }


def calibration_case(level, seed=1, **kw):
    """60-day gap at 1.5 AU with a 0.3 m/s/axis anomaly 2 days after the start, as in the full-simulator
    study (studies/gap_study.py): weekly DSN with ΔDOR up to the start of the gap (the initial state already
    comes from the DSN), circular orbit. Returns the velocity error at the end of the gap and the result."""
    cfg = DeepConfig(r0_au=1.5, days=60.0, dsn_pass_days=0.0, p0_km=1.0, v0_mmps=1.0, anomaly_days=(2.0,),
                     cmb_on=level is not None, cmb_level=level or "realistic", seed=seed, **kw)
    r = cfg.r0_au * AU
    x0 = np.array([r, 0.0, 0.0, 0.0, np.sqrt(MU / r), 0.0])
    res = run_deep(cfg, x0=x0)
    return deep_metrics(res)["v error total [m/s]"], res
