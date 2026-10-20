"""Demonstrator physics: constants, reference frames, CMB dipole, ephemerides,
Lambert solver and heliocentric dynamics. Units: km, s, K.
"""
import numpy as np

# ----------------------------------------------------------------------------- constants
C_KMS = 299_792.458
T0_CMB = 2.7255
T0_CMB_SIG = 0.0006                     # monopole uncertainty [K]
AU_KM = 149_597_870.7
MU_SUN = 1.32712440018e11
MU_PLANETS = {"venus": 324_858.59, "jupiter": 126_712_764.1, "saturn": 37_940_585.2,
              "earth": 403_503.24, "mars": 42_828.37,
              "earth_c": 398_600.4418, "moon": 4_902.800}
P_SR_1AU = 4.56e-6                      # solar radiation pressure at 1 AU [N/m²]
EPS_J2000 = np.deg2rad(23.4392911)

# Velocity of the Sun relative to the CMB (Planck 2018 results I, Table 3)
V_SSB = 369.82; V_SSB_SIG = 0.11        # km/s
L_SSB = 264.021; L_SSB_SIG = 0.011      # degrees
B_SSB = 48.253; B_SSB_SIG = 0.005       # degrees

DAY = 86400.0
ARCSEC = np.deg2rad(1 / 3600)


# ----------------------------------------------------------------------------- reference frames
_A_G = np.array([
    [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
    [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
    [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669],
])
GAL_TO_EQ = _A_G.T
c, s = np.cos(EPS_J2000), np.sin(EPS_J2000)
EQ_TO_ECL = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def unit(v, axis=-1):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=axis, keepdims=True)


def lb_to_unit(l_deg, b_deg):
    l, b = np.deg2rad(l_deg), np.deg2rad(b_deg)
    return np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)])


def radec_to_ecl(ra_deg, dec_deg):
    return EQ_TO_ECL @ lb_to_unit(ra_deg, dec_deg)


def v_ssb(mag=V_SSB, l_deg=L_SSB, b_deg=B_SSB):
    """Velocity of the Sun (SSB) relative to the CMB, J2000 ecliptic [km/s]."""
    return EQ_TO_ECL @ GAL_TO_EQ @ (mag * lb_to_unit(l_deg, b_deg))


def v_ssb_random(rng):
    return v_ssb(V_SSB + rng.normal(0, V_SSB_SIG), L_SSB + rng.normal(0, L_SSB_SIG),
                 B_SSB + rng.normal(0, B_SSB_SIG))


_VSSB_COV = {}


def v_ssb_cov(n=20000):
    """Cartesian covariance [km²/s²] of the error in the Planck solar velocity (amplitude and direction), by Monte Carlo
    from the same distribution as v_ssb_random (an isotropic prior would underestimate the dipole-axis error)."""
    if "C" not in _VSSB_COV:
        rng = np.random.default_rng(12345)
        d = np.array([v_ssb_random(rng) for _ in range(n)]) - v_ssb()
        _VSSB_COV["C"] = d.T @ d / n
    return _VSSB_COV["C"]


def perpendicular_basis(axis):
    a = unit(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = unit(np.cross(a, ref))
    return e1, np.cross(a, e1)


# ----------------------------------------------------------------------------- CMB
def dipole_factor(v_obs, n_hat):
    """D = sqrt(1-β²)/(1-β·n): T = T0·D (Eq. 6.2 of the author's doctoral thesis)."""
    beta = np.asarray(v_obs, float) / C_KMS
    n_hat = np.asarray(n_hat, float)
    return np.sqrt(1.0 - np.sum(beta * beta, axis=-1)) / (1.0 - np.sum(beta * n_hat, axis=-1))


H_PLANCK, K_BOLTZ = 6.62607015e-34, 1.380649e-23


def linearizer(nu_ghz, band_frac=0.25, T0=None, nodes=9):
    """"Linearized" temperature delivered by a broadband detector.

    The detector measures intensity (in-band power). Calibration converts the intensity change
    into temperature using the blackbody derivative at T0 (linear), as in WMAP and Planck.
    For a sky thermodynamic temperature T:
        T_lin = T0 + [I(T) − I(T0)] / I'(T0),   I(T) = band average of ν³ / (exp(hν/kT) − 1)
    To 2nd order: T_lin ≈ T + (ΔT²/2T0)·(x·coth(x/2) − 2), x = hν/kT0 (Kamionkowski &
    Knox 2003, PRD 67, 063001: dipole and its 2nd-order spectral distortion). Flat band of relative width band_frac.
    """
    T0 = T0_CMB if T0 is None else T0
    nus = nu_ghz * 1e9 * (1.0 + band_frac * np.linspace(-0.5, 0.5, nodes))
    c = H_PLANCK * nus / K_BOLTZ

    def I(T):
        T = np.asarray(T, float)[..., None]
        return np.mean(nus**3 / np.expm1(c / T), axis=-1)
    x0 = c / T0
    dI = np.mean(nus**3 * (x0 / T0) * np.exp(x0) / np.expm1(x0) ** 2)
    I0 = float(I(T0))

    def f(T):
        return T0 + (I(T) - I0) / dI
    return f


def lin_boost(nu_ghz, T0=None):
    """2nd-order factor x·coth(x/2) − 2 at the band centre (0 in the Rayleigh-Jeans limit)."""
    T0 = T0_CMB if T0 is None else T0
    x = H_PLANCK * nu_ghz * 1e9 / (K_BOLTZ * T0)
    return x / np.tanh(x / 2) - 2.0


def tetra_dirs(axis, roll=0.0):
    """4 sensors in a regular tetrahedron: 1 on the axis + 3 at 109.47° spaced by 120°.
    roll [rad]: rotation of the set about the axis (free; used to avoid the Galactic plane)."""
    a = unit(axis)
    e1, e2 = perpendicular_basis(a)
    th = np.arccos(-1 / 3)
    out = [a]
    for phi in roll + np.deg2rad([0.0, 120.0, 240.0]):
        out.append(np.cos(th) * a + np.sin(th) * (np.cos(phi) * e1 + np.sin(phi) * e2))
    return np.array(out)


def velocity_sigma_per_uK(v_obs, n):
    """1σ error of the velocity vector [m/s] for 1 µK of noise per sensor (Fisher, free T0)."""
    beta = v_obs / C_KMS
    g = np.sqrt(1 - beta @ beta)
    den = 1 - n @ beta
    D = g / den
    dD = (-beta[None, :] / (g * den[:, None]) + g * n / den[:, None] ** 2) / C_KMS
    J = np.hstack([T0_CMB * dD, D[:, None]]) / 1e-6
    cov = np.linalg.inv(J.T @ J)[:3, :3]
    return float(np.sqrt(np.trace(cov)) * 1e3)


# ----------------------------------------------------------------------------- ephemerides
# JPL approximate Keplerian elements (Standish), J2000 ecliptic, 1800-2050
_ELEMENTS = {
    "venus": ([0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
              [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418]),
    "earth": ([1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
              [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0]),
    "mars": ([1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
             [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343]),
    "jupiter": ([5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
                [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106]),
    "saturn": ([9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
               [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794]),
}
JD_J2000 = 2451545.0


def julian_date(year, month, day, hour=0.0):
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5 + hour / 24


# Ephemerides: 'kepler' (Standish mean elements, JPL; error ~10³–10⁴ km) or 'de421' (JPL DE421, 2020–2040
# excerpt in data/de421_2020_2040.npz; 'earth' = Earth-Moon barycentre, as in the Standish elements)
EPHEM = {"source": "kepler"}


def set_ephemeris(source):
    if source == "de421" and not de421_available():
        source = "kepler"
    EPHEM["source"] = source
    return source


def de421_available():
    from pathlib import Path
    return (Path(__file__).with_name("data") / "de421_2020_2040.npz").exists()


_DE = {}


def _de421():
    if not _DE:
        from pathlib import Path
        d = np.load(Path(__file__).with_name("data") / "de421_2020_2040.npz")
        _DE.update({k: d[k] for k in d.files})
    return _DE


def _cheb(body, jd):
    """Position [km] of the body relative to the Solar System barycentre (ICRF), JPL Chebyshev."""
    d = _de421()
    C = d[body]
    ja, span = float(d[body + "_jalpha"]), float(d[body + "_span"])
    jd = np.asarray(jd, float)
    jb = ja + C.shape[0] * span
    if np.any(jd < ja - 1e-9) or np.any(jd > jb + 1e-9):        # outside the excerpt the result would be silently wrong
        raise ValueError(f"DE421 truncated to JD {ja:.1f}–{jb:.1f} (2020–2040); date out of range. "
                         "Use ephem='kepler' or another date.")
    idx, off = np.divmod(jd - ja, span)
    last = idx >= C.shape[0]                                    # exact end of the table: last record, s = 1
    idx = np.where(last, C.shape[0] - 1, idx)
    off = np.where(last, span, off)
    idx = idx.astype(int)
    s_ = 2.0 * off / span - 1.0
    n = C.shape[2]
    T = [np.ones_like(s_), s_]
    for _ in range(2, n):
        T.append(2 * s_ * T[-1] - T[-2])
    T = np.stack(T[:n], axis=-1)                         # (..., n)
    coef = C[idx]                                       # (..., 3, n)
    return np.einsum("...kn,...n->...k", coef, T)


def planet_position(name, jd):
    """Heliocentric position [km], J2000 ecliptic. 'earth' = Earth-Moon barycentre; with DE421, 'earth_c'
    (Earth centre) and 'moon' (Moon) are also available."""
    if EPHEM["source"] == "de421":
        if name in ("earth_c", "moon"):
            d = _de421()
            emb = _cheb("earthmoon", jd)
            gm = _cheb("moon", jd)                              # Moon relative to the Earth
            k = float(d["EMRAT"])
            r = emb - gm / (1 + k) if name == "earth_c" else emb + gm * k / (1 + k)
            return (r - _cheb("sun", jd)) @ EQ_TO_ECL.T
        b = "earthmoon" if name == "earth" else name
        r = _cheb(b, jd) - _cheb("sun", jd)
        return r @ EQ_TO_ECL.T
    if name in ("earth_c", "moon"):
        name = "earth"
    return _planet_kepler(name, jd)


def sun_bary_velocity(jd, dt_days=0.05):
    """Velocity of the Sun relative to the Solar System barycentre [km/s, ecliptic] (DE421; ~10–15 m/s, dominated
    by Jupiter). The CMB dipole is measured in the barycentric frame: v_probe,SSB = v_helio + v_Sun,SSB + v_SSB.
    With the mean elements (no JPL Sun), returns zero."""
    if EPHEM["source"] != "de421":
        return np.zeros(3)
    d = (_cheb("sun", jd + dt_days) - _cheb("sun", jd - dt_days)) / (2 * dt_days * DAY)
    return d @ EQ_TO_ECL.T


def default_planets():
    """Perturbing bodies: with DE421, Earth and Moon separately; with the mean elements, their barycentre."""
    if EPHEM["source"] == "de421" and "moon" in _de421_files():
        return ("venus", "earth_c", "moon", "mars", "jupiter", "saturn")
    return ("venus", "earth", "mars", "jupiter", "saturn")


def _de421_files():
    try:
        return set(_de421().keys())
    except Exception:
        return set()


def _planet_kepler(name, jd):
    el0, rate = _ELEMENTS[name]
    T = (np.asarray(jd, float) - JD_J2000) / 36525.0
    a, e, I, L, varpi, Om = [el0[k] + rate[k] * T for k in range(6)]
    a = a * AU_KM
    I, L, varpi, Om = map(np.deg2rad, (I, L, varpi, Om))
    w = varpi - Om
    M = np.mod(L - varpi + np.pi, 2 * np.pi) - np.pi
    E = M + e * np.sin(M)
    for _ in range(12):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    xp, yp = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
    cw, sw, cO, sO, cI, sI = np.cos(w), np.sin(w), np.cos(Om), np.sin(Om), np.cos(I), np.sin(I)
    x = (cw * cO - sw * sO * cI) * xp + (-sw * cO - cw * sO * cI) * yp
    y = (cw * sO + sw * cO * cI) * xp + (-sw * sO + cw * cO * cI) * yp
    z = (sw * sI) * xp + (cw * sI) * yp
    return np.stack([x, y, z], axis=-1)


def planet_state(name, jd, dt=60.0):
    r = planet_position(name, jd)
    v = (planet_position(name, jd + dt / DAY) - planet_position(name, jd - dt / DAY)) / (2 * dt)
    return r, v


# ----------------------------------------------------------------------------- Lambert
def _C(z):
    if z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    if z < 0:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)
    return 0.5


def _S(z):
    if z > 0:
        q = np.sqrt(z)
        return (q - np.sin(q)) / q**3
    if z < 0:
        q = np.sqrt(-z)
        return (np.sinh(q) - q) / q**3
    return 1 / 6


def lambert(r1, r2, tof, mu=MU_SUN):
    """Lambert solver with universal variables (Curtis, Orbital Mechanics for Engineering Students, Alg. 5.2),
    prograde trajectory, 0 revolutions."""
    R1, R2 = np.linalg.norm(r1), np.linalg.norm(r2)
    dth = np.arccos(np.clip(r1 @ r2 / (R1 * R2), -1, 1))
    if np.cross(r1, r2)[2] < 0:
        dth = 2 * np.pi - dth
    A = np.sin(dth) * np.sqrt(R1 * R2 / (1 - np.cos(dth)))
    y = lambda z: R1 + R2 + A * (z * _S(z) - 1) / np.sqrt(_C(z))
    F = lambda z: (y(z) / _C(z)) ** 1.5 * _S(z) + A * np.sqrt(y(z)) - np.sqrt(mu) * tof
    z = -100.0
    while y(z) <= 0 or F(z) < 0:
        z += 0.1
    for _ in range(200):                      # Newton with numerical derivative
        h = 1e-6 * max(1.0, abs(z))
        dz = F(z) / ((F(z + h) - F(z - h)) / (2 * h))
        z -= dz
        if abs(dz) < 1e-10:
            break
    else:                                      # did not converge
        raise RuntimeError("Lambert: Newton did not converge (time of flight or geometry outside the 0-revolution range)")
    if abs(np.sin(dth)) < 1e-6:
        raise RuntimeError("Lambert: ~180° transfer is singular in this formulation")
    yz = y(z)
    f, g, gd = 1 - yz / R1, A * np.sqrt(yz / mu), 1 - yz / R2
    return (r2 - f * r1) / g, (gd * r2 - r1) / g


# ----------------------------------------------------------------------------- dynamics
C_KMS_GR = 299792.458


def gr_accel(r, v, mu=None):
    """First-order post-Newtonian relativistic correction of the Sun (Schwarzschild term, PPN β = γ = 1):
    a = μ/(c² r³)·[(4μ/r − v²)·r + 4(r·v)·v]  (IERS Conventions 2010, eq. 10.12; Moyer 2003; the same expression as
    GMAT's RelativisticCorrection.cpp, without the geodesic and Lense-Thirring terms, which are zero or negligible
    for the Sun). ~0.2 nm/s² at 1 AU: ~60 km and ~5 mm/s accumulated over a 300-day cruise."""
    mu = MU_SUN if mu is None else mu
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    v2 = np.sum(v * v, axis=-1, keepdims=True)
    rv = np.sum(r * v, axis=-1, keepdims=True)
    return mu / (C_KMS_GR**2 * rn**3) * ((4 * mu / rn - v2) * r + 4 * rv * v)


class Dynamics:
    """Sun + perturbing planets (point masses) + solar radiation pressure.

    Venus, Earth, Mars, Jupiter and Saturn. Departure and arrival (inside the spheres of influence)
    are not modelled: the cruise starts 5 days after departure and ends 5 days before arrival.

    State with 6 components (r, v) or 7 (r, v, s): s is the relative error of the radiation-pressure
    coefficient (true C_R·A/m = nominal × (1 + s)), estimated by the filter.
    `a_extra` (km/s², 3) is an unmodelled acceleration added only to the "truth".
    """

    def __init__(self, jd0, planets=None, mass_kg=500.0,
                 area_m2=1.0, CR=1.3, srp_scale=1.0, gr=True):
        self.jd0 = jd0
        planets = default_planets() if planets is None else planets
        self.gr = gr                # relativistic correction of the Sun (Schwarzschild), as in GMAT
        self.planets = planets
        self.mu = np.array([MU_PLANETS[p] for p in planets])
        self.k_srp_nom = CR * P_SR_1AU * AU_KM**2 * 1e-3 * area_m2 / mass_kg
        self.k_srp = self.k_srp_nom * srp_scale
        self.a_extra = None
        self.thrust = None          # function (t, x) -> thrust acceleration (km/s²), state (..., 6+)
        self._tab = None

    def tabulate(self, t0, t1, step=600.0):
        """Precomputes the planet positions (linear interpolation every 600 s: error up to ~0.5 km, the same
        for the truth and the filter)."""
        tt = np.arange(t0 - step, t1 + 2 * step, step)
        jd = self.jd0 + tt / DAY
        self._tab = (tt, np.stack([planet_position(p, jd) for p in self.planets], axis=1))

    def positions(self, t):
        if self._tab is not None and self._tab[0][0] <= t <= self._tab[0][-1]:
            tt, P = self._tab
            i = min(int((t - tt[0]) // (tt[1] - tt[0])), len(tt) - 2)
            w = (t - tt[i]) / (tt[i + 1] - tt[i])
            return P[i] * (1 - w) + P[i + 1] * w
        jd = self.jd0 + t / DAY
        return np.array([planet_position(p, jd) for p in self.planets])

    def accel(self, r, P, srp=None):
        rn3 = np.sum(r * r, axis=-1, keepdims=True) ** 1.5
        k = self.k_srp if srp is None else self.k_srp * (1.0 + srp)
        a = (-MU_SUN + k) * r / rn3
        for mu, p in zip(self.mu, P):
            d = r - p
            d3 = np.sum(d * d, axis=-1, keepdims=True) ** 1.5
            a = a - mu * (d / d3 + p / (p @ p) ** 1.5)
        if self.a_extra is not None:
            a = a + self.a_extra
        return a

    def deriv(self, t, x, P=None):
        P = self.positions(t) if P is None else P
        if x.shape[-1] == 7:
            a = self.accel(x[..., :3], P, x[..., 6:7])
        else:
            a = self.accel(x[..., :3], P)
        if self.gr:
            a = a + gr_accel(x[..., :3], x[..., 3:6])
        if self.thrust is not None:
            a = a + self.thrust(t, x)
        if x.shape[-1] == 7:
            return np.concatenate([x[..., 3:6], a, np.zeros_like(x[..., 6:7])], axis=-1)
        return np.concatenate([x[..., 3:], a], axis=-1)

    def rk4(self, x, t, dt):
        Pa, Pm, Pb = self.positions(t), self.positions(t + dt / 2), self.positions(t + dt)
        k1 = self.deriv(t, x, Pa)
        k2 = self.deriv(t, x + dt / 2 * k1, Pm)
        k3 = self.deriv(t, x + dt / 2 * k2, Pm)
        k4 = self.deriv(t, x + dt * k3, Pb)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def propagate(self, x, t, dt, max_step=3600.0):
        n = max(1, int(np.ceil(abs(dt) / max_step)))
        h = dt / n
        for i in range(n):
            x = self.rk4(x, t + i * h, h)
        return x
