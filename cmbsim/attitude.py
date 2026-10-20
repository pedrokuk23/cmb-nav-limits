"""Star-tracker attitude, ported from the author's 2017 undergraduate thesis star-tracker code (ITA).

Origin of each routine (original MATLAB code):
- quatrmx.m / sangvel.m / propQuat.m / angularVel.m  -> quatrmx, sangvel, propagate_quat, angular_velocity
- att_triad.m, att_qmethod.m, att_quest.m, att_esoq2.m -> att_triad, att_qmethod, att_quest, att_esoq2
- camera: FOV 12°, f = 58.4536 mm, 12 µm pixel, GSC6 catalogue (V <= 6), as in the
  attitude benchmark of the same code.

Convention (Carrara): quaternion q = [q1, q2, q3, q4] with the scalar part last; the attitude
matrix A = quatrmx(q) maps vectors from the inertial frame to the body frame: b = A r.

QUEST and ESOQ2 were ported literally (λ_max approximated by n, no Newton iteration).
To avoid the QUEST singularity at 180° rotations, the tracking mode is used: the reference
directions are pre-rotated by the previous attitude and the algorithm only solves for the
small correction (as also happens in a real sensor after acquisition).

Model for the CMB demonstrator: the CMB sensors and the star tracker share the same body;
the body z axis is the anti-Sun axis (CMB sensor 1) and the star tracker points 90° away
from it (+x; the optional 2nd head points along +y). An attitude error δθ (inertial) makes the
filter use rotated sensor directions: equivalent to measuring with velocity v + δθ × v. With
|v| ≈ 370 km/s, 1″ of cross-track error ≈ 1.8 m/s.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import physics as ph

ALGORITHMS = {
    "quest": "QUEST (2017 thesis code)",
    "esoq2": "ESOQ2 (2017 thesis code)",
    "qmethod": "Davenport q-method",
    "triad": "TRIAD (2 brightest stars)",
}


# ----------------------------------------------------------------------------- quaternions
def quatrmx(q):
    """quatrmx.m (Carrara, 1998): attitude matrix from the quaternion (scalar last)."""
    q1, q2, q3, q4 = q
    return np.array([
        [q1*q1 - q2*q2 - q3*q3 + q4*q4, 2*(q1*q2 + q3*q4), 2*(q1*q3 - q2*q4)],
        [2*(q1*q2 - q3*q4), -q1*q1 + q2*q2 - q3*q3 + q4*q4, 2*(q2*q3 + q1*q4)],
        [2*(q1*q3 + q2*q4), 2*(q2*q3 - q1*q4), -q1*q1 - q2*q2 + q3*q3 + q4*q4]])


def rmxquat(A):
    """Inverse of quatrmx (Shepperd's method, numerically stable)."""
    tr = np.trace(A)
    c = np.array([tr, A[0, 0], A[1, 1], A[2, 2]])
    k = int(np.argmax(c))
    if k == 0:
        q4 = 0.5 * np.sqrt(1 + tr)
        q = np.array([(A[1, 2] - A[2, 1]) / (4 * q4), (A[2, 0] - A[0, 2]) / (4 * q4),
                      (A[0, 1] - A[1, 0]) / (4 * q4), q4])
    elif k == 1:
        q1 = 0.5 * np.sqrt(1 + 2 * A[0, 0] - tr)
        q = np.array([q1, (A[0, 1] + A[1, 0]) / (4 * q1), (A[0, 2] + A[2, 0]) / (4 * q1),
                      (A[1, 2] - A[2, 1]) / (4 * q1)])
    elif k == 2:
        q2 = 0.5 * np.sqrt(1 + 2 * A[1, 1] - tr)
        q = np.array([(A[0, 1] + A[1, 0]) / (4 * q2), q2, (A[1, 2] + A[2, 1]) / (4 * q2),
                      (A[2, 0] - A[0, 2]) / (4 * q2)])
    else:
        q3 = 0.5 * np.sqrt(1 + 2 * A[2, 2] - tr)
        q = np.array([(A[0, 2] + A[2, 0]) / (4 * q3), (A[1, 2] + A[2, 1]) / (4 * q3), q3,
                      (A[0, 1] - A[1, 0]) / (4 * q3)])
    return q / np.linalg.norm(q)


def sangvel(w):
    """sangvel.m (Carrara, 1998): 4x4 matrix of the kinematics q' = ½ Ω(w) q."""
    return np.array([[0, w[2], -w[1], w[0]],
                     [-w[2], 0, w[0], w[1]],
                     [w[1], -w[0], 0, w[2]],
                     [-w[0], -w[1], -w[2], 0]])


def propagate_quat(q, w, dt):
    """propQuat.m integrated in closed form (w constant over the step): q(t+dt) = exp(½Ω dt) q."""
    wn = np.linalg.norm(w)
    if wn < 1e-15:
        return np.asarray(q, float).copy()
    a = 0.5 * wn * dt
    qn = (np.cos(a) * np.eye(4) + np.sin(a) / wn * sangvel(w)) @ q
    return qn / np.linalg.norm(qn)


def angular_velocity(q_prev, q_next, dt):
    """angularVel.m: angular velocity (body) from two consecutive quaternions."""
    dq = (np.asarray(q_next) - np.asarray(q_prev)) / dt
    a = q_prev
    Qt = np.array([[a[3], a[2], -a[1], -a[0]],
                   [-a[2], a[3], a[0], -a[1]],
                   [a[1], -a[0], a[3], -a[2]]])
    return 2 * Qt @ dq


def small_angle(A_est, A_true):
    """Attitude error (body, rad) such that A_est ≈ (I − [δθ×]) A_true."""
    D = A_est @ A_true.T
    return 0.5 * np.array([D[1, 2] - D[2, 1], D[2, 0] - D[0, 2], D[0, 1] - D[1, 0]])


def rot_small(dth):
    """Orthonormalized (I − [δθ×]) (small rotation)."""
    x, y, z = dth
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    th = np.linalg.norm(dth)
    if th < 1e-15:
        return np.eye(3)
    Kn = K / th
    return np.eye(3) - np.sin(th) * Kn + (1 - np.cos(th)) * Kn @ Kn


# ----------------------------------------------------------------------------- algorithms
def _bzs(r, b):
    B = b.T @ r                         # B = Σ b_i r_iᵀ  (att_*.m: B = b*r')
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([B[1, 2] - B[2, 1], B[2, 0] - B[0, 2], B[0, 1] - B[1, 0]])
    return B, S, sigma, Z


def att_triad(r1, r2, b1, b2):
    """att_triad.m: matrix D with D r1 = b1 and D r2 ≈ b2."""
    t1b = ph.unit(b1); t2b = ph.unit(np.cross(b1, b2)); t3b = np.cross(t1b, t2b)
    t1r = ph.unit(r1); t2r = ph.unit(np.cross(r1, r2)); t3r = np.cross(t1r, t2r)
    return np.column_stack([t1b, t2b, t3b]) @ np.column_stack([t1r, t2r, t3r]).T


def att_qmethod(r, b):
    """att_qmethod.m: eigenvector of the largest eigenvalue of Davenport's K matrix."""
    _, S, sigma, Z = _bzs(r, b)
    K = np.block([[S - sigma * np.eye(3), Z[:, None]], [Z[None, :], np.array([[sigma]])]])
    _, V = np.linalg.eigh(K)
    return V[:, -1]


def att_quest(r, b):
    """att_quest.m (literal, λ = n): Gibbs vector p and q = [p; 1]/√(1+pᵀp)."""
    _, S, sigma, Z = _bzs(r, b)
    lam = len(b)
    p = np.linalg.solve((lam + sigma) * np.eye(3) - S, Z)
    return np.append(p, 1.0) / np.sqrt(1 + p @ p)


def att_esoq2(r, b):
    """att_esoq2.m (literal, λ = n), as in the attitude benchmark of the 2017 thesis code."""
    _, S, sigma, Z = _bzs(r, b)
    lam = len(b)
    sa = S - (sigma + lam) * np.eye(3)
    m = (sigma - lam) * sa - np.outer(Z, Z)
    ma, mb, mc = np.diag(m)
    mx, my, mz = m[1, 0], m[2, 0], m[2, 1]
    minors = np.array([mb * mc - mz**2, ma * mc - my**2, ma * mb - mx**2])
    i = int(np.argmax(np.abs(minors)))
    if i == 0:
        y = np.array([minors[0], my * mz - mx * mc, mx * mz - my * mb])
    elif i == 1:
        y = np.array([my * mz - mx * mc, minors[1], mx * my - mz * ma])
    else:
        y = np.array([mx * mz - my * mb, mx * my - mz * ma, minors[2]])
    q = np.r_[(lam - sigma) * y, Z @ y]
    nq = np.linalg.norm(q)
    return q / nq if np.isfinite(nq) and nq > 1e-14 else np.full(4, np.nan)


def solve_attitude(method, r, b, mags=None):
    """Attitude matrix A (b = A r) with the selected algorithm. r, b: (n,3)."""
    if method == "triad":
        o = np.argsort(mags) if mags is not None else np.arange(len(r))
        i, j = o[0], o[1]
        return att_triad(r[i], r[j], b[i], b[j])
    q = {"quest": att_quest, "esoq2": att_esoq2, "qmethod": att_qmethod}[method](r, b)
    if not np.all(np.isfinite(q)):
        q = att_qmethod(r, b)
    return quatrmx(q)


# ----------------------------------------------------------------------------- catalogue and camera
CATALOGS = {"gsc6": "GSC6 from the author's undergraduate star-tracker project (V ≤ 6; 4401 stars)",
            "hipparcos": "Hipparcos (V ≤ 7.5; 25 713 stars, cedar-solve file from the author's undergraduate star-tracker project)"}


@lru_cache(maxsize=2)
def catalogue(name="gsc6"):
    """Star catalogue: directions in the J2000 ecliptic frame and V magnitudes.
    'gsc6': GSC6 from the 2017 thesis code (V ≤ 6). 'hipparcos': Hipparcos (ESA 1997) up to V = 7.5,
    from the file hipparcos_v8_cedarsolve051.npz (ICRS vectors)."""
    if name == "hipparcos":
        d = np.load(Path(__file__).with_name("data") / "hip75.npz")
        return d["vec"].astype(np.float64) @ ph.EQ_TO_ECL.T, d["vmag"].astype(np.float64)
    d = np.loadtxt(Path(__file__).with_name("data") / "gsc6.csv", delimiter=",")
    ra, de = np.deg2rad(d[:, 2]), np.deg2rad(d[:, 3])
    eq = np.column_stack([np.cos(ra) * np.cos(de), np.sin(ra) * np.cos(de), np.sin(de)])
    return eq @ ph.EQ_TO_ECL.T, d[:, 1]


# Identification (2017 thesis code): lost-in-space requires > 4 identified stars (compare.m); tracking
# (Strack.m) continues with ≥ 3 starting from the previous attitude propagated by the angular velocity;
# with the gyroscope as prior (INS-aided benchmark), 2 stars suffice.
IDENT_MODES = {"lost": "Lost-in-space every frame (≥ 5 stars)",
               "tracking": "Tracking (acquisition with ≥ 5, continues with ≥ 3)",
               "ins": "Gyro-aided (≥ 2 stars; requires gyroscope)"}
N_ACQ, N_TRACK, N_AIDED = 5, 3, 2


def ident_valid(counts, mode, spin, has_gyro):
    """Frames (spin phases, in order) with identified attitude, for the given identification mode."""
    c = np.asarray(counts)
    if mode == "ins" and has_gyro:
        return c >= N_AIDED
    if mode == "lost" or not spin:
        return c >= N_ACQ
    ok = np.zeros(len(c), bool)
    trk = False
    for rep in range(2):                     # 2 passes: the 2nd starts from the steady-state tracking state
        for k, n in enumerate(c):
            if trk and n >= N_TRACK:
                ok[k] = True
            elif n >= N_ACQ:
                trk, ok[k] = True, True
            else:
                trk, ok[k] = False, False
    return ok


# Exclusion by bright bodies in the field of view (half-angle beyond the apparent radius) [degrees]
R_BODY_KM = {"earth": 6378.0, "mars": 3390.0, "venus": 6052.0, "jupiter": 71492.0, "saturn": 60268.0}
MOON_ORBIT_KM = 384400.0
EARTH_NEAR_KM = 2.0e6


@dataclass
class StarTracker:
    fov_deg: float = 12.0
    focal_mm: float = 58.4536
    pixel_mm: float = 0.012
    mag_limit: float = 6.0
    centroid_px: float = 0.25
    frame_hz: float = 5.0
    exposure_s: float = 0.1
    heads: int = 1
    method: str = "quest"
    boresight_deg: float = 90.0     # angle between the optical axis and the spin axis (body z)
    catalog: str = "gsc6"
    ident: str = "tracking"         # IDENT_MODES
    excl_on: bool = False           # exclusion by Sun, Earth/Moon and planets in the field of view
    sun_excl_deg: float = 45.0
    earth_excl_deg: float = 25.0
    moon_excl_deg: float = 15.0
    planet_excl_deg: float = 2.0

    @property
    def ifov_rad(self):
        return self.pixel_mm / self.focal_mm

    @property
    def sigma_star(self):
        """Angular error of one star per frame [rad]."""
        return self.centroid_px * self.ifov_rad

    def smear_px(self, rate_dps):
        """Typical star smear during spin: the apparent rate is ω·sin θ (θ = angle from the star to the
        spin axis); with the optical axis near the spin axis, the smear at the field edge remains."""
        s_eff = max(np.sin(np.deg2rad(self.boresight_deg)), np.sin(np.deg2rad(self.fov_deg / 3)))
        return np.deg2rad(rate_dps) * s_eff * self.exposure_s / self.ifov_rad

    def degraded(self, rate_dps):
        """Effect of smear (spin): worse centroid and fewer detected stars.
        Assumption: read noise dominates -> SNR ∝ 1/√L, with L the smear in pixels."""
        L = max(1.0, self.smear_px(rate_dps))
        return self.centroid_px * np.sqrt(L), self.mag_limit - 1.25 * np.log10(L)


def mount(axis, phase=0.0):
    """Body->inertial matrix: z = anti-Sun axis, x = head 1 (rotated by 'phase' about z)."""
    z = ph.unit(axis)
    e1, e2 = ph.perpendicular_basis(z)
    x = np.cos(phase) * e1 + np.sin(phase) * e2
    return np.column_stack([x, np.cross(z, x), z])          # columns: body x, y, z in the inertial frame


HEAD_BORESIGHT = [np.array([1.0, 0, 0]), np.array([0, 1.0, 0])]


def head_frames(C, boresight_deg=90.0):
    """For each head: inertial->camera matrix (camera z = optical axis), with the optical axis at
    boresight_deg from the spin axis (body z), towards x (head 1) or y (head 2)."""
    out = []
    be = np.deg2rad(boresight_deg)
    for k, bs in enumerate(HEAD_BORESIGHT):
        zc = C @ (np.cos(be) * np.array([0, 0, 1.0]) + np.sin(be) * bs)
        ref = C @ np.array([0, 0, 1.0]) if boresight_deg > 30 else C @ bs
        xc = ph.unit(ref - (ref @ zc) * zc)
        out.append(np.vstack([xc, np.cross(zc, xc), zc]))
    return out


def visible(st: StarTracker, Mcam, mag_limit=None):
    """Stars in the (square) field of view of one head: inertial and camera directions, and magnitudes."""
    cat, mag = catalogue(st.catalog)
    lim = st.mag_limit if mag_limit is None else mag_limit
    bc = cat @ Mcam.T
    half = np.tan(np.deg2rad(st.fov_deg / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        xp, yp = bc[:, 0] / bc[:, 2], bc[:, 1] / bc[:, 2]
    ok = (bc[:, 2] > 0) & (np.abs(xp) <= half) & (np.abs(yp) <= half) & (mag <= lim)
    return cat[ok], bc[ok], mag[ok]


def field(st: StarTracker, C, mag_limit=None):
    """Stars from all heads in use: (inertial r, body b, mag, head index)."""
    R, Bv, M, H = [], [], [], []
    for k, Mc in enumerate(head_frames(C, st.boresight_deg)[:st.heads]):
        r, _, m = visible(st, Mc, mag_limit)
        R.append(r); Bv.append(r @ C); M.append(m); H.append(np.full(len(r), k))
    return np.vstack(R), np.vstack(Bv), np.concatenate(M), np.concatenate(H)


def frame_cov(b, sigma):
    """Cramér-Rao bound (Shuster's QUEST model) of the per-frame error, in the body frame [rad²]."""
    F = sum(np.eye(3) - np.outer(bi, bi) for bi in b) / sigma**2
    return np.linalg.inv(F) if len(b) >= 2 else np.full((3, 3), np.nan)


def solve_noisy(st: StarTracker, r, b, mags, sigma, rng, method=None):
    """One attitude solution with angular noise σ on each star. Returns δθ in the body frame [rad].

    r: catalogue directions (inertial); b: the same in the body frame (noise-free); A_true = b ⟵ r.
    QUEST uses tracking mode (reference pre-rotated by the previous attitude, here the true one)
    to avoid the Gibbs-vector singularity at 180°; the others receive the catalogue reference,
    as in the original code."""
    method = method or st.method
    n = len(b)
    ref = np.where(np.abs(b[:, [2]]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
    e1 = ph.unit(np.cross(b, ref))
    e2 = np.cross(b, e1)
    g = rng.normal(0, sigma, (n, 2))
    bn = ph.unit(b + g[:, :1] * e1 + g[:, 1:] * e2)
    if method == "quest":
        return small_angle(solve_attitude(method, b, bn, mags), np.eye(3))
    A_true = np.linalg.lstsq(r, b, rcond=None)[0].T          # b = A r
    return small_angle(solve_attitude(method, r, bn, mags), A_true)


def benchmark(st: StarTracker, C, n_frames=300, seed=0, methods=None):
    """Per-frame error (body) for each algorithm, with the same noise: RMS per axis [″]."""
    rng = np.random.default_rng(seed)
    r, b, m, _ = field(st, C)
    out = {}
    if len(b) < 2:
        return out, len(b)
    states = rng.bit_generator.state
    for meth in (methods or list(ALGORITHMS)):
        rng.bit_generator.state = states                    # same noise realizations
        E = np.array([solve_noisy(st, r, b, m, st.sigma_star, rng, meth) for _ in range(n_frames)])
        out[meth] = np.sqrt(np.mean(E**2, axis=0)) / ph.ARCSEC
    out["crlb"] = np.sqrt(np.diag(frame_cov(b, st.sigma_star))) / ph.ARCSEC
    return out, len(b)


# Gyroscopes (orders of magnitude; 1σ): angle random walk (ARW) and bias instability
GYROS = {
    "none": None,
    "mems": dict(label="MEMS (tactical grade)", arw_deg_rth=0.1, bias_deg_h=1.0),
    "fog": dict(label="Fiber-optic (space grade)", arw_deg_rth=0.003, bias_deg_h=0.01),
    "hrg": dict(label="Hemispherical resonator (HRG)", arw_deg_rth=0.0003, bias_deg_h=0.001),
}


@lru_cache(maxsize=256)
def gyro_star_window_sigma(sig_meas_as, avail, frame_hz, n_frames, arw_deg_rth, bias_deg_h, chains=200, seed=2026):
    """Window-averaged error (1σ, ″) of a per-axis gyro-stellar Kalman filter.

    State [θ, b]: the gyroscope propagates the attitude (angle noise σ_v, random-walk bias σ_u);
    each star-tracker frame (with probability 'avail') measures θ with σ_meas. This is the classic
    per-axis filter of Farrenkopf (1978) / Lefferts, Markley & Shuster (1982).

    The filter starts from steady state (Riccati iterated to convergence and initial errors drawn
    from that covariance), with 'chains' chains (default 200); a warm-up that depends on the window
    length with few chains would give a pessimistic result."""
    dt = 1.0 / frame_hz
    sv = arw_deg_rth * np.pi / 180 / 60                    # rad/√s
    su = bias_deg_h * np.pi / 180 / 3600 / 60              # rad/s/√s: the bias walks B in 1 h
    sm = sig_meas_as * np.pi / 180 / 3600
    N = int(min(n_frames, 20000))
    rng = np.random.default_rng(seed)
    q11, q12, q22 = sv**2 * dt + su**2 * dt**3 / 3, -su**2 * dt**2 / 2, su**2 * dt
    Lq = np.linalg.cholesky(np.array([[q11, q12], [q12, q22]]) + np.eye(2) * 1e-40)
    # 1) steady-state covariance (same random availability sequence, until convergence)
    P11, P12, P22 = sm**2, 0.0, (su * 3600) ** 2 + 1e-30
    for k in range(200000):
        P11o, P22o = P11, P22
        P11, P12, P22 = P11 - 2 * P12 * dt + P22 * dt**2 + q11, P12 - P22 * dt + q12, P22 + q22
        if rng.random() < avail:
            S = P11 + sm**2
            K1, K2 = P11 / S, P12 / S
            P11, P12, P22 = (1 - K1) * P11, (1 - K1) * P12, P22 - K2 * P12
        if k > 1000 and abs(P11 - P11o) < 1e-9 * P11 and abs(P22 - P22o) < 1e-9 * P22:
            break
    # 2) initial errors drawn from the steady-state covariance
    Pss = np.array([[P11, P12], [P12, P22]])
    L0 = np.linalg.cholesky(Pss + np.eye(2) * 1e-40)
    z0 = L0 @ rng.normal(0, 1, (2, chains))
    e, be = z0[0].copy(), z0[1].copy()
    acc = np.zeros(chains)
    for k in range(N):
        w = Lq @ rng.normal(0, 1, (2, chains))
        e = e - be * dt - w[0]
        be = be - w[1]
        P11, P12, P22 = P11 - 2 * P12 * dt + P22 * dt**2 + q11, P12 - P22 * dt + q12, P22 + q22
        if rng.random() < avail:
            S = P11 + sm**2
            K1, K2 = P11 / S, P12 / S
            y = -e + rng.normal(0, sm, chains)             # innovation (measurement − estimate)
            e = e + K1 * y
            be = be + K2 * y
            P11, P12, P22 = (1 - K1) * P11, (1 - K1) * P12, P22 - K2 * P12
        acc += e
    mean_e = acc / N * np.sqrt(N / max(n_frames, 1)) ** (1 if n_frames > N else 0)
    return float(np.sqrt(np.mean(mean_e**2)) * 180 / np.pi * 3600)

class AttitudeModel:
    """Attitude error per CMB measurement window.

    Three contributions:
    - fixed misalignment between the star tracker and the CMB sensors (body frame);
    - slow thermal drift (1st-order Gauss-Markov, body frame);
    - star-tracker noise averaged over the window (inertial frame): one solution of the selected
      algorithm with σ_star/√N, N = valid frames in the window."""

    def __init__(self, st: StarTracker, tau_s, rng, misalign_arcsec, drift_arcsec, drift_tau_h,
                 spin=False, spin_rate_dps=0.0, gyro="none", coast_max_arcsec=30.0):
        self.st, self.tau, self.rng = st, tau_s, rng
        self.gyro = GYROS.get(gyro)
        self.spin, self.rate = spin, spin_rate_dps
        self.sig_mis = misalign_arcsec * ph.ARCSEC
        self.sig_lf = drift_arcsec * ph.ARCSEC
        self.tc = max(drift_tau_h, 1e-3) * 3600.0
        self.mis_b = rng.normal(0, self.sig_mis, 3)
        self.lf_b = rng.normal(0, self.sig_lf, 3)
        self.t_last = None
        if spin:
            sig_px, self.mag_lim = st.degraded(spin_rate_dps)
            self.sig_frame = sig_px * st.ifov_rad
        else:
            self.sig_frame, self.mag_lim = st.sigma_star, st.mag_limit
        self.n_frames = max(1, int(round(tau_s * st.frame_hz)))
        self.t_fix, self.coast = None, None         # 1st window: known attitude (acquisition phase)
        # separate generator for the window noise and the gyroscope: the drift and misalignment (which dominate)
        # stay the same when the catalogue, identification mode or gyroscope is changed (fair comparison)
        self.rng_w = np.random.default_rng(int(rng.integers(2**31)) + 99)
        self.n_coast, self.n_reduced = 0, 0
        self.coast_max = coast_max_arcsec * ph.ARCSEC

    def set_pointing(self, axis, roll=0.0):
        """New pointing: stores the star fields of all phases (and heads) and the per-window noise
        covariance without exclusions."""
        self.axis = ph.unit(axis)
        phases = np.linspace(0, 2 * np.pi, 36, endpoint=False) if self.spin else [roll]
        self.all_fields = []
        bores = []
        for p in phases:
            C = mount(self.axis, p)
            r, b, m, h = field(self.st, C, self.mag_lim)
            self.all_fields.append((C, r, b, m, h))
            bores.append([M_[2] for M_ in head_frames(C, self.st.boresight_deg)[:self.st.heads]])
        self.bores = np.array(bores)                                # (phases, heads, 3) inertial
        self._cache = {}
        self.coast = None
        st0 = self._stats(np.zeros(self.bores.shape[:2], bool))
        self._apply(st0)
        self.avail0 = self.avail
        return self

    def _stats(self, blocked):
        """Valid phases and per-window noise with the heads blocked in 'blocked' (phases × heads)."""
        key = blocked.tobytes()
        if key in self._cache:
            return self._cache[key]
        flds, counts = [], []
        for k, (C, r, b, m, h) in enumerate(self.all_fields):
            keep = ~blocked[k][h] if len(h) else np.zeros(0, bool)
            flds.append((C, r[keep], b[keep], m[keep]))
            counts.append(int(keep.sum()))
        ok = ident_valid(counts, self.st.ident, self.spin, self.gyro is not None) & (np.array(counts) >= 2)
        fields = [f for f, o in zip(flds, ok) if o]
        out = dict(fields=fields, avail=len(fields) / len(flds), cov_white=None, n_stars=0.0,
                   sig_win_gyro=None, cov_crlb=None)
        if fields:
            avail = out["avail"]
            covs = [f[0] @ frame_cov(f[2], self.sig_frame) @ f[0].T for f in fields]
            out["cov_crlb"] = np.mean(covs, axis=0) / (self.n_frames * avail)
            # covariance of the selected ALGORITHM (e.g. TRIAD is far from the bound): Monte Carlo
            rng_c = np.random.default_rng(12345)
            sig = self.sig_frame / np.sqrt(self.n_frames * avail)
            K = max(200, 6 * len(fields))
            D = np.array([fields[k % len(fields)][0] @ solve_noisy(self.st, *fields[k % len(fields)][1:], sig, rng_c)
                          for k in range(K)])
            out["cov_white"] = D.T @ D / K
            out["n_stars"] = float(np.mean([len(f[2]) for f in fields]))
            if self.gyro is not None:
                # gyroscope + stars: window-averaged error per body axis, from the Kalman filter
                fb = np.mean([frame_cov(f[2], self.sig_frame) for f in fields], axis=0)
                sm = np.sqrt(np.clip(np.diag(fb), 0, None)) / ph.ARCSEC
                sg = np.array([gyro_star_window_sigma(round(float(x), 2), round(avail, 3), float(self.st.frame_hz),
                                                      int(self.n_frames), self.gyro["arw_deg_rth"],
                                                      self.gyro["bias_deg_h"]) for x in sm]) * ph.ARCSEC
                # the window reconstruction can smooth (forward and backward): never worse than the frame average
                sg = np.minimum(sg, sm * ph.ARCSEC / np.sqrt(max(self.n_frames * avail, 1.0)))
                out["sig_win_gyro"] = sg
                out["cov_white"] = np.mean([f[0] @ np.diag(sg**2) @ f[0].T for f in fields], axis=0)
        self._cache[key] = out
        return out

    def _apply(self, st_):
        self.fields, self.avail, self.cov_white = st_["fields"], st_["avail"], st_["cov_white"]
        self.n_stars, self.sig_win_gyro, self.cov_crlb = st_["n_stars"], st_["sig_win_gyro"], st_["cov_crlb"]

    def blocked(self, pos_km, jd):
        """Heads with the Sun, Earth/Moon or a planet inside the exclusion zone, per phase (phases × heads)."""
        st = self.st
        bodies = [(-np.asarray(pos_km, float), st.sun_excl_deg)]
        for nm, Rk in R_BODY_KM.items():
            d = ph.planet_position(nm, jd) - pos_km
            dn = float(np.linalg.norm(d))
            rad = np.degrees(np.arcsin(min(1.0, Rk / dn)))
            if nm == "earth":
                # limb stray light: full exclusion up to ~2 million km, beyond which the Earth becomes a bright point
                half = max(st.planet_excl_deg, st.earth_excl_deg * min(1.0, EARTH_NEAR_KM / dn)) + rad
                # the Moon is up to 384 400 km from the Earth
                half = max(half, max(st.planet_excl_deg, st.moon_excl_deg * min(1.0, EARTH_NEAR_KM / dn))
                           + np.degrees(np.arcsin(min(1.0, MOON_ORBIT_KM / dn))))
            else:
                half = st.planet_excl_deg + rad
            bodies.append((d, half))
        blk = np.zeros(self.bores.shape[:2], bool)
        for d, half in bodies:
            u = ph.unit(d)
            blk |= (self.bores @ u) > np.cos(np.deg2rad(half))
        return blk

    @property
    def var_body(self):
        """Per-axis variance (rad²) of the body-frame error (misalignment + steady-state drift)."""
        return self.sig_mis**2 + self.sig_lf**2

    @property
    def q_drift(self):
        """Process-noise intensity of the drift (rad²/s per axis, body frame)."""
        return 2 * self.sig_lf**2 / self.tc

    def sample(self, t, pos_km=None, jd=None):
        """Error of the window centred at t: (inertial noise, body drift, body misalignment [rad],
        inertial noise covariance [rad²]). None if no attitude is available (CMB measurement discarded).

        With bright-body exclusion (pos_km, jd), blocked phases are removed from the window. If no phase
        remains, there is a gap: with a gyroscope, the attitude is propagated from the last window with stars
        (ARW + residual bias, error correlated between windows) while σ ≤ coast_max; without a gyroscope, the
        measurement is lost."""
        if self.t_last is not None:
            phi = np.exp(-(t - self.t_last) / self.tc)
            self.lf_b = phi * self.lf_b + np.sqrt(max(1 - phi**2, 0.0)) * self.rng.normal(0, self.sig_lf, 3)
        dt_ = 0.0 if self.t_last is None else t - self.t_last
        if self.t_fix is None:
            self.t_fix = t
        self.t_last = t
        if self.st.excl_on and pos_km is not None and jd is not None:
            st_ = self._stats(self.blocked(pos_km, jd))
            if st_["avail"] < self.avail0 - 1e-9:
                self.n_reduced += 1
        else:
            st_ = self._stats(np.zeros(self.bores.shape[:2], bool))
        self._apply(st_)
        if not self.fields:
            if self.gyro is None or self.t_fix is None:
                return None
            arw = self.gyro["arw_deg_rth"] * np.pi / 180 / 60       # rad/√s
            bias = self.gyro["bias_deg_h"] * np.pi / 180 / 3600     # rad/s
            if self.coast is None:                                   # start of the gap: draw the residual bias
                self.coast = dict(e=np.zeros(3), b=self.rng_w.normal(0, bias, 3))
            dt_c = t - self.t_fix
            self.coast["e"] = self.coast["e"] + self.coast["b"] * dt_ + self.rng_w.normal(0, arw * np.sqrt(max(dt_, 0.0)), 3)
            sig = np.sqrt(arw**2 * dt_c + (bias * dt_c) ** 2)
            if sig > self.coast_max:
                return None
            self.n_coast += 1
            C = self.all_fields[0][0]
            return C @ self.coast["e"], self.lf_b.copy(), self.mis_b.copy(), np.eye(3) * sig**2
        self.t_fix, self.coast = t, None
        C, r, b, m = self.fields[self.rng_w.integers(len(self.fields))]
        if self.sig_win_gyro is not None:
            white = C @ (self.rng_w.normal(0, 1, 3) * self.sig_win_gyro)
        else:
            sig = self.sig_frame / np.sqrt(self.n_frames * self.avail)
            white = C @ solve_noisy(self.st, r, b, m, sig, self.rng_w)
        return white, self.lf_b.copy(), self.mis_b.copy(), self.cov_white
