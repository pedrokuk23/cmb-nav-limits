"""Unscented Kalman filter (Eqs. 6.26-6.36), additive noise, scaled unscented transform
(Wan & Van der Merwe, 2000). Replaces MATLAB's 'unscentedKalmanFilter'."""
import numpy as np


class UKF:
    def __init__(self, fx, hx, x0, P0, Q, R, alpha=1e-3, beta=2.0, kappa=0.0):
        """
        fx(X, t, dt) -> propagated X; X: (2n+1, n) sigma points
        hx(X, t)     -> Z: (2n+1, m)
        """
        self.fx, self.hx = fx, hx
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.Q, self.R = np.asarray(Q, float), np.asarray(R, float)
        n = self.x.size
        self.n = n
        self.lam = alpha**2 * (n + kappa) - n
        self.Wm = np.full(2 * n + 1, 1.0 / (2 * (n + self.lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lam / (n + self.lam)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    def sigma_points(self):
        P = 0.5 * (self.P + self.P.T)
        try:
            S = np.linalg.cholesky((self.n + self.lam) * P)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(P)
            S = V @ np.diag(np.sqrt(np.clip(w, 1e-30, None) * (self.n + self.lam)))
        return np.vstack([self.x, self.x + S.T, self.x - S.T])     # Eq. 6.26

    def predict(self, t, dt):
        X = self.fx(self.sigma_points(), t, dt)                      # Eq. 6.27
        self.x = self.Wm @ X                                         # Eq. 6.28
        d = X - self.x
        self.P = d.T @ (self.Wc[:, None] * d) + self.Q               # Eq. 6.29
        self._X = X

    def update(self, z, t, R=None, hx=None):
        R = self.R if R is None else R
        hx = self.hx if hx is None else hx
        X = self.sigma_points()
        Z = hx(X, t)                                                 # Eq. 6.30
        zm = self.Wm @ Z                                             # Eq. 6.31
        dz = Z - zm
        dx = X - self.x
        S = dz.T @ (self.Wc[:, None] * dz) + R                       # Eq. 6.32
        Pxz = dx.T @ (self.Wc[:, None] * dz)                         # Eq. 6.33
        K = np.linalg.solve(S.T, Pxz.T).T                            # Eq. 6.34
        self.x = self.x + K @ (z - zm)                               # Eq. 6.35
        self.P = self.P - K @ S @ K.T                                # Eq. 6.36 (equivalent form)
        self.P = 0.5 * (self.P + self.P.T)


class EKF:
    """Extended Kalman filter with central-difference Jacobians, same interface as the UKF
    (fx/hx vectorized over rows). The step for each state is eps·σ_i, so the linearization follows the
    uncertainty (states with σ = 0 do not contribute). Joseph-form update."""

    def __init__(self, fx, hx, x0, P0, Q, R, eps=1e-2, fjac=None):
        """fjac(x, t, dt) -> (propagated x, F): prediction with an analytic Jacobian (no finite differences)."""
        self.fx, self.hx, self.fjac = fx, hx, fjac
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.Q, self.R = np.asarray(Q, float), np.asarray(R, float)
        self.n = self.x.size
        self.eps = eps

    def _points(self):
        s = np.sqrt(np.clip(np.diag(self.P), 0, None))
        d = np.where(s > 0, self.eps * s, 1e-6)
        E = np.diag(d)
        return np.vstack([self.x, self.x + E, self.x - E]), d

    def predict(self, t, dt):
        if self.fjac is not None:
            self.x, F = self.fjac(self.x.copy(), t, dt)
            self.P = F @ self.P @ F.T + self.Q
            self.P = 0.5 * (self.P + self.P.T)
            return
        X, d = self._points()
        Y = self.fx(X, t, dt)
        n = self.n
        F = ((Y[1:n + 1] - Y[n + 1:]) / (2 * d[:, None])).T
        self.x = Y[0]
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, z, t, R=None, hx=None):
        R = self.R if R is None else R
        hx = self.hx if hx is None else hx
        X, d = self._points()
        Z = hx(X, t)
        n = self.n
        H = ((Z[1:n + 1] - Z[n + 1:]) / (2 * d[:, None])).T
        S = H @ self.P @ H.T + R
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T
        self.x = self.x + K @ (z - Z[0])
        A = np.eye(n) - K @ H
        self.P = A @ self.P @ A.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)


class GPF(UKF):
    """Gaussian particle filter with an unscented proposal (Kotecha & Djurić 2003; van der Merwe et al. 2000).

    The prediction is the UKF's (propagating thousands of particles through the full dynamics would be costly).
    At each measurement: (1) the UKF update gives the proposal N(x_u, P_u); (2) N particles are drawn from it;
    (3) weight w ∝ p(z|x)·N(x; x⁻, P⁻)/N(x; x_u, P_u), with the exact h(x) at each particle; (4) the weighted
    mean and covariance become the new state. If the effective sample size drops below twice the number of
    sampled directions, the sample covariance is unreliable: the UKF covariance is kept (only the mean is
    corrected); these fallbacks are counted in `n_fallback`. For a nearly linear measurement model the weights are
    nearly uniform and the result matches the UKF; that is precisely the test. The filter shares the UKF
    prediction, so it checks the nonlinearity of the measurement update, not of the dynamics."""

    def __init__(self, fx, hx, x0, P0, Q, R, n_particles=1000, seed=0, **kw):
        super().__init__(fx, hx, x0, P0, Q, R, **kw)
        self.N = int(n_particles)
        self.rng = np.random.default_rng(seed + 31337)
        self.ess = []
        self.n_fallback = 0
        self.n_dim = []

    @staticmethod
    def _factor(P):
        """P = (D·V·diag(s))(…)ᵀ on the correlation matrix (states with very different units: km, km/s, ppm,
        nK); returns (D, V, s) restricted to the directions of nonzero variance."""
        P = 0.5 * (P + P.T)
        d = np.sqrt(np.clip(np.diag(P), 0, None))
        d = np.where(d > 0, d, 1.0)
        C = P / np.outer(d, d)
        w, V = np.linalg.eigh(C)
        ok = w > max(w.max(), 0) * 1e-12
        return d, V[:, ok], np.sqrt(w[ok])

    def update(self, z, t, R=None, hx=None):
        R = self.R if R is None else R
        hx = self.hx if hx is None else hx
        x0, P0 = self.x.copy(), self.P.copy()
        super().update(z, t, R=R, hx=hx)
        xu, Pu = self.x.copy(), self.P.copy()
        du, Vu, su = self._factor(Pu)
        xi = self.rng.standard_normal((self.N // 2, su.size))
        xi = np.vstack([xi, -xi])                                    # antithetic sampling
        self.N = xi.shape[0]
        Xp = xu + ((xi * su) @ Vu.T) * du
        Zp = hx(Xp, t)
        Ri = np.linalg.inv(R)
        dz = z - Zp
        lw = -0.5 * np.einsum("ij,jk,ik->i", dz, Ri, dz)
        d0, V0, s0 = self._factor(P0)
        y0 = (((Xp - x0) / d0) @ V0) / s0
        lw += -0.5 * np.sum(y0**2, 1) + 0.5 * np.sum(xi**2, 1)
        lw -= lw.max()
        w = np.exp(lw)
        w /= w.sum()
        ess = 1.0 / np.sum(w**2)
        self.ess.append(ess / self.N)
        self.n_dim.append(su.size)
        # control variate: subtract the UNWEIGHTED mean/covariance of the same particles (whose expected
        # value is exactly x_u, P_u). Without it, the sampling noise (σ/√N at each measurement) accumulates
        # over thousands of updates without P accounting for it, and the filter becomes inconsistent.
        d = Xp - xu
        u = np.full(self.N, 1.0 / self.N)
        dx = (w - u) @ d
        self.x = xu + dx
        if ess > 2 * su.size:
            Pn = Pu + d.T @ ((w - u)[:, None] * d) - np.outer(dx, dx)
            Pn = 0.5 * (Pn + Pn.T)
            if np.linalg.eigvalsh(Pn).min() > -1e-12 * np.abs(np.diag(Pn)).max():
                self.P = Pn
                return
        self.n_fallback += 1
        self.P = Pu
