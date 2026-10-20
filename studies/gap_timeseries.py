"""Article figure data: velocity error along the 60 d gap (with anomaly) for the CMB levels and the
references, with the filter 3σ band. One seed.

Usage (from the repository root): python studies/gap_timeseries.py
Output: results/gap_timeseries.npz (plotted in the article figure)."""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import gap_study as EL                                          # noqa: E402
import ratio_study as R                                             # noqa: E402
from cmbsim import physics as ph                                    # noqa: E402
from cmbsim.simulator import Config, run                            # noqa: E402

CASES = {"dyn": R.REFS_GAP["dyn"][1], "imu": R.REFS_GAP["imu"][1], "xnav": R.REFS_GAP["xnav"][1],
         "today": R.LEVELS["today"][1], "metro": R.LEVELS["metro"][1], "limit": R.LEVELS["limit"][1]}
SEED = 3


def one(k):
    a, b = EL.GAP_START, EL.GAP_START + R.GAP
    kw = {**EL.BASE, **CASES[k], "seed": SEED, "dsn_gaps": ((a, b),), "end_day": b + 2.0,
          "anomaly_days": (a + 2.0,), "anomaly_dv_mps": EL.ANOM_DV}
    r = run(Config(**kw))
    d = r.t / ph.DAY
    m = (d >= a - 5) & (d <= b)
    ev = np.linalg.norm(r.x_est[m, 3:6] - r.x_true[m, 3:], axis=1) * 1e3
    sv = np.sqrt(r.P_diag[m][:, 3:6].sum(1)) * 1e3
    return k, d[m] - a, ev, sv


if __name__ == "__main__":
    with Pool(2) as p:
        out = p.map(one, list(CASES))
    np.savez(os.path.join(EL.OUT, "gap_timeseries.npz"), **{f"{k}_{n}": v for k, t, e, s in out
                                                          for n, v in (("t", t), ("ev", e), ("sv", s))})
    for k, t, e, s in out:
        print(k, e[-1], s[-1])
