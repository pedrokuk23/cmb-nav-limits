"""Trims the JPL DE421 ephemerides (PyPI package 'de421', Brandon Rhodes, MIT license) to 2020–2040.

Input: the de421 package folder (files jpl-<body>.npy and constants.npy, JPL Chebyshev coefficients in km,
ICRF frame, centred on the Solar System barycentre), downloaded from PyPI into data_prep/raw/.
Output: cmbsim/data/de421_2020_2040.npz with the intervals of the period and the adjusted limits ("moon" is
geocentric). Usage:
    cd data_prep/raw && pip download de421 --no-deps && tar xzf de421-2008.1.tar.gz && cd ../..
    python data_prep/prepare_de421.py data_prep/raw/de421-2008.1/de421 cmbsim/data/de421_2020_2040.npz
"""
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
c = {k.decode(): v for k, v in np.load(f"{src}/constants.npy")}
ja, jo = float(c["jalpha"]), float(c["jomega"])
J0, J1 = 2458849.5, 2466154.5                     # 2020-01-01 to 2040-01-01
res = {"EMRAT": c["EMRAT"], "AU": c["AU"]}
for b in ("sun", "earthmoon", "moon", "venus", "mars", "jupiter", "saturn"):
    a = np.load(f"{src}/jpl-{b}.npy")
    span = (jo - ja) / a.shape[0]
    i0, i1 = int((J0 - ja) // span), int(np.ceil((J1 - ja) / span))
    res[b] = a[i0:i1]
    res[b + "_jalpha"] = ja + i0 * span
    res[b + "_span"] = span
np.savez_compressed(out, **res)
