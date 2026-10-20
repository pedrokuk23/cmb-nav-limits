"""Downgrades the Planck CO maps (HFI_CompMap_CO-Type1_512_R2.00_ring.fits, the templates of the PySM3 co1
model) to nside 64, smoothed to 1°, and writes cmbsim/data/co_planck_n64.npz.

The input file (not distributed here) is downloaded from the PySM3 data portal into data_prep/raw/:
  https://portal.nersc.gov/project/cmb/pysm-data/co/HFI_CompMap_CO-Type1_512_R2.00_ring.fits
Columns: CO 1-0, 2-1 and 3-2 lines, in K_CMB as seen in the HFI 100, 217 and 353 GHz channels (Planck 2015 X).
Usage: python data_prep/prepare_co.py data_prep/raw/HFI_CompMap_CO-Type1_512_R2.00_ring.fits cmbsim/data/co_planck_n64.npz
"""
import sys

import healpy as hp
import numpy as np

m = hp.read_map(sys.argv[1], field=[0, 1, 2])
out = {}
for k, name in enumerate(("co10", "co21", "co32")):
    s = hp.smoothing(m[k], fwhm=np.deg2rad(1.0), lmax=3 * 512 - 1, iter=1)
    out[name] = hp.ud_grade(s, 64).astype(np.float32) * 1e6          # µK_CMB in the Planck channel
np.savez_compressed(sys.argv[2], **out)
