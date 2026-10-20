"""Reconstructs, at nside 64, the large-scale part of the PySM3 d11 (dust) and s6 (synchrotron) models.

Input: a_ℓm (ℓ ≤ 160) extracted from the PySM3 "raw" files, downloaded into data_prep/raw/ from the PySM data
repository (https://portal.nersc.gov/project/cmb/pysm-data):
  dust_gnilc/raw/gnilc_dust_largescale_template_logpoltens_alm_nside2048_lmax1024_complex64.fits.gz (I, Q, U)
  dust_gnilc/raw/gnilc_dust_largescale_template_beta_alm_nside2048_lmax1024.fits.gz
  dust_gnilc/raw/gnilc_dust_largescale_template_Td_alm_nside2048_lmax1024.fits.gz
  synch/raw/synch_largescale_template_logpoltens_alm_lmax128_2023.02.24.fits.gz (I, Q, U)
  synch/raw/synch_largescale_beta_alm_nside512_lmax768.fits.gz
The a_ℓm are stored in pysm3_alm_lmax160.npz as <name>_idx (FITS a_ℓm index ℓ² + ℓ + m + 1), <name>_re and
<name>_im, for names dust_I/Q/U, dust_beta, dust_Td, synch_I/Q/U and synch_beta.
Same procedure as pysm3.models.ModifiedBlackBodyRealization / PowerLawRealization, without the
small scales: map = log_pol_tens_to_map(alm2map(a_ℓm)); dust × 0.911 (colour correction);
β_s = map − 3.1.

Usage: python data_prep/pysm3_large_scale.py data_prep/raw/pysm3_alm_lmax160.npz data_prep/raw/pysm3_large_n64.npz
"""
import sys

import healpy as hp
import numpy as np
from pysm3 import utils

d = np.load(sys.argv[1])
NS = 64


def alm(name, L):
    idx = d[name + "_idx"]
    l = np.floor(np.sqrt(idx - 1)).astype(int)
    m = idx - 1 - l * l - l
    a = np.zeros(hp.Alm.getsize(L), complex)
    a[hp.Alm.getidx(L, l, m)] = d[name + "_re"] + 1j * d[name + "_im"]
    return hp.alm2map(a, NS, lmax=L)


dust = utils.log_pol_tens_to_map(np.array([alm("dust_" + c, 160) for c in "IQU"])) * 0.911
synch = utils.log_pol_tens_to_map(np.array([alm("synch_" + c, 128) for c in "IQU"]))
np.savez_compressed(sys.argv[2], dust_A353=dust[0].astype(np.float32),
                    dust_beta=alm("dust_beta", 160).astype(np.float32), dust_T=alm("dust_Td", 160).astype(np.float32),
                    synch_A23=synch[0].astype(np.float32), synch_beta=(alm("synch_beta", 160) - 3.1).astype(np.float32))
