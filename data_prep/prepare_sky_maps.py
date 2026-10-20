"""Builds cmbsim/data/sky_n64.npz from the following raw inputs (not distributed; download them into
data_prep/raw/):
- PySM v1 templates (Thorne et al. 2017, MNRAS 469, 2821; github.com/bthorne93/PySM_public), nside 512, µK_RJ,
  Galactic coordinates: synchrotron (408 MHz) + spectral index, free-free (30 GHz), AME (22.8 and 41 GHz),
  thermal dust (545 GHz) + β_d and T_d;
- WMAP 3-year foreground-cleaned, Wiener-filtered CMB map (Tegmark, de Oliveira-Costa & Hamilton
  2003, PRD 68, 123523; tegmark.org/wmap/wiener3yr_map.fits), downgraded beforehand to nside 128 and saved
  as a .npy array in µK.

- PySM3 (Zonca et al. 2021, JOSS 6, 3783; Pan-Experiment Galactic Science Group 2025, ApJ 991, 23):
  large-scale part of the d11 (GNILC dust: amplitude at 353 GHz, β_d, T_d) and s6
  (synchrotron at 23 GHz and β_s) models, reconstructed from the a_ℓm (ℓ ≤ 160) of the "raw" files of the
  PySM data repository (portal.nersc.gov/project/cmb/pysm-data), without the stochastic small scales.
  Optional file pysm3_large_n64.npz (produced by data_prep/pysm3_large_scale.py).

Usage: python data_prep/prepare_sky_maps.py <pysm_templates_dir> <tegmark_wiener3yr_n128.npy> [pysm3_large_n64.npz]
"""
import sys
from pathlib import Path

import healpy as hp
import numpy as np

tpl, cmbfile = Path(sys.argv[1]), sys.argv[2]
NS_FG, NS_CMB = 64, 128


def rd(name):
    return hp.read_map(tpl / name, field=0, dtype=np.float64)


fg = {
    "synch_A": rd("synch_t_new.fits"), "synch_beta": rd("synch_beta.fits"),
    "ff_A": rd("ff_t_new.fits"),
    "ame1_A": rd("ame_t_new.fits"), "ame1_nup": rd("ame_nu_peak_0.fits"), "ame2_A": rd("ame2_t_new.fits"),
    "dust_A": rd("dust_t_new.fits"), "dust_T": rd("dust_temp.fits"), "dust_beta": rd("dust_beta.fits"),
}
out = {k: hp.ud_grade(v, NS_FG).astype(np.float32) for k, v in fg.items()}
emis = np.loadtxt(tpl / "emissivity.txt", unpack=True)
out["ame_emis_nu"], out["ame_emis_j"] = emis[0].astype(np.float32), emis[1].astype(np.float32)
cmb = np.load(cmbfile).astype(np.float32)
assert cmb.size == 12 * NS_CMB**2
out["cmb_uK"] = cmb
if len(sys.argv) > 3:
    p3 = np.load(sys.argv[3])
    for k in p3.files:
        out["p3_" + k] = p3[k].astype(np.float32)
dest = Path(__file__).resolve().parents[1] / "cmbsim" / "data" / "sky_n64.npz"
np.savez_compressed(dest, **out)
print(dest, dest.stat().st_size / 1e6, "MB")
for k, v in out.items():
    print(f"{k:12s} {v.size:7d}  min {v.min():10.3g}  med {np.median(v):10.3g}  max {v.max():10.3g}")
