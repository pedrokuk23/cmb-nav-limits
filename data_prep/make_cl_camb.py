"""ΛCDM TT power spectrum (Planck 2018, TT,TE,EE+lowE+lensing) computed with CAMB 2 (Lewis 2026,
arXiv:2607.14854; pip install camb). Writes cmbsim/data/cl_tt_camb.txt with ℓ and D_ℓ [µK²] (lensed).

No raw input is needed: the spectrum is computed directly by CAMB.
Usage: python data_prep/make_cl_camb.py
"""
from pathlib import Path

import camb
import numpy as np

p = camb.set_params(H0=67.32, ombh2=0.02238, omch2=0.1201, mnu=0.06, omk=0, tau=0.0543,
                    As=2.100e-9, ns=0.9660, lmax=3000)
res = camb.get_results(p)
dl = res.get_cmb_power_spectra(p, CMB_unit="muK")["total"][:, 0]
ell = np.arange(len(dl))
dest = Path(__file__).resolve().parents[1] / "cmbsim" / "data" / "cl_tt_camb.txt"
np.savetxt(dest, np.column_stack([ell[2:3001], dl[2:3001]]), fmt=["%d", "%.4f"],
           header=f"CAMB {camb.__version__}, Planck 2018 (H0=67.32, ombh2=0.02238, omch2=0.1201, tau=0.0543, "
                  "As=2.1e-9, ns=0.966, mnu=0.06). Columns: ell, D_ell=ell(ell+1)C_ell/2pi [uK^2], lensed TT")
print(dest)
