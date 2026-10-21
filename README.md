# cmb-nav-limits

Code and data to reproduce the results of the article

> P. Kukulka de Albuquerque and L. Galembeck, "Limits of Cosmic Microwave Background Dipole Velocimetry for Autonomous
> Deep-Space Navigation: Present Technology Versus the Photon-Noise Floor," submitted to *IEEE Access*, 2026.

The repository contains `cmbsim`, a navigation simulator for spacecraft velocity estimation from the kinematic dipole
of the cosmic microwave background (CMB). It also contains the scripts of every study in the article, the result
tables they produced, and the scripts that draw the figures.

## What the simulator does

- **Truth.** It propagates a heliocentric Earth–Mars cruise using the JPL DE421 ephemerides, solar radiation pressure,
  unmodeled acceleration, desaturations and ΔV anomalies. The sky includes the CMB (WMAP map), PySM3 dust and
  synchrotron, Planck CO, point sources, zodiacal emission computed along the trajectory with ZodiPy, and kinematic
  aberration.
- **CMB instrument.** It models four sensors with 2–3 bands, six detector families, NET, 1/f noise or chopper
  modulation, and gain calibration and drift. The receiver can be differential or total power (the dipole-to-monopole
  ratio method), with a drifting reference load. The scan uses spin, tilt and precession, and a star-tracker attitude
  reference adds misalignment and drift.
- **Competing sensors.** It includes DSN range, Doppler and ΔDOR (a solar-conjunction cut and the Cassini plasma model
  are available but not used in the article, whose tracking gaps are prescribed),
  IMU, solar Doppler, asteroid OpNav, XNAV (Ely et al. 2022, "best 4") and planet-based CeleNav. They are modeled at
  the level of their observables (e.g., the IMU's measured ΔV, the pulsar time of arrival as a range along the pulsar
  direction), with error levels from the literature, not as full hardware chains.
- **Filter.** An unscented Kalman filter estimates position, velocity, monopole temperature, sky biases, gains,
  reference-load offsets, zodiacal scale, misalignment and drift, 1/f offsets and empirical acceleration. An EKF and a
  particle filter are available as cross-checks.
- **Reduced model.** `cmbsim/deep.py` is a linear model for year-long cruises at 5–300 AU. Its CMB velocimeter is
  abstract (velocity + Gauss–Markov bias + white noise) and calibrated against the full simulator on a common 60-day
  gap case at 1.5 AU; `studies/deep_space_calibration_fit.py` reproduces the stored calibration. The 100 AU results
  are therefore conditional on that error model.

## Installation

Python ≥ 3.11 is required (tested with 3.13).

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests/          # 44 tests, ~15 min on 2 cores
```

All input data needed to run the studies is included in `cmbsim/data/` (4.8 MB). The first run computes the
zodiacal-emission cache in `~/.cache/cmbsim` (about 15 s per band).

## Reproducing the article

Every script runs from the repository root. Run times are wall-clock estimates on 2 CPU cores. The studies resume
from their CSV if interrupted; delete the CSV to start over. `--nproc N` sets the number of worker processes.

| Article item | Command | Output | Time |
|---|---|---|---|
| Fig. 1, eq. (6): photon-noise floor | `python figures/make_figures.py` | `figures/fig_photon_limit.pdf` | seconds |
| Fig. 2: simulator diagram | `pdflatex -output-directory figures figures/fig_simulator.tex` | `figures/fig_simulator.pdf` | seconds |
| Sec. VI-A: tracking gaps of 15–85 d, detector families, NET sweep | `python studies/gap_study.py` | `results/gap_study.{csv,md}` | ~2.2 h |
| Sec. VI-A: 1/f, chopper and spin rate; alignment and gain requirements | `python studies/one_over_f_study.py` | `results/one_over_f_study.{csv,md}` | ~35 min |
| Sec. VI-A–C and VI-E, Tables IV–VI (gap column): ratio receiver, load and alignment requirements, zodiacal modeling, CMB levels, lost-state recovery | `python studies/ratio_study.py` | `results/ratio_study.{csv,md}` | ~45 min |
| Sec. VI-D, Table VI (100 AU columns), reduced-model calibration | `python studies/deep_space_study.py` then `python studies/deep_space_calibrated_study.py` | `results/deep_space_study.{csv,md}`, `results/deep_space_calibrated_study.{csv,md}` | ~15 min each |
| Fig. 4: velocity error during the 60-day gap | `python studies/gap_timeseries.py` | `results/gap_timeseries.npz` | ~3 min |
| Fig. 5: requirement curves (from `results/ratio_study.csv`) | `python figures/make_figures.py` | `figures/fig_requirements.pdf` | seconds |
| Sec. V-C: calibration of the reduced-model CMB levels (one scale factor per level; run after `consistency_study.py`) | `python studies/deep_space_calibration_fit.py` | `results/deep_space_calibration_fit.{csv,md}` | ~4 min |
| Sec. VI-E, Fig. 7: decomposition of the photon-floor penalty | `python studies/floor_budget_study.py` | `results/floor_budget_study.{csv,md}` | ~20 min |
| Sec. V-D, Fig. 3: filter consistency (NEES; UKF vs EKF vs particle filter) | `python studies/consistency_study.py` | `results/consistency_study.{csv,md}`, `results/consistency_series.npz` | ~30 min |
| Sec. VI-B, VI-E: load requirement at the photon-limit level; relativistic velocity composition | `python studies/robustness_study.py` | `results/robustness_study.{csv,md}` | ~1.5 h |
| Figs. 3–7 | `python figures/make_figures.py` | `figures/fig_consistency.pdf`, `fig_timeseries.pdf`, `fig_requirements.pdf`, `fig_ladder.pdf`, `fig_budget.pdf` | seconds |

**Safeguards.** If a requested component cannot run (for example, `healpy` or `zodipy` is missing, or a data file
is absent), the simulator stops with an error instead of silently using a reduced model; `allow_fallback=True` turns
this into a warning, and `Results.info["components"]` records what actually ran. Each study that resumes from its CSV
keeps a code fingerprint in `results/<study>.csv.provenance.json` and refuses to resume rows computed with different
code (`--resume-anyway` overrides and records the mix).

`./run_all.sh` runs everything in this order. The `results/` folder already contains the outputs used in the
article, so `python figures/make_figures.py` works right after installation. Every run is deterministic: a seed fixes
all random draws, so re-running a study reproduces the tables to the last digit.

Each study can also print its summary from an existing CSV without running new simulations, e.g.
`python studies/ratio_study.py --summary`.

## Repository layout

```
cmbsim/            simulator package
  simulator.py     full simulator: Config, truth, measurements, UKF (run(Config(...)) -> Results)
  deep.py          reduced linear model for 5-300 AU cruises (run_deep(DeepConfig(...)))
  instrument.py    detector presets and design families
  skymaps.py       sky model from maps (CMB + foregrounds); sky.py: statistical sky
  zodi.py          zodiacal emission along the trajectory (ZodiPy)
  attitude.py      star tracker, catalogues, attitude algorithms
  beacons.py       synthetic main-belt population for asteroid OpNav
  physics.py       constants, ephemerides, dipole model, Lambert solver
  ukf.py           UKF, EKF and Gaussian particle filter
  data/            bundled inputs (see "Data sources")
studies/           one script per study of the article
figures/           figure scripts and the TikZ source of Fig. 2
results/           CSV and Markdown outputs used in the article
data_prep/         scripts that rebuilt cmbsim/data from public raw data (not needed to run)
tests/             automated tests
```

Minimal example:

```python
from cmbsim.simulator import Config, run
r = run(Config(tech="tes_3b", spin=True, tau_s=7200.0, end_day=30.0, seed=1))
print(r.x_est[-1, 3:6] - r.x_true[-1, 3:])      # velocity error [km/s] at the end
```

## Data sources (bundled in `cmbsim/data/`)

| File | Content | Source |
|---|---|---|
| `sky_n64.npz` | CMB map and Galactic foreground templates (HEALPix) | WMAP 3-year Wiener-filtered map (Tegmark, de Oliveira-Costa & Hamilton 2003); PySM v1 templates (Thorne et al. 2017) and the large-scale part of PySM3 d11/s6 (Zonca et al. 2021) |
| `co_planck_n64.npz` | CO line maps | Planck HFI CompMap CO Type 1 (Planck Legacy Archive) |
| `de421_2020_2040.npz` | Planet and Moon ephemerides, 2020–2040 | JPL DE421 |
| `hip75.npz` | Star catalogue, V ≤ 7.5 | Hipparcos (ESA 1997) |
| `gsc6.csv` | Bright-star catalogue, V ≤ 6 (4401 stars) | From the author's undergraduate star-tracker code |
| `cl_tt_camb.txt` | Lensed TT power spectrum | CAMB with Planck 2018 parameters |

The scripts in `data_prep/` document how each file was produced and where to download the raw inputs. The raw files
(about 40 MB) are not distributed.

## Citation

If you use this code, please cite the article and this repository (see `CITATION.cff`).

## License

MIT (see `LICENSE`). The bundled data products keep the terms of their original providers.
