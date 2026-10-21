#!/usr/bin/env bash
# Reproduces every result and figure of the article (about 7 h on 2 cores).
# Delete results/*.csv first to recompute from scratch; otherwise the studies resume from the existing CSVs.
set -euo pipefail
cd "$(dirname "$0")"
NPROC="${NPROC:-$(python -c 'import os; print(os.cpu_count() or 2)')}"
python studies/gap_study.py --nproc "$NPROC"
python studies/one_over_f_study.py --nproc "$NPROC"
python studies/ratio_study.py --nproc "$NPROC"
python studies/deep_space_study.py
python studies/deep_space_calibrated_study.py
python studies/gap_timeseries.py
python studies/floor_budget_study.py --nproc "$NPROC"
python studies/consistency_study.py --nproc "$NPROC"
python studies/deep_space_calibration_fit.py          # needs results/consistency_study.csv
python studies/robustness_study.py --nproc "$NPROC"
python studies/chromatic_study.py --nproc "$NPROC"
python studies/seed_intervals.py
python figures/make_figures.py
