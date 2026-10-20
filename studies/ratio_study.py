"""Dipole/monopole ratio method and the "metrology" and "limit" levels in the full simulator (IEEE Access article).

Until now the "metrology" and "limit" levels came from analytic estimates (abstract model in cmbsim/deep.py).
This study simulates them in the full simulator, with the ratio method modelled:

- total-power receiver against a cold reference load (receiver='total'): the DC channel of each sensor measures
  (1 + g)·T, the monopole becomes the gain lever arm and the dipole/monopole ratio is independent of the gain;
- chopper at 10 Hz (the detector 1/f vanishes; the load drift remains);
- reference-load drift: Gauss-Markov per sensor (load_drift_nK, τ = 24 h), estimated by the filter (4 states);
- spin axis tilted 20° from the anti-Sun direction with precession in 4 steps: the ring harmonics also see the
  component along the axis (without this, the axial component only appears in the DC channels and is degenerate
  with the relative gain between the axis sensor and the ring sensors).

Groups (60 d gap starting on day 160, anomaly of 0.3 m/s/axis on day 162, as in the gap and 1/f studies):
1. gain: differential × total receiver, gain drift 1, 10 and 100 ppm (no attitude error, effective NET 0.3);
2. load: total, gain 10 ppm, load drift 3 nK–3 µK (no attitude error, effective NET 0.083 ~ 0.01 m²·sr);
3. attitude: total, gain 10 ppm, load 30 nK, NET 0.083, alignment drift 1″–1 mas (Hipparcos star tracker);
1b (group 6). gain with effective NET 0.083;
4. levels: today (family D), requirements, ratio, metrology, limit + references (dynamics only, IMU, XNAV);
5. lost-state recovery (40 days without DSN from day 100), same levels + asteroids, XNAV, CeleNav.

Usage (from the repository root): python studies/ratio_study.py [--summary] [--nproc N] [--groups 1,2,...]
Outputs: results/ratio_study.csv and results/ratio_study.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import provenance                                                    # noqa: E402
import gap_study as EL                                          # noqa: E402
from cmbsim import physics as ph                                    # noqa: E402
from cmbsim.instrument import family_config                         # noqa: E402
from cmbsim.simulator import Config, run                            # noqa: E402

OUT = EL.OUT
CSV = os.path.join(OUT, "ratio_study.csv")
MD = os.path.join(OUT, "ratio_study.md")
GAP = 60.0
SEEDS = tuple(range(1, 7))
ST = dict(st_catalog="hipparcos", st_mag_limit=7.0, st_exposure_s=0.01, st_boresight_deg=20.0)
NOATT = dict(att_on=False)
TILT = dict(spin_tilt_deg=20.0, prec_n=4)
CHOP = dict(chop_hz=10.0, chop_penalty=2.0)
# effective NET [µK√s] per level: 0.3 (requirements); 0.083 and 0.026 are equivalent to the white noise of 0.01 and
# 0.1 m²·sr at the photon limit (5.4 and 1.7 cm/s per day; see the article), with the same simulator scaling
# (NET 1 ↔ 0.65 m/s/day)
NET_REQ, NET_MET, NET_LIM = 0.3, 0.083, 0.026


def cmb(net_eff, **kw):
    return dict(EL.CMB, **CHOP, net_uK_rts=net_eff / EL.nf_tes(), **kw)


RATIO = dict(receiver="total", **TILT, zodi_estimate=True)
# mas-class attitude reference (guide telescope / interferometric metrology), modelled as a star tracker
# with white noise per window of ~5 mas (metrology) and ~0.6 mas (limit)
ST_MET = dict(ST, st_centroid_px=0.05, st_heads=2)
ST_LIM = dict(ST, st_centroid_px=0.01, st_heads=4, st_frame_hz=10.0)
LEVELS = {
    "today": ("Today, realistic (family D, 1 rpm spin, all systematics)", dict(EL.CMB)),
    "req": ("Requirements (differential, chopper, 0.1″, 1 ppm, NET 0.3)",
            dict(cmb(NET_REQ, **ST, att_drift_arcsec=0.1, gain_drift_ppm=1.0))),
    "ratio": ("Ratio (total power, tilted, 0.1″, gain 10 ppm, load 30 nK, NET 0.3)",
              dict(cmb(NET_REQ, **ST, **RATIO, att_drift_arcsec=0.1, gain_drift_ppm=10.0, load_drift_nK=30.0))),
    "metro": ("Metrology (ratio, 10 mas, load 30 nK, NET 0.083)",
              dict(cmb(NET_MET, **ST_MET, **RATIO, att_drift_arcsec=0.01, gain_drift_ppm=10.0, load_drift_nK=30.0))),
    "limit": ("Limit (ratio, 1 mas, load 3 nK, NET 0.026)",
               dict(cmb(NET_LIM, **ST_LIM, **RATIO, att_drift_arcsec=0.001, gain_drift_ppm=10.0, load_drift_nK=3.0))),
}
# variants of the limit level that isolate the zodiacal light (group 4 only)
LIM_VAR = {
    "limit_z": ("Limit, zodiacal light treated as a random walk (10 % model, scale not estimated)",
                 dict(LEVELS["limit"][1], zodi_estimate=False)),
    "limit_no_z": ("Limit, no zodiacal light (proxy for > 5 AU)", dict(LEVELS["limit"][1], zodi_on=False)),
}
REFS_GAP = {
    "dyn": ("Dynamics only", EL.NOCMB),
    "imu": ("IMU", {**EL.NOCMB, **EL.IMU}),
    "xnav": ("XNAV (Ely, best 4)", {**EL.NOCMB, **EL.XNAV}),
}
REFS_REC = {
    "ast": ("Asteroids", {**EL.NOCMB, **EL.AST}),
    "xnav": ("XNAV (Ely, best 4)", {**EL.NOCMB, "xnav_on": True, "xnav_model": "ely"}),
    "cel": ("CeleNav (0.5″, 6 h)", {**EL.NOCMB, "celenav": True}),
}
GAINS = (1.0, 10.0, 100.0)
LOADS = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0)
ATTS = (1.0, 0.1, 0.01, 0.001)
REC_START, REC_LEN = 100.0, 40.0
REC_T = (1.0, 5.0, 10.0, 25.0, 39.0)
REC_TH = (100.0, 10.0, 1.0, 0.1)
FIELDS = ["group", "case", "param", "seed", "ev_end", "sv_end", "er_end", "within3", "extra", "t_s"]


def gap_jobs(groups):
    out = []
    for s in SEEDS:
        if 1 in groups:
            for rc in ("differential", "total"):
                for g in GAINS:
                    kw = cmb(NET_REQ, **NOATT, **TILT, receiver=rc, gain_drift_ppm=g, zodi_estimate=True)
                    out.append(dict(group="gain", case=rc, param=g, seed=s, kw=kw))
        if 6 in groups:
            for rc in ("differential", "total"):
                for g in GAINS:
                    kw = cmb(NET_MET, **NOATT, **TILT, receiver=rc, gain_drift_ppm=g, zodi_estimate=True)
                    out.append(dict(group="gain_met", case=rc, param=g, seed=s, kw=kw))
        if 2 in groups:
            for ld in LOADS:
                kw = cmb(NET_MET, **NOATT, **RATIO, gain_drift_ppm=10.0, load_drift_nK=ld)
                out.append(dict(group="load", case="total", param=ld, seed=s, kw=kw))
        if 3 in groups:
            for ad in ATTS:
                kw = cmb(NET_MET, **ST_LIM, **RATIO, gain_drift_ppm=10.0, load_drift_nK=30.0, att_drift_arcsec=ad)
                out.append(dict(group="attitude", case="total", param=ad, seed=s, kw=kw))
        if 4 in groups:
            for k, (_, kw) in {**LEVELS, **LIM_VAR, **REFS_GAP}.items():
                out.append(dict(group="levels", case=k, param=0, seed=s, kw=kw))
        if 5 in groups:
            for k, (_, kw) in {**LEVELS, **REFS_REC}.items():
                out.append(dict(group="recovery", case=k, param=0, seed=s, kw=kw))
    return out


def key(j):
    return (j["group"], j["case"], f"{float(j['param']):g}", str(int(j["seed"])))


def one(j):
    t0 = time.time()
    if j["group"] != "recovery":
        r = EL.one(dict(group="ratio", case="B", gap_d=GAP, anomaly=True, seed=int(j["seed"]), net_eff=1.0,
                        over=j["kw"]))
        return dict(group=j["group"], case=j["case"], param=j["param"], seed=j["seed"], ev_end=r["ev_end"],
                    sv_end=r["sv_end"], er_end=r["er_end"], within3=r["within3"], extra="", t_s=round(time.time() - t0, 1))
    kw = dict(j["kw"], start_day=REC_START, end_day=REC_START + REC_LEN, dsn_calibration=False, obs_analysis=False,
              maneuvers_days=(), seed=int(j["seed"]))
    r = run(Config(**kw))
    d = (r.t - r.t[0]) / ph.DAY
    ev = np.linalg.norm(r.x_est[:, 3:6] - r.x_true[:, 3:], axis=1) * 1e3
    er = np.linalg.norm(r.x_est[:, :3] - r.x_true[:, :3], axis=1)
    at = [f"{float(ev[np.where(d <= t)[0][-1]]):.4g}/{float(er[np.where(d <= t)[0][-1]]):.4g}" for t in REC_T]
    first = []
    for th in REC_TH:
        bad = np.where(ev >= th)[0]
        first.append("0" if not len(bad) else (f"{d[bad[-1] + 1]:.3g}" if bad[-1] + 1 < len(d) else "nan"))
    return dict(group="recovery", case=j["case"], param=0, seed=j["seed"], ev_end=round(float(ev[-1]), 5), sv_end="",
                er_end=round(float(er[-1]), 3), within3="", extra=";".join(at) + "|" + ";".join(first),
                t_s=round(time.time() - t0, 1))


def load():
    if not os.path.exists(CSV):
        return []
    with open(CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def rms(x):
    x = [float(v) for v in x]
    return float(np.sqrt(np.mean(np.square(x)))) if x else np.nan


def summarize():
    rows = load()
    if not rows:
        print("no results")
        return
    sel = lambda g, c=None, p=None: [r for r in rows if r["group"] == g and (c is None or r["case"] == c)
                                     and (p is None or float(r["param"]) == p)]
    ref = rms([r["ev_end"] for r in sel("levels", "dyn")])
    ns = len({r["seed"] for r in rows})
    L = ["# Ratio method and metrology/limit levels in the full simulator — results", "",
         f"Generated by `studies/ratio_study.py` on {time.strftime('%Y-%m-%d %H:%M')}. {GAP:g}-day gap starting on day "
         f"{EL.GAP_START:g} with an anomaly of {EL.ANOM_DV} m/s per axis on day {EL.GAP_START + 2:g}; weekly DSN with ΔDOR "
         f"before and after. Velocity error [m/s] at the end of the gap, RMS over up to {ns} seeds (in parentheses: "
         "filter 1σ and % of time within ±3σ). Chopper at 10 Hz in all CMB cases except \"today\".", ""]
    if sel("gain"):
        L += ["## 1. Gain immunity (axis tilted 20°, no attitude error, effective NET 0.3)", "",
              "| Receiver | " + " | ".join(f"gain {g:g} ppm" for g in GAINS) + " |", "|---|" + "---|" * len(GAINS)]
        for rc in ("differential", "total"):
            L.append(f"| {rc} | " + " | ".join(_cell(sel("gain", rc, g)) for g in GAINS) + " |")
        L.append("")
    if sel("gain_met"):
        L += ["## 1b. Gain immunity with lower noise (effective NET 0.083, axis tilted 20°, no attitude error)", "",
              "| Receiver | " + " | ".join(f"gain {g:g} ppm" for g in GAINS) + " |", "|---|" + "---|" * len(GAINS)]
        for rc in ("differential", "total"):
            L.append(f"| {rc} | " + " | ".join(_cell(sel("gain_met", rc, g)) for g in GAINS) + " |")
        L.append("")
    if sel("load"):
        L += ["## 2. Reference-load requirement (total power, gain 10 ppm, no attitude error, effective NET 0.083)", "",
              "| Load drift (1σ, τ 24 h) [nK] | v error [m/s] |", "|---|---|"]
        for ld in LOADS:
            L.append(f"| {ld:g} | {_cell(sel('load', 'total', ld))} |")
        L.append("")
    if sel("attitude"):
        L += ["## 3. Alignment requirement (total power, gain 10 ppm, load 30 nK, effective NET 0.083, mas-class attitude reference)", "",
              "| Alignment drift (1σ, τ 24 h) | v error [m/s] |", "|---|---|"]
        for ad in ATTS:
            L.append(f"| {ad * 1e3:g} mas | {_cell(sel('attitude', 'total', ad))} |")
        L.append("")
    if sel("levels"):
        L += ["## 4. CMB levels and references (60 d gap with anomaly)", "",
              "| Case | v error [m/s] | r error [km] | Ratio to dynamics only |", "|---|---|---|---|"]
        for k, (nm, _) in {**LEVELS, **LIM_VAR, **REFS_GAP}.items():
            s_ = sel("levels", k)
            if s_:
                ev = rms([r["ev_end"] for r in s_])
                L.append(f"| {nm} | {_cell(s_)} | {rms([r['er_end'] for r in s_]):.3g} | {ev / ref:.2f} |")
        L.append("")
    if sel("recovery"):
        L += ["## 5. Lost-state recovery (day 100, 10⁴ km and 1 km/s per axis, no DSN)", "",
              "Velocity [m/s] / position [km] error after t days (RMS over the seeds):", "",
              "| Case | " + " | ".join(f"{t:g} d" for t in REC_T) + " |", "|---|" + "---|" * len(REC_T)]
        names = {**LEVELS, **REFS_REC}
        for k, (nm, _) in names.items():
            s_ = sel("recovery", k)
            if not s_:
                continue
            at = [r["extra"].split("|")[0].split(";") for r in s_]
            cells = []
            for i in range(len(REC_T)):
                v = [float(a[i].split("/")[0]) for a in at]
                p = [float(a[i].split("/")[1]) for a in at]
                cells.append(f"{rms(v):.3g} / {rms(p):.3g}")
            L.append(f"| {nm} | " + " | ".join(cells) + " |")
        L += ["", "Days until the velocity error drops below the threshold and stays there (worst seed):", "",
              "| Case | " + " | ".join(f"< {t:g} m/s" for t in REC_TH) + " |", "|---|" + "---|" * len(REC_TH)]
        for k, (nm, _) in names.items():
            s_ = sel("recovery", k)
            if not s_:
                continue
            fs = [r["extra"].split("|")[1].split(";") for r in s_]
            cells = []
            for i in range(len(REC_TH)):
                v = [float(f[i]) for f in fs]
                cells.append("not reached" if any(np.isnan(v)) else f"{max(v):.2g}")
            L.append(f"| {nm} | " + " | ".join(cells) + " |")
        L.append("")
    with open(MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def _cell(s_):
    if not s_:
        return "—"
    ev = rms([r["ev_end"] for r in s_])
    sv = rms([r["sv_end"] for r in s_]) if s_[0]["sv_end"] != "" else np.nan
    d3 = np.mean([float(r["within3"]) for r in s_]) if s_[0]["within3"] != "" else np.nan
    return f"{ev:.3g} ({sv:.2g}; {d3:.0f} %)"


def main():
    ap = argparse.ArgumentParser(description="Dipole/monopole ratio method and metrology/limit levels.")
    ap.add_argument("--summary", action="store_true", help="only rebuild the tables from the CSV")
    ap.add_argument("--nproc", type=int, default=os.cpu_count() or 2, help="number of worker processes")
    ap.add_argument("--groups", default="1,2,3,4,5,6", help="comma-separated groups to run (1-6)")
    ap.add_argument("--resume-anyway", action="store_true", help="resume even if the CSV was written by other code")
    a = ap.parse_args()
    if a.summary:
        summarize()
        return
    groups = {int(x) for x in a.groups.split(",")}
    mixed = provenance.check_resume(CSV, a.resume_anyway)
    done = {(r["group"], r["case"], f"{float(r['param']):g}", r["seed"]) for r in load()}
    todo = [j for j in gap_jobs(groups) if key(j) not in done]
    print(f"{len(todo)} runs", flush=True)
    os.makedirs(OUT, exist_ok=True)
    new = not os.path.exists(CSV)
    t0 = time.time()
    with open(CSV, "a", newline="") as fh, Pool(a.nproc) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(one, todo)):
            w.writerow(row)
            fh.flush()
            print(f"[{i + 1}/{len(todo)} {time.time() - t0:.0f} s] {row['group']} {row['case']} {row['param']} "
                  f"s{row['seed']}: {row['ev_end']}", flush=True)
    provenance.stamp(CSV, mixed)
    summarize()


if __name__ == "__main__":
    main()
