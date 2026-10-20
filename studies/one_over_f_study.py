"""1/f × modulation study.

In the tracking-gap study, 1/f noise was the dominant CMB systematic: the TES knee (0.15 Hz) is 9× above the
1 rpm spin frequency (17 mHz). Here the same case (family D, 60-day gap starting on day 160, anomaly of
0.3 m/s/axis on day 162, weekly DSN with ΔDOR before and after) is rerun with ways of beating the 1/f noise:

- spin of 1, 3 and 10 rpm;
- chopper (Dicke, ×2 on the NET) at 10 Hz: the detector 1/f only appears at the modulation frequency;
- exact destriping + offset calibration with an internal load every 1 h (already in the simulator);
- reference without 1/f.

For the star tracker to follow the fast spin, all variants use the Hipparcos catalogue (V ≤ 7), 10 ms exposure
and the optical axis at 20° from the spin axis (100 % availability up to 10 rpm). Sweeps the effective NET
(0.3–10 µK√s) with all systematics and, at NET 1, isolates the 1/f noise (no attitude or gain error).

Usage (from the repository root): python studies/one_over_f_study.py [--summary] [--nproc N]
Outputs: results/one_over_f_study.csv and results/one_over_f_study.md. Cost: ~1 h with 2 cores.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import gap_study as EL                                          # noqa: E402
import provenance                                               # noqa: E402

OUT = EL.OUT
CSV = os.path.join(OUT, "one_over_f_study.csv")
MD = os.path.join(OUT, "one_over_f_study.md")
GAP = 60.0
SEEDS = (1, 2)
NETS = (0.3, 1.0, 3.0, 10.0)
ST = dict(st_catalog="hipparcos", st_mag_limit=7.0, st_exposure_s=0.01, st_boresight_deg=20.0)
ONLY_1F = dict(att_on=False, gain_cal_ppm=0.0, gain_drift_ppm=0.0)   # isolates the 1/f noise
VARIANTS = {
    "spin1": ("Spin 1 rpm", dict(spin_rate_dps=6.0)),
    "spin3": ("Spin 3 rpm", dict(spin_rate_dps=18.0)),
    "spin10": ("Spin 10 rpm", dict(spin_rate_dps=60.0)),
    "chop10": ("Chopper 10 Hz (Dicke ×2), spin 1 rpm", dict(spin_rate_dps=6.0, chop_hz=10.0, chop_penalty=2.0)),
    "destripe": ("Destriping + offset calibration every 1 h, spin 1 rpm",
                 dict(spin_rate_dps=6.0, destripe=True, cal_period_h=1.0)),
    "no1f": ("No 1/f (reference)", dict(spin_rate_dps=6.0, one_over_f=False)),
}
# attitude and gain requirements, with the 1/f beaten by the chopper: thermal alignment drift [″] × gain drift [ppm]
REQ_ATT = (1.0, 0.3, 0.1)
REQ_GAIN = (10.0, 1.0, 0.1)
REQ_NETS = (1.0, 0.3)
FIELDS = ["variant", "name", "systematics", "net_eff", "seed", "ev_end", "sv_end", "er_end", "within3", "noise_fac",
          "t_s"]


def jobs():
    out = []
    for s in SEEDS:
        for v in VARIANTS:
            for net in NETS:
                out.append(dict(variant=v, systematics="all", net_eff=net, seed=s))
            if v != "no1f":
                out.append(dict(variant=v, systematics="1/f only", net_eff=1.0, seed=s))
        for net in REQ_NETS:
            for ad in REQ_ATT:
                for gd in REQ_GAIN:
                    if net != 1.0 and not (ad == 0.1 and gd <= 1.0):
                        continue
                    out.append(dict(variant="chop10", systematics=f"req att={ad:g} gain={gd:g}", net_eff=net, seed=s))
    return out


def key(j):
    return (j["variant"], j["systematics"], f"{float(j['net_eff']):g}", str(int(j["seed"])))


def one(j):
    nf = EL.nf_tes()
    name, extra = VARIANTS[j["variant"]]
    over = dict(EL.CMB, **ST, **extra, net_uK_rts=float(j["net_eff"]) / nf)
    if j["systematics"] == "1/f only":
        over.update(ONLY_1F)
    if j["systematics"].startswith("req"):
        ad, gd = (float(x.split("=")[1]) for x in j["systematics"].split()[1:])
        over.update(att_drift_arcsec=ad, gain_drift_ppm=gd)
    t0 = time.time()
    r = EL.one(dict(group="1f", case="B", gap_d=GAP, anomaly=True, seed=int(j["seed"]), net_eff=j["net_eff"],
                    over=over))
    return dict(variant=j["variant"], name=name, systematics=j["systematics"], net_eff=float(j["net_eff"]),
                seed=int(j["seed"]), ev_end=r["ev_end"], sv_end=r["sv_end"], er_end=r["er_end"],
                within3=r["within3"], noise_fac=_factor(j), t_s=round(time.time() - t0, 1))


def _factor(j):
    """Harmonic noise factor due to 1/f (spin) or to the modulation (chopper), for the table."""
    _, extra = VARIANTS[j["variant"]]
    fk = 0.15                                                       # knee of the tes_3b preset
    if extra.get("chop_hz"):
        return round(float(extra["chop_penalty"] * np.sqrt(1 + fk / extra["chop_hz"])), 2)
    if extra.get("one_over_f") is False:
        return 1.0
    return round(float(np.sqrt(1 + fk / (extra["spin_rate_dps"] / 360.0))), 2)


def load():
    if not os.path.exists(CSV):
        return []
    with open(CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def summarize():
    rows = load()
    for r in rows:
        for k in ("net_eff", "ev_end", "sv_end", "er_end", "within3", "noise_fac"):
            r[k] = float(r[k])
    if not rows:
        print("no results")
        return
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan
    el = EL.load()
    refA = rms([float(r["ev_end"]) for r in el if r["group"] == "cases" and r["case"] == "A" and r["anomaly"] == "True"
                and float(r["gap_d"]) == GAP]) if el else np.nan
    L = ["# 1/f × modulation study — results", "",
         f"Generated by `studies/one_over_f_study.py` on {time.strftime('%Y-%m-%d %H:%M')}. Case B (family D, 90/150/220 GHz) with "
         f"a {GAP:g}-day gap starting on day {EL.GAP_START:g}, anomaly of {EL.ANOM_DV} m/s per axis; Hipparcos star "
         "tracker V ≤ 7, 10 ms, optical axis at 20° from the spin axis. Velocity error at the end of the gap, RMS over "
         f"{len(SEEDS)} seeds. Reference: dynamics only = **{refA:.3g} m/s**; target (half) = {refA / 2:.3g} m/s.", "",
         "## All systematics (attitude, gain and 1/f)", "",
         "| Variant | Harmonic noise factor | " + " | ".join(f"NET {n:g}" for n in NETS) + " | 1/f only, NET 1 |",
         "|---|---|" + "---|" * len(NETS) + "---|"]
    for v, (name, _) in VARIANTS.items():
        sel = [r for r in rows if r["variant"] == v]
        if not sel:
            continue
        cells = []
        for n in NETS:
            x = [r["ev_end"] for r in sel if r["systematics"] == "all" and r["net_eff"] == n]
            val = rms(x)
            cells.append("—" if not x else (f"**{val:.3g}**" if val < refA / 2 else f"{val:.3g}"))
        s1 = [r["ev_end"] for r in sel if r["systematics"] == "1/f only"]
        L.append(f"| {name} | {sel[0]['noise_fac']:g} | " + " | ".join(cells) + f" | {rms(s1):.3g} |" if s1 else
                 f"| {name} | {sel[0]['noise_fac']:g} | " + " | ".join(cells) + " | — |")
    L += ["", "Bold: error below half of the dynamics-only case (useful NET).", "",
          "## Highest useful NET per variant", ""]
    for v, (name, _) in VARIANTS.items():
        ok = [n for n in sorted(NETS, reverse=True)
              if rms([r["ev_end"] for r in rows if r["variant"] == v and r["systematics"] == "all" and r["net_eff"] == n]) < refA / 2]
        L.append(f"- {name}: " + (f"**{ok[0]:g} µK√s**" if ok else "no NET in the sweep") + ".")
    rq = [r for r in rows if r["systematics"].startswith("req")]
    if rq:
        L += ["", "## Attitude and gain requirements (chopper 10 Hz, 1/f beaten)", "",
              "Velocity error [m/s] at the end of the gap. Rows: thermal alignment drift between star tracker and CMB "
              "sensor (1σ, τ = 24 h; the fixed 3″ misalignment is estimated). Columns: gain drift (1σ, τ = 10 days).", ""]
        for net in REQ_NETS:
            L += [f"**Effective NET {net:g} µK√s**", "", "| Alignment drift | " + " | ".join(f"gain {g:g} ppm" for g in REQ_GAIN) + " |",
                  "|---|" + "---|" * len(REQ_GAIN)]
            for ad in REQ_ATT:
                cells = []
                for gd in REQ_GAIN:
                    x = [r["ev_end"] for r in rq if r["net_eff"] == net and r["systematics"] == f"req att={ad:g} gain={gd:g}"]
                    v_ = rms(x)
                    cells.append("—" if not x else (f"**{v_:.3g}**" if v_ < refA / 2 else f"{v_:.3g}"))
                L.append(f"| {ad:g}″ | " + " | ".join(cells) + " |")
            L.append("")
    os.makedirs(OUT, exist_ok=True)
    with open(MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="1/f × modulation study.")
    ap.add_argument("--summary", action="store_true", help="only rebuild the tables from the CSV")
    ap.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2)), help="number of worker processes")
    ap.add_argument("--resume-anyway", action="store_true", help="resume even if the CSV was written by other code")
    a = ap.parse_args()
    if a.summary:
        summarize()
        return
    os.makedirs(OUT, exist_ok=True)
    mixed = provenance.check_resume(CSV, a.resume_anyway)
    done = {key(r) for r in load()}
    todo = [j for j in jobs() if key(j) not in done]
    print(f"{len(todo)} simulations to run ({len(done)} already in the CSV)", flush=True)
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as fh, Pool(a.nproc) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(one, todo)):
            w.writerow(row)
            fh.flush()
            print(f"[{i + 1}/{len(todo)}] {row['variant']} ({row['systematics']}) NET {row['net_eff']:g} seed "
                  f"{row['seed']}: v error = {row['ev_end']:.3g} m/s ({row['t_s']:.0f} s)", flush=True)
    provenance.stamp(CSV, mixed)
    summarize()


if __name__ == "__main__":
    main()
