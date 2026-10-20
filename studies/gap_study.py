"""Tracking-gap study (use case 1: navigation through a DSN tracking gap).

Weekly DSN (8 h + ΔDOR) up to day 160; gap of 15, 30, 60 or 85 days; with and without an unknown ΔV anomaly
(0.3 m/s per axis, 2 days after the start of the gap, known epoch). Compares the sensor combinations
A, A′, B, B′, C, C′, D (± CMB) and E (± CMB), and sweeps the CMB NET (3–300 µK√s, plus 0.3 and 1 to find the
threshold) and the CMB sensor families A–F.

Usage (from the repository root):
    python studies/gap_study.py              # run everything (resumes from the CSV if it already has rows)
    python studies/gap_study.py --quick      # 1 seed, only the 30 and 85 day gaps (test)
    python studies/gap_study.py --summary    # only rebuild the tables from the CSV

Outputs: results/gap_study.csv (one row per simulation) and results/gap_study.md (tables).
Cost: ~2.5 h with 2 cores (242 simulations; the CMB cases use a 2 h window to keep the run time manageable).
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
from cmbsim import physics as ph                                   # noqa: E402
from cmbsim.instrument import FAMILY_PRESETS, family_config        # noqa: E402
from cmbsim.simulator import Config, band_weights, gap_stats, run  # noqa: E402
from cmbsim import skymaps                                          # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
CSV = os.path.join(OUT, "gap_study.csv")
MD = os.path.join(OUT, "gap_study.md")

GAP_START = 160.0
GAPS = (15.0, 30.0, 60.0, 85.0)
SEEDS = (1, 2)
ANOM_DV = 0.3

# common baseline: 2026 trajectory, TCMs outside the gap, weekly DSN with ΔDOR, dynamics with 5 % SRP error,
# 1 nm/s² unmodelled acceleration (τ = 1 d) and 1 mm/s desaturations (Config defaults)
BASE = dict(dsn_pass_days=7.0, dsn_pass_h=8.0, dsn_ddor=True, obs_analysis=False, maneuvers_days=(10.0, 60.0, 150.0, 250.0))
# CMB of case B: family D (trichroic TES, 90/150/220 GHz), 1 rpm spin, 2 h window
CMB = dict(**family_config("tes_3b"), spin=True, tau_s=7200.0)
NOCMB = dict(cmb_on=False, att_on=False)
IMU = dict(imu_on=True)
SDOP = dict(sdop_on=True)
AST = dict(ast_on=True)
XNAV = dict(xnav_on=True, xnav_model="ely", xnav_period_h=24.0)

CASES = {
    "A": ("Dynamics only", NOCMB),
    "A'": ("Dynamics + IMU", {**NOCMB, **IMU}),
    "B": ("CMB", CMB),
    "B'": ("CMB + IMU", {**CMB, **IMU}),
    "C": ("CMB + solar Doppler + IMU", {**CMB, **SDOP, **IMU}),
    "C'": ("Solar Doppler + IMU", {**NOCMB, **SDOP, **IMU}),
    "S": ("Solar Doppler (no IMU)", {**NOCMB, **SDOP}),
    "D": ("Asteroids", {**NOCMB, **AST}),
    "D+CMB": ("Asteroids + CMB", {**CMB, **AST}),
    "E": ("Compact XNAV", {**NOCMB, **XNAV}),
    "E+CMB": ("Compact XNAV + CMB", {**CMB, **XNAV}),
}
NET_SWEEP = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)      # effective (combined) NET [µK√s]
SWEEP_GAPS = (60.0, 85.0)
# ideal systematics: no attitude error, no 1/f and no gain error (only white noise + sky + zodiacal light):
# separates the NET limit from the systematics limit
IDEAL = dict(att_on=False, one_over_f=False, gain_cal_ppm=0.0, gain_drift_ppm=0.0)
IDEAL_GAP = 60.0

FIELDS = ["group", "case", "name", "gap_d", "anomaly", "seed", "net_eff", "ev_end", "sv_end", "ev_max", "er_end",
          "sr_end", "er_max", "within3", "ev_dyn", "ev_pre", "n_cmb", "t_s"]


def nf_tes():
    c = Config(**CMB)
    _, w, _ = band_weights(c)
    return skymaps.combo_noise_factor(w)


def jobs(quick=False):
    gaps = (30.0, 85.0) if quick else GAPS
    seeds = SEEDS[:1] if quick else SEEDS
    out = []
    for g in gaps:
        for an in (False, True):
            for s in seeds:
                for k in CASES:
                    out.append(dict(group="cases", case=k, gap_d=g, anomaly=an, seed=s))
    nf = nf_tes()
    for g in (SWEEP_GAPS[-1:] if quick else SWEEP_GAPS):
        for s in seeds:
            for net in NET_SWEEP:
                out.append(dict(group="net", case="B", gap_d=g, anomaly=True, seed=s, net_eff=net,
                                over=dict(net_uK_rts=net / nf)))
            if g == IDEAL_GAP:
                for net in NET_SWEEP:
                    out.append(dict(group="net_ideal", case="B", gap_d=g, anomaly=True, seed=s, net_eff=net,
                                    over=dict(IDEAL, net_uK_rts=net / nf)))
            for fam in FAMILY_PRESETS:
                out.append(dict(group="family", case=fam, gap_d=g, anomaly=True, seed=s,
                                over=dict(**family_config(fam), spin=True, tau_s=7200.0)))
    return out


def key(j):
    an = j["anomaly"] if isinstance(j["anomaly"], bool) else j["anomaly"] == "True"
    net = f"{float(j.get('net_eff') or 0):g}" if j["group"].startswith("net") else ""
    return (j["group"], j["case"], f"{float(j['gap_d']):g}", str(an), str(int(j["seed"])), net)


def one(j):
    g = float(j["gap_d"])
    a, b = GAP_START, GAP_START + g
    name, extra = CASES.get(j["case"], (j["case"], CMB))
    kw = {**BASE, **extra, **j.get("over", {})}
    kw.update(seed=int(j["seed"]), dsn_gaps=((a, b),), end_day=b + 2.0,
              anomaly_days=((a + 2.0,) if j["anomaly"] else ()), anomaly_dv_mps=ANOM_DV)
    t0 = time.time()
    r = run(Config(**kw))
    row = gap_stats(r, [(a, b)])[0]
    d = r.t / ph.DAY
    ipre = np.where(d < a)[0][-1]
    tech = r.info["tech"]
    net_eff = j.get("net_eff") or (float(tech.net_uK_rts) * float(r.info["noise_factor"]) if kw.get("cmb_on", True) else 0.0)
    return dict(group=j["group"], case=j["case"], name=name if j["group"] == "cases" else tech.name, gap_d=g,
                anomaly=bool(j["anomaly"]), seed=int(j["seed"]), net_eff=round(float(net_eff), 3),
                ev_end=row["v error end [m/s]"], sv_end=row["1σ v end [m/s]"], ev_max=row["v error max [m/s]"],
                er_end=row["r error end [km]"], sr_end=row["1σ r end [km]"], er_max=row["r error max [km]"],
                within3=row["% within ±3σ"], ev_dyn=row["Dynamics only: v error end [m/s]"],
                ev_pre=round(float(np.linalg.norm(r.x_est[ipre, 3:6] - r.x_true[ipre, 3:]) * 1e3), 4),
                n_cmb=int(r.info["n_cmb"]), t_s=round(time.time() - t0, 1))


def load():
    if not os.path.exists(CSV):
        return []
    with open(CSV, newline="") as fh:
        return list(csv.DictReader(fh))


def summarize():
    rows = load()
    if not rows:
        print("no results")
        return
    for r in rows:
        for k in ("gap_d", "net_eff", "ev_end", "sv_end", "ev_max", "er_end", "sr_end", "er_max", "within3", "ev_pre"):
            r[k] = float(r[k])
        r["anomaly"] = r["anomaly"] == "True"
    rms = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else np.nan
    L = ["# Tracking-gap study — results", "",
         f"Generated by `studies/gap_study.py` on {time.strftime('%Y-%m-%d %H:%M')}. Gap starting on day "
         f"{GAP_START:g}; anomaly of {ANOM_DV} m/s per axis on day {GAP_START + 2:g}; weekly 8 h DSN passes with ΔDOR "
         "before and after. Values: RMS over the seeds, at the last instant before the end of the gap.", ""]
    cases = [c for c in CASES if any(r["case"] == c and r["group"] == "cases" for r in rows)]
    gaps = sorted({r["gap_d"] for r in rows if r["group"] == "cases"})
    for an in (False, True):
        L += [f"## {'With' if an else 'Without'} ΔV anomaly — velocity [m/s] / position [km] error at the end of the gap", "",
              "| Case | Sensors | " + " | ".join(f"{g:g} d" for g in gaps) + " | Consistency (±3σ) |",
              "|---|---|" + "---|" * len(gaps) + "---|"]
        best = {}
        for g in gaps:
            sc = {c: rms([r["ev_end"] for r in rows if r["group"] == "cases" and r["case"] == c
                          and r["gap_d"] == g and r["anomaly"] == an]) for c in cases}
            best[g] = min((v, c) for c, v in sc.items() if np.isfinite(v))[1] if sc else None
        for c in cases:
            cells, ins = [], []
            for g in gaps:
                sel = [r for r in rows if r["group"] == "cases" and r["case"] == c and r["gap_d"] == g and r["anomaly"] == an]
                if not sel:
                    cells.append("—")
                    continue
                v, p = rms([r["ev_end"] for r in sel]), rms([r["er_end"] for r in sel])
                ins += [r["within3"] for r in sel]
                cell = f"{v:.3g} / {p:.3g}"
                cells.append(f"**{cell}**" if best.get(g) == c else cell)
            L.append(f"| {c} | {CASES[c][0]} | " + " | ".join(cells) + f" | {np.mean(ins):.0f} % |")
        L.append("")
    sw = [r for r in rows if r["group"] == "net"]
    if sw:
        L += ["## CMB NET sweep (case B, with anomaly)", "",
              "Effective NET = array NET × component-separation factor (90/150/220 GHz). Reference: case A "
              "(dynamics only) for the same gaps.", "",
              "| Effective NET [µK√s] | " + " | ".join(f"{g:g} d: v error [m/s]" for g in SWEEP_GAPS) + " |",
              "|---|" + "---|" * len(SWEEP_GAPS)]
        for net in sorted({r["net_eff"] for r in sw}):
            L.append(f"| {net:g} | " + " | ".join(
                f"{rms([r['ev_end'] for r in sw if r['net_eff'] == net and r['gap_d'] == g]):.3g}" for g in SWEEP_GAPS) + " |")
        refA = {g: rms([r["ev_end"] for r in rows if r["group"] == "cases" and r["case"] == "A" and r["anomaly"]
                        and r["gap_d"] == g]) for g in SWEEP_GAPS}
        L.append("| A (dynamics only) | " + " | ".join(f"{refA[g]:.3g}" for g in SWEEP_GAPS) + " |")
        L.append("")
        for g in SWEEP_GAPS:
            ok = [net for net in sorted({r["net_eff"] for r in sw}, reverse=True)
                  if rms([r["ev_end"] for r in sw if r["net_eff"] == net and r["gap_d"] == g]) < 0.5 * refA[g]]
            L.append(f"- {g:g} d gap: highest useful NET (error < half of case A) = "
                     + (f"**{ok[0]:g} µK√s**" if ok else "**no value in the sweep**") + ".")
        L.append("")
    swi = [r for r in rows if r["group"] == "net_ideal"]
    if swi:
        refA = rms([r["ev_end"] for r in rows if r["group"] == "cases" and r["case"] == "A" and r["anomaly"]
                    and r["gap_d"] == IDEAL_GAP])
        L += [f"## NET sweep with ideal systematics (case B, anomaly, {IDEAL_GAP:g} d gap)", "",
              "No attitude error, no 1/f and no gain error: what remains is white noise, the sky and the zodiacal light. "
              "The difference from the previous table shows how much the systematics cost.", "",
              "| Effective NET [µK√s] | v error [m/s] | Filter 1σ [m/s] | r error [km] |", "|---|---|---|---|"]
        for net in sorted({r["net_eff"] for r in swi}):
            sel = [r for r in swi if r["net_eff"] == net]
            L.append(f"| {net:g} | {rms([r['ev_end'] for r in sel]):.3g} | {rms([r['sv_end'] for r in sel]):.3g} | "
                     f"{rms([r['er_end'] for r in sel]):.3g} |")
        L.append(f"| A (dynamics only) | {refA:.3g} | — | — |")
        ok = [net for net in sorted({r["net_eff"] for r in swi}, reverse=True)
              if rms([r["ev_end"] for r in swi if r["net_eff"] == net]) < 0.5 * refA]
        L += ["", "- Highest useful NET with ideal systematics (error < half of case A) = "
              + (f"**{ok[0]:g} µK√s**" if ok else "**no value in the sweep**") + ".", ""]
    fam = [r for r in rows if r["group"] == "family"]
    if fam:
        L += ["## CMB sensor families, case B with anomaly", "",
              "| Family | Preset | Effective NET [µK√s] | Mass [kg] | Power [W] | " + " | ".join(f"{g:g} d: v error [m/s]" for g in SWEEP_GAPS) + " |",
              "|---|---|---|---|---|" + "---|" * len(SWEEP_GAPS)]
        for k, t in FAMILY_PRESETS.items():
            sel = [r for r in fam if r["case"] == k]
            if not sel:
                continue
            L.append(f"| {t.family} | `{k}` | {sel[0]['net_eff']:.3g} | {t.mass_kg[0]:g}–{t.mass_kg[1]:g} | "
                     f"{t.power_w[0]:g}–{t.power_w[1]:g} | " + " | ".join(
                         f"{rms([r['ev_end'] for r in sel if r['gap_d'] == g]):.3g}" for g in SWEEP_GAPS) + " |")
        L.append("")
    os.makedirs(OUT, exist_ok=True)
    with open(MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="Tracking-gap study (use case 1).")
    ap.add_argument("--quick", action="store_true", help="1 seed, only the 30 and 85 day gaps (test)")
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
    todo = [j for j in jobs(a.quick) if key(j) not in done]
    print(f"{len(todo)} simulations to run ({len(done)} already in the CSV)", flush=True)
    new = not os.path.exists(CSV)
    with open(CSV, "a", newline="") as fh, Pool(a.nproc) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(one, todo)):
            w.writerow(row)
            fh.flush()
            print(f"[{i + 1}/{len(todo)}] {row['group']} {row['case']} gap {row['gap_d']:g} d "
                  f"anomaly={row['anomaly']} seed {row['seed']}: v error = {row['ev_end']:.3g} m/s, "
                  f"r = {row['er_end']:.3g} km ({row['t_s']:.0f} s)", flush=True)
    provenance.stamp(CSV, mixed)
    summarize()


if __name__ == "__main__":
    main()
