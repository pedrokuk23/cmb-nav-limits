"""Builds Figures 1 and 3–7 of the article from the analytic expression and the files in results/.

  Fig. 1  fig_photon_limit.pdf  photon-noise floor of a CMB velocimeter, eq. (6) of the article (no data needed)
  Fig. 4  fig_timeseries.pdf    velocity error during the 60-day gap        <- results/gap_timeseries.npz
  Fig. 6  fig_ladder.pdf        CMB sensor levels versus competing sensors  <- results/ratio_study.csv,
                                                                               results/deep_space_study.csv,
                                                                               results/deep_space_calibrated_study.csv
  Fig. 5  fig_requirements.pdf  ratio receiver: gain, load and alignment     <- results/ratio_study.csv
  Fig. 7  fig_budget.pdf        floor budget, analytic floor to instrument   <- results/floor_budget_study.csv
  Fig. 3  fig_consistency.pdf   filter consistency                           <- results/consistency_study.csv,
                                                                               results/consistency_series.npz
Fig. 2 (simulator block diagram) is a TikZ source: compile figures/fig_simulator.tex with pdflatex.

Usage (from the repository root): python figures/make_figures.py   -> figures/*.pdf and *.png
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

C_LIGHT = 299792458.0
K_B = 1.380649e-23
H_P = 6.62607015e-34
T0 = 2.7255
ZETA4_24 = 25.976                                                # 4! zeta(4) = integral of x^4 e^x / (e^x - 1)^2


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"))
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=300)
    plt.close(fig)
    print("wrote", os.path.join("figures", name + ".pdf"))


def rms(x):
    x = [float(v) for v in x]
    return float(np.sqrt(np.mean(np.square(x))))


def read_csv(name):
    with open(os.path.join(RES, name), newline="") as fh:
        return list(csv.DictReader(fh))


def fig_photon_limit():
    """Fig. 1: sigma_v = c / sqrt(F_G G tau), F_G = (2/3) (k T0 / h)^3 4! zeta(4) / c^2."""
    f_g = (2 / 3) * (K_B * T0 / H_P) ** 3 * ZETA4_24 / C_LIGHT ** 2
    tau = np.logspace(np.log10(600), np.log10(30 * 86400), 200)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for g, ls in [(1e-3, ":"), (1e-2, "-."), (1e-1, "--"), (1, "-")]:
        ax.loglog(tau / 86400, C_LIGHT / np.sqrt(f_g * g * tau), "k", ls=ls, lw=1, label=f"$G={g:g}$ m$^2$ sr")
    # present-day focal plane: effective NET 1 uK sqrt(s), full-sky scan -> 0.65 m/s per day (white noise only)
    ax.loglog(tau / 86400, 0.65 * np.sqrt(86400 / tau), color="tab:red", lw=1.2,
              label="Today, NET$_{\\rm eff}$=1 $\\mu$K$\\sqrt{\\rm s}$ (white only)")
    for y, lab in [(1.79, "1$^{\\prime\\prime}$ pointing"), (0.37, "1 ppm gain"), (0.0018, "1 mas pointing")]:
        ax.axhline(y, color="tab:blue", lw=0.6, alpha=0.7)
        ax.text(0.0075, y * 1.12, lab, color="tab:blue", fontsize=6)
    ax.set_xlabel("Integration time [days]")
    ax.set_ylabel("$\\sigma_v$ per axis [m/s]")
    ax.set_xlim(tau[0] / 86400, 30)
    ax.set_ylim(1e-3, 20)
    ax.legend(fontsize=5.5, loc="upper right", frameon=False)
    ax.grid(alpha=0.2, which="both", lw=0.3)
    fig.tight_layout()
    save(fig, "fig_photon_limit")


def fig_timeseries():
    """Fig. 4: velocity error and filter 1-sigma during the 60-day gap (one seed)."""
    z = np.load(os.path.join(RES, "gap_timeseries.npz"))
    cases = [("dyn", "Dynamics only", "#1f77b4", 2.4), ("today", "CMB, present", "#d62728", 1.0),
             ("metro", "CMB, metrology", "#9467bd", 1.3), ("limit", "CMB, photon limit", "#ff7f0e", 1.3),
             ("imu", "IMU", "#2ca02c", 1.3), ("xnav", "XNAV", "#000000", 1.1)]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for k, lab, c, lw in cases:
        t, e, s = z[k + "_t"], z[k + "_ev"], z[k + "_sv"]
        if t[-1] < 60:                                           # hold the last filter value to the end of the gap
            t, e, s = np.append(t, 60.0), np.append(e, e[-1]), np.append(s, s[-1])
        ax.semilogy(t, e, color=c, lw=lw, label=lab, drawstyle="steps-post")
        ax.semilogy(t, s, color=c, lw=0.7, ls=(0, (3, 2)), drawstyle="steps-post")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.axvline(2, color="0.6", lw=0.6, ls=":")
    ax.text(2.6, 4.5e-4, "$\\Delta V$ anomaly", fontsize=6, color="0.35")
    ax.set_xlabel("Days since start of tracking gap")
    ax.set_ylabel("Velocity error [m/s]")
    ax.set_xlim(-5, 61)
    ax.set_ylim(4e-4, 3)
    ax.grid(axis="y", which="major", color="0.9", lw=0.5)
    h, l_ = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color="0.4", lw=0.7, ls=(0, (3, 2))))
    l_.append("Filter 1$\\sigma$ (3D)")
    fig.legend(h, l_, fontsize=6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.0),
               handlelength=1.6, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    save(fig, "fig_timeseries")


def fig_ladder():
    """Fig. 6: velocity error per CMB level in the three scenarios of Table VI of the article."""
    rr = read_csv("ratio_study.csv")
    gap = lambda c: rms([r["ev_end"] for r in rr if r["group"] == "levels" and r["case"] == c])
    cal = {r["case"]: r for r in read_csv("deep_space_calibrated_study.csv")}
    deep = read_csv("deep_space_calibrated_study.csv")
    base = read_csv("deep_space_study.csv")
    sky = "v error plane of sky [m/s]"
    pick = lambda rows, grp, case: float(next(r[sky] for r in rows if r["group"] == grp and r["case"] == case))
    levels = [("No\nCMB", "dyn", None), ("Pre-\nsent", "today", "realistic"), ("Requi-\nrem.", "req", "requirements_sim"),
              ("Ratio", "ratio", "ratio_sim"), ("Metro-\nlogy", "metro", "metrology_sim"),
              ("Photon\nlimit", "limit", "limit_sim"), ("Photon\n(analyt.)", None, "limit")]
    g_v, d_v, a_v = [], [], []
    for _, rc, dl in levels:
        g_v.append(gap(rc) if rc else float(cal["limit"]["v error total [m/s]"]))
        d_v.append(pick(deep, "3", "Monthly DSN") if dl is None else pick(deep, "3", f"+ CMB {dl}"))
        a_v.append(pick(deep, "4b", "Dynamics only") if dl is None else pick(deep, "4b", f"CMB {dl}"))
    refs = [(gap("imu"), "IMU (gap)", "#2a78d6"),
            (pick(base, "2", "Weekly DSN + ΔDOR 5 nrad"), "DSN + weekly $\\Delta$DOR (100 AU)", "#eb6834"),
            (pick(deep, "4b", "XNAV"), "XNAV (100 AU, auton.)", "#1baf7a")]
    x = np.arange(len(levels))
    w = 0.27
    fig, ax = plt.subplots(figsize=(3.5, 3.1))
    ax.bar(x - w, g_v, w * 0.92, color="#2a78d6", label="60-d gap + $\\Delta V$, 1.5 AU")
    ax.bar(x, d_v, w * 0.92, color="#eb6834", label="100 AU, monthly DSN")
    ax.bar(x + w, a_v, w * 0.92, color="#1baf7a", label="100 AU, autonomous + $\\Delta V$")
    ax.axvspan(5.5, 6.5, color="0.94", zorder=0)
    for y, lab, c in refs:
        ax.axhline(y, color=c, lw=0.8, ls=(0, (4, 2)), label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([lv[0] for lv in levels], fontsize=6.3)
    ax.set_ylabel("Velocity error [m/s]")
    fig.legend(fontsize=5.6, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.55, 1.0), columnspacing=0.8,
               handlelength=1.8)
    ax.set_ylim(1e-3, 1.5)
    ax.set_xlim(-0.55, 6.5)
    ax.grid(axis="y", color="0.9", lw=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    save(fig, "fig_ladder")


def fig_requirements():
    """Fig. 5: velocity error at the end of the 60-day gap versus gain drift, load drift and alignment drift."""
    rr = read_csv("ratio_study.csv")

    def curve(group, case):
        ps = sorted({float(r["param"]) for r in rr if r["group"] == group and r["case"] == case})
        ev = [rms([r["ev_end"] for r in rr if r["group"] == group and r["case"] == case and float(r["param"]) == p])
              for p in ps]
        sv = [rms([r["sv_end"] for r in rr if r["group"] == group and r["case"] == case and float(r["param"]) == p])
              for p in ps]
        return np.array(ps), np.array(ev), np.array(sv)

    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.35))
    ax = axs[0]
    for grp, net, ls in (("gain", "0.3", "--"), ("gain_met", "0.083", "-")):
        for case, lab, c, mk in (("differential", "Differential", "#1f77b4", "o"), ("total", "Ratio (total power)",
                                                                                      "#d62728", "s")):
            x, y, _ = curve(grp, case)
            ax.plot(x, y, ls=ls, color=c, marker=mk, ms=4, lw=1.3, label=f"{lab}, NET$_{{\\rm eff}}$ {net}")
    ax.set_xscale("log")
    ax.set_xlabel("Gain drift [ppm]")
    ax.set_ylabel("Velocity error [m/s]")
    ax.set_title("(a) Gain", fontsize=8)
    ax.set_ylim(0, 0.46)
    ax.legend(fontsize=5.2, frameon=False, loc="upper left", ncol=1)
    ax = axs[1]
    x, y, s_ = curve("load", "total")
    ax.plot(x, y, color="#d62728", marker="s", ms=4, lw=1.3, label="Error")
    ax.plot(x, s_, color="#d62728", lw=0.8, ls=(0, (3, 2)), label="Filter 1$\\sigma$")
    ax.axvspan(1000, x.max() * 1.2, color="0.93", zorder=0)
    ax.text(1600, 0.02, "degrades", fontsize=5.5, color="0.4", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("Reference-load drift [nK]")
    ax.set_title("(b) Reference load", fontsize=8)
    ax.set_ylim(0, 0.2)
    ax.set_xlim(x.min() / 1.2, x.max() * 1.2)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    ax = axs[2]
    x, y, s_ = curve("attitude", "total")
    ax.plot(x * 1e3, y, color="#d62728", marker="s", ms=4, lw=1.3, label="Error")
    ax.plot(x * 1e3, s_, color="#d62728", lw=0.8, ls=(0, (3, 2)), label="Filter 1$\\sigma$")
    ax.axvspan(300, x.max() * 1e3 * 1.2, color="0.93", zorder=0)
    ax.text(560, 0.03, "degrades", fontsize=5.5, color="0.4", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("Alignment drift [mas]")
    ax.set_title("(c) Beam alignment", fontsize=8)
    ax.set_ylim(0, 0.45)
    ax.set_xlim(x.min() * 1e3 / 1.2, x.max() * 1e3 * 1.2)
    ax.legend(fontsize=5.5, frameon=False, loc="upper left")
    for a_ in axs:
        a_.grid(color="0.9", lw=0.5)
    fig.tight_layout()
    save(fig, "fig_requirements")


def fig_budget():
    """Fig. 7: floor budget from the analytic photon floor to the simulated photon-limit instrument."""
    rows = read_csv("floor_budget_study.csv")
    steps = [("A", "Ana-\nlytic"), ("B", "4\nbeams"), ("C", "White\nonly"),
             ("D", "+ sky"), ("E", "+ zodi"), ("F", "+ gain"), ("G", "+ load"), ("H", "+ att.")]
    ev = [rms([r["ev_end"] for r in rows if r["step"] == k]) for k, _ in steps]
    sv = [rms([r["sv_end"] for r in rows if r["step"] == k]) if k not in ("A", "B") else np.nan for k, _ in steps]
    x = np.arange(len(steps))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    cols = ["0.65", "0.65"] + ["#1f77b4"] * 6
    ax.bar(x, np.array(ev) * 1e3, 0.62, color=cols)
    ax.plot(x, np.array(sv) * 1e3, "k_", ms=14, mew=1.2, label="Filter 1$\\sigma$")
    for xi, v in zip(x, ev):
        ax.text(xi, v * 1e3 * 1.04, f"{v * 1e3:.1f}", ha="center", va="bottom", fontsize=5.8)
    ax.axhline(6.4, color="#2ca02c", lw=0.9, ls=(0, (4, 2)), label="IMU (6.4 mm/s)")
    ax.axhline(2.8, color="#000000", lw=0.9, ls=(0, (4, 2)), label="XNAV (2.8 mm/s)")
    ax.axvline(1.5, color="0.6", lw=0.6, ls=":")
    ax.text(0.5, 13, "reduced\nmodel", ha="center", fontsize=5.8, color="0.35")
    ax.text(4.5, 36, "full simulator, photon-limit instrument", ha="center", fontsize=5.8, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([s_[1] for s_ in steps], fontsize=5.6)
    ax.set_ylabel("Velocity error after 60-d gap [mm/s]")
    ax.set_ylim(0, 40)
    ax.legend(fontsize=5.6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    ax.grid(axis="y", color="0.9", lw=0.5)
    fig.tight_layout()
    save(fig, "fig_budget")


def fig_consistency():
    """Fig. 3: (a) normalized per-axis velocity error during the gap, one seed; (b) mean NEES per case."""
    z = np.load(os.path.join(RES, "consistency_series.npz"))
    rows = read_csv("consistency_study.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.35), gridspec_kw=dict(width_ratios=[1.6, 1]))
    ax = axs[0]
    for case, lab, c in (("limit", "CMB, photon limit", "#ff7f0e"), ("xnav", "XNAV", "#000000")):
        t, zz = z[case + "_t"], z[case + "_z"]
        for i, ls in enumerate(("-", "--", ":")):
            ax.plot(t, zz[:, i], color=c, lw=0.8, ls=ls, label=f"{lab}, {'xyz'[i]}")
    ax.axhspan(-3, 3, color="0.93", zorder=0)
    ax.axhline(3, color="0.5", lw=0.6)
    ax.axhline(-3, color="0.5", lw=0.6)
    ax.set_xlim(0, 60)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("Days since start of tracking gap")
    ax.set_ylabel("Velocity error / filter $\\sigma$")
    ax.set_title("(a) Normalized error per axis (one seed)", fontsize=8)
    ax.legend(fontsize=5.3, frameon=False, ncol=2, loc="lower right")
    ax = axs[1]
    order = [("dyn", "Dyn. only"), ("today", "CMB present"), ("req", "CMB req."), ("ratio", "CMB ratio"),
             ("metro", "CMB metrol."), ("limit", "CMB limit"), ("imu", "IMU"), ("xnav", "XNAV")]
    vals = np.array([np.mean([float(r["nees_mean"]) for r in rows if r["case"] == k and r["filter"] == "ukf"])
                     for k, _ in order]) / 3.0
    y = np.arange(len(order))
    cols = ["#d62728" if v > 1.5 else "#1f77b4" for v in vals]
    ax.barh(y, vals, 0.6, color=cols)
    for yi, v in zip(y, vals):
        ax.text(max(v, 1.0) + 0.05, yi, f"{v:.2f}", va="center", fontsize=5.8)
    ax.axvline(1, color="0.3", lw=0.8, ls=(0, (4, 2)))
    ax.text(1.03, -0.62, "consistent = 1", fontsize=5.8, color="0.3", va="center")
    ax.text(2.25, 1.3, "filter\noptimistic", fontsize=5.8, color="#d62728", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([o[1] for o in order], fontsize=6)
    ax.set_ylim(len(order) - 0.5, -1.0)
    ax.set_xlim(0, 2.6)
    ax.set_xlabel("Mean velocity NEES / 3")
    ax.set_title("(b) Consistency per case (6 seeds)", fontsize=8)
    for a_ in axs:
        a_.grid(color="0.9", lw=0.5)
        a_.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_consistency")


if __name__ == "__main__":
    fig_photon_limit()
    fig_timeseries()
    fig_ladder()
    fig_requirements()
    for fn, needs in ((fig_budget, "floor_budget_study.csv"), (fig_consistency, "consistency_study.csv")):
        if os.path.exists(os.path.join(RES, needs)):
            fn()
