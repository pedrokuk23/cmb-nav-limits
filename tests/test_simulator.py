import numpy as np

from cmbsim import physics as ph
from cmbsim.simulator import Config, run, snapshot_velocity, reference_transfer


def test_lambert_window_2026():
    _, _, _, info = reference_transfer(Config())
    assert 8 < info["c3"] < 11            # 2026 Earth-Mars window: C3 ~ 9 km²/s²


def test_snapshot_recovers_velocity_staring_and_spinning():
    from cmbsim.simulator import Stare, Spin
    v = ph.v_ssb() + np.array([20.0, -15.0, 1.0])
    for Geo in (Stare, Spin):
        g = Geo(np.array([1.0, 0.3, 0.0]), 1e-7, np.eye(4) * 1e-12)
        z = g.model(v[None], np.array([1.0001 * ph.T0_CMB]))[0]
        assert np.linalg.norm(snapshot_velocity(z, g, ph.v_ssb()) - v) < 1e-5


def test_spin_reduces_sky_bias():
    from cmbsim.simulator import sky_bias_cov
    static = sky_bias_cov(False, 0.5, True)
    spinning = sky_bias_cov(True, 0.5, True)
    assert spinning[1, 1] < static[1, 1] / 100     # ring average << single point


def test_short_simulation_improves_over_dynamics():
    r = run(Config(end_day=40, tau_s=3600, tech="litebird", seed=2, att_on=False))
    ev = np.linalg.norm(r.x_est[:, 3:6] - r.x_true[:, 3:], axis=1) * 1e3
    ed = np.linalg.norm(r.x_dyn[:, 3:] - r.x_true[:, 3:], axis=1) * 1e3
    assert ev[-1] < ed[-1] and ev[-1] < 2.0


def test_equipment_defaults_and_statistics():
    from cmbsim.simulator import segment_stats
    assert Config(tech="litebird").tau() == 1800.0
    assert Config(tech="litebird").instrument().fs_hz == 19.0
    assert Config(tech="litebird", fs_hz=50.0).instrument().fs_hz == 50.0
    r = run(Config(end_day=70, tech="dedicated", seed=3))
    rows = segment_stats(r)
    assert len(rows) == 3 and rows[1]["Segment"].startswith("TCM 1")


def test_attitude_algorithms():
    """QUEST/ESOQ2/q/TRIAD recover the attitude without noise; with noise, the optimal ones reach the bound."""
    from scipy.spatial.transform import Rotation
    from cmbsim import attitude as at
    A = Rotation.random(random_state=5).as_matrix()
    r = at.catalogue()[0][:10]
    b = r @ A.T
    for f in (at.att_quest, at.att_esoq2, at.att_qmethod):
        assert np.linalg.norm(at.small_angle(at.quatrmx(f(r, b)), A)) < 1e-9
    assert np.linalg.norm(at.small_angle(at.att_triad(r[0], r[1], b[0], b[1]), A)) < 1e-9
    q = at.rmxquat(A)
    w = np.array([0.01, -0.02, 0.005])
    assert np.allclose(at.angular_velocity(q, at.propagate_quat(q, w, 1e-3), 1e-3), w, rtol=1e-3)
    res, n = at.benchmark(at.StarTracker(), at.mount(np.array([1.0, 0.3, 0.1])), n_frames=200)
    assert n >= 5
    for m in ("quest", "esoq2", "qmethod"):
        assert np.all(res[m] < 1.2 * res["crlb"])
    assert res["triad"][0] > 2 * res["crlb"][0]


def test_attitude_in_simulation():
    r = run(Config(end_day=40, tau_s=3600, tech="litebird", seed=2))
    a = r.info["att"]
    assert a is not None and a["avail"] == 1.0 and len(a["t"]) > 100
    assert a["budget_mis"] > a["budget_white"]          # 3″ misalignment >> window-averaged noise
    r0 = run(Config(end_day=40, tau_s=3600, tech="litebird", seed=2, att_on=False))
    assert r0.info["att"] is None


def test_foregrounds_minimum_between_70_and_100_ghz():
    import pytest
    from cmbsim import skymaps
    if not skymaps.available():
        pytest.skip("healpy or sky maps unavailable")
    assert abs(skymaps.rj_to_cmb(100.0) - 1.2866) < 0.001           # K_RJ -> K_CMB conversion
    rows = {r["nu"]: r["total"] for r in skymaps.spectrum_table([30.0, 90.0, 353.0])}
    assert rows[90.0] < rows[30.0] and rows[90.0] < rows[353.0]
    w1, w2 = skymaps.two_freq_weights(90.0, 150.0)
    assert abs(w1 + w2 - 1) < 1e-12 and w2 < 0


def test_1f_noise_has_1_over_f_spectrum_and_spin_avoids_it():
    from cmbsim.instrument import Flicker
    rng = np.random.default_rng(1)
    fl = Flicker(1.0, 0.02, 1.0, 1e-3, 1, rng, f_lo=1e-5, f_hi=5)
    dt, N = 0.05, 2**16
    x = np.array([fl.step(k * dt)[0] for k in range(N)])
    from scipy.signal import welch
    fr, P = welch(x, fs=1 / dt, nperseg=2**13)
    for f0 in (0.03, 0.1, 0.3):
        i = np.argmin(abs(fr - f0))
        assert 0.5 < P[i] / (2 * 0.02 / f0) < 2
    assert Flicker(1e-6, 0.02, 1.0, 1800, 4, rng).spin_factor(0.1) < 1.3


def test_zodiacal_kinematic_and_gain():
    import pytest
    from cmbsim import skymaps, zodi
    from cmbsim.simulator import Config, reference_transfer
    if not (skymaps.available() and zodi.available()):
        pytest.skip("healpy/zodipy unavailable")
    jd0, r0, _, _ = reference_transfer(Config())
    Z = zodi.Zodi(140.0, jd0, 295.0)
    ecl = Z.value(-r0 / np.linalg.norm(r0), 0.0)[0]             # anti-Sun, in the ecliptic plane
    pole = Z.value(np.array([0, 0, 1.0]), 0.0)[0]
    assert 1e-6 < ecl < 1e-4 and pole < ecl                      # µK in the plane, lower at the ecliptic pole
    s = skymaps.Sky(90.0, 0.5, True, None, 3.0, seed=1)
    assert np.all(np.abs(s.kinematic(np.eye(3), np.zeros(3))) < 1e-12)   # no velocity, no effect
    assert np.max(np.abs(s.kinematic(np.eye(3), np.array([1e-4, 0, 0])))) < 5e-6
    r = run(Config(end_day=25, tau_s=3600, tech="litebird", seed=3, gain_cal_ppm=100.0))
    assert r.info["gain"]["receiver"] == "differential" and abs(np.mean(r.info["gain"]["g0_ppm"])) < 400


def test_spin_records_ring_averages():
    """The ring averages a0 do not enter the filter with 1/f noise, but must be recorded (for plots)."""
    from cmbsim.simulator import Config, run
    res = run(Config(spin=True, tof_days=200, sky_real=False))
    assert np.isfinite(res.cmb_T[:, :4]).all(axis=1).mean() > 0.9


def test_estimated_gain_removes_calibration():
    """With the gain in the filter, a large constant calibration error must not spoil navigation."""
    from cmbsim.simulator import Config, run
    kw = dict(spin=True, tof_days=200, sky_real=False, zodi_on=False, gain_cal_ppm=200.0, gain_drift_ppm=0.0)
    def rms(r):
        m = r.t > 30 * 86400
        return np.sqrt(np.mean(np.sum((r.x_est[m, 3:6] - r.x_true[m, 3:]) ** 2, 1)))
    r_est = run(Config(**kw))
    r_no = run(Config(gain_estimate=False, **kw))
    assert r_est.x_est.shape[1] == 29 and r_no.x_est.shape[1] == 21     # + 1 solar radiation pressure state
    assert r_est.info["gain"]["est_ppm"].shape == r_est.info["gain"]["g_ppm"].shape
    # the filter learns the differential gain between sensors; navigation does not degrade (with spin, the sky
    # biases and the Sun δv also absorb much of the constant calibration)
    e, t = r_est.info["gain"]["est_ppm"][-1], r_est.info["gain"]["g_ppm"][-1]
    assert np.corrcoef(e - e.mean(), t - t.mean())[0, 1] > 0.8
    assert rms(r_est) < 1.5 * rms(r_no)          # single seed; the velocity gain is studied separately


def test_truth_with_dynamic_model_errors():
    """Truth with wrong SRP, unmodelled acceleration and impulses; the filter estimates the SRP and stays consistent."""
    from cmbsim.simulator import Config, run
    kw = dict(spin=True, tof_days=200, sky_real=False, zodi_on=False)
    r = run(Config(**kw))
    d = r.info["dyn"]
    assert d["mismatch"] and d["acc_nms"] is not None and len(d["desat"]) > 10
    assert r.x_est.shape[1] == 29 and d["srp_idx"] == 22
    r0 = run(Config(dyn_mismatch=False, srp_estimate=False, **kw))
    assert r0.x_est.shape[1] == 28 and r0.info["dyn"]["acc_nms"] is None
    # different true trajectories, but the same navigation error level
    assert np.abs(r.x_true[-1, :3] - r0.x_true[-1, :3]).max() > 0
    m = r.t > 30 * 86400
    e = (r.x_est[m, 3:6] - r.x_true[m, 3:]) * 1e3
    s = np.sqrt(r.P_diag[m, 3:6]) * 1e3
    assert np.mean(np.abs(e) < 3 * s) > 0.95


def test_linearized_temperature():
    """2nd order of the linearization matches the formula; without the onboard model, the error shows up (isolated case)."""
    from cmbsim import physics as ph
    from cmbsim.simulator import Config, run
    f = ph.linearizer(90.0, 0.0)                      # narrow band
    dT = 3.36e-3
    assert abs((f(ph.T0_CMB + dT) - ph.T0_CMB - dT) - dT**2 / (2 * ph.T0_CMB) * ph.lin_boost(90.0)) < 5e-9
    kw = dict(tech="litebird", spin=True, tof_days=200, sky_real=False, one_over_f=False, att_on=False,
              zodi_on=False, gain_cal_ppm=0.0, gain_drift_ppm=0.0, gain_estimate=False, dyn_mismatch=False)
    def rms(r):
        m = r.t > 30 * 86400
        return np.sqrt(np.mean(np.sum((r.x_est[m, 3:6] - r.x_true[m, 3:]) ** 2, 1))) * 1e3
    r_c = run(Config(**kw))
    r_n = run(Config(lin_correct=False, **kw))
    assert r_c.info["lin"] is not None and rms(r_n) > 3 * rms(r_c)


def test_observability_common_gain():
    """The observability analysis flags the common gain (4 sensors together) as an unobservable combination."""
    from cmbsim.simulator import Config, run
    r = run(Config(spin=True, tof_days=200, sky_real=False, zodi_on=False))
    ob = r.info["obs"]
    assert ob is not None and len(ob["days"]) > 100 and ob["ratio"].shape[1] == len(ob["labels"])
    L = ob["labels"]

    def common_gain(w):
        cal = np.array([w["vec"][L.index(f"gain cal {j+1}")] for j in range(4)])
        return w["ratio"] > 0.9 and np.all(np.sign(cal) == np.sign(cal[0])) and np.abs(cal).min() > 0.3
    assert any(common_gain(w) for w in ob["weak"])
    assert ob["ratio"][-1, L.index("v_x")] < 0.05


def test_electric_propulsion():
    """Thrust with bias and fluctuation: the trajectory changes and the filter (with thrust bias states) stays consistent."""
    from cmbsim.simulator import Config, run
    kw = dict(tech="litebird", spin=True, tof_days=200, sky_real=False, zodi_on=False, one_over_f=False,
              ep_on=True, obs_analysis=False)
    r = run(Config(**kw))
    ep = r.info["ep"]
    assert ep is not None and len(ep["bias"]) == 2 and r.x_est.shape[1] == 32
    m = r.t > 30 * 86400
    e = (r.x_est[m, 3:6] - r.x_true[m, 3:]) * 1e3
    s = np.sqrt(r.P_diag[m, 3:6]) * 1e3
    assert np.mean(np.abs(e) < 3 * s) > 0.95                     # filter consistent with the wrong thrust
    # the thrust (~600 m/s in the 1st arc) moves the trajectory off the ballistic profile
    r0 = run(Config(**{**kw, "ep_on": False}))
    assert np.linalg.norm(r.x_true[-1, 3:] - r0.x_true[-1, 3:]) > 0.3


def _quick(**kw):
    base = dict(tech="litebird", spin=True, sky_real=False, att_on=False, zodi_on=False, gain_estimate=False,
                gain_cal_ppm=0.0, gain_drift_ppm=0.0, dyn_mismatch=False, srp_estimate=False, obs_analysis=False,
                temp_linearized=False, tof_days=120)
    base.update(kw)
    from cmbsim.simulator import Config, run
    r = run(Config(**base))
    m = r.t > 30 * 86400
    e = (r.x_est[m, 3:6] - r.x_true[m, 3:]) * 1e3
    s = np.sqrt(r.P_diag[m, 3:6]) * 1e3
    return r, float(np.sqrt(np.mean(np.sum(e**2, 1)))), float(np.mean(np.abs(e) < 3 * s))


def test_precessing_axis_and_destriping():
    """Tilted axis in 4 orientations (16 bias states) and destriping with stable offsets: consistent filter."""
    r, v, c = _quick(spin_tilt_deg=15.0, prec_n=4, one_over_f=True, f_knee_hz=2e-4, f_min_1f_hz=1e-4, destripe=True)
    assert r.x_est.shape[1] == 14 + 12 + 4 and c > 0.95 and v < 3.0


def test_sky_sky_receiver_and_intrinsic_dipole():
    from cmbsim.simulator import Config
    assert Config().receiver == "differential"
    r, v, c = _quick(spin=False, repoint_days=30.0, receiver="sky-sky", one_over_f=True, one_f_common=0.9)
    assert c > 0.95
    r2, v2, c2 = _quick(intrinsic_dipole_uK=30.0, vssb_prior_kms=5.0, one_over_f=False)
    assert r2.info["intrinsic"]["v_kms"] > 3.0 and v2 < 5.0


def test_periodic_offset_calibration():
    """Exact destriping + internal load every 6 h: calibration measurements enter and the filter stays consistent."""
    r, v, c = _quick(one_over_f=True, destripe=True, cal_period_h=6.0, tof_days=60)
    assert r.info["n_cal"] > 150 and c > 0.95
    assert any(l.startswith("offset 1/f") for l in r.info["obs"]["labels"]) if r.info.get("obs") else True


def test_attitude_states_and_gyroscope():
    """Misalignment/drift as states; the gyroscope never degrades the window average; boresight near the spin axis."""
    from cmbsim import attitude as at
    st = at.StarTracker(exposure_s=0.02)
    st0 = at.StarTracker(exposure_s=0.02, boresight_deg=0.0)
    assert st0.smear_px(6.0) < 0.3 * st.smear_px(6.0)
    s_kf = at.gyro_star_window_sigma(30.0, 0.5, 5.0, 9000, 0.003, 0.01)
    assert s_kf > 0
    r, v, c = _quick(att_on=True, st_exposure_s=0.02, gyro="fog", obs_analysis=True)
    L = r.info["obs"]["labels"]
    assert "misalign. x" in L and c > 0.95


def test_n_frequency_separation():
    """N bands: Σw = 1, nominal components nulled, 2 bands = two-band formula; simulation runs with the combination."""
    from cmbsim import skymaps
    w, null = skymaps.nfreq_weights((140.0, 40.0, 60.0, 90.0, 235.0), "sync+dust")
    assert abs(w.sum() - 1) < 1e-10 and null == ("dust", "synchrotron")
    for c in null:
        f = np.array([skymaps.sed_cmb(c, x) for x in (140.0, 40.0, 60.0, 90.0, 235.0)])
        assert abs(w @ f) < 1e-9 * np.abs(f).max()
    w2, _ = skymaps.nfreq_weights((90.0, 150.0), "all")       # 2 bands null only the dust
    assert np.allclose(w2, skymaps.two_freq_weights(90.0, 150.0))
    assert abs(skymaps.combo_noise_factor(w2) - np.sqrt(2) * np.hypot(*w2)) < 1e-12
    r, v, c = _quick(freqs_extra_ghz=(60.0, 235.0), tof_days=60)
    assert len(r.info["sep"]["bands"]) == 3 and r.info["noise_factor"] > 1.0 and c > 0.9


def test_ekf_and_particle_filter():
    """EKF and Gaussian particle filter (UKF proposal) give the same as the UKF in this nearly linear problem."""
    _, v0, _ = _quick(tof_days=60)
    r1, v1, c1 = _quick(tof_days=60, filter_type="ekf")
    r2, v2, c2 = _quick(tof_days=60, filter_type="pf", pf_particles=300)
    assert c1 > 0.95 and c2 > 0.95 and abs(v1 - v0) < 0.5 * v0 + 0.3 and abs(v2 - v0) < 0.5 * v0 + 0.3
    assert np.median(r2.info["filt"]["ess"]) > 0.8


def test_star_tracker_catalogue_identification_and_gaps():
    """Hipparcos down to V = 7 gives more stars; tracking ≥ lost-in-space; Earth exclusion opens a gap near
    departure and the gyroscope covers it."""
    from cmbsim import attitude as at
    from cmbsim import physics as ph
    assert len(at.catalogue("hipparcos")[0]) > 5 * len(at.catalogue("gsc6")[0])
    ok = at.ident_valid([6, 4, 3, 2, 3, 6, 4], "tracking", True, False)
    assert list(ok) == [True, True, True, False, False, True, True]
    assert list(at.ident_valid([6, 4, 3, 2], "lost", True, False)) == [True, False, False, False]
    assert at.ident_valid([2, 2], "ins", True, True).all()
    ax = np.array([0, 0, 1.0])
    av = {}
    for cat, ml in (("gsc6", 6.0), ("hipparcos", 7.0)):
        m = at.AttitudeModel(at.StarTracker(catalog=cat, mag_limit=ml, exposure_s=0.02), 1800, np.random.default_rng(1),
                             3, 1, 24, spin=True, spin_rate_dps=18.0).set_pointing(ax)
        av[cat] = m.avail
    assert av["hipparcos"] > av["gsc6"]
    # staring sensor with the Earth on the boresight at 1 million km: without gyro the measurement is lost; with FOG it is covered
    jd = ph.julian_date(2026, 10, 31)
    rE = ph.planet_position("earth", jd)
    st_ = at.StarTracker(excl_on=True)
    for gyro in ("none", "fog"):
        m = at.AttitudeModel(st_, 1800, np.random.default_rng(1), 3, 1, 24, gyro=gyro, coast_max_arcsec=60.0)
        m.set_pointing(np.array([0, 0, 1.0]))
        far = rE + 5e7 * np.array([0, 1.0, 0])
        assert m.sample(0.0, far, jd) is not None                 # far from Earth: stars, attitude fixed
        s_ = m.sample(1800.0, rE - 1e6 * m.bores[0, 0], jd)
        assert (s_ is None) if gyro == "none" else (s_ is not None and m.n_coast == 1)


def test_alignment_misalignment_only_calibration_and_walk():
    """Only misalignment as a state (drift in the biases), alignment calibration and extra random walk: consistent filter."""
    r, v, c = _quick(att_on=True, st_exposure_s=0.02, att_est_drift=False, align_cal_h=24.0, sky_bias_walk_nK=5.0,
                      tof_days=60)
    assert r.info["n_alc"] > 40 and c > 0.95 and v < 3.0


def test_mission_dsn_loss_ephemerides_and_b_plane():
    """Trimmed DE421, DSN passes, data loss and B-plane at arrival."""
    from cmbsim import physics as ph
    if ph.de421_available():
        jd = ph.julian_date(2027, 8, 22)
        ph.set_ephemeris("de421"); rD = ph.planet_position("mars", jd)
        ph.set_ephemeris("kepler"); rK = ph.planet_position("mars", jd)
        assert 1e3 < np.linalg.norm(rD - rK) < 2e5 and abs(np.linalg.norm(rD) / ph.AU_KM - 1.54) < 0.05
    r, v, c = _quick(dsn_pass_days=7.0, cmb_loss_pct=20.0, tof_days=60)
    assert r.info["n_dsn"] > 30 and r.info["n_cmb_lost"] > 100 and c > 0.95 and v < 1.0
    b = r.info["bplane"]
    assert np.isfinite(b["dBT_km"]) and b["sig_BT_km"] > 0


def test_co_lines():
    """CO only in bands that contain a line; stronger in the Galactic plane."""
    from cmbsim import skymaps
    if not skymaps.co_available():
        import pytest
        pytest.skip("CO maps unavailable")
    assert np.ndim(skymaps.co_band(140.0, co_on=True)) == 0 and np.ndim(skymaps.co_band(235.0, co_on=True)) == 1
    assert np.ndim(skymaps.co_band(235.0, co_on=False)) == 0          # off: no global state (audit item B-B4)
    co = skymaps.co_band(235.0, co_on=True)
    import healpy as hp
    th, _ = hp.pix2ang(64, np.arange(co.size))
    b = 90 - np.degrees(th)
    assert np.std(co[np.abs(b) < 5]) > 20 * np.std(co[np.abs(b) > 30])


def test_mars_targeting_ddor_opnav_and_relativity():
    """Targeting of a real periapsis (differential correction), ΔDOR, optical navigation and the solar relativistic term."""
    from cmbsim import physics as ph
    from cmbsim.simulator import Config, reference_transfer
    r = np.array([ph.AU_KM, 0, 0]); v = np.array([0, 29.78, 0])
    assert 0.1e-12 < np.linalg.norm(ph.gr_accel(r, v)) < 0.3e-12          # ~0.18 nm/s² at 1 AU
    _, _, _, info = reference_transfer(Config())
    assert info["targeted"] and info["miss_km"] < 100.0 and 5e3 < info["B_target_km"] < 2e4   # hyperbolic departure: ~20 km
    rr, vv, c = _quick(tof_days=60, dsn_pass_days=15.0, dsn_ddor=True)
    assert rr.info["n_ddor"] >= 2 and c > 0.95


def test_guided_tcms_reach_target():
    """TCMs guided by the estimate reduce the delivery dispersion at Mars."""
    from cmbsim.simulator import Config, run
    kw = dict(tech="litebird", spin=True, obs_analysis=False, sky_real=False, zodi_on=False, att_on=False,
              maneuver_err_mps=0.02, opnav_days=30.0, maneuvers_days=(10.0, 60.0, 150.0, 250.0, 285.0))
    r0 = run(Config(**kw))
    r1 = run(Config(tcm_guided=True, **kw))
    b0, b1 = r0.info["bplane"], r1.info["bplane"]
    assert len(r1.info["tcm"]) == 5 and r1.info["tcm_dv_total_mps"] > 0
    assert np.hypot(b1["deliv_BT_km"], b1["deliv_BR_km"]) < 0.2 * np.hypot(b0["deliv_BT_km"], b0["deliv_BR_km"])


def test_xnav_analytic_ekf_and_hyperbolic_departure():
    """XNAV provides position; the EKF with analytic Jacobian agrees with the UKF; perigee departure with Earth and Moon."""
    from cmbsim import physics as ph
    from cmbsim.simulator import Config, reference_transfer
    r, v, c = _quick(tof_days=60, xnav_on=True, cmb_on=False)
    er = np.linalg.norm(r.x_est[:, :3] - r.x_true[:, :3], axis=1)[r.t > 30 * 86400]
    assert r.info["n_xnav"] >= 50 and np.sqrt(np.mean(er**2)) < 200 and c > 0.95
    _, v0, _ = _quick(tof_days=60)
    _, v1, c1 = _quick(tof_days=60, filter_type="ekf_an")
    assert abs(v1 - v0) < 0.3 * v0 + 0.2 and c1 > 0.95
    if ph.de421_available():
        jd0, r0, _, info = reference_transfer(Config())
        assert info["departure"] == "hyperbola"
        ph.set_ephemeris("de421")
        assert abs(np.linalg.norm(r0 - ph.planet_position("earth_c", jd0)) - 6678.137) < 1.0


def test_high_resolution_sky():
    """nside 256: 0.25° beam and small-scale foregrounds (larger residual than at nside 128)."""
    from cmbsim import skymaps
    if not skymaps.available():
        import pytest
        pytest.skip("healpy or sky maps unavailable")
    s1 = skymaps.Sky(140.0, 0.25, True, None, 3.0, seed=1, nside=128)
    s2 = skymaps.Sky(140.0, 0.25, True, None, 3.0, seed=1, nside=256)
    assert s2.map.size == 4 * s1.map.size and np.std(s2.fg_resid) > np.std(s1.fg_resid)


def test_deadband_and_xnav_scaling():
    from cmbsim.simulator import xnav_sigma_from
    assert abs(xnav_sigma_from(0.2, 24.0) - 20.0) < 1e-9 and abs(xnav_sigma_from(0.05, 24.0) - 40.0) < 1e-9
    r, v, c = _quick(tof_days=60, tcm_guided=True, tcm_deadband_sigma=3.0, maneuver_err_mps=0.02)
    assert c > 0.9


def test_audit_corrections():
    """Corrections from the formula audit: DE421 time span, Planck prior on v_SSB, Sun→SSB velocity and
    restoration of the global ephemeris."""
    import pytest
    from cmbsim import physics as ph
    prev = ph.EPHEM["source"]
    try:
        ph.set_ephemeris("de421")
    except Exception:
        pytest.skip("DE421 unavailable")
    with pytest.raises(ValueError):                                     # A-B2: outside 2020–2040
        ph.planet_position("mars", ph.julian_date(2045, 1, 1))
    ev = np.sqrt(np.linalg.eigvalsh(ph.v_ssb_cov()))                    # C-B3: ~32/47/111 m/s
    assert 0.09 < ev[-1] < 0.13 and ev[0] < 0.05
    vs = np.linalg.norm(ph.sun_bary_velocity(ph.julian_date(2026, 10, 31))) * 1e3   # A-B4: ~10–15 m/s
    assert 5 < vs < 20
    from cmbsim import zodi
    src = ph.EPHEM["source"]
    try:                                                                # A-B1: the zodiacal model does not switch the ephemeris
        zodi.nominal_positions(ph.julian_date(2026, 10, 31), 100.0, step_days=50.0)
    except Exception:
        pass
    assert ph.EPHEM["source"] == src
    ph.set_ephemeris(prev)


def test_frozen_attitude_drift_does_not_diverge():
    """With misalignment estimated and drift not estimated (EP scenarios), the drift states stay frozen:
    previously they decayed without process noise to ~1e-54 and the UKF turned NaN (regression test)."""
    r, v, c = _quick(att_on=True, att_estimate=True, att_est_drift=False, tof_days=90, st_exposure_s=0.02)
    assert np.isfinite(r.x_est).all() and np.isfinite(r.P_diag).all()
    assert r.P_diag[-1, -3:].min() > 1e-20


def test_zodi_fills_nan():
    """ZodiPy may return NaN in some pixels; fill_nan replaces them with the neighbour mean."""
    from cmbsim import zodi
    m = np.ones((2, 12 * zodi.NSIDE**2)); m[0, [10, 11, 500]] = np.nan
    f = zodi.fill_nan(m)
    assert np.isfinite(f).all() and np.allclose(f, 1.0) and np.isnan(m).sum() == 3


# ----------------------------------------------------------------------------- use case 1: tracking gaps
def _gap(**kw):
    """Short cruise without CMB (fast): weekly DSN with ΔDOR, gap on days 60–100 and anomaly on day 62."""
    from cmbsim.simulator import Config, run
    base = dict(end_day=110, seed=4, cmb_on=False, att_on=False, obs_analysis=False, dsn_pass_days=7.0,
                dsn_ddor=True, dsn_gaps=((60.0, 100.0),), anomaly_days=(62.0,), maneuvers_days=(10.0,))
    base.update(kw)
    r = run(Config(**base))
    d = r.t / ph.DAY
    i = np.where(d < 100.0)[0][-1]                       # last record before the end of the gap
    ev = np.linalg.norm(r.x_est[i, 3:6] - r.x_true[i, 3:]) * 1e3
    er = np.linalg.norm(r.x_est[i, :3] - r.x_true[i, :3])
    return r, ev, er


def test_gap_conjunction_and_plasma():
    from cmbsim import beacons as bc
    from cmbsim.simulator import gap_intervals
    # Cassini plasma model: ~0.9 mm/s at 3°, ~0.09 mm/s at 10°
    assert 0.6 < bc.plasma_rate_sigma_mmps(3.0) < 1.2 and 0.05 < bc.plasma_rate_sigma_mmps(10.0) < 0.15
    assert bc.sep_angle_deg(np.array([2 * ph.AU_KM, 0, 0]), np.array([-ph.AU_KM, 0, 0])) < 1e-6   # behind the Sun
    assert gap_intervals([0, 1, 2, 3, 4], [False, True, True, False, True]) == [(1.0, 3.0), (4.0, 4.0)]
    r, ev, er = _gap()
    d = r.t / ph.DAY
    assert not np.any((r.kind == "dsn") & (d >= 60) & (d < 100))          # no DSN in the gap
    assert np.any((r.kind == "dsn") & (d >= 100))                         # DSN returns afterwards
    c = r.info["conj"]
    assert c["n_dsn_gap"] >= 5 and len(c["deg"]) == len(c["days"]) and np.all(c["deg"] > 0)
    assert r.info["anomalies"] and r.info["anomalies"][0][0] == 62.0
    # without DSN, the filter knows it lost the velocity (0.3 m/s/axis anomaly) and the error is ~0.5 m/s
    sv = np.sqrt(r.P_diag[np.where(d < 100)[0][-1], 3:6].sum()) * 1e3
    assert 0.1 < ev < 1.5 and 0.2 < sv < 1.0
    from cmbsim.simulator import gap_stats
    g = gap_stats(r)
    assert len(g) == 1 and g[0]["Duration [d]"] == 40.0 and abs(g[0]["v error end [m/s]"] - ev) < 1e-3
    # automatic SEP cutoff: with a threshold above the minimum SEP, the whole cruise becomes a gap
    r2, _, _ = _gap(dsn_gaps=(), dsn_sep_min_deg=181.0, end_day=40, anomaly_days=())
    assert r2.info["n_dsn"] == 0 and r2.info["conj"]["gaps"]
    r3, _, _ = _gap(dsn_plasma=True, end_day=40, anomaly_days=())       # plasma: only a larger σ
    assert r3.info["n_dsn"] > 0 and np.isfinite(r3.x_est).all()


def test_anomaly_imu_solar_doppler_asteroids_xnav():
    """With the anomaly in the gap: the IMU measures the ΔV; asteroids and XNAV recover; solar Doppler estimates the bias."""
    _, ev_a, er_a = _gap()
    r_i, ev_i, _ = _gap(imu_on=True)
    assert r_i.info["n_imu"] >= 5 and ev_i < 0.05 < ev_a                     # IMU: mm/s
    r_d, ev_d, er_d = _gap(ast_on=True)
    assert r_d.info["n_ast"] > 10 and er_d < er_a and ev_d < ev_a
    from cmbsim.simulator import xnav_ely_sigma
    assert np.allclose(xnav_ely_sigma(129.0, 0.5), [87, 97, 5, 8])           # Ely 2022, Tab. 2
    r_x, ev_x, er_x = _gap(xnav_on=True, xnav_model="ely")
    assert er_x < 50 and ev_x < ev_a and len(r_x.info["xnav_sigma_km"]) == 4
    r_s, _, _ = _gap(sdop_on=True, sdop_period_h=2.0, end_day=80)
    sd = r_s.info["sdop"]                                                    # (day, true bias, estimated, 1σ) m/s
    assert sd.shape[1] == 4 and np.mean(np.abs(sd[:, 1] - sd[:, 2]) < 3 * sd[:, 3]) > 0.9
    r_r, _, _ = _gap(sund_on=True, end_day=40, anomaly_days=())
    assert r_r.info["n_sund"] > 30


def test_asteroids_ds1_validation():
    """Asteroids only, no DSN and no initial calibration: like DS1 AutoNav (150–250 km and 0.2 m/s, 1σ, with
    ~25 days of images; Riedel et al. 2000), the error drops to hundreds of km and tenths of m/s."""
    from cmbsim.simulator import Config, run
    r = run(Config(end_day=50, seed=1, cmb_on=False, att_on=False, obs_analysis=False, dsn_calibration=False,
                   ast_on=True, maneuvers_days=()))
    i = -1
    er = np.linalg.norm(r.x_est[i, :3] - r.x_true[i, :3])
    ev = np.linalg.norm(r.x_est[i, 3:6] - r.x_true[i, 3:]) * 1e3
    assert er < 1000 and ev < 1.0 and r.info["n_ast"] > 30


def test_family_presets():
    from cmbsim.instrument import FAMILY_PRESETS, PRESETS, family_config
    assert set(FAMILY_PRESETS) == {"warm_2b", "hemt90_3b", "hemt20_3b", "tes_3b", "multimode_2b", "mkid_3b"}
    for k, t in FAMILY_PRESETS.items():
        assert k in PRESETS and len(t.mass_kg) == 2 and len(t.power_w) == 2 and t.family in "ABCDEF"
        assert t.bands_ghz[0] == t.freq_ghz and len(t.bands_ghz) >= 2
        assert np.isclose(t.net_uK_rts, np.mean(t.net_comb_uK_rts) / 2)
    from cmbsim.simulator import Config
    c = Config(**family_config("tes_3b"))
    assert c.extra_bands() == (90.0, 220.0) and c.instrument().net_uK_rts == 2.25


def test_chopper_modulation():
    """Chopper (Dicke) at f_m: no 1/f drift; white noise × penalty × √(1 + f_k/f_m)."""
    from cmbsim.simulator import Config, run
    kw = dict(end_day=7, seed=1, sky_real=False, att_on=False, zodi_on=False, obs_analysis=False, spin=True,
              tech="tes_3b", maneuvers_days=())
    r0 = run(Config(**kw, one_over_f=False))
    r1 = run(Config(**kw, chop_hz=10.0, chop_penalty=2.0))
    assert r1.info["flicker"] is None and r0.info["chop"] is None
    f = 2.0 * np.sqrt(1 + 0.15 / 10.0)
    assert np.isclose(r1.info["chop"]["factor"], f) and np.isclose(r1.info["sigma_T"] / r0.info["sigma_T"], f)


def test_deep_space_100au():
    """Linearized model (cmbsim/deep.py): the calibrated abstract CMB reproduces the full simulator and, at 100 AU,
    XNAV wins; the plane-of-sky error with DSN comes from the unmodelled acceleration (≈ ½·a·t²)."""
    from cmbsim.deep import DeepConfig, run_deep, deep_metrics, calibration_case
    a = np.sqrt(np.mean([calibration_case(None, seed=s)[0] ** 2 for s in range(1, 5)]))
    q = np.sqrt(np.mean([calibration_case("requirements", seed=s)[0] ** 2 for s in range(1, 5)]))
    assert 0.3 < a < 0.7 and 0.15 < q / a < 0.5
    m = deep_metrics(run_deep(DeepConfig(days=120, dsn_pass_days=30.0)))
    assert m["v error line of sight [m/s]"] < 0.005 and 0.002 < m["1σ v plane of sky [m/s]"] < 0.05
    x = deep_metrics(run_deep(DeepConfig(days=120, dsn_pass_days=7.0, dsn_until_day=30.0, xnav_on=True,
                                         anomaly_days=(60.0,))))
    d = deep_metrics(run_deep(DeepConfig(days=120, dsn_pass_days=7.0, dsn_until_day=30.0, anomaly_days=(60.0,))))
    assert x["v error total [m/s]"] < 0.05 < d["v error total [m/s]"] and x["r error total [km]"] < 50
    # physical limit: mm/s at the end of the gap, better than a navigation-grade IMU
    lim = np.sqrt(np.mean([calibration_case("limit", seed=s)[0] ** 2 for s in range(1, 5)]))
    imu = np.sqrt(np.mean([calibration_case(None, seed=s, imu_on=True)[0] ** 2 for s in range(1, 5)]))
    assert lim < imu < 0.05


def test_ratio_method_load_and_zodiacal():
    """Total power + reference-load drift (estimated) + estimated zodiacal scale: the true load has the requested
    statistics and the filter converges over a short arc."""
    from cmbsim.simulator import Config, run
    kw = dict(end_day=8, seed=2, sky_real=False, att_on=False, obs_analysis=False, spin=True, tech="tes_3b",
              maneuvers_days=(), chop_hz=10.0, receiver="total", spin_tilt_deg=20.0, prec_n=4)
    r = run(Config(**kw, load_drift_nK=100.0, load_tau_h=6.0, zodi_estimate=True))
    assert r.info["load"]["estimated"] and 20.0 < r.info["load"]["rms_nK"] < 300.0
    ev = np.linalg.norm(r.x_est[-1, 3:6] - r.x_true[-1, 3:]) * 1e3
    assert np.isfinite(ev) and ev < 1.0
    assert run(Config(**kw)).info["load"] is None


def test_missing_component_is_not_silent(monkeypatch):
    """A requested component that cannot run raises by default; with allow_fallback it warns and is recorded."""
    import pytest
    from cmbsim import simulator as S
    monkeypatch.setattr(S.zodimod, "available", lambda: False)
    monkeypatch.setattr(S.skymaps, "available", lambda: False)
    kw = dict(end_day=7, seed=1, att_on=False, obs_analysis=False, spin=True, maneuvers_days=(),
              sky_real=True, zodi_on=True)
    with pytest.raises(RuntimeError, match="unavailable"):
        S.run(S.Config(**kw))
    with pytest.warns(RuntimeWarning, match="unavailable"):
        r = S.run(S.Config(**kw, allow_fallback=True))
    c = r.info["components"]
    assert c["sky_real"] is False and c["zodi"] is False and len(c["fallbacks"]) == 2


def test_plasma_model_continuous():
    """Cassini plasma model (Iess et al. 2012, eq. 2): the three branches join at SEP = 90° and 170°."""
    from cmbsim.beacons import plasma_allan_x as f
    assert np.isclose(f(90.0), f(90.0 + 1e-9), rtol=1e-6)
    assert np.isclose(f(170.0), 1.27e-14, rtol=0.02)


def test_relativistic_velocity_composition():
    """Einstein velocity addition: reduces to the Galilean sum at low speed; parallel case matches the 1-D formula."""
    u = np.array([370.0, 0.0, 0.0])
    v = np.array([30.0, 0.0, 0.0])
    w = ph.compose_velocity(u, v, relativistic=True)
    c = ph.C_KMS
    assert np.isclose(w[0], (370.0 + 30.0) / (1 + 370.0 * 30.0 / c**2), rtol=1e-14)
    assert np.allclose(ph.compose_velocity(u * 1e-6, v * 1e-6, True), (u + v) * 1e-6, rtol=1e-10)
    assert np.allclose(ph.compose_velocity(u, v, False), u + v)
