import functools
import multiprocessing

import numpy as np

from kantab.splits import folds, scale_inputs

PV_TASKS = ("pv_mp5", "pv_chain6")
MAP_TASKS = ("tyre4", "msis7", "flame5")
RULE_TASKS = ("gasz7", "ign5", "igrf4", "msis_d3", "msis_d4", "msis_d5", "msis_d6")
TASKS = PV_TASKS + MAP_TASKS + RULE_TASKS
SEED = {**{t: 41_000 + i for i, t in enumerate(PV_TASKS)}, **{t: 42_000 + i for i, t in enumerate(MAP_TASKS)},
        **{t: 43_000 + i for i, t in enumerate(RULE_TASKS)}}
N, K, FOLD = 60_000, 5, 0
MSIS_ORDER = (0, 1, 2, 3, 5, 4, 6)
MSIS_FIXED = np.array([500.0, 0.0, 12.0, 183.5, 157.5, 157.5, 40.0])
BOX = {
    "pv_mp5": [(100, 1200), (-10, 70), (0, 85), (1, 10), (0.5, 5)],
    "pv_chain6": [(0, 1000), (20, 500), (0, 85), (-180, 180), (-10, 45), (0, 12)],
    "tyre4": [(-0.3, 0.3), (-0.3, 0.3), (0.0, 0.1), (0.2, 1.1)],
    "msis7": [(200, 800), (-90, 90), (0, 24), (1, 366), (65, 250), (65, 250), (0, 80)],
    "flame5": [(300, 800), (1, 30), (0.5, 1.5), (0.0, 1.0), (0.0, 0.3)],
    "gasz7": [(280, 350), (5, 120), (0.0, 0.10), (0.0, 0.04), (0.0, 0.15), (0.0, 0.10), (0.0, 0.20)],
    "ign5": [(900, 1400), (10, 60), (0.5, 1.5), (0.0, 1.0), (0.0, 0.3)],
    "igrf4": [(-90, 90), (-180, 180), (300, 800), (2020, 2030)],
}
BOX.update({f"msis_d{d}": [BOX["msis7"][c] for c in MSIS_ORDER[:d]] for d in (3, 4, 5, 6)})
FZ = 4000.0
WORKERS = 10


def _pv_power(g_eff, t_cell):
    import pvlib
    m = pvlib.pvsystem.retrieve_sam("CECMod")["Canadian_Solar_Inc__CS5P_220M"]
    il, i0, rs, rsh, nnsvth = pvlib.pvsystem.calcparams_cec(
        np.maximum(g_eff, 0.0), t_cell, m["alpha_sc"], m["a_ref"], m["I_L_ref"], m["I_o_ref"], m["R_sh_ref"], m["R_s"], m["Adjust"])
    return np.asarray(pvlib.pvsystem.singlediode(il, i0, rs, rsh, nnsvth)["p_mp"], dtype=np.float64)


def _pv(task, x):
    import pvlib
    if task == "pv_mp5":
        g, t, aoi, am, pw = x.T
        eff = g * pvlib.iam.physical(aoi) * pvlib.spectrum.spectral_factor_firstsolar(pw, am, module_type="multisi")
        return _pv_power(eff, t)
    dni, dhi, zen, rel_az, t_air, wind = x.T
    tilt, az = 30.0, 180.0
    ghi = dni * np.cos(np.radians(zen)) + dhi
    sun_az = az + rel_az
    poa = pvlib.irradiance.get_total_irradiance(tilt, az, zen, sun_az, dni, ghi, dhi, dni_extra=1361.0,
                                                model="haydavies", albedo=0.2)
    iam = pvlib.iam.physical(pvlib.irradiance.aoi(tilt, az, zen, sun_az))
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    t_cell = pvlib.temperature.sapm_cell(poa["poa_global"], t_air, wind, **params)
    am = pvlib.atmosphere.get_absolute_airmass(pvlib.atmosphere.get_relative_airmass(zen, model="kastenyoung1989"))
    spec = pvlib.spectrum.spectral_factor_firstsolar(np.full_like(am, 1.5), am, module_type="multisi")
    eff = (np.nan_to_num(poa["poa_direct"]) * iam + np.nan_to_num(poa["poa_diffuse"])) * spec
    return _pv_power(eff, t_cell)


def _tyre(x):
    # F_y is proportional to F_z in this model, so F_z is divided out; camber stays one-sided because the
    # implementation takes sign(γ), which jumps at 0
    import copy
    from vehiclemodels.parameters_vehicle2 import parameters_vehicle2
    from vehiclemodels.utils.tire_model import formula_lateral, formula_lateral_comb
    base = parameters_vehicle2().tire
    out = np.empty(len(x))
    for i, (kappa, alpha, gamma, lam) in enumerate(x):
        p = copy.copy(base)
        p.p_dx1, p.p_dy1 = base.p_dx1 * lam, base.p_dy1 * lam
        f0y, mu_y = formula_lateral(alpha, gamma, FZ, p)
        out[i] = formula_lateral_comb(kappa, alpha, gamma, mu_y, FZ, f0y, p) / FZ
    return out


def _msis(x):
    # pymsis.calculate would pass the whole day of year and the rest as UT; the kernel takes a continuous day
    import pymsis
    from pymsis import msis, msis21f
    alt, lat, lst, doy, f107, f107a, ap = x.T
    pymsis.calculate(np.datetime64("2021-01-01T12:00"), 0.0, 0.0, 400.0, 150.0, 150.0, [[15.0] * 7])
    f = lambda v: np.ascontiguousarray(v, dtype=np.float32)
    with msis._lock:
        out = msis21f.pymsiscalc(f(doy), f(np.full(len(x), 43_200.0)), f(np.mod(15 * (lst - 12), 360)), f(lat), f(alt),
                                 f(f107), f(f107a), f(np.repeat(ap[:, None], 7, 1)))
    return np.log10(np.asarray(out)[:, pymsis.Variable.MASS_DENSITY].astype(np.float64))


def _msis_slice(d, x):
    full = np.tile(MSIS_FIXED, (len(x), 1))
    full[:, list(MSIS_ORDER[:d])] = x
    return _msis(full)


def _flame(x):
    import cantera as ct
    gas = ct.Solution("gri30.yaml")
    i_co2 = gas.species_index("CO2")
    out = np.empty(len(x))
    for i, (t0, p, phi, h2, co2) in enumerate(x):
        gas.set_equivalence_ratio(phi, {"CH4": 1 - h2, "H2": h2}, {"O2": 1.0, "N2": 3.76})
        X = gas.X * (1 - co2)
        X[i_co2] += co2
        gas.TPX = t0, p * 1e5, X
        gas.equilibrate("HP")
        out[i] = gas.T
    return out


def _gasz(x):
    import CoolProp.CoolProp as CP
    st = CP.AbstractState("HEOS", "Methane&Ethane&Propane&Nitrogen&CarbonDioxide&Hydrogen")
    out = np.empty(len(x))
    for i, (t, p, c2, c3, n2, co2, h2) in enumerate(x):
        st.set_mole_fractions([1 - c2 - c3 - n2 - co2 - h2, c2, c3, n2, co2, h2])
        st.specify_phase(CP.iphase_gas)
        st.update(CP.PT_INPUTS, p * 1e5, t)
        out[i] = st.compressibility_factor()
    return out


def _ignition(x):
    import cantera as ct
    gas = ct.Solution("gri30.yaml")
    i_co2 = gas.species_index("CO2")
    out = np.empty(len(x))
    for i, (t0, p, phi, h2, co2) in enumerate(x):
        gas.set_equivalence_ratio(phi, {"CH4": 1 - h2, "H2": h2}, {"O2": 1.0, "N2": 3.76})
        X = gas.X * (1 - co2)
        X[i_co2] += co2
        gas.TPX = t0, p * 1e5, X
        r = ct.IdealGasConstPressureReactor(gas)
        net = ct.ReactorNet([r])
        while net.time < 10.0 and r.T < t0 + 400:
            net.step()
        out[i] = np.log10(net.time) if r.T >= t0 + 400 else np.nan
    return out


def _igrf(x):
    # the Gauss coefficients are linear in time within an epoch, so interpolating the components is exact
    import datetime
    import ppigrf
    lat, lon, alt, year = x.T
    B = [np.stack(ppigrf.igrf(lon, lat, alt, datetime.datetime(e, 1, 1)))[:, 0] for e in (2020, 2025, 2030)]
    k = np.clip(((year - 2020) // 5).astype(int), 0, 1)
    w = (year - 2020 - 5 * k) / 5
    lo, hi = np.where(k == 0, B[0], B[1]), np.where(k == 0, B[1], B[2])
    return np.sqrt(((lo + w * (hi - lo)) ** 2).sum(0))


def _pooled(fn, x):
    if multiprocessing.current_process().daemon or len(x) < 2000:   # pool workers cannot start pools
        return fn(x)
    with multiprocessing.Pool(WORKERS) as p:
        return np.concatenate(p.map(fn, np.array_split(x, 4 * WORKERS)))


def model(task, x):
    x = np.asarray(x, dtype=np.float64)
    if task in PV_TASKS:
        return _pv(task, x)
    if task.startswith("msis_d"):
        return _msis_slice(int(task[-1]), x)
    return {"tyre4": _tyre, "msis7": _msis, "flame5": _flame, "igrf4": _igrf,
            "gasz7": lambda v: _pooled(_gasz, v), "ign5": lambda v: _pooled(_ignition, v)}[task](x)


@functools.lru_cache(maxsize=None)
def load(task, n=N):
    rng = np.random.default_rng(SEED[task])
    x = np.stack([rng.uniform(lo, hi, n) for lo, hi in BOX[task]], axis=1)
    return x, model(task, x), rng.permutation(n)


def _parts(task, fold=FOLD):
    x, y, g = load(task)
    fs = folds(g, K)
    test, val = np.isin(g, fs[fold]), np.isin(g, fs[(fold + 1) % K])
    return x, y, ~(test | val), val, test


def split(task, fold=FOLD):
    x, y, train, val, test = _parts(task, fold)
    xtr, xva, xte = scale_inputs(x[train], x[val], x[test])
    mean, std = y[train].mean(), y[train].std()
    z = lambda v: ((v - mean) / std).astype(np.float32)
    return (xtr, z(y[train])), (xva, z(y[val])), (xte, z(y[test])), (float(mean), float(std))


def nodes(task, ns, fold=FOLD):
    x, _, train, _, _ = _parts(task, fold)
    lo, hi = x[train].min(0), x[train].max(0)
    axes = [lo[a] + (hi[a] - lo[a]) * np.linspace(0, 1, n) for a, n in enumerate(ns)]
    return np.stack([m.ravel() for m in np.meshgrid(*axes, indexing="ij")], axis=1)


def node_values(task, ns, fold=FOLD):
    x, y, train, _, _ = _parts(task, fold)
    ns = (ns,) * x.shape[1] if isinstance(ns, int) else tuple(ns)
    return (model(task, nodes(task, ns, fold)) - y[train].mean()) / y[train].std()
