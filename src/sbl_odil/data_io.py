"""load the data from the LES paper"""

from pathlib import Path
import numpy as np
import jax.numpy as jnp
import scipy.io

from .cases import COOLING_RATES, SBLCase


def collapse_profile(arr, n_z: int):
    """average over samples if shape is (n_sample, data)"""
    if arr is None:
        return arr
    if jnp.ndim(arr) == 0:
        return arr

    arr_jax = jnp.array(arr).flatten()
    if arr_jax.size == n_z:
        return arr_jax
    if arr_jax.size % n_z == 0:
        n_samples = arr_jax.size // n_z
        return arr_jax.reshape(n_samples, n_z).mean(axis=0)
    return arr_jax


def load_sbl_data(
    case: SBLCase,
    data_dir: Path,
    z_top: float,
    u_geostrophic: float,
) -> dict:
    """Load SBL LES from paper"""
    data_dir = Path(data_dir)
    path = data_dir / f"{case.value}_samples_10min.mat"
    print(f"Loading SBL data from: {path}")

    mat = scipy.io.loadmat(path)
    u_raw = mat["ubar"] * u_geostrophic
    v_raw = mat["vbar"] * u_geostrophic
    k_raw = mat["k"] * u_geostrophic**2
    theta_raw = mat["Tbar"]
    wT_raw = mat["wT"] * u_geostrophic
    uw_raw = mat["uw"] * u_geostrophic**2
    vw_raw = mat["vw"] * u_geostrophic**2

    nz = k_raw.shape[1]
    z_les = jnp.linspace(z_top / nz / 2, z_top - z_top / nz / 2, nz)

    axes = tuple(i for i in range(uw_raw.ndim) if i != 1)
    tau_z = np.sqrt(np.mean(uw_raw, axis=axes) ** 2 + np.mean(vw_raw, axis=axes) ** 2)
    u_star_les = float(np.sqrt(tau_z[0]))

    z_les_np = np.array(z_les)
    idx = np.where(tau_z < 0.05 * tau_z[0])[0]
    delta_abl = float(z_les_np[idx[0]] / 0.95) if len(idx) > 0 else z_top * 0.6
    
    ##approx epsilon
    dthetadz = np.gradient(theta_raw, z_les, axis = 1)
    N2 = 9.81 / theta_raw[0] * dthetadz
    N2 = np.maximum(N2, 1e-6)
    dudz = np.gradient(u_raw, z_les, axis= 1)
    dvdz = np.gradient(v_raw, z_les, axis = 1)
    shear_mag = np.sqrt(dudz**2 + dvdz**2)
    T_shear = 1.0 / np.maximum(shear_mag, 1e-6)
    T_buoy = 1.0 / np.sqrt(N2)

    T = 1.0 / np.sqrt(1.0 / T_shear**2 + 1.0 / T_buoy**2)
    eps = np.maximum(k_raw / T, 1e-6)

    print(f"  LES-derived u_star = {u_star_les:.4f} m/s")
    print(f"  Estimated ABL height = {delta_abl:.1f} m")
    print(f"  Surface cooling rate = {COOLING_RATES[case] * 3600:.2f} K/h")

    return {
        "u": jnp.array(u_raw),
        "v": jnp.array(v_raw),
        "k": jnp.array(k_raw),
        "eps": jnp.array(eps),
        "theta": jnp.array(theta_raw),
        "wT": jnp.array(wT_raw),
        "uw": jnp.array(uw_raw),
        "vw": jnp.array(vw_raw),
        "u_star": u_star_les,
        "delta_abl": delta_abl,
        "z_les": z_les,
        "nz_les": nz,
    }
