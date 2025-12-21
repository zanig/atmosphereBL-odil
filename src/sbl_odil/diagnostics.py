"""diagnostics and physics validation"""

import jax.numpy as jnp

from .config import g, sigma_theta, theta_0
from .data_io import collapse_profile
from .model import (
    compute_buoyancy_production,
    compute_eddy_viscosity,
    compute_monin_obukhov_length,
    compute_shear_production,
    compute_tke_source_term,
)


def validate_sbl_physics(state, params, forcing, u_star, z):
    """Physics sanity checks for SBL."""
    diagnostics = {}

    P = compute_shear_production(state.u, state.v, state.k, state.eps, z, params)
    B = compute_buoyancy_production(state.theta, state.k, state.eps, z, params)
    w_theta_surface = forcing["surface_heat_flux"]
    S_k = compute_tke_source_term(z, u_star, w_theta_surface, params)

    TKE_budget = P + B - state.eps + S_k
    diagnostics["TKE_imbalance"] = float(jnp.mean(jnp.abs(TKE_budget)))
    diagnostics["P_mean"] = float(jnp.mean(P))
    diagnostics["B_mean"] = float(jnp.mean(B))
    diagnostics["eps_mean"] = float(jnp.mean(state.eps))

    L = compute_monin_obukhov_length(u_star, w_theta_surface)
    diagnostics["L_obukhov"] = float(L)
    diagnostics["zeta_max"] = float(jnp.max(z / L))

    dtheta_dz = jnp.gradient(state.theta, z)
    du_dz = jnp.gradient(state.u, z)
    dv_dz = jnp.gradient(state.v, z)
    shear_sq = du_dz**2 + dv_dz**2 + 1e-10
    Ri = (g / theta_0) * dtheta_dz / shear_sq

    BL_mask = z < 200.0
    Ri_BL = Ri[BL_mask]

    shear_magnitude = jnp.sqrt(shear_sq)
    valid_mask = (z < 200.0) & (shear_magnitude > 1e-3)
    Ri_valid = Ri[valid_mask]

    diagnostics["Ri_BL_mean"] = float(jnp.mean(Ri_BL))
    diagnostics["Ri_BL_max"] = float(jnp.max(Ri_BL))
    diagnostics["Ri_valid_mean"] = float(jnp.mean(Ri_valid))
    diagnostics["Ri_valid_max"] = float(jnp.max(Ri_valid))

    diagnostics["Ri_full_mean"] = float(jnp.mean(Ri))
    diagnostics["Ri_full_max"] = float(jnp.max(Ri))

    nu_t = compute_eddy_viscosity(state.k, state.eps, params)
    w_theta = -(nu_t / sigma_theta) * dtheta_dz
    diagnostics["w_theta_surface_model"] = float(w_theta[0])
    diagnostics["w_theta_surface_prescribed"] = float(w_theta_surface)

    diagnostics["k_min"] = float(jnp.min(state.k))
    diagnostics["eps_min"] = float(jnp.min(state.eps))

    diagnostics["shear_mean"] = float(jnp.mean(shear_magnitude[BL_mask]))
    diagnostics["shear_max"] = float(jnp.max(shear_magnitude))
    diagnostics["u_deficit"] = float(forcing["u_G"] - state.u[0])

    return diagnostics


def diagnose_velocity_collapse(state, les_data, z):
    """Check whether the velocity profile has collapsed."""
    du_dz = jnp.gradient(state.u, z)
    dv_dz = jnp.gradient(state.v, z)
    dtheta_dz = jnp.gradient(state.theta, z)

    print("=" * 70)
    print("VELOCITY COLLAPSE DIAGNOSIS")
    print("=" * 70)

    print("\nVelocity profiles:")
    print(f"  u_surface = {state.u[0]:.4f} m/s")
    print(f"  u_top     = {state.u[-1]:.4f} m/s")
    print(f"  u_mean    = {jnp.mean(state.u):.4f} m/s")
    print(f"  u_std     = {jnp.std(state.u):.4f} m/s")

    print("\nLES velocity for comparison:")
    les_u = collapse_profile(les_data["u"], len(z))
    print(f"  LES u_surface = {les_u[0]:.4f} m/s")
    print(f"  LES u_top     = {les_u[-1]:.4f} m/s")
    print(f"  LES u_mean    = {jnp.mean(les_u):.4f} m/s")

    print("\nGradients:")
    print(f"  |du/dz| mean = {jnp.mean(jnp.abs(du_dz)):.6e} 1/s")
    print(f"  |du/dz| max  = {jnp.max(jnp.abs(du_dz)):.6e} 1/s")
    print(f"  |dtheta/dz| mean = {jnp.mean(jnp.abs(dtheta_dz)):.6e} K/m")

    shear = jnp.sqrt(du_dz**2 + dv_dz**2)
    print("\nShear magnitude:")
    print(f"  Mean shear = {jnp.mean(shear):.6e} 1/s")
    print(f"  Max shear  = {jnp.max(shear):.6e} 1/s")

    if jnp.max(shear) < 1e-3:
        print("  The flow has nearly zero shear >>> data weight bigger??ß")
