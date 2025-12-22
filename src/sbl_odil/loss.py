"""ODIL loss definitions."""

import jax.numpy as jnp
from jax import jit
from jax.nn import softplus
from .config import gamma_above, theta_0, z_inversion
from .model import (
    ABLState,
    Turbulence,
    compute_eps_residuals,
    compute_momentum_residuals,
    compute_surface_bc_residuals,
    compute_temperature_residuals,
    compute_tke_residuals,
    compute_top_bc_residuals,
)


def compute_odil_loss(
    state_array,
    params_array,
    bc_params,
    les_data,
    forcing,
    weights,
    z,
    z0,
    z_top,
):
    """Compute ODIL loss"""
    n_z = len(z)
    state = ABLState.from_array(state_array, n_z, z)
    params = Turbulence.from_array(params_array)

    u_star = bc_params[0]
    w_theta_surface = forcing.get("surface_heat_flux", 0.0)

    res_u, res_v = compute_momentum_residuals(state.u, state.v, state.k, state.eps, z, params, forcing)
    res_theta = compute_temperature_residuals(state.theta, state.k, state.eps, z, params, forcing)
    res_k = compute_tke_residuals(
        state.u, state.v, state.theta, state.k, state.eps, z, params, u_star, w_theta_surface
    )
    res_eps = compute_eps_residuals(
        state.u, state.v, state.theta, state.k, state.eps, z, params, u_star, w_theta_surface
    )

    L_PDE = (
        jnp.mean(res_u**2)
        + jnp.mean(res_v**2)
        + jnp.mean(res_theta**2)
        + jnp.mean(res_k**2)
        + jnp.mean(res_eps**2)
    )

    res_u_bc, res_v_bc, res_k_bc, res_eps_bc, res_heatflux_bc, theta_surf_model = (
        compute_surface_bc_residuals(
            state.u,
            state.v,
            state.k,
            state.eps,
            state.theta,
            z,
            z0,
            u_star,
            params,
            w_theta_surface,
        )
    )

    theta_surf_les = les_data["theta"][0] ## match "surface " temperature
    res_theta_surf_bc = theta_surf_model - theta_surf_les

    theta_top_val = theta_0 + gamma_above * (z_top - z_inversion)
    res_u_top, res_v_top, res_theta_top = compute_top_bc_residuals(
        state.u, state.v, state.theta, forcing["u_G"], forcing["v_G"], theta_top_val
    )

    res_u_bc *= 0 # set 0 because im not sure if this is correct
    res_v_bc *= 0 # set 0 

    L_BC = (
        res_u_bc**2
        + res_v_bc**2
        + res_k_bc**2
        + res_eps_bc**2
        + res_heatflux_bc**2
        + res_theta_surf_bc**2
        + res_u_top**2
        + res_v_top**2
        + res_theta_top**2
    )

    L_data = (
        jnp.mean((state.u - les_data["u"]) ** 2)
        + jnp.mean((state.v - les_data["v"]) ** 2)
        + jnp.mean((state.k - les_data["k"]) ** 2)
    )
    L_pen = 1e-2 * jnp.mean(softplus(state.eps / -1e-10)) 
    L_pen += 1e-6 * jnp.mean(softplus(state.k / -1e-10))
    L_pen = 0
    total_loss = (
        weights["lambda_pde"] * L_PDE
        + weights["lambda_bc"] * L_BC
        + weights["lambda_data"] * L_data
        + L_pen
    )

    return total_loss, {"L_PDE": L_PDE, "L_BC": L_BC, "L_data": L_data, "u_star": u_star}


@jit
def loss_fn(state_array, params_array, bc_params, les_data, forcing, weights, z, z0, z_top):
    """JIT wrapper for ODIL loss."""
    loss, _ = compute_odil_loss(state_array, params_array, bc_params, les_data, forcing, weights, z, z0, z_top)
    return loss
