"""Core model state, turbulence parameters, and physics operators."""

import jax
import jax.numpy as jnp
from jax import jit

from .config import (
    beta_stability,
    gamma_1,
    gamma_2,
    gamma_above,
    g,
    kappa,
    sigma_theta,
    theta_0,
    z_inversion,
)


def init_theta_profile(z: jnp.ndarray) -> jnp.ndarray:
    """GABLS SBL initial potential temperature profile."""
    theta = jnp.where(
        z >= z_inversion,
        theta_0 + gamma_above * (z - z_inversion),
        theta_0 * jnp.ones_like(z),
    )
    return theta


class ABLState:
    """state for SBL (u, v, theta, k, eps)."""

    def __init__(self, n_z: int, z: jnp.ndarray):
        self.u = jnp.zeros(n_z)
        self.v = jnp.zeros(n_z)
        self.theta = init_theta_profile(z)
        self.k = jnp.ones(n_z) * 0.1
        self.eps = jnp.ones(n_z) * 0.01

    def to_array(self) -> jnp.ndarray:
        return jnp.concatenate([self.u, self.v, self.theta, self.k, self.eps])

    @classmethod
    def from_array(cls, arr: jnp.ndarray, n_z: int, z: jnp.ndarray):
        state = cls(n_z, z)
        state.u = arr[:n_z]
        state.v = arr[n_z : 2 * n_z]
        state.theta = arr[2 * n_z : 3 * n_z]
        state.k = arr[3 * n_z : 4 * n_z]
        state.eps = arr[4 * n_z : 5 * n_z]
        return state


class Turbulence:
    """Turbulence model parameters with constraints (squish them into bounds with sigmoid)"""

    BOUNDS = {
        "C_mu": (0.01, 0.12),
        "C_1": (0.25, 3.0),
        "C_2": (0.35, 4.8),
        "sigma_k": (0.55, 1.5),
        "sigma_eps": (0.8, 1.95),
    }

    def __init__(self):
        self.C_mu = 0.0123123
        self.C_1 = 1.
        self.C_2 = 2.
        self.sigma_k = 1.0
        self.sigma_eps = 1.3333212

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + jnp.exp(-x))

    @staticmethod
    def _inv_sigmoid(y):
        y = jnp.clip(y, 1e-6, 1.0 - 1e-6)
        return jnp.log(y / (1.0 - y))

    @staticmethod
    def _constrain(unconstrained, lo, hi):
        return lo + (hi - lo) * Turbulence._sigmoid(unconstrained)

    @staticmethod
    def _unconstrain(constrained, lo, hi):
        normalized = (constrained - lo) / (hi - lo)
        return Turbulence._inv_sigmoid(normalized)

    def to_array(self) -> jnp.ndarray:
        b = self.BOUNDS
        return jnp.array(
            [
                self._unconstrain(self.C_mu, *b["C_mu"]),
                self._unconstrain(self.C_1, *b["C_1"]),
                self._unconstrain(self.C_2, *b["C_2"]),
                self._unconstrain(self.sigma_k, *b["sigma_k"]),
                self._unconstrain(self.sigma_eps, *b["sigma_eps"]),
            ]
        )

    @classmethod
    def from_array(cls, arr: jnp.ndarray):
        params = cls.__new__(cls)
        b = cls.BOUNDS
        params.C_mu = cls._constrain(arr[0], *b["C_mu"])
        params.C_1 = cls._constrain(arr[1], *b["C_1"])
        params.C_2 = cls._constrain(arr[2], *b["C_2"])
        params.sigma_k = cls._constrain(arr[3], *b["sigma_k"])
        params.sigma_eps = cls._constrain(arr[4], *b["sigma_eps"])
        return params


jax.tree_util.register_pytree_node(
    Turbulence,
    lambda t: ((t.C_mu, t.C_1, t.C_2, t.sigma_k, t.sigma_eps), None),
    lambda _, children: Turbulence.from_array(jnp.array(children)),
)


@jit
def compute_eddy_viscosity(k, eps, params):
    nu_t = params.C_mu * k**2 / (eps + 1e-10)
    return nu_t


@jit
def compute_turbulent_fluxes(u, v, theta, k, eps, z, params):
    nu_t = compute_eddy_viscosity(k, eps, params)
    du_dz = jnp.gradient(u, z)
    dv_dz = jnp.gradient(v, z)
    dtheta_dz = jnp.gradient(theta, z)

    uw = -nu_t * du_dz
    vw = -nu_t * dv_dz
    w_theta = -nu_t / sigma_theta * dtheta_dz

    return uw, vw, w_theta, nu_t


@jit
def compute_momentum_residuals(u, v, k, eps, z, params, forcing):
    uw, vw, _, _ = compute_turbulent_fluxes(u, v, jnp.zeros_like(u), k, eps, z, params)

    d_uw_dz = jnp.gradient(uw, z)
    d_vw_dz = jnp.gradient(vw, z)

    f = forcing["f_coriolis"]
    u_g, v_g = forcing["u_G"], forcing["v_G"]

    residual_u = -d_uw_dz + f * (v - v_g)
    residual_v = -d_vw_dz - f * (u - u_g)

    return residual_u, residual_v


@jit
def compute_temperature_residuals(theta, k, eps, z, params, forcing):
    nu_t = compute_eddy_viscosity(k, eps, params)
    dtheta_dz = jnp.gradient(theta, z)
    w_theta = -nu_t / sigma_theta * dtheta_dz
    d_wtheta_dz = jnp.gradient(w_theta, z)

    residual_theta = -d_wtheta_dz
    return residual_theta


@jit
def compute_shear_production(u, v, k, eps, z, params):
    nu_t = compute_eddy_viscosity(k, eps, params)
    du_dz = jnp.gradient(u, z)
    dv_dz = jnp.gradient(v, z)
    S2 = du_dz**2 + dv_dz**2
    P = nu_t * S2
    return P


@jit
def compute_buoyancy_production(theta, k, eps, z, params):
    nu_t = compute_eddy_viscosity(k, eps, params)
    dtheta_dz = jnp.gradient(theta, z)
    B = -g / theta_0 * (nu_t / sigma_theta) * dtheta_dz
    return B


@jit
def compute_stability_functions(zeta):
    """MOST stability functions with Dyer constants (clipped zeta)."""
    zeta_clipped = jnp.clip(zeta, -2.0, 1.0)
    zeta_unstable = jnp.minimum(zeta_clipped, -1e-6)
    zeta_stable = jnp.maximum(zeta_clipped, 1e-6)

    phi_m_unstable = (1.0 - gamma_1 * zeta_unstable) ** (-0.25)
    phi_h_unstable = sigma_theta * (1.0 - gamma_2 * zeta_unstable) ** (-0.5)

    phi_m_stable = 1.0 + beta_stability * zeta_stable
    phi_h_stable = sigma_theta + beta_stability * zeta_stable

    phi_m = jnp.where(zeta_clipped < 0, phi_m_unstable, phi_m_stable)
    phi_h = jnp.where(zeta_clipped < 0, phi_h_unstable, phi_h_stable)

    return phi_m, phi_h


@jit
def compute_phi_eps(zeta, phi_m):
    """Phi_epsilon per van der Laan 2017 (clipped zeta)."""
    zeta_clipped = jnp.clip(zeta, -2.0, 1.0)
    zeta_unstable = jnp.minimum(zeta_clipped, -1e-6)
    zeta_stable = jnp.maximum(zeta_clipped, 1e-6)
    phi_eps_unstable = 1.0 - zeta_unstable
    phi_eps_stable = phi_m - zeta_stable
    return jnp.where(zeta_clipped < 0, phi_eps_unstable, phi_eps_stable)


@jit
def compute_tke_source_term(z, u_star, w_theta_surface, params):
    L = compute_monin_obukhov_length(u_star, w_theta_surface)
    L_safe = jnp.maximum(jnp.abs(L), 100.0)

    zeta = jnp.clip(z / L_safe, -2.0, 1.0)

    phi_m, phi_h = compute_stability_functions(zeta)
    phi_eps = compute_phi_eps(zeta, phi_m)
    phi_m_safe = jnp.clip(phi_m, 0.1, 100.0)

    C_kD = kappa**2 / (params.sigma_k * jnp.sqrt(params.C_mu + 1e-10) + 1e-10)

    prefactor = u_star**3 / (kappa * L_safe + 1e-10)
    sigma_theta_local = sigma_theta

    f_st = (2.0 - zeta) - 2.0 * beta_stability * zeta * (
        1.0 - 2 * zeta + 2 * beta_stability * zeta
    )

    S_k = prefactor * (
        1.0
        - phi_h / (sigma_theta_local * phi_m_safe + 1e-10)
        - (C_kD / 4.0) * phi_m_safe ** (-3.5) * phi_eps ** (-1.5) * f_st
    )

    decay = jnp.exp(-z / (0.1 * L_safe + 10.0))
    S_k = S_k * decay
    S_k = jnp.clip(S_k, -10.0, 10.0)

    return S_k


@jit
def compute_tke_residuals(u, v, theta, k, eps, z, params, u_star, w_theta_surface):
    P_shear = compute_shear_production(u, v, k, eps, z, params)
    B_buoyancy = compute_buoyancy_production(theta, k, eps, z, params)

    nu_t = compute_eddy_viscosity(k, eps, params)
    dk_dz = jnp.gradient(k, z)
    transport = nu_t / params.sigma_k * dk_dz
    d_transport_dz = jnp.gradient(transport, z)

    S_k = compute_tke_source_term(z, u_star, w_theta_surface, params)

    residual_k = P_shear + B_buoyancy - eps + d_transport_dz + S_k
    return residual_k


@jit
def compute_monin_obukhov_length(u_star, w_theta_surface):
    denominator = kappa * g * w_theta_surface
    safe_denom = jnp.where(
        jnp.abs(denominator) < 1e-10,
        jnp.sign(denominator + 1e-20) * 1e-10,
        denominator,
    )
    L_raw = -(u_star**3 * theta_0) / safe_denom
    L = jnp.clip(L_raw, -1e10, 1e10)
    return L


@jit
def compute_c3(zeta, params):
    """Stability function C3 for the epsilon equation."""
    zeta_clipped = jnp.clip(zeta, -2.0, 1.0)
    phi_m, phi_h = compute_stability_functions(zeta_clipped)
    phi_eps = compute_phi_eps(zeta_clipped, phi_m)

    f_eps_unstable = phi_m ** (-0.5) * (1.0 - 0.75 * gamma_1 * zeta_clipped)
    f_eps_stable = phi_m ** (-1.5) * (2.0 * phi_m - 1.0)
    f_eps = jnp.where(zeta_clipped < 0, f_eps_unstable, f_eps_stable)

    zeta_safe = jnp.where(
        jnp.abs(zeta_clipped) < 1e-6,
        jnp.where(zeta_clipped >= 0, 1e-6, -1e-6),
        zeta_clipped,
    )

    term = (
        params.C_1 * phi_m
        - params.C_2 * phi_eps
        + (params.C_2 - params.C_1) * phi_eps ** (-0.5) * f_eps
    )
    C3 = (sigma_theta * phi_m) / (zeta_safe * phi_h) * term
    return C3


def compute_eps_source_term(u, v, theta, k, eps, z, params, u_star, w_theta_surface):
    P_shear = compute_shear_production(u, v, k, eps, z, params)
    B = compute_buoyancy_production(theta, k, eps, z, params)

    L_mo = compute_monin_obukhov_length(u_star, w_theta_surface)
    L_mo_safe = jnp.where(
        jnp.abs(L_mo) < 1e-6,
        jnp.where(L_mo >= 0, 1e-6, -1e-6),
        L_mo,
    )
    zeta = z / L_mo_safe
    C_3 = compute_c3(zeta, params)
    # This avoids explicit branching while preserving stable/unstable behavior.

    production_term = params.C_1 * (eps / (k + 1e-10)) * P_shear
    buoyancy_term = C_3 * params.C_1 * (eps / (k + 1e-10)) * B
    destruction_term = -params.C_2 * (eps**2 / (k + 1e-10))

    return production_term + buoyancy_term + destruction_term


@jit
def compute_eps_residuals(u, v, theta, k, eps, z, params, u_star, w_theta_surface):
    S_eps = compute_eps_source_term(u, v, theta, k, eps, z, params, u_star, w_theta_surface)

    nu_t = compute_eddy_viscosity(k, eps, params)
    deps_dz = jnp.gradient(eps, z)
    transport = nu_t / params.sigma_eps * deps_dz
    d_transport_dz = jnp.gradient(transport, z)

    residual_eps = d_transport_dz + S_eps
    return residual_eps


@jit
def compute_surface_bc_residuals(u, v, k, eps, theta, z, z0, u_star, params, w_theta_surface):
    """Surface BCs (just heat and temp for now? )"""
    u_surf = u[0]
    v_surf = v[0]
    k_surf = k[0]
    eps_surf = eps[0]
    z_surf = z[0]

    u_ex = u_star / kappa * jnp.log(z_surf / z0)
    k_ex = u_star**2 / jnp.sqrt(params.C_mu)
    eps_ex = u_star**3 / (z_surf * kappa)

    residual_u_bc = u_surf - u_ex
    residual_v_bc = v_surf - 0.0
    residual_k_bc = k_surf - k_ex
    residual_eps_bc = eps_surf - eps_ex

    nu_t_surf = params.C_mu * k_surf**2 / (eps_surf + 1e-10)
    dtheta_dz_surf = (theta[1] - theta[0]) / (z[1] - z[0])
    w_theta_model = -nu_t_surf / sigma_theta * dtheta_dz_surf
    residual_heatflux_bc = w_theta_model - w_theta_surface

    theta_surf = theta[0]

    return 0, 0, 0, 0, residual_heatflux_bc, theta_surf


@jit
def compute_top_bc_residuals(u, v, theta, u_G, v_G, theta_top):
    """top BCs"""
    residual_u_top = u[-1] - u_G
    residual_v_top = v[-1] - v_G
    residual_theta_top = theta[-1] - theta_top
    return residual_u_top, residual_v_top, residual_theta_top
