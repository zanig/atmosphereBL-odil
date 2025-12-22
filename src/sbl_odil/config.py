"""Configuration and constants for the SBL ODIL experiments."""

import jax.numpy as jnp

# Physical constants
kappa = 0.4
g = 9.81
# Dyer constants for MOST functions
sigma_theta = 1.0
omega_earth = 7.2921e-5

# Grid defaults
DEFAULT_N_Z = 64
DEFAULT_Z_TOP = 400.0
DEFAULT_Z0 = 0.1

# GABLS SBL temperature profile parameters
theta_0 = 265.0
z_inversion = 100.0
gamma_above = 0.01

# Stability function constants
gamma_1 = 16.0
gamma_2 = 16.0
beta_stability = 5.0

# Default forcing parameters
DEFAULT_LATITUDE = 73.0
DEFAULT_U_GEOSTROPHIC = 8.0

DEFAULT_WEIGHTS = {
    "lambda_pde": 1e0,
    "lambda_bc": 1e0,
    "lambda_data": 4e0,
}


def make_grid(n_z: int, z0: float, z_top: float) -> jnp.ndarray:
    """Create a vertical grid for the SBL column."""
    return jnp.linspace(z0, z_top, n_z)
