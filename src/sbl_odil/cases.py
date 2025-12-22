"""taken from the paper"""

from enum import Enum
import jax.numpy as jnp

from .config import omega_earth


class SBLCase(Enum):
    """Stable boundary layer cases from GABLS."""

    WEAK = "sblw"
    MODERATE = "sblm"
    TNBL = "tnbl"


COOLING_RATES = {
    SBLCase.WEAK: 0.05 / 3600,
    SBLCase.MODERATE: 0.25 / 3600,
    SBLCase.TNBL: 0.0,
}

CASE_U_GEOSTROPHIC = {
    SBLCase.TNBL: 12.0,
}

CASE_THETA_TOP = {
    SBLCase.TNBL: 300.0,
}


def coriolis_parameter(latitude_deg: float, omega: float = omega_earth) -> jnp.ndarray:
    """Compute the Coriolis parameter for a given latitude."""
    return 2.0 * omega * jnp.sin(jnp.radians(latitude_deg))


def get_forcing(
    case: SBLCase,
    f_coriolis: float,
    u_geostrophic: float,
    BL_height_estimate: float = 200.0,
) -> dict:
    """Build the forcing dictionary for a given SBL case."""
    cooling_rate_K_per_s = COOLING_RATES[case]
    w_theta_surface = -cooling_rate_K_per_s * BL_height_estimate / 2.0
    forcing = {
        "u_G": u_geostrophic,
        "v_G": 0.0, #idk if this is correct????
        "f_coriolis": f_coriolis,
        "surface_heat_flux": w_theta_surface,
    }
    theta_top = CASE_THETA_TOP.get(case)
    if theta_top is not None:
        forcing["theta_top"] = theta_top
    return forcing
