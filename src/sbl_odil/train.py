"""Training utilities for SBL ODIL."""

import jax
import jax.numpy as jnp

from jax import jit
import optax


from .config import kappa
from .data_io import collapse_profile
from .loss import compute_odil_loss, loss_fn
from .model import ABLState, Turbulence


def init_state(les_data: dict, z: jnp.ndarray, n_z: int, key: jax.random.PRNGKey, noise : float = 0.05) -> ABLState:
    """Initialize state from LES data or simple heuristics."""
    key, *subkeys = jax.random.split(key, 6)
    def add_relative_noise(x, key, rel_std): #decays with height
        return x * (1.0 + rel_std * jax.random.normal(key, x.shape) * jnp.exp(-z / z.max()) )
    def add_log_noise(x, key, sigma): #for eps and k
        return x * jnp.exp(sigma  * jax.random.normal(key, x.shape) * jnp.exp(-z / z.max()))
    state = ABLState(n_z, z)

    state.u = add_relative_noise(
                        collapse_profile(les_data["u"], n_z), subkeys[0], noise)
    state.v = add_relative_noise(
                        collapse_profile(les_data["v"], n_z), subkeys[1], noise)
    state.k = add_log_noise(
                        jnp.maximum(collapse_profile(les_data["k"], n_z), 1e-6), subkeys[2], noise)
    state.theta = add_relative_noise(
                        collapse_profile(les_data["theta"], n_z), subkeys[3], noise*0.1)
    state.eps = add_log_noise(
                        jnp.maximum(collapse_profile(les_data["eps"], n_z), 1e-6), subkeys[4], noise)

    return state


def initialize_params() -> Turbulence:
    """Initialize turbulence parameters with defaults."""
    return Turbulence()


@jit
def _loss_only(state_array, params_array, bc_params, les_data, forcing, weights, z, z0, z_top):
    return loss_fn(state_array, params_array, bc_params, les_data, forcing, weights, z, z0, z_top)


def train_odil(
    les_data,
    forcing,
    weights,
    z,
    n_z,
    n_epochs=1000000,
    lr=1e-3,
    print_every=5000,
    u_star_init=0.3,
    z0=0.1,
    init_noise=0.0
):
    """Train the SBL ODIL model for a single case."""
    key = jax.random.PRNGKey(1)
    state = init_state(les_data, z, n_z, key=key, noise=init_noise)
    params = initialize_params()

    state_array = state.to_array()
    params_array = params.to_array()
    bc_params = jnp.array([u_star_init])

    les_data_jax = {k: collapse_profile(v, n_z) for k, v in les_data.items()}

    n_state = len(state_array)
    n_params = len(params_array)

    optimizer = optax.chain(optax.novograd(learning_rate=lr))

    combined = jnp.concatenate([state_array, params_array, bc_params])
    opt_state = optimizer.init(combined)

    z_top = float(z[-1])

    @jit
    def step(combined, opt_state):
        def loss_all(c):
            s = c[:n_state]
            p = c[n_state : n_state + n_params]
            bc = c[n_state + n_params :]
            return _loss_only(s, p, bc, les_data_jax, forcing, weights, z, z0, z_top)

        loss_val, grads = jax.value_and_grad(loss_all)(combined)
        updates, opt_state_new = optimizer.update(grads, opt_state, combined)
        combined_new = optax.apply_updates(combined, updates)
        return combined_new, opt_state_new, loss_val

    history = {"loss": [], "L_PDE": [], "L_BC": [], "L_data": [], "u_star": [], "epoch": []}

    print("Starting SBL ODIL optimization...")
    print("-" * 70)

    current = combined

    for epoch in range(n_epochs):
        current, opt_state, loss_val = step(current, opt_state)

        if epoch % print_every == 0 or epoch == n_epochs - 1:
            s_arr = current[:n_state]
            p_arr = current[n_state : n_state + n_params]
            bc_arr = current[n_state + n_params :]

            _, components = compute_odil_loss(
                s_arr,
                p_arr,
                bc_arr,
                les_data_jax,
                forcing,
                weights,
                z,
                z0,
                z_top,
            )

            history["epoch"].append(epoch)
            history["loss"].append(float(loss_val))
            history["L_PDE"].append(float(components["L_PDE"]))
            history["L_BC"].append(float(components["L_BC"]))
            history["L_data"].append(float(components["L_data"]))
            history["u_star"].append(float(components["u_star"]))

            print(
                f"Epoch {epoch:6d} | Loss: {loss_val:10.2e} | "
                f"PDE: {components['L_PDE']:8.2e} | BC: {components['L_BC']:8.2e} | "
                f"Data: {components['L_data']:8.2e} | u*: {components['u_star']:.4f}"
            )

    print("-" * 70)
    print("Optimization complete!")

    final_state = ABLState.from_array(current[:n_state], n_z, z)
    final_params = Turbulence.from_array(current[n_state : n_state + n_params])
    final_bc_params = current[n_state + n_params :]
    final_u_star = float(final_bc_params[0])

    return final_state, final_params, final_u_star, history
