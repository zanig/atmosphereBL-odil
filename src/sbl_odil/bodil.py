"""B-ODIL sampling utilities."""

import jax
import jax.numpy as jnp
import jax.random as random
import optax

from .data_io import collapse_profile
from .loss import compute_odil_loss, loss_fn
from .model import Turbulence, log_turbulence_jacobian
from .train import init_state, train_odil


def run_bodil_sampling(
    les_data,
    forcing,
    weights,
    z,
    n_z,
    n_samples=100,
    u_star_init=0.3,
    save_path="bodil_samples.npz",
    z0=0.1,
    init_noise = 0.0
):
    
    """Sample the parameter posterior using B-ODIL."""
    key = random.PRNGKey(1)
    print("\n" + "=" * 70)
    print("B-ODIL PARAMETER SAMPLING")
    print("=" * 70)

    prior_means = jnp.array([0.03, 1.21, 1.92, 1.0, 1.3])
    prior_stds = jnp.array([0.015, 0.605, 0.96, 0.5, 0.65])

    samples = []
    log_posteriors = []
    states = []

    print("Finding MAP estimate first...")
    state_map, params_map, u_star_map, _ = train_odil(
        les_data,
        forcing,
        weights,
        z,
        n_z,
        n_epochs=100000,
        lr=1e-4,
        print_every=10000,
        u_star_init=u_star_init,
        z0=z0,
        init_noise=init_noise
    )

    params_map_unconstrained = params_map.to_array()

    current_params_unc = params_map_unconstrained
    current_u_star = u_star_map
    current_state_opt = state_map.to_array()

    z_top = float(z[-1])

    def collapse_first_sample(arr):
        arr_jax = jnp.array(arr)
        if arr_jax.ndim > 1:
            arr_jax = arr_jax[:1]
        return collapse_profile(arr_jax, n_z)

    les_first = {k: collapse_first_sample(v) for k, v in les_data.items()}

    state_current = init_state(les_data, z, n_z, key, init_noise)
    state_current_array = state_current.to_array()
    params_current_array = Turbulence.from_array(current_params_unc).to_array()
    bc_params_current = jnp.array([current_u_star])

    loss_current, _ = compute_odil_loss(
        state_current_array,
        params_current_array,
        bc_params_current,
        les_first,
        forcing,
        weights,
        z,
        z0,
        z_top,
    )
    log_post_current = -loss_current

    params_constrained = Turbulence.from_array(current_params_unc)
    params_vec = jnp.array(
        [
            params_constrained.C_mu,
            params_constrained.C_1,
            params_constrained.C_2,
            params_constrained.sigma_k,
            params_constrained.sigma_eps,
        ]
    )
    log_prior_current = -0.5 * jnp.sum(((params_vec - prior_means) / prior_stds) ** 2)
    log_post_current += log_prior_current + log_turbulence_jacobian(current_params_unc)

    accepted = 0
    proposal_std = .666


    for i in range(n_samples):
        key, subkey = random.split(key)
        proposal_params_unc = current_params_unc + proposal_std * random.normal(
            subkey, current_params_unc.shape
        )

        print(f"\nSample {i + 1}/{n_samples}: Optimizing u*(theta_proposed)...")

        state_init = init_state(les_data, z, n_z, key, init_noise)
        params_fixed = Turbulence.from_array(proposal_params_unc)
        state_array = state_init.to_array()
        params_array_fixed = params_fixed.to_array()
        bc_params = jnp.array([current_u_star])
        optimizer = optax.novograd(learning_rate=1e-4) ## idk why but novograd > adam
        opt_state = optimizer.init(state_array)

        @jax.jit
        def quick_step_state_only(state_arr, opt_state, params_arr_fixed, bc_fixed):
            def loss_state(s): ### only do the state / keep turb params
                return loss_fn(s, params_arr_fixed, bc_fixed, les_first, forcing, weights, z, z0, z_top)

            loss_val, grads = jax.value_and_grad(loss_state)(state_arr)
            updates, opt_state_new = optimizer.update(grads, opt_state, state_arr)
            state_new = optax.apply_updates(state_arr, updates)
            return state_new, opt_state_new, loss_val

        current_state = state_array
        for _ in range(75000):
            current_state, opt_state, loss_val = quick_step_state_only(
                current_state, opt_state, params_array_fixed, bc_params
            )

        state_proposed_array = current_state

        loss_proposed, _ = compute_odil_loss(
            state_proposed_array,
            proposal_params_unc,
            bc_params,
            les_first,
            forcing,
            weights,
            z,
            z0,
            z_top,
        )
        log_post_proposed = -loss_proposed

        params_prop_constrained = Turbulence.from_array(proposal_params_unc)
        params_prop_vec = jnp.array(
            [
                params_prop_constrained.C_mu,
                params_prop_constrained.C_1,
                params_prop_constrained.C_2,
                params_prop_constrained.sigma_k,
                params_prop_constrained.sigma_eps,
            ]
        )
        log_prior_proposed = -0.5 * jnp.sum(((params_prop_vec - prior_means) / prior_stds) ** 2)
        log_post_proposed += log_prior_proposed + log_turbulence_jacobian(proposal_params_unc)

        log_alpha = log_post_proposed - log_post_current
        alpha = float(jnp.minimum(1.0, jnp.exp(log_alpha)))

        key, subkey = random.split(key)
        if random.uniform(subkey) < alpha:
            current_params_unc = proposal_params_unc
            log_post_current = log_post_proposed
            current_state_opt = state_proposed_array
            accepted += 1
            print(f"  ACCEPTED (alpha={alpha:.3f}, loss={loss_proposed:.2e})")
        else:
            print(f"  REJECTED (alpha={alpha:.3f}, loss={loss_proposed:.2e})")

        params_constrained = Turbulence.from_array(current_params_unc)
        samples.append(
            [
                params_constrained.C_mu,
                params_constrained.C_1,
                params_constrained.C_2,
                params_constrained.sigma_k,
                params_constrained.sigma_eps,
                current_u_star,
            ]
        )
        log_posteriors.append(float(log_post_current))
        states.append(current_state_opt)

    samples = jnp.array(samples)
    log_posteriors = jnp.array(log_posteriors)
    states = jnp.array(states)

    acceptance_rate = accepted / n_samples
    print(f"\nAcceptance rate: {acceptance_rate:.2%}")

    jnp.savez(
        save_path,
        samples=samples,
        states=states,
        log_posteriors=log_posteriors,
        param_names=["C_mu", "C_1", "C_2", "sigma_k", "sigma_eps", "u_star"],
    )

    return samples, log_posteriors, states
