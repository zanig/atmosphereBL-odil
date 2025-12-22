import jax
import jax.numpy as jnp
import jax.random as random
import optax
from jax import jit

from .loss import joint_loss
from .model import Turbulence, ABLState
from .train_joint import train_joint_odil
from .data_io import collapse_profile

def run_joint_bodil_sampling(
    les_data,
    forcings,
    weights,
    z_list,
    n_samples=100,
    z0=0.1,
    init_noise=0.0,
    epochs = 10000,
    save_path="joint_bodil_samples.npz"
):
    key = random.PRNGKey(1)
    n_cases = len(les_data)
    
    prior_means = jnp.array([0.03, 1.21, 1.92, 1.0, 1.3])
    prior_stds = jnp.array([0.015, 0.605, 0.96, 0.5, 0.65])
    
    ## idk if this is actually a MAP but close enough
    map_states, map_params, map_u_stars = train_joint_odil(
        les_data,
        forcings,
        weights,
        z_list,
        n_epochs=epochs, 
        lr=3e-5,
        print_every=epochs//20,
        z0=z0,
        init_noise=init_noise
    )
    
    collapsed_data_list = []
    for i in range(n_cases):
        raw = les_data[i]
        n_z_i = len(z_list[i])
        col = {k: collapse_profile(v, n_z_i) for k, v in raw.items()}
        collapsed_data_list.append(col)

    fixed_bcs_list = [jnp.array([u]) for u in map_u_stars]
    

    current_params_unc = map_params.to_array()

    current_states_list = [s.to_array() for s in map_states]
    state_sizes = [len(s) for s in current_states_list]
    current_states_concat = jnp.concatenate(current_states_list)
    params_array_map = map_params.to_array()
    
    loss_current, _ = joint_loss(
                current_states_list,params_array_map,fixed_bcs_list,
                collapsed_data_list, weights, forcings, z_list, z0)
    
    
    params_constrained = map_params
    p_vec = jnp.array([params_constrained.C_mu, params_constrained.C_1, 
                       params_constrained.C_2, params_constrained.sigma_k, 
                       params_constrained.sigma_eps])
    log_prior_current = -0.5 * jnp.sum(((p_vec - prior_means) / prior_stds) ** 2)
    log_post_current = -loss_current + log_prior_current
    
    inner_optimizer = optax.novograd(learning_rate=1e-4)
    opt_state = inner_optimizer.init(current_states_concat)
    def unpack_states(concat_states):
        extracted = []
        curr = 0
        for size in state_sizes:
            extracted.append(concat_states[curr : curr + size])
            curr += size
        return extracted

    @jit
    def inner_step(concat_states, opt_state, fixed_params_arr):
        def loss_wrt_states(c_states):
            list_s = unpack_states(c_states)
            val, _ = joint_loss(
                list_s,
                fixed_params_arr,
                fixed_bcs_list,
                collapsed_data_list,
                weights,
                forcings,
                z_list,
                z0
            )
            return val
        
        loss_val, grads = jax.value_and_grad(loss_wrt_states)(concat_states)
        updates, opt_state_new = inner_optimizer.update(grads, opt_state, concat_states)
        concat_states_new = optax.apply_updates(concat_states, updates)
        return concat_states_new, opt_state_new, loss_val
    
    #actual MCMC loop :)
    samples = []
    log_posteriors = []
    
    
    saved_states = []
    
    accepted = 0
    proposal_std = 0.76621
    
    for i in range(n_samples):
        key, subkey = random.split(key)
        
        # Propose new params random walky
        proposal_params_unc = current_params_unc + proposal_std * random.normal(
            subkey, current_params_unc.shape)
        print(f"Sample {i+1}/{n_samples}")
        temp_states = current_states_concat
        
        # do inner opt
        fixed_params_array = Turbulence.from_array(proposal_params_unc).to_array()
        for _ in range(55555): 
            temp_states, opt_state, _ = inner_step(temp_states, opt_state, fixed_params_array)
        state_proposed_concat = temp_states
        list_states_prop = unpack_states(state_proposed_concat)
        loss_proposed, _ = joint_loss(list_states_prop, proposal_params_unc,
                                      fixed_bcs_list, collapsed_data_list,  weights,
                                      forcings, z_list,z0)
        
        #now calc prior
        p_prop_obj = Turbulence.from_array(proposal_params_unc)
        p_prop_vec = jnp.array([p_prop_obj.C_mu, p_prop_obj.C_1, p_prop_obj.C_2, 
                                p_prop_obj.sigma_k, p_prop_obj.sigma_eps])
        
        log_prior_proposed = -0.5 * jnp.sum(((p_prop_vec - prior_means) / prior_stds) ** 2)
        log_post_proposed = -loss_proposed + log_prior_proposed

        # accept/ reject step
        log_alpha = log_post_proposed - log_post_current
        alpha = float(jnp.minimum(1.0, jnp.exp(log_alpha)))

        key, subkey = random.split(key)
        if random.uniform(subkey) < alpha:
            current_params_unc = proposal_params_unc
            log_post_current = log_post_proposed
            current_states_concat = state_proposed_concat
            accepted += 1
            print(f"  ACCEPTED (alpha={alpha:.3f}, loss={loss_proposed:.2e})")
        else:
            print(f"  REJECTED (alpha={alpha:.3f}, loss={loss_proposed:.2e})")
        
        pc = Turbulence.from_array(current_params_unc)
        # [C_mu, C_1, C_2, sig_k, sig_e, u_star_weak, u_star_mod, u_star_tnbl]
        row = [pc.C_mu, pc.C_1, pc.C_2, pc.sigma_k, pc.sigma_eps] + [float(u[0]) for u in fixed_bcs_list]
        samples.append(row)
        log_posteriors.append(float(log_post_current))
        saved_states.append(current_states_concat)


    samples = jnp.array(samples)
    log_posteriors = jnp.array(log_posteriors)
    saved_states = jnp.array(saved_states)
    acc_rate = accepted / n_samples
    print(f"\nJoint Sampling Complete. Acceptance rate: {acc_rate:.2%}")
    
    
    p_names = ["C_mu", "C_1", "C_2", "sigma_k", "sigma_eps"] + [f"u_star_case{i}" for i in range(n_cases)]

    jnp.savez(
        save_path,
        samples=samples,
        states=saved_states,
        log_posteriors=log_posteriors,
        param_names=p_names,
        state_sizes=state_sizes
    )

    return samples, log_posteriors, saved_states