import jax
import jax.numpy as jnp
import optax
from jax import jit

from .model import ABLState, Turbulence
from .train import init_state
from .loss import joint_loss, loss_fn
from .data_io import collapse_profile

def train_joint_odil(
    list_of_les_data,
    list_of_forcing,
    weights,
    z,
    n_epochs=100000,
    lr=1e-3,
    print_every=1000,
    z0=0.1,
    init_noise=0.0
):
    n_cases = len(list_of_les_data)
    key = jax.random.PRNGKey(1)
    states_list = []
    bc_params_list = []
    state_sizes = []
    
    collapsed_les_data = []

    for i in range(n_cases):
        key, subkey = jax.random.split(key)
        z_i = z[i]
        n_z_i = len(z_i)
        
        raw_data = list_of_les_data[i]
        collapsed_data = {k: collapse_profile(v, n_z_i) for k, v in raw_data.items()}
        collapsed_les_data.append(collapsed_data)
        
        
        s = init_state(list_of_les_data[i], z_i, n_z_i, subkey, noise=init_noise)
        bc_params_list.append(jnp.array([list_of_les_data[i]["u_star"]]))
        s_arr = s.to_array()
        
        states_list.append(s_arr)
        state_sizes.append(len(s_arr))
        
    params_shared = Turbulence()
    params_array = params_shared.to_array()
    
    
    combined_parts = states_list + [params_array] + bc_params_list
    combined_init = jnp.concatenate(combined_parts)
    
    len_params = len(params_array)
    len_bc = 1
    
    optimizer = optax.novograd(learning_rate=lr)
    opt_state = optimizer.init(combined_init)
    
    @jit
    def step(combined, opt_state):
        
        def unpack_and_compute_loss(c):
            extracted_states = []
            curr_idx = 0
            
            for size in state_sizes:
                extracted_states.append(c[curr_idx : curr_idx + size])
                curr_idx += size
            
            extracted_params = c[curr_idx : curr_idx + len_params]
            curr_idx += len_params
        
            extracted_bcs = []
            for _ in range(n_cases):
                extracted_bcs.append(c[curr_idx : curr_idx + len_bc])
                curr_idx += len_bc
            
            loss, loss_comps  = joint_loss(
                extracted_states,  
                extracted_params,
                extracted_bcs,
                collapsed_les_data,
                weights,
                list_of_forcing,
                z,            
                z0
            )
            return loss, loss_comps
     
        (loss_val, loss_comps), grads = jax.value_and_grad(unpack_and_compute_loss, has_aux=True)(combined)
        updates, opt_state_new = optimizer.update(grads, opt_state, combined)
        combined_new = optax.apply_updates(combined, updates)
        return combined_new, opt_state_new, loss_val, loss_comps

    print(f"{'='*70}\nstart opt\n{'='*70}")
    
    current_combined = combined_init
        
    for epoch in range(n_epochs):
        current_combined, opt_state, loss_val, loss_comps = step(current_combined, opt_state)
        
        if epoch % print_every == 0:
            print(f"Epoch {epoch} | Loss: {loss_val:.2e}  | L_pde: {loss_comps['L_PDE']:.2e}  | L_data: {loss_comps['L_data']:.2e}  | L_bc: {loss_comps['L_BC']:.2e}")
            
    final_states = []
    curr_idx = 0
    for i in range(n_cases):
        size = state_sizes[i]
        s_arr = current_combined[curr_idx : curr_idx + size]
        final_states.append(ABLState.from_array(s_arr, len(z[i]), z[i]))
        curr_idx += size
        
    p_arr = current_combined[curr_idx : curr_idx + len_params]
    final_params = Turbulence.from_array(p_arr)
    curr_idx += len_params
    
    final_u_stars = []
    for _ in range(n_cases):
        bc_arr = current_combined[curr_idx : curr_idx + len_bc]
        final_u_stars.append(float(bc_arr[0]))
        curr_idx += len_bc

    return final_states, final_params, final_u_stars