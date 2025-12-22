
import argparse
from pathlib import Path
import sys
import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sbl_odil.cases import CASE_U_GEOSTROPHIC, SBLCase, coriolis_parameter, get_forcing
from sbl_odil.config import (
    DEFAULT_LATITUDE,
    DEFAULT_WEIGHTS,
    DEFAULT_Z0,
    make_grid,
)
from sbl_odil.data_io import load_sbl_data
from sbl_odil.train_joint import train_joint_odil 
from sbl_odil.plotting import plot_loss_history, plot_profiles_comparison

def main() -> None:
    parser = argparse.ArgumentParser(description="Joint cases")
    parser.add_argument(
        "--epochs", type=int, default=100000, help="Number of optimization epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the .mat LES files",
    )
    parser.add_argument("--cases", type=str, default="nmw", help="which cases to include. e.g. --cases nmw for all three or --cases mn for moderate and neutral")
    parser.add_argument("--init_noise", type=float, default=0.0, help="Noise added to the initial condition")
    args = parser.parse_args()


  

 
    all_configs = {
        'w': {"case_enum": SBLCase.WEAK, "name": "Weak", "z_top": 400.0, "nz": 64},
        'm': {"case_enum": SBLCase.MODERATE, "name": "Moderate", "z_top": 400.0, "nz": 64},
        'n': {"case_enum": SBLCase.TNBL, "name": "TNBL", "z_top": 5000.0, "nz": 360},
        }
    case_configs = []
    if 'w' in args.cases:
        case_configs.append(all_configs['w'])
    if 'm' in args.cases:
        case_configs.append(all_configs['m'])
    if 'n' in args.cases:
        case_configs.append(all_configs['n'])
    if not case_configs: 
        print("wrong case IDs")
        return
        
        
    f_coriolis = coriolis_parameter(DEFAULT_LATITUDE)


    data_list = []
    forcing_list = []
    z_list = []
    case_names = []

    for config in case_configs:
        case = config["case_enum"]
        name = config["name"]
        z_top = config["z_top"]
        nz = config["nz"]
        
        print(f"  -> Loaded {name:<10} (height {z_top}m, nz={nz})")


        z_grid = make_grid(nz, DEFAULT_Z0, z_top)
        z_list.append(z_grid)
        case_names.append(name)

 
        u_geostrophic = CASE_U_GEOSTROPHIC.get(case, 8.0)


        les_data = load_sbl_data(case, args.data_dir, z_top, u_geostrophic)
        data_list.append(les_data)


        forcing = get_forcing(case, f_coriolis, u_geostrophic)
        forcing_list.append(forcing)


    final_states, final_params, final_u_stars = train_joint_odil(
        list_of_les_data=data_list,
        list_of_forcing=forcing_list,
        weights=DEFAULT_WEIGHTS,
        z=z_list,      
        n_epochs=args.epochs,
        lr=args.lr,
        print_every=max(args.epochs // 20, 1000),
        z0=DEFAULT_Z0,
        init_noise=args.init_noise
    )

    print("=" * 70)

    print(f"\nTurbulence Parameters :")
    print(f"  C_mu       = {final_params.C_mu:.4f}")
    print(f"  C_1        = {final_params.C_1:.4f}")
    print(f"  C_2        = {final_params.C_2:.4f}")
    print(f"  sigma_k    = {final_params.sigma_k:.4f}")
    print(f"  sigma_eps  = {final_params.sigma_eps:.4f}")

    print(f"\nCase-Specific Results (example for u*):")
    print(f"  {'Case':<10} | {'Learned u*':<12} | {'LES u*':<12} | {'Error (pct) ':<10}")
    print("-" * 22)
    
    for i, name in enumerate(case_names):
        u_star_learned = final_u_stars[i]
        u_star_les = data_list[i]["u_star"]
        err = abs(u_star_learned - u_star_les) / u_star_les * 100
        print(f"  {name:<10} | {u_star_learned:<12.4f} | {u_star_les:<12.4f} | {err:<10.2f}")


    save_path = "joint_results.npz"

    jnp.savez(
        save_path,
        C_mu=final_params.C_mu,
        C_1=final_params.C_1,
        C_2=final_params.C_2,
        sigma_k=final_params.sigma_k,
        sigma_eps=final_params.sigma_eps,
        u_stars=jnp.array(final_u_stars)
    )
    print(f"\nSaved parameters to {save_path}")

    for i, name in enumerate(case_names):
        plot_profiles_comparison(
            final_states[i], 
            data_list[i], 
            z_list[i], 
            f"{name}", 
            save_path=f"joint_profile_{name}.png"
        )


if __name__ == "__main__":
    main()
    
    
##todo add loss history plto? 