
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sbl_odil.cases import CASE_U_GEOSTROPHIC, SBLCase, coriolis_parameter, get_forcing
from sbl_odil.config import (
    DEFAULT_LATITUDE,
    DEFAULT_N_Z,
    DEFAULT_U_GEOSTROPHIC,
    DEFAULT_WEIGHTS,
    DEFAULT_Z0,
    DEFAULT_Z_TOP,
    make_grid,
)
from sbl_odil.data_io import load_sbl_data
from sbl_odil.diagnostics import diagnose_velocity_collapse, validate_sbl_physics
from sbl_odil.plotting import (
    plot_corner,
    plot_loss_history,
    plot_parameter_posteriors,
    plot_profiles_comparison,
    plot_profiles_with_uncertainty,
)
from sbl_odil.train import train_odil
from sbl_odil.bodil import run_bodil_sampling


def main() -> None:
    parser = argparse.ArgumentParser(description="SBL ODIL Optimization")
    parser.add_argument(
        "--case",
        type=str,
        default="weak",
        choices=["weak", "moderate", "tnbl"],
        help="SBL case: weak (0.05 K/h), moderate (0.25 K/h), or tnbl (neutral)",
    )
    parser.add_argument("--init_noise", type=float, default=0.0, help="Noise added to the initial condition")
    parser.add_argument("--epochs", type=int, default=100000, help="Number of optimization epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing the .mat LES files",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=128,
        help="Number of B-ODIL samples (set to 0 to skip)",
    )
    args = parser.parse_args()

    case_lookup = {
        "weak": SBLCase.WEAK,
        "moderate": SBLCase.MODERATE,
        "tnbl": SBLCase.TNBL,
    }
    case_name_lookup = {
        "weak": "W",
        "moderate": "M",
        "tnbl": "TNBL",
    }
    case = case_lookup[args.case]
    case_name = case_name_lookup[args.case]
    u_geostrophic = CASE_U_GEOSTROPHIC.get(case, DEFAULT_U_GEOSTROPHIC)

    print(f"\n{'=' * 70}")
    print(f"STABLE BOUNDARY LAYER ODIL - {args.case.upper()} STABILITY")
    print(f"{'=' * 70}\n")

    z = make_grid(DEFAULT_N_Z, DEFAULT_Z0, DEFAULT_Z_TOP)
    f_coriolis = coriolis_parameter(DEFAULT_LATITUDE)

    les_data = load_sbl_data(case, args.data_dir, DEFAULT_Z_TOP, u_geostrophic)
    forcing = get_forcing(case, f_coriolis, u_geostrophic)

    final_state, final_params, final_u_star, history = train_odil(
        les_data=les_data,
        forcing=forcing,
        weights=DEFAULT_WEIGHTS,
        z=z,
        n_z=DEFAULT_N_Z,
        n_epochs=args.epochs,
        lr=args.lr,
        print_every=max(args.epochs // 20, 1000),
        u_star_init=les_data["u_star"],
        z0=DEFAULT_Z0,
        init_noise=args.init_noise
    )

    print("\nPlotting results...")
    plot_loss_history(history, case_name)
    plot_profiles_comparison(final_state, les_data, z, case_name)

    print(f"\nFinal learned parameters (SBL-{case_name}):")
    print(f"  u_star (learned) = {final_u_star:.4f}")
    print(f"  C_mu = {final_params.C_mu:.4f}")
    print(f"  C_1  = {final_params.C_1:.4f}")
    print(f"  C_2  = {final_params.C_2:.4f}")
    print(f"  sigma_k  = {final_params.sigma_k:.4f}")
    print(f"  sigma_eps  = {final_params.sigma_eps:.4f}")

    if args.n_samples > 0:
        samples, _, state_samples = run_bodil_sampling(
            les_data,
            forcing,
            DEFAULT_WEIGHTS,
            z,
            DEFAULT_N_Z,
            n_samples=args.n_samples,
            u_star_init=final_u_star,
            z0=DEFAULT_Z0,
            init_noise= args.init_noise
        )

        plot_parameter_posteriors(samples, case_name)
        plot_corner(samples, case_name, save_path=f"sbl_{case_name}_corner.png")
        plot_profiles_with_uncertainty(state_samples, les_data, case_name, z, DEFAULT_N_Z)

    print("\n" + "=" * 70)
    print("PHYSICS VALIDATION")
    print("=" * 70)
    physics_diag = validate_sbl_physics(final_state, final_params, forcing, final_u_star, z)
    for key, val in physics_diag.items():
        print(f"  {key:<30s}: {val:>12.6e}")

    diagnose_velocity_collapse(final_state, les_data, z)


if __name__ == "__main__":
    main()
