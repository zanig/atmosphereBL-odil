"""Plotting utilities for SBL ODIL diagnostics and posteriors."""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from scipy.ndimage import gaussian_filter

from .data_io import collapse_profile

def les_percentile_band(x, n_z, lo=2.5, hi=97.5):
    x_c = collapse_profile(x, n_z)           # mean
    x_lo = jnp.percentile(x, lo, axis=0)
    x_hi = jnp.percentile(x, hi, axis=0)
    return x_c, x_lo, x_hi

def plot_loss_history(history, case_name, save_path=None):
    """Plot training loss history and components."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = history["epoch"]

    axes[0].semilogy(epochs, history["loss"], "b-", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_title(f"SBL({case_name}) Training Loss")


    axes[1].semilogy(epochs, history["L_PDE"], "r-", label="L_PDE", linewidth=2)
    axes[1].semilogy(epochs, history["L_BC"], "g-", label="L_BC", linewidth=2)
    axes[1].semilogy(epochs, history["L_data"], "b-", label="L_data", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss Component")
    axes[1].set_title("Loss Components")
    axes[1].legend()


    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_profiles_comparison(odil_state, les_data, z, case_name, save_path=None):
    """Compare ODIL profiles against LES mean profiles."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))

    z_km = z / 1000.0

    les_u, les_u_lo, les_u_hi = les_percentile_band(les_data["u"], len(z))
    les_v, les_v_lo, les_v_hi = les_percentile_band(les_data["v"], len(z))
    les_k, les_k_lo, les_k_hi = les_percentile_band(les_data["k"], len(z))
    les_theta, les_theta_lo, les_theta_hi = les_percentile_band(les_data["theta"], len(z))
    axes[0].fill_betweenx(z_km,les_u_lo,les_u_hi,color="gray",alpha=0.3,label="LES 95%")
    axes[0].plot(les_u, z_km, "k-", linewidth=2, label="LES")
    axes[0].plot(odil_state.u, z_km, "r--", linewidth=2, label="ODIL")
    axes[0].set_xlabel("U [m/s]")
    axes[0].set_ylabel("Height [km]")
    axes[0].set_title("U Velocity")
    axes[0].legend()


    axes[1].fill_betweenx(z_km,les_v_lo,les_v_hi,color="gray",alpha=0.3,label="LES 95%")
    axes[1].plot(les_v, z_km, "k-", linewidth=2, label="LES")
    axes[1].plot(odil_state.v, z_km, "r--", linewidth=2, label="ODIL")
    axes[1].set_xlabel("V [m/s]")
    axes[1].set_ylabel("Height [km]")
    axes[1].set_title("V Velocity")
    axes[1].legend()


    axes[2].fill_betweenx(z_km,les_k_lo,les_k_hi,color="gray",alpha=0.3,label="LES 95%")
    axes[2].plot(les_k, z_km, "k-", linewidth=2, label="LES")
    axes[2].plot(odil_state.k, z_km, "r--", linewidth=2, label="ODIL")
    axes[2].set_xlabel("TKE [m^2/s^2]")
    axes[2].set_ylabel("Height [km]")
    axes[2].set_title("Turbulent Kinetic Energy")
    axes[2].legend()

    axes[3].fill_betweenx(z_km,les_theta_lo,les_theta_hi,color="gray",alpha=0.3,label="LES 95%")
    axes[3].plot(les_theta, z_km, "k-", linewidth=2, label="LES")
    axes[3].plot(odil_state.theta, z_km, "r--", linewidth=2, label="ODIL")
    axes[3].set_xlabel("theta [K]")
    axes[3].set_ylabel("Height [km]")
    axes[3].set_title("Potential Temperature")
    axes[3].legend()


    plt.suptitle(f"SBL({case_name}) ODIL vs LES Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_parameter_posteriors(samples, case_name, save_path=None):
    """Plot posterior distributions of parameters."""
    param_names = ["C_mu", "C_1", "C_2", "sigma_k", "sigma_eps", "u_star"]

    fig, axes = plt.subplots(2, 3, figsize=(10, 5))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.hist(samples[:, i], bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(
            samples[:, i].mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {samples[:, i].mean():.4f}",
        )
        ax.set_xlabel(name)
        ax.set_ylabel("freq")
        ax.set_title(f"{name} Posterior")
        ax.legend()


    plt.suptitle(f"SBL({case_name}) Parameter Posteriors", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig


def plot_corner(samples, case_name, save_path=None):
    param_names = [r"$C_\mu$", r"$C_1$", r"$C_2$", r"$\sigma_k$", r"$\sigma_\varepsilon$"]
    default_values = [0.03, 1.21, 1.92, 1.0, 1.3]

    n_params = 5
    data = np.array(samples[:, :n_params])

    fig, axes = plt.subplots(n_params, n_params, figsize=(6, 6),)

    color = "#5BA3A8"
    color_dark = "#3D7A7F"

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i < j:
                ax.axis("off")
            elif i == j:
                x = data[:, i]
                kde = stats.gaussian_kde(x)
                x_grid = np.linspace(
                    x.min() - 0.1 * (x.max() - x.min()),
                    x.max() + 0.1 * (x.max() - x.min()),
                    200,
                )
                ax.fill_between(x_grid, kde(x_grid), alpha=0.7, color=color)
                ax.plot(x_grid, kde(x_grid), color=color_dark, linewidth=1.5)

                ax.text(0.95, 0.95, param_names[i], transform=ax.transAxes, fontsize=14, ha="right", va="top")

                ax.set_yticks([])
                if i < n_params - 1:
                    ax.set_xticklabels([])
            else:
                x = data[:, j]
                y = data[:, i]

                H, xedges, yedges = np.histogram2d(x, y, bins=30)
                H = gaussian_filter(H, sigma=1.5)

                H_sorted = np.sort(H.flatten())[::-1]
                H_cumsum = np.cumsum(H_sorted) / H_sorted.sum()
                levels = [H_sorted[np.argmax(H_cumsum > q)] for q in [0.68, 0.95]]
                levels = sorted(levels)

                X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2)
                ax.contourf(
                    X,
                    Y,
                    H.T,
                    levels=[levels[0], levels[1], H.max()],
                    colors=[color, color_dark],
                    alpha=[0.3, 0.6],
                )
                ax.contour(X, Y, H.T, levels=levels, colors=[color_dark], linewidths=0.5, alpha=0.8)

                ax.scatter(default_values[j], default_values[i], marker="*", s=150, c="black", zorder=10)
                ax.set_xlim(default_values[j]*0.45, default_values[j]*1.55)
                if i < n_params - 1:
                    ax.set_xticklabels([])
                if j > 0:
                    ax.set_yticklabels([])

            if i == n_params - 1:
                ax.set_xlabel(param_names[j], fontsize=12)
            if j == 0 and i > 0:
                ax.set_ylabel(param_names[i], fontsize=12)

    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="black", markersize=15, label=r"Default"),
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.5, label=r"$p(\theta|(u,k)$"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", frameon=True, fontsize=12, title="Legend")

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return fig


def plot_profiles_with_uncertainty(state_samples, les_data, case_name, z, n_z, save_path=None):
    """Plot profiles with uncertainty from B-ODIL state samples."""
    state_samples = jnp.array(state_samples)
    n_samples_plot = state_samples.shape[0]

    print(f"Plotting {n_samples_plot} profile samples...")

    u_profiles = state_samples[:, :n_z]
    theta_profiles = state_samples[:, 2 * n_z : 3 * n_z]
    k_profiles = state_samples[:, 3 * n_z : 4 * n_z]

    u_mean = u_profiles.mean(axis=0)
    u_std = u_profiles.std(axis=0)
    k_mean = k_profiles.mean(axis=0)
    k_std = k_profiles.std(axis=0)
    theta_mean = theta_profiles.mean(axis=0)
    theta_std = theta_profiles.std(axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(8, 4))
    z_km = z / 1000.0

    les_u, les_u_lo, les_u_hi = les_percentile_band(les_data["u"], len(z))
    les_k, les_k_lo, les_k_hi = les_percentile_band(les_data["k"], len(z))
    les_theta, les_theta_lo, les_theta_hi = les_percentile_band(les_data["theta"], len(z))

    axes[0].plot(les_u, z_km, "k-", linewidth=2, label="LES")
    axes[0].plot(u_mean, z_km, "r-", linewidth=2, label="B-ODIL mean")
    axes[0].fill_betweenx(z_km, u_mean - 2 * u_std, u_mean + 2 * u_std,
                          color="red", alpha=0.3, label="95% CI")
    axes[0].fill_betweenx(z_km,les_u_lo,les_u_hi,color="gray",
                          alpha=0.3,label="LES 95%")
    axes[0].set_xlabel("U [m/s]")
    axes[0].set_ylabel("Height [km]")
    axes[0].legend()
    axes[0].set_title("U Velocity with Uncertainty")

    axes[1].plot(les_k, z_km, "k-", linewidth=2, label="LES")
    axes[1].plot(k_mean, z_km, "r-", linewidth=2, label="B-ODIL mean")
    axes[1].fill_betweenx(z_km, k_mean - 2 * k_std, k_mean + 2 * k_std,
                          color="red", alpha=0.3, label="95% CI")
    axes[1].fill_betweenx(z_km,les_k_lo,les_k_hi,color="gray",alpha=0.3,label="LES 95%")
    axes[1].set_xlabel("TKE [m^2/s^2]")
    axes[1].set_ylabel("Height [km]")
    axes[1].legend()
    axes[1].set_title("TKE with Uncertainty")

    axes[2].plot(les_theta, z_km, "k-", linewidth=2, label="LES")
    axes[2].plot(theta_mean, z_km, "r-", linewidth=2, label="B-ODIL mean")
    axes[2].fill_betweenx( z_km, theta_mean - 2 * theta_std,
                          theta_mean + 2 * theta_std, color="red",
                          alpha=0.3,label="95% CI")
    axes[2].fill_betweenx(z_km,les_theta_lo,les_theta_hi,color="gray",
                          alpha=0.3,label="LES 95%")
    axes[2].set_xlabel("theta [K]")
    axes[2].set_ylabel("Height [km]")
    axes[2].legend()
    axes[2].set_title("Temperature with Uncertainty")

    plt.suptitle(f"SBL({case_name}) Profiles with Parameter Uncertainty", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path = save_path or f"sbl_{case_name}_uncertainty.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    return fig
