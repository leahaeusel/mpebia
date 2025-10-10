"""Utility functions for information gain calculations."""

import matplotlib.pyplot as plt
import numpy as np

import mpebia.plotting.rc_params  # pylint: disable=unused-import
from mpebia.plotting import colors


def plot_observations_d(d_gt, d_obs, path):
    """Plot displacement observations and ground truth values."""
    x_d = np.array(range(len(d_obs) // 2)) + 1

    _, ax = plt.subplots(2, figsize=(6, 8))

    for i in range(2):
        ax[i].scatter(
            x_d,
            d_gt[i * len(d_gt) // 2 : (i + 1) * len(d_gt) // 2],
            label="Ground truth",
            marker="_",
            color=colors.GROUND_TRUTH,
            zorder=3,
        )
        ax[i].scatter(
            x_d,
            d_obs[i * len(d_gt) // 2 : (i + 1) * len(d_gt) // 2],
            label="Observations",
            color=colors.OBSERVATIONS_DARK,
        )
        ax[i].set_xticks(x_d, minor=False)
        ax[i].set_xlabel("location")
        ax[i].set_ylabel(r"tissue displacement $d^t_{}$ [mm]".format(i + 1))
        ax[i].grid(True)
        ax[i].legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_observations_v(v_gt, v_obs, path):
    """Plot blood volume fraction observations and ground truth values."""
    x_v = np.array(range(len(v_obs))) + 1

    _, ax = plt.subplots(figsize=(6, 4))

    ax.scatter(x_v, v_gt, label="Ground truth", marker="_", color=colors.GROUND_TRUTH, zorder=3)
    ax.scatter(x_v, v_obs, label="Observations", color=colors.OBSERVATIONS_DARK)
    ax.set_xticks(x_v, minor=False)
    ax.set_xlabel("location")
    ax.set_ylabel(r"blood volume fraction $\varepsilon^b$ [-]")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_posterior(Ehs, Eds, posterior, path):
    """Plot blood volume fraction observations and ground truth values."""
    fig, ax = plt.subplots(figsize=(6, 4))

    contour = ax.contourf(Ehs, Eds, posterior.T, 6, cmap=colors.CMAP)
    cbar = fig.colorbar(contour)
    ax.set_xlabel(r"$E^h$ [kPa]")
    ax.set_ylabel(r"$E^d$ [kPa]")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
