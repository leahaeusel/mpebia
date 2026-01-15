"""Evaluate posteriors at indicated SNRs and #observations."""

import matplotlib.pyplot as plt
import numpy as np

import mpebia.plotting.rc_params  # pylint: disable=unused-import
from mpebia.electromechanical_model.cube_model import CubeModel
from mpebia.electromechanical_model.parameters_riig_grid import ParametersRIIGGrid
from mpebia.electromechanical_model.parameters_shared import ParametersShared
from mpebia.entropies import information_gain_2d
from mpebia.logging import get_logger
from mpebia.observations import (
    gaussian_log_likelihood,
    get_equidistant_indices,
    sample_sobol,
    snr_to_std,
)
from mpebia.output import get_directory, get_object_attributes
from mpebia.plotting import colors
from mpebia.spacing import evenly_log_spaced
from mpebia.truncated_gaussian_prior import TruncatedNormalPrior

# SETUP

logger = get_logger(__file__)
directory = get_directory(__file__)

params = ParametersShared()
params_riig = ParametersRIIGGrid()

cube_model = CubeModel(params.l0, params.U, params.rho)
prior = TruncatedNormalPrior(
    params.mean_prior,
    params.std_prior,
    params.truncations_E,
    params.truncations_nu,
    (params.num_grid_points_E, params.num_grid_points_nu),
    params.offset_ppf,
)

logger.info("Parameters shared:\n%s", get_object_attributes(params))
logger.info("Parameters for information gain grid:\n%s\n", get_object_attributes(params_riig))

# Load data from npz file
data = np.load(directory / "data.npz", allow_pickle=True)
nums_obs_I = data["nums_obs_I"]
snrs_I = data["snrs_I"]
information_gain_sp = data["information_gain_sp"]
F_gt = data["F_gt"]
d_gt = data["d_gt"]
I_gt = data["I_gt"]
F_obs_d = data["F_obs_d"]
d_obs = data["d_obs"]
all_I_sol_grid = data["all_I_sol_grid"]
log_likelihood_grid_d = data["log_likelihood_grid_d"]
log_prior_grid = data["log_prior_grid"]


def plot_observations(F_obs_gt, d_gt, I_gt, F_obs_d, F_obs_I, d_obs, I_obs, name="observations"):
    """Plot and save observations."""
    fig, ax = plt.subplots(2)
    fig.set_figheight(9)
    ax[0].plot(F_obs_gt, d_gt, label="d - ground truth", color=colors.GROUND_TRUTH)
    ax[1].plot(F_obs_gt, I_gt, label="I - ground truth", color=colors.GROUND_TRUTH)
    ax[0].scatter(F_obs_d, d_obs, label="d - obs", color=colors.OBSERVATIONS_DARK)
    ax[1].scatter(F_obs_I, I_obs, label="I - obs", color=colors.OBSERVATIONS_DARK)

    for ax_i in ax:
        ax_i.set_xlabel("F")
        ax_i.grid(True)
        ax_i.legend()

    plt.savefig(directory / f"{name}.png")
    plt.close()


# Compute the posteriors at specific points
nums_obs_plot_post = [
    params.nums_obs_I[0],
    params.nums_obs_I[1],
    params.nums_obs_I[1],
]
snrs_plot_post = [
    params.snrs_I[0],
    params.snrs_I[1],
    params.snrs_I[0],
]
posteriors_to_plot = []
log_likelihood_d_to_plot = []
log_likelihood_I_to_plot = []
for num_obs_I, snr_I in zip(nums_obs_plot_post, snrs_plot_post):
    # Generate observations
    i_obs_I = get_equidistant_indices(num_obs_I, len(F_gt))
    I_obs_wo_noise = I_gt[i_obs_I]
    noise = sample_sobol(len(I_obs_wo_noise), params.seed)
    I_sol_grid = all_I_sol_grid[:, :, i_obs_I]
    std_I = snr_to_std(snr_I, I_obs_wo_noise)
    I_obs = I_obs_wo_noise + std_I * noise
    # Compute log-likelihood
    log_likelihood_grid_I = gaussian_log_likelihood(I_obs, I_sol_grid, std_I)
    # Compute the posterior
    log_likelihood_mp_grid = log_likelihood_grid_d + log_likelihood_grid_I
    information_gain_mp, posterior_mp = information_gain_2d(
        log_likelihood_mp_grid,
        log_prior_grid,
        prior.grid_points_1,
        prior.grid_points_2,
        return_posterior=True,
    )
    riig = (information_gain_mp - information_gain_sp) / information_gain_sp

    plot_observations(
        F_gt, d_gt, I_gt, F_obs_d, F_gt[i_obs_I], d_obs, I_obs, f"obs_N{num_obs_I}_SNR{snr_I}"
    )
    logger.info(
        f"Information gain at {num_obs_I} observations and SNR {snr_I}: {information_gain_mp}"
    )
    logger.info(f"RIIG at {num_obs_I} observations and SNR {snr_I}: {riig}")

    posteriors_to_plot.append(posterior_mp)
    log_likelihood_d_to_plot.append(log_likelihood_grid_d)
    log_likelihood_I_to_plot.append(log_likelihood_grid_I)

posteriors_to_plot = np.array(posteriors_to_plot)
log_likelihood_d_to_plot = np.array(log_likelihood_d_to_plot)
log_likelihood_I_to_plot = np.array(log_likelihood_I_to_plot)

np.savez_compressed(
    directory / "data_indicated_posteriors",
    nums_obs_plot_post=nums_obs_plot_post,
    snrs_plot_post=snrs_plot_post,
    posteriors_to_plot=posteriors_to_plot,
    log_likelihood_d_to_plot=log_likelihood_d_to_plot,
    log_likelihood_I_to_plot=log_likelihood_I_to_plot,
)
