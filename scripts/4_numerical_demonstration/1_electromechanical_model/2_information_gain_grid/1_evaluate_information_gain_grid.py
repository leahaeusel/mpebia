"""Evaluate the multi-physics information gain on a grid."""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

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


# COMPUTATIONS

# Set up values to iterate over for different information gains
nums_obs_I = 2 ** np.arange(0, params_riig.num_obs_I_max_exp + 1)
snrs_I = evenly_log_spaced(params_riig.snr_min, params_riig.snr_max, params_riig.num_snrs)
logger.info("Difference in information gain will be evaluated on the following grid:")
logger.info("Number of observations of I: \n%s", str(nums_obs_I))
logger.info("Signal-to-noise ratios of I: \n%s\n", str(snrs_I))

# Compute prior
log_prior_grid = prior.get_log_prior_on_grid()
prior_grid = np.exp(log_prior_grid)

# Compute displacement observations
cube_model.set_material_parameters(params.E_gt, params.nu_gt)
F_gt = np.linspace(0.0, params.F_max, np.max(nums_obs_I) + 2)
d_gt, I_gt = cube_model.solve(F_gt)

i_obs_d = get_equidistant_indices(params.num_obs_d, len(F_gt))
F_obs_d = F_gt[i_obs_d]

d_obs_wo_noise, _ = cube_model.solve(F_obs_d)

std_d = snr_to_std(params.snr_d, d_obs_wo_noise)
logger.info("Standard deviation d: %f", std_d)

# Sample noise for d from a Sobol sequence
noise = sample_sobol(len(d_obs_wo_noise), params.seed)
d_obs = d_obs_wo_noise + noise * std_d

# Compute information gain for displacement observations only
log_likelihood_grid_d = cube_model.get_log_likelihood_on_grid(
    prior, d_obs, std_d, F_obs_d, index_sol=0
)

information_gain_sp, posterior_sp = information_gain_2d(
    log_likelihood_grid_d,
    log_prior_grid,
    prior.grid_points_1,
    prior.grid_points_2,
    return_posterior=True,
)
logger.info("Information gain for displacement data only: %f", information_gain_sp)

# Sample noise for I from a Sobol sequence
sobol_samples = sp.stats.qmc.Sobol(d=1, seed=params_riig.seed_noise).random(len(I_gt))
all_noise = sp.stats.norm.ppf(sobol_samples).squeeze()

# Compute solution for electric current I on grid
all_I_sol_grid = cube_model.solve_on_grid(
    prior.grid_points_1, prior.grid_points_2, F_gt, index_sol=1
)  # shape: (num_grid_points_E, num_grid_points_nu, len(Fs))

posteriors_mp_grid = np.zeros(
    (
        len(nums_obs_I),
        len(snrs_I),
        len(prior.grid_points_1),
        len(prior.grid_points_2),
    )
)
information_gain_mp_grid = np.zeros((len(nums_obs_I), len(snrs_I)))
F_obs_I_grid = np.zeros((len(nums_obs_I), len(snrs_I)), dtype=object)
I_obs_grid = np.zeros((len(nums_obs_I), len(snrs_I)), dtype=object)
eval_points = []

for i_num_obs_I, num_obs_I in enumerate(nums_obs_I):
    logger.info("\n\nCurrent number of observations of I: %d\n", num_obs_I)
    i_obs_I = get_equidistant_indices(num_obs_I, len(F_gt))
    I_obs_wo_noise = I_gt[i_obs_I]
    noise = all_noise[:num_obs_I]
    I_sol_grid = all_I_sol_grid[:, :, i_obs_I]

    for i_snr, snr_I in enumerate(snrs_I):
        logger.info("\nCurrent signal-to-noise ratio of I: %f", snr_I)

        std_I = snr_to_std(snr_I, I_obs_wo_noise)
        logger.info("Corresponding standard deviation of I: %f", std_I)

        # Compute the log-likelihood for the current SNR
        I_obs = I_obs_wo_noise + std_I * noise
        log_likelihood_grid_I = gaussian_log_likelihood(I_obs, I_sol_grid, std_I)

        # Compute the resulting information gain
        log_likelihood_mp_grid = log_likelihood_grid_d + log_likelihood_grid_I
        information_gain_mp, posterior_mp = information_gain_2d(
            log_likelihood_mp_grid,
            log_prior_grid,
            prior.grid_points_1,
            prior.grid_points_2,
            return_posterior=True,
        )

        posteriors_mp_grid[i_num_obs_I, i_snr] = posterior_mp
        information_gain_mp_grid[i_num_obs_I, i_snr] = information_gain_mp
        F_obs_I_grid[i_num_obs_I, i_snr] = F_gt[i_obs_I]
        I_obs_grid[i_num_obs_I, i_snr] = I_obs
        eval_points.append((num_obs_I, snr_I))

eval_points = np.array(eval_points)
difference_in_information_gain_grid = information_gain_mp_grid - information_gain_sp
relative_increase_information_gain_grid = difference_in_information_gain_grid / information_gain_sp

# PLOTTING

fig, ax = plt.subplots(figsize=(5, 4))

contour = ax.contourf(
    nums_obs_I, snrs_I, relative_increase_information_gain_grid.T, 10, cmap=colors.CMAP
)
fig.colorbar(contour, label=r"$\text{RIIG} \left[ q_\text{mp}, q_\text{sp}, p \right]$")

ax.set_xlabel(r"$N_{\text{obs}, 2}$ [-]")
ax.set_ylabel(r"$SNR_2$ [-]")
ax.grid(True)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_aspect("equal", "box")

plt.tight_layout()
plt.savefig(directory / "riig_grid.png")

np.savez_compressed(
    directory / "data",
    nums_obs_I=nums_obs_I,
    snrs_I=snrs_I,
    eval_points=eval_points,
    F_gt=F_gt,
    d_gt=d_gt,
    I_gt=I_gt,
    F_obs_d=F_obs_d,
    F_obs_I_grid=F_obs_I_grid,
    d_obs=d_obs,
    I_obs_grid=I_obs_grid,
    all_noise=all_noise,
    all_I_sol_grid=all_I_sol_grid,
    log_prior_grid=log_prior_grid,
    log_likelihood_grid_d=log_likelihood_grid_d,
    posteriors_mp_grid=posteriors_mp_grid,
    information_gain_sp=information_gain_sp,
    information_gain_mp_grid=information_gain_mp_grid,
    relative_increase_information_gain_grid=relative_increase_information_gain_grid,
)
