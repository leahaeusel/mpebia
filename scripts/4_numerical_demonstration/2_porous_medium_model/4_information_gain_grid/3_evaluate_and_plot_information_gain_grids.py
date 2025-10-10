"""Evaluate the multi-physics information gain on a grid."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from queens.utils.io import load_result

import mpebia.plotting.rc_params  # pylint: disable=unused-import
from mpebia.entropies import information_gain_2d, relative_increase_in_information_gain
from mpebia.logging import get_logger
from mpebia.observations import gaussian_log_likelihood, sample_sobol, snr_to_std
from mpebia.output import get_directory, get_object_attributes
from mpebia.plotting import colors
from mpebia.plotting.positioning import get_axis_bbox
from mpebia.plotting.style import get_arrow_kwargs, get_label_kwargs
from mpebia.porous_medium_model.parameters_riig_grid import ParametersRIIGGrid as params_riig
from mpebia.porous_medium_model.parameters_shared import ParametersShared as params
from mpebia.porous_medium_model.plotting import (
    plot_observations_d,
    plot_observations_v,
    plot_posterior,
)
from mpebia.spacing import evenly_log_spaced
from mpebia.truncated_gaussian_prior import TruncatedNormalPrior

# SETUP

logger = get_logger(__file__)
directory = get_directory(__file__)

pickle_path_grid_data = directory / "mpebia_poro_grid_kj.pickle"
pickle_path_ground_truth = directory / "mpebia_poro_ground_truth_different_kjs.pickle"

prior = TruncatedNormalPrior(
    params.mean_prior,
    params.std_prior,
    params.truncations_Eh,
    params.truncations_Ed,
    (params.num_grid_points_Eh, params.num_grid_points_Ed),
    params.offset_ppf,
)

logger.info("Parameters shared:\n%s", get_object_attributes(params))
logger.info("Parameters for information gain grid:\n%s\n", get_object_attributes(params_riig))

# Load QUEENS data from pickle files
data_grid = load_result(pickle_path_grid_data)
data_obs = load_result(pickle_path_ground_truth)

output_grid = data_grid["raw_output_data"]["result"]  # list of [(7, 2), (2,)] of len 25000
input_grid = data_grid["input_data"]  # (25000, 4)
output_obs = data_obs["raw_output_data"]["result"]  # list of [(7, 2), (2,)] of len 10
kjs_obs = data_obs["input_data"][:, 2]  # (10, 4)
num_evals = len(output_grid)
num_kjs = len(kjs_obs)
num_evals_per_kj = int(num_evals / num_kjs)

# Check for None outputs (non-converged simulations)
job_ids_with_none_output = []
inputs_with_none_output = []
for job_id, (input, output) in enumerate(zip(input_grid, output_grid)):
    for output_i in output:
        if np.any(output_i is None):
            job_ids_with_none_output.append(job_id)
            inputs_with_none_output.append(input)

if len(job_ids_with_none_output) > 0:
    inputs_with_none_output = np.array(inputs_with_none_output)
    raise ValueError(
        "Inputs {} with Job IDs {} produced None output.".format(
            inputs_with_none_output, job_ids_with_none_output
        )
    )


# COMPUTATIONS

# Set up values to iterate over for different information gains
snrs_1 = evenly_log_spaced(params_riig.snr_1_min, params_riig.snr_1_max, params_riig.num_snrs_1)
snrs_2 = evenly_log_spaced(params_riig.snr_2_min, params_riig.snr_2_max, params_riig.num_snrs_2)
logger.info("Difference in information gain will be evaluated on the following grid:")
logger.info("Parameters k_j: \n%s", str(kjs_obs))
logger.info("Signal-to-noise ratios of 1st field: \n%s\n", str(snrs_1))
logger.info("Signal-to-noise ratios of 2nd field: \n%s\n", str(snrs_2))

# Sample noise for 1st and 2nd field from a Sobol sequence
num_obs_d = len(output_obs[0][0].flatten())
num_obs_v = len(output_obs[0][1].flatten())
noise_1 = sample_sobol(num_obs_d, params.seed_noise)
noise_2 = sample_sobol(num_obs_v, params.seed_noise)

# Get input grid points
young_h = np.sort(np.unique(np.array(input_grid[:, 0], dtype=np.float64)))  # shape(50,)
young_d = np.sort(np.unique(np.array(input_grid[:, 1], dtype=np.float64)))  # shape(50,)

# Compute prior
prior.grid_points_1 = young_h
prior.grid_points_2 = young_d
log_prior_grid = prior.get_log_prior_on_grid()
prior_grid = np.exp(log_prior_grid)


def reconstruct_outputs_on_input_grid(input_grid, output_lst, kj, num_evals_per_kj):
    """Reconstruct inputs and outputs on grid for a given kj."""
    tol = 0.001
    i_evals_with_kj = np.where(np.abs(input_grid[:, 2] - kj) <= tol)[0]
    input_kj = input_grid[i_evals_with_kj]  # shape (2500, 4)
    output_kj = [output_lst[i] for i in i_evals_with_kj]  # len 2500

    # Check that k_j inputs of evaluations and observations were the same
    np.testing.assert_array_almost_equal(input_kj[:, 2], kj * np.ones(input_kj.shape[0]))
    assert len(young_h) * len(young_d) == num_evals_per_kj

    d_grid = np.zeros((len(young_h), len(young_d), num_obs_d))
    v_grid = np.zeros((len(young_h), len(young_d), num_obs_v))

    # Create grid of E1 and E2 to check that grid reconstruction works correctly
    young_h_grid = np.zeros((len(young_h), len(young_d)))
    young_d_grid = np.zeros((len(young_h), len(young_d)))

    # Iterate over evaluations
    for input, output in zip(input_kj, output_kj):
        i_young_h = np.where(np.abs(input[0] - young_h) <= tol)
        i_young_d = np.where(np.abs(input[1] - young_d) <= tol)

        if not np.all(d_grid[i_young_h, i_young_d] == 0):
            raise ValueError("Entry in d_grid has already been set")

        d_grid[i_young_h, i_young_d] = output[0].reshape(-1, order="F")
        v_grid[i_young_h, i_young_d] = output[1].flatten()
        young_h_grid[i_young_h, i_young_d] = young_h[i_young_h]
        young_d_grid[i_young_h, i_young_d] = young_d[i_young_d]

    # Safety checks: Check that all rows/columns of coordinate grids are equal
    assert np.all(np.all(young_h_grid == young_h_grid[:, 0][:, np.newaxis], axis=1))
    assert np.all(np.all(young_d_grid == young_d_grid[0, :], axis=0))

    return d_grid, v_grid


def get_ground_truth_observations(ys_obs, i_kj_obs):
    """Recover the ground truth observations from the QUEENS output.

    Args:
        ys_obs (np.ndarray): QUEENS output for all coupling parameters k_j.
        i_kj_obs (int): Index of the coupling parameter k_j for which to recover the observations.

    Returns:
        np.ndarray: Displacement ground truth.
        np.ndarray: Blood volume fraction ground truth.
    """
    y_obs_kj_gt = ys_obs[i_kj_obs]
    d_gt = y_obs_kj_gt[0]
    v_gt = y_obs_kj_gt[1].flatten()

    return d_gt, v_gt


evaluation_points_vary_kj_snr1 = []
evaluation_points_vary_kj_snr2 = []
evaluation_points_vary_snr1_snr2 = []
information_gains_sp_vary_kj_snr1 = np.zeros((len(kjs_obs), len(snrs_1)))
information_gains_sp_vary_kj_snr2 = np.zeros((len(kjs_obs), 1))
information_gains_sp_vary_snr1_snr2 = np.zeros((len(snrs_1), 1))
information_gains_mp_vary_kj_snr1 = np.zeros((len(kjs_obs), len(snrs_1)))
information_gains_mp_vary_kj_snr2 = np.zeros((len(kjs_obs), len(snrs_2)))
information_gains_mp_vary_snr1_snr2 = np.zeros((len(snrs_1), len(snrs_2)))

for i_kj, kj in enumerate(kjs_obs):
    # Reconstruct d_grid with shape (num_E1, num_E2, num_obs_d)
    # and         v_grid with shape (num_E1, num_E2, num_obs_v)
    d_grid, v_grid = reconstruct_outputs_on_input_grid(
        input_grid, output_grid, kj, num_evals_per_kj
    )

    # Get ground truth outputs for observations with shape (num_obs,)
    d_gt_2d, v_gt = get_ground_truth_observations(output_obs, i_kj)
    d1_gt = d_gt_2d[:, 0]
    d2_gt = d_gt_2d[:, 1]

    # Compute fixed standard deviations
    std_d_fixed = snr_to_std(params_riig.fixed_snr_1, d_gt_2d)
    std_v_fixed = snr_to_std(params_riig.fixed_snr_2, v_gt)
    logger.info("Fixed std_d: %f", std_d_fixed)
    logger.info("Fixed std_v: %f", std_v_fixed)

    # Compute fixed observations
    d1_obs = d1_gt + noise_1[: num_obs_d // 2] * std_d_fixed
    d2_obs = d2_gt + noise_1[num_obs_d // 2 :] * std_d_fixed
    d_obs_fixed = np.vstack((d1_obs, d2_obs)).T.reshape(-1, order="F")
    d_gt = np.vstack((d1_gt, d2_gt)).T.reshape(-1, order="F")
    v_obs_fixed = v_gt + noise_2 * std_v_fixed

    if i_kj == 5:
        plot_observations_d(
            d_gt, d_obs_fixed, directory / f"d_obs_fixed_snr{round(params_riig.fixed_snr_1)}.png"
        )
        plot_observations_v(
            v_gt, v_obs_fixed, directory / f"v_obs_fixed_snr{round(params_riig.fixed_snr_2)}.png"
        )

    # Compute log likelihoods and information gain for fixed SNRs
    fixed_log_likelihood_grid_d = gaussian_log_likelihood(d_obs_fixed, d_grid, std_d_fixed)
    fixed_log_likelihood_grid_v = gaussian_log_likelihood(v_obs_fixed, v_grid, std_v_fixed)
    fixed_information_gain_sp, posterior_sp = information_gain_2d(
        fixed_log_likelihood_grid_d,
        log_prior_grid,
        prior.grid_points_1,
        prior.grid_points_2,
        return_posterior=True,
    )
    information_gains_sp_vary_kj_snr2[i_kj] = fixed_information_gain_sp
    logger.info("Fixed information_gain single-physics: %f", fixed_information_gain_sp)

    # Compute information gain for varying SNRs of 1st field
    for i_snr1, snr1 in enumerate(snrs_1):
        std_d = snr_to_std(snr1, d_gt_2d)
        d_obs = d_gt + std_d * noise_1

        if i_kj == 5 and i_snr1 == len(snrs_1) - 1:
            plot_observations_d(d_gt, d_obs, directory / f"d_obs_snr{round(snr1)}.png")

        log_likelihood_grid_d = gaussian_log_likelihood(d_obs, d_grid, std_d)
        log_likelihood_grid_mp = log_likelihood_grid_d + fixed_log_likelihood_grid_v

        information_gain_sp, posterior_sp = information_gain_2d(
            log_likelihood_grid_d,
            log_prior_grid,
            prior.grid_points_1,
            prior.grid_points_2,
            return_posterior=True,
        )
        information_gain_mp, posterior_mp = information_gain_2d(
            log_likelihood_grid_mp,
            log_prior_grid,
            prior.grid_points_1,
            prior.grid_points_2,
            return_posterior=True,
        )

        information_gains_sp_vary_kj_snr1[i_kj, i_snr1] = information_gain_sp
        information_gains_mp_vary_kj_snr1[i_kj, i_snr1] = information_gain_mp
        evaluation_points_vary_kj_snr1.append((kj, snr1))

    # Compute information gain for varying SNRs of 2nd field
    for i_snr2, snr2 in enumerate(snrs_2):
        std_v = snr_to_std(snr2, v_gt)
        v_obs = v_gt + std_v * noise_2

        if i_kj == 0 and i_snr2 == 0:
            plot_observations_v(v_gt, v_obs, directory / f"v_obs_snr{round(snr2)}.png")

        log_likelihood_grid_v = gaussian_log_likelihood(v_obs, v_grid, std_v)
        log_likelihood_grid_mp = fixed_log_likelihood_grid_d + log_likelihood_grid_v

        information_gain_mp, posterior_mp = information_gain_2d(
            log_likelihood_grid_mp,
            log_prior_grid,
            prior.grid_points_1,
            prior.grid_points_2,
            return_posterior=True,
        )

        information_gains_mp_vary_kj_snr2[i_kj, i_snr2] = information_gain_mp
        evaluation_points_vary_kj_snr2.append((kj, snr2))

    # Compute information gain for varying SNRs of 1st and 2nd field
    if i_kj == 0:
        for i_snr1, snr1 in enumerate(snrs_1):
            std_d = snr_to_std(snr1, d_gt_2d)
            d_obs = d_gt + std_d * noise_1
            log_likelihood_grid_d = gaussian_log_likelihood(d_obs, d_grid, std_d)
            information_gain_sp = information_gain_2d(
                log_likelihood_grid_d,
                log_prior_grid,
                prior.grid_points_1,
                prior.grid_points_2,
            )
            information_gains_sp_vary_snr1_snr2[i_snr1] = information_gain_sp

            for i_snr2, snr2 in enumerate(snrs_2):
                std_v = snr_to_std(snr2, v_gt)
                v_obs = v_gt + std_v * noise_2
                log_likelihood_grid_v = gaussian_log_likelihood(v_obs, v_grid, std_v)
                log_likelihood_grid_mp = log_likelihood_grid_d + log_likelihood_grid_v
                information_gain_mp = information_gain_2d(
                    log_likelihood_grid_mp,
                    log_prior_grid,
                    prior.grid_points_1,
                    prior.grid_points_2,
                )
                information_gains_mp_vary_snr1_snr2[i_snr1, i_snr2] = information_gain_mp
                evaluation_points_vary_snr1_snr2.append((snr1, snr2))

    logger.info("Finished kj=%f", kj)

evaluation_points_vary_kj_snr1 = np.array(evaluation_points_vary_kj_snr1)
evaluation_points_vary_kj_snr2 = np.array(evaluation_points_vary_kj_snr2)
evaluation_points_vary_snr1_snr2 = np.array(evaluation_points_vary_snr1_snr2)
relative_increase_information_gain_vary_kj_snr1 = relative_increase_in_information_gain(
    information_gains_sp_vary_kj_snr1, information_gains_mp_vary_kj_snr1
)
relative_increase_information_gain_vary_kj_snr2 = relative_increase_in_information_gain(
    information_gains_sp_vary_kj_snr2, information_gains_mp_vary_kj_snr2
)
relative_increase_information_gain_vary_snr1_snr2 = relative_increase_in_information_gain(
    information_gains_sp_vary_snr1_snr2, information_gains_mp_vary_snr1_snr2
)

# PLOTTING

fig = plt.figure(
    figsize=(11.9, 11.0),
)
offset = 0.015
gs = GridSpec(
    2,
    3,
    height_ratios=[1.0, 1.8],
    hspace=0.35,
    wspace=0.3,
    left=0.05,
    right=0.985,
    top=0.99,
    bottom=0.06,
)

######### PLOT RIIG GRID #########

######### 2D contour plots of the RIIG planes #########
snrs_lst = [[snrs_2, snrs_1], snrs_2, snrs_1]
riigs = [
    relative_increase_information_gain_vary_snr1_snr2,
    relative_increase_information_gain_vary_kj_snr2,
    relative_increase_information_gain_vary_kj_snr1,
]
riig_min = np.min([np.min(riig) for riig in riigs])
riig_max = np.max([np.max(riig) for riig in riigs])
evaluation_points = [
    evaluation_points_vary_snr1_snr2,
    evaluation_points_vary_kj_snr2,
    evaluation_points_vary_kj_snr1,
]
contour_kwargs = dict(
    levels=14,
    cmap=colors.CMAP,
    vmin=riig_min,
    vmax=riig_max,
)
logger.info("Min. RIIG: %f", riig_min)
logger.info("Max. RIIG: %f", riig_max)
contours = []
axes_riig = []
for i, (snrs, riig, eval_points) in enumerate(zip(snrs_lst, riigs, evaluation_points)):
    ax_riig = fig.add_subplot(gs[0, i])
    axes_riig.append(ax_riig)

    if i == 0:
        contour = ax_riig.contourf(snrs[0], snrs[1], riig, **contour_kwargs)
        ax_riig.set_xlabel(r"$\text{SNR}_2$  [-]")
        ax_riig.set_ylabel(r"$\text{SNR}_1$ [-]")
        ax_riig.set_xscale("log")
    else:
        contour = ax_riig.contourf(kjs_obs, snrs, riig.T, **contour_kwargs)
        ax_riig.set_xlabel(r"coupling parameter $k_J$  [-]")
        ax_riig.set_ylabel(r"$\text{SNR}_%d$ [-]" % (-i + 3))

    ax_riig.grid(True)
    ax_riig.set_yscale("log")

    contours.append(contour)

######### 3D plot of the two RIIG planes #########
ax_3d = fig.add_subplot(gs[1, 0:3], projection="3d")
kj_mesh, snr1_mesh = np.meshgrid(kjs_obs, np.log(snrs_1))
alpha = 0.9
ax_3d.contourf(
    kj_mesh,
    relative_increase_information_gain_vary_kj_snr1.T,
    snr1_mesh,
    zdir="y",
    offset=np.log(params_riig.fixed_snr_2),
    alpha=alpha,
    **contour_kwargs,
)
kj_mesh, snr2_mesh = np.meshgrid(kjs_obs, np.log(snrs_2))
ax_3d.contourf(
    kj_mesh,
    snr2_mesh,
    relative_increase_information_gain_vary_kj_snr2.T,
    zdir="z",
    offset=np.log(params_riig.fixed_snr_1),
    alpha=alpha,
    **contour_kwargs,
)
snr1_mesh, snr2_mesh = np.meshgrid(np.log(snrs_1), np.log(snrs_2))
ax_3d.contourf(
    relative_increase_information_gain_vary_snr1_snr2.T,
    snr2_mesh,
    snr1_mesh,
    zdir="x",
    offset=kjs_obs[0],
    alpha=alpha,
    **contour_kwargs,
)
ax_3d.set_xbound((np.min(kjs_obs), np.max(kjs_obs)))
ax_3d.set_ybound((np.min(snr2_mesh), np.max(snr2_mesh)))
ax_3d.set_zbound((np.min(snr1_mesh), np.max(snr1_mesh)))
ax_3d.set_xlabel(r"coupling parameter $k_J$  [-]")
ax_3d.set_ylabel(r"$\text{SNR}_2$ [-]")
ax_3d.set_zlabel(r"$\text{SNR}_1$ [-]")
ax_3d.set_xticks([-1.5, -1.0, -0.5, 0.0])
snr_exponents = np.arange(1, 6)
snr_ticks = np.log(10**snr_exponents)
ax_3d.set_yticks(snr_ticks, labels=[f"$10^{exp}$" for exp in snr_exponents])
ax_3d.set_zticks(snr_ticks, labels=[f"$10^{exp}$" for exp in snr_exponents])

######### Add arrows with labels #########
arrow_kwargs = get_arrow_kwargs(ax=None)
label_kwargs = get_label_kwargs(ax=None, fontsize=14)
pos_arrows = (
    (-0.09, 0.0, -0.08, -0.11),
    (0.115, 0.112, 0.045, -0.03),
    (0.111, 0.061, -0.04, -0.092),
)
for x1, x2, y1, y2 in pos_arrows:
    ax_3d.annotate(
        "",
        xy=(x1, y1),  # tip
        xytext=(x2, y2),
        **arrow_kwargs,
    )
x1, x2, x3 = -0.055, 0.153, 0.121
y1, y2, y3 = -0.103, 0.015, -0.067
pos_labels = (
    (x1, y1),
    (x1, y1 - 0.008),
    (x2, y2),
    (x2, y2 - 0.008),
    (x2, y2 - 0.016),
    (x3, y3),
    (x3, y3 - 0.008),
    (x3, y3 - 0.016),
)
labels = (
    "Stronger",
    "coupling",
    "Higher",
    "signal-to-noise ratio",
    "in the 1st field",
    "Higher",
    "signal-to-noise ratio",
    "in the 2nd field",
)
for pos_label, label in zip(pos_labels, labels):
    ax_3d.annotate(
        r"\textbf{" + label + r"}",
        xy=pos_label,
        **label_kwargs,
    )


########## Add colorbar #########
pos_riig = axes_riig[1].get_position()
bbox_riig = get_axis_bbox(axes_riig[1], fig)
width_riig = pos_riig.x1 - pos_riig.x0
cbar_ax = fig.add_axes([pos_riig.x0, bbox_riig.y0 - 0.05, width_riig, 0.02])
fig.colorbar(
    contours[1],
    cax=cbar_ax,
    label=r"Relative increase in information gain $\text{RIIG}\!\left( \boldsymbol{y}_{1, \text{obs}}, \boldsymbol{y}_{2, \text{obs}} \right)$ [\hspace{1pt}-\hspace{1pt}]",
    location="bottom",
)

plt.savefig(directory / "riig_grids.png")
