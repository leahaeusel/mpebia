"""Evaluate the single-physics and multi-physics posteriors."""

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
from mpebia.porous_medium_model.parameters_shared import ParametersShared
from mpebia.truncated_gaussian_prior import TruncatedNormalPrior

# SETUP

logger = get_logger(__file__)
directory = get_directory(__file__)

pickle_path_3phase_model = (
    directory / ".." / "3_posteriors_with_observations" / "mpebia_poro_grid_with_blood.pickle"
)
pickle_path_ground_truth = (
    directory / ".." / "1_ground_truth" / "mpebia_poro_ground_truth_at_dline.pickle"
)

params = ParametersShared()
prior = TruncatedNormalPrior(
    params.mean_prior,
    params.std_prior,
    params.truncations_Eh,
    params.truncations_Ed,
    (params.num_grid_points_Eh, params.num_grid_points_Ed),
    params.offset_ppf,
)

logger.info("Parameters shared:\n%s", get_object_attributes(params))


# COMPUTATIONS

# Load model outputs on grid


def get_data_from_pickle(pickle_path):
    """Extract data from pickle file from QUEENS run.

    Args:
        pickle_path (str, Path): Path to pickle file

    Returns:
        np.array: Grid points of healthy Young's modulus
        np.array: Grid points of diseased Young's modulus
        np.array: Displacement results on grid
        np.array: Blood volume fraction results on grid
    """
    data = load_result(pickle_path)
    young_h = np.sort(np.unique(np.array(data["input_data"][:, 0], dtype=np.float64)))  # shape(50,)
    young_d = np.sort(np.unique(np.array(data["input_data"][:, 1], dtype=np.float64)))  # shape(50,)

    output = data["raw_output_data"]["result"]  # shape (2500,)
    input = data["input_data"]

    # Check for None outputs (non-converged simulations)
    job_ids_with_none_output = []
    inputs_with_none_output = []
    for job_id, (input_i, output_i) in enumerate(zip(input, output)):
        if np.any(output_i is None):
            job_ids_with_none_output.append(job_id)
            inputs_with_none_output.append(input_i)
            continue
        for output_j in output_i:
            if np.any(output_j is None):
                job_ids_with_none_output.append(job_id)
                inputs_with_none_output.append(input_i)

    if len(job_ids_with_none_output) > 0:
        inputs_with_none_output = np.array(inputs_with_none_output)
        raise ValueError(
            "Inputs {} with Job IDs {} from pickle path {} produced None output.".format(
                inputs_with_none_output, job_ids_with_none_output, pickle_path
            )
        )

    displ = np.array([o[0] for o in output])  # shape (2500, 7, 2)
    d_grid = displ.reshape((len(young_h), len(young_d), -1), order="F")  # shape (50, 50, 14)

    v_grid = None
    if len(output[0]) > 1:
        volfrac = np.array([o[1] for o in output])  # shape (2500, 2)
        v_grid = volfrac.reshape((len(young_h), len(young_d), -1), order="F")  # shape (50, 50, 2)

    return young_h, young_d, d_grid, v_grid


young_h, young_d, d_grid, v_grid = get_data_from_pickle(pickle_path_3phase_model)

# Compute observations
data = load_result(pickle_path_ground_truth)
d_gt = data["output"]["result"][0][0]
d1_gt = d_gt[:, 0]
d2_gt = d_gt[:, 1]
v_gt = data["output"]["result"][0][1]
std_1 = snr_to_std(params.snr_1, d_gt)  # 0.05
std_2 = snr_to_std(params.snr_2, v_gt)  # 0.005
num_obs_d = len(d_gt.flatten())
noise_1 = sample_sobol(num_obs_d, params.seed_noise)
noise_2 = sample_sobol(len(v_gt), params.seed_noise)
d1_obs = d1_gt + noise_1[: num_obs_d // 2] * std_1
d2_obs = d2_gt + noise_1[num_obs_d // 2 :] * std_1
d_obs_flat = np.vstack((d1_obs, d2_obs)).T.reshape(-1, order="F")
v_obs = v_gt + noise_2 * std_2

logger.info("Standard deviation displacement: %f\n", std_1)
logger.info("Standard deviation volfrac: %f\n", std_2)

# Compute prior
prior.grid_points_1 = young_h
prior.grid_points_2 = young_d
log_prior_grid_3 = prior.get_log_prior_on_grid()
prior_grid_3 = np.exp(log_prior_grid_3)
prior.plot_prior_and_grid_points(prior_grid_3, [params.young_h_gt, params.young_d_gt], directory)

prior.grid_points_1 = young_h
prior.grid_points_2 = young_d
log_prior_grid = prior.get_log_prior_on_grid()

# Compute likelihoods and posteriors

log_like_grid_d = gaussian_log_likelihood(d_obs_flat, d_grid, std_1)
info_gain_sp, posterior_sp = information_gain_2d(
    log_like_grid_d,
    log_prior_grid,
    young_h,
    young_d,
    return_posterior=True,
)

log_like_grid_v = gaussian_log_likelihood(v_obs, v_grid, std_2)
log_like_grid_mp = log_like_grid_d + log_like_grid_v
info_gain_mp, posterior_mp = information_gain_2d(
    log_like_grid_mp,
    log_prior_grid,
    young_h,
    young_d,
    return_posterior=True,
)
riig = relative_increase_in_information_gain(info_gain_sp, info_gain_mp)


# PLOTTING

# General plot parameters
plt.rcParams["hatch.color"] = "grey"
plt.rcParams["hatch.linewidth"] = 3.0
size_observation_markers = 30
size_ground_truth_markers = 80

num_rows = 1
num_cols = 7
width_ratio_gap = 0.53
width_ratios = [1, width_ratio_gap, 1, width_ratio_gap, 1, width_ratio_gap, 1]

fig = plt.figure(figsize=(11.9, 3.1))
gs = GridSpec(
    num_rows,
    num_cols,
    width_ratios=[1, width_ratio_gap, 1, width_ratio_gap, 1, width_ratio_gap, 1],
    wspace=0.0,
    hspace=0.0,
    left=0.14,
    right=0.978,
    top=0.8,
    bottom=0.22,
)


######### PLOT PRIOR, LIKELIHOOD, AND POSTERIOR #########

col_prior = 0
col_liked = 2
col_likeI = 4
col_post = 6
cols = [col_prior, col_liked, col_likeI, col_post]
distrs_to_plot = [
    np.exp(log_prior_grid),
    np.exp(log_like_grid_d),
    np.exp(log_like_grid_v),
    posterior_mp,
]
axes_distr = np.empty((len(cols),), dtype=object)
for j, (col, distr) in enumerate(zip(cols, distrs_to_plot)):
    ax = fig.add_subplot(gs[0, col])
    contour = ax.contourf(
        young_h,
        young_d,
        distr.T,
        6,
        cmap=colors.CMAP,
    )

    ax.set_xlabel(r"$x_1 = E^h$ [kPa]")
    ax.set_ylabel(r"$x_2 = E^d$ [kPa]")
    ax.grid(True)

    # Add column titles
    if col == col_prior:
        title = r"Prior"
    elif col == col_liked:
        title = r"First-field likelihood"
    elif col == col_likeI:
        title = r"Second-field likelihood"
    elif col == col_post:
        title = r"Posterior"
    else:
        raise ValueError("Invalid column.")
    ax.set_title(title, fontsize=16, pad=25)

    axes_distr[j] = ax

######## ADD COLORED BOX ########

pad = 0.01
width_primary_label = 0.03
width_secondary_label = 0.037
linewidth = 5

# Get column positions
bbox_col0 = get_axis_bbox(axes_distr[0], fig)
bbox_col3 = get_axis_bbox(axes_distr[3], fig)
pos_col0 = axes_distr[0].get_position()
offset_box_left = 2.2 * pad + width_secondary_label + width_primary_label
pos = (bbox_col0.x0 - offset_box_left, bbox_col0.y0 - 3 * pad)
width_box = bbox_col3.x1 - bbox_col0.x0 + 1.5 * pad + offset_box_left
height_box = pos_col0.y1 - bbox_col0.y0 + 9 * pad

color_frame = colors.MULTIPHYSICS_LIGHT
zorder = -1000

fig.patches.append(
    plt.Rectangle(
        pos,
        width_box,
        height_box,
        fill=False,
        edgecolor=color_frame,
        linewidth=linewidth,
        transform=fig.transFigure,
        figure=fig,
        zorder=zorder,
    )
)

# background for primary label
fig.patches.append(
    plt.Rectangle(
        (pos[0], pos[1]),
        width_primary_label,
        height_box,
        fill=True,
        facecolor=color_frame,
        edgecolor=color_frame,
        linewidth=linewidth,
        transform=fig.transFigure,
        figure=fig,
        zorder=zorder,
    )
)

# background for secondary label
fig.patches.append(
    plt.Rectangle(
        (pos[0] + width_primary_label, pos[1]),
        width_secondary_label,
        height_box,
        fill=True,
        facecolor=colors.MULTIPHYSICS_LIGHTER,
        edgecolor=None,
        linewidth=linewidth,
        transform=fig.transFigure,
        figure=fig,
        zorder=zorder - 1,
    )
)

# Add primary label
fig.text(
    pos[0] + (width_primary_label / 2),
    pos[1] + height_box / 2,
    r"\textbf{Multi-physics BIA}",
    verticalalignment="center",
    horizontalalignment="center",
    fontsize=16,
    weight="bold",
    rotation="vertical",
)

# Add secondary label
pos_number = np.array([-0.55, 0.36])
fontsize_marker = 14
ax = axes_distr[0]
ax.scatter(
    pos_number[0],
    pos_number[1],
    color="k",
    s=250,
    marker="o",
    zorder=10,
    transform=ax.transAxes,
    clip_on=False,
)
ax.scatter(
    pos_number[0],
    pos_number[1],
    color="w",
    s=200,
    marker="o",
    zorder=11,
    transform=ax.transAxes,
    clip_on=False,
)
ax.annotate(
    text=r"$\bf{1}$",
    xy=(0, 0),
    xytext=pos_number + (0.005, -0.007),
    color="k",
    fontsize=fontsize_marker,
    ha="center",
    va="center",
    zorder=1000,
    xycoords=ax.transAxes,
    textcoords=ax.transAxes,
)


plt.savefig(directory / "poro_prior_likelihood_posterior.png", dpi=300)
