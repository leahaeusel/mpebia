"""Evaluate the single-physics and multi-physics posteriors."""

import matplotlib.pyplot as plt
import numpy as np
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

pickle_path_3phase_model = directory / "mpebia_poro_grid_with_blood.pickle"
pickle_path_2phase_model = directory / "mpebia_poro_grid_without_blood.pickle"
pickle_path_ground_truth = (
    directory / ".." / "1_ground_truth" / "mpebia_poro_ground_truth_at_dline.pickle"
)

snr_1 = 50
snr_2 = 50000

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


young_h_3, young_d_3, d_grid_3, v_grid_3 = get_data_from_pickle(pickle_path_3phase_model)
young_h_2, young_d_2, d_grid_2, _ = get_data_from_pickle(pickle_path_2phase_model)

youngs_h = [young_h_2, young_h_3, young_h_3]
youngs_d = [young_d_2, young_d_3, young_d_3]
d_grids = [d_grid_2, d_grid_3, d_grid_3]
v_grids = [None, None, v_grid_3]

# Compute observations
data = load_result(pickle_path_ground_truth)
d_gt = data["output"]["result"][0][0]
d1_gt = d_gt[:, 0]
d2_gt = d_gt[:, 1]
v_gt = data["output"]["result"][0][1]
std_1 = snr_to_std(snr_1, d_gt)  # 0.05
std_2 = snr_to_std(snr_2, v_gt)  # 0.005
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
prior.grid_points_1 = young_h_3
prior.grid_points_2 = young_d_3
log_prior_grid_3 = prior.get_log_prior_on_grid()
prior_grid_3 = np.exp(log_prior_grid_3)
prior.plot_prior_and_grid_points(prior_grid_3, [params.young_h_gt, params.young_d_gt], directory)

prior.grid_points_1 = young_h_2
prior.grid_points_2 = young_d_2
log_prior_grid_2 = prior.get_log_prior_on_grid()

log_priors = [log_prior_grid_2, log_prior_grid_3, log_prior_grid_3]

# Compute likelihoods and posteriors
posteriors_sp = []
info_gains_sp = []
log_likes_grid_d = []
posterior_mp = None
info_gain_mp = None
riig = None
for young_h, young_d, d_grid, v_grid, log_prior_grid in zip(
    youngs_h, youngs_d, d_grids, v_grids, log_priors
):
    log_like_grid_d = gaussian_log_likelihood(d_obs_flat, d_grid, std_1)
    info_gain_sp, posterior_sp = information_gain_2d(
        log_like_grid_d,
        log_prior_grid,
        young_h,
        young_d,
        return_posterior=True,
    )
    info_gains_sp.append(info_gain_sp)
    posteriors_sp.append(posterior_sp)
    log_likes_grid_d.append(log_like_grid_d)

    if v_grid is not None:
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

posteriors_to_plot = [posteriors_sp[0], posteriors_sp[1], posterior_mp]
info_gains_to_plot = [info_gains_sp[0], info_gains_sp[1], info_gain_mp]


# PLOTTING

# General plot parameters
plt.rcParams["hatch.color"] = "grey"
plt.rcParams["hatch.linewidth"] = 3.0
size_observation_markers = 30
size_ground_truth_markers = 80

row_obs_d1 = 0
row_obs_d2 = row_obs_d1 + 1
row_obs_v = row_obs_d2 + 1
row_contours = row_obs_v + 1
row_ig = row_contours + 1
row_riig = row_ig + 1
num_rows = row_riig + 1

col_obs_d_no_blood = 0
col_obs_d = col_obs_d_no_blood + 1
col_obs_dv = col_obs_d + 1
cols_obs = [col_obs_d_no_blood, col_obs_d, col_obs_dv]
num_columns = col_obs_dv + 1

height_ratios = [4, 4, 4, 5.6, 0.9, 0.9]

fig, ax = plt.subplots(
    num_rows,
    num_columns,
    figsize=(11.9, 12.6),
    gridspec_kw={"height_ratios": height_ratios, "wspace": 0.55, "hspace": 0.55},
)
fig.subplots_adjust(left=0.125, right=0.83, top=0.88, bottom=0.065)

################## Plot observations ##################
ax[row_obs_v, col_obs_d_no_blood].axis("off")
ax[row_obs_v, col_obs_d].axis("off")
# Set observations x-axis labels and grid
for col in cols_obs:
    for row in [row_obs_d1, row_obs_d2, row_obs_v]:
        ax[row, col].set_xlabel("location")
        ax[row, col].grid(True)

x_d = np.array(range(len(d1_obs))) + 1
x_v = np.array(range(len(v_obs))) + 1
# Plot displacement data
for col in [col_obs_d_no_blood, col_obs_d, col_obs_dv]:
    for i, (d_obs, d_gt) in enumerate(zip([d1_obs, d2_obs], [d1_gt, d2_gt])):
        row = i + row_obs_d1
        ax[row, col].scatter(
            x_d,
            d_gt,
            label="Ground truth",
            marker="_",
            color=colors.GROUND_TRUTH,
            s=size_ground_truth_markers,
            zorder=3,  # Set a higher z-order to ensure markers are in front of the grid
        )
        ax[row, col].scatter(
            x_d,
            d_obs,
            label="Observations",
            color=colors.OBSERVATIONS_DARK,
            s=size_observation_markers,
        )
        ax[row, col].set_ylabel(r"$y_{1," + str(i + 1) + r"}= d^t_" + str(i + 1) + r"$ [mm]")
        ax[row, col].set_xticks(x_d, minor=False)
# Plot blood volume fraction data
for col in [
    col_obs_dv,
]:
    ax[row_obs_v, col].scatter(
        x_v,
        v_gt,
        marker="_",
        color=colors.GROUND_TRUTH,
        s=size_ground_truth_markers,
        zorder=3,  # Set a higher z-order to ensure markers are in front of the grid
    )
    ax[row_obs_v, col].scatter(
        x_v,
        v_obs,
        color=colors.OBSERVATIONS_DARK,
        s=size_observation_markers,
    )
    ax[row_obs_v, col].set_ylabel(r"$y_2 = \varepsilon^b$ [-]")
    ax[row_obs_v, col].set_xticks(x_v, minor=False)
    ax[row_obs_v, col].set_xlim([0.5, 2.5])
# Add observations legend
ax[row_obs_d2, col_obs_dv].legend(bbox_to_anchor=(1.15, 0.47), loc="center left")

################## Plot posteriors ##################
for i, (distr, sub) in enumerate(zip(posteriors_to_plot, ["sp", "sp", "mp"])):
    col = i + col_obs_d_no_blood
    contour = ax[row_contours, col].contourf(
        young_h,
        young_d,
        distr.T,
        6,
        vmin=np.min(posteriors_to_plot),
        vmax=np.max(posteriors_to_plot),
        cmap=colors.CMAP,
    )

    ax[row_contours, col].scatter(
        params.young_h_gt,
        params.young_d_gt,
        marker="x",
        color=colors.GROUND_TRUTH,
        linewidths=1.5,
        s=100,
        label="Ground truth",
        zorder=3,
    )
    ax[row_contours, col].scatter(
        params.young_h_gt,
        params.young_d_gt,
        marker="x",
        color="white",
        linewidths=3.0,
        s=120,
    )

    ax[row_contours, col].set_xlabel(r"$x_1 = E^h$ [kPa]")
    ax[row_contours, col].set_ylabel(r"$x_2 = E^d$ [kPa]")
    ax[row_contours, col].grid(True)

################## Plot IG ##################
ig_upper_lim = np.max(info_gains_to_plot) * 1.05
for col, info_gain in zip(range(num_columns), info_gains_to_plot):
    if col == 2:
        # Dummy bar for legend
        ax[row_ig, col].barh(
            0.0,
            0.0,
            color="grey",
            label="Single-physics",
        )
        # Actual bars
        ax[row_ig, col].barh(
            0.0,
            info_gain,
            color=colors.MULTIPHYSICS,
            label="Multi-physics",
        )
        ax[row_ig, col].barh(0.0, info_gains_to_plot[1], color="none", hatch="//")
        ax[row_ig, col].vlines(info_gains_to_plot[1], -0.4, 0.4, color="grey")
        ax[row_ig, col].set_xlabel(
            r"$\text{IG}\!\left( \boldsymbol{y}_\text{obs} \right)$ [nat]", labelpad=4.0
        )
    else:
        ax[row_ig, col].barh(0.0, info_gain, color="grey")
        ax[row_ig, col].set_xlabel(
            r"$\text{IG}\!\left( \boldsymbol{y}_\text{obs} = \boldsymbol{y}_{1, \text{obs}} \right)$ [nat]",
            labelpad=4.0,
        )
    ax[row_ig, col].set_xlim(0.0, ig_upper_lim)
    ax[row_ig, col].set_ylim(-0.6, 0.6)
    ax[row_ig, col].get_yaxis().set_visible(False)
    ax[row_ig, col].spines["top"].set_visible(False)
    ax[row_ig, col].spines["right"].set_visible(False)
    ax[row_ig, col].grid(True)
    logger.info("IG for col %d: %f", col, info_gain)
# Add IG legend
ax[row_ig, 2].legend(bbox_to_anchor=(1.15, 0.25), loc="center left")

################## Plot RIIG ##################
ax[row_riig, 0].axis("off")
ax[row_riig, 1].axis("off")
col = 2
box = ax[row_riig, col].get_position()
left_offset = info_gain_sp / ig_upper_lim
ax[row_riig, col].set_position(
    [
        box.x0 + box.width * left_offset,  # move it right
        box.y0,
        box.width * (1 - left_offset),  # make it narrower
        box.height,
    ]
)
ax[row_riig, col].barh(
    0.0,
    riig,
    color=colors.MULTIPHYSICS,
)
logger.info("RIIG for col %d: %f", col, riig)

ax[row_riig, col].set_xlim(0.0, riig * (1.0 + 0.05 / (1 - left_offset)))
ax[row_riig, col].set_ylim(-0.6, 0.6)
xlabel = r"$\text{RIIG}\!\left( \boldsymbol{y}_{1, \text{obs}}, \boldsymbol{y}_{2, \text{obs}} \right)$ [\hspace{1pt}-\hspace{1pt}]"
ax[row_riig, col].set_xlabel(xlabel, labelpad=4.0, loc="right")
ax[row_riig, col].get_yaxis().set_visible(False)
ax[row_riig, col].spines["top"].set_visible(False)
ax[row_riig, col].spines["right"].set_visible(False)
ax[row_riig, col].grid(True)

################## Add colored boxes ##################
gap = 0.01
pad = 0.01
extra_pad = 0.01
height_text = 0.03
height_boxes = 1 - 2 * pad - height_text
width_legend = 0.18
width_labels = 0.04
linewidth = 5
label_sp = r"\textbf{Single-physics BIA}"
label_mp = r"\textbf{Multi-physics BIA}"
upper_labels = [label_sp, label_mp]

# Get column positions
bbox_col1 = get_axis_bbox(ax[1, 0], fig)
bbox_col2 = get_axis_bbox(ax[1, 1], fig)
bbox_col3 = get_axis_bbox(ax[1, 2], fig)
pos_col3 = ax[1, 2].get_position()
x_center_col23 = (bbox_col2.x1 + bbox_col3.x0) / 2 + 0.002
positions = [
    (bbox_col1.x0 - pad, pad),
    (x_center_col23 + gap / 2, pad),
]
width_left_box = x_center_col23 - positions[0][0] - gap / 2
width_right_box = pos_col3.x1 - positions[1][0] + 1.5 * pad
widths = [width_left_box, width_right_box]
colors_frame = [colors.MODEL, colors.MULTIPHYSICS_LIGHT]
zorders = [-1000, -1000]
zipped = zip(upper_labels, positions, widths, colors_frame, zorders)
for upper_label, pos, width, color, zorder in zipped:
    # frame
    fig.patches.append(
        plt.Rectangle(
            pos,
            width,
            height_boxes,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            transform=fig.transFigure,
            figure=fig,
            zorder=zorder,
        )
    )
    if upper_label == label_sp:
        # Additional box inbetween single-physics axes
        fig.patches.append(
            plt.Rectangle(
                pos,
                width / 2,
                height_boxes,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                transform=fig.transFigure,
                figure=fig,
                zorder=zorder,
            )
        )
    # background for label
    fig.patches.append(
        plt.Rectangle(
            (pos[0], pos[1] + height_boxes),
            width,
            height_text,
            fill=True,
            facecolor=color,
            edgecolor=color,
            linewidth=linewidth,
            transform=fig.transFigure,
            figure=fig,
            zorder=zorder,
        )
    )
    # label
    fig.text(
        pos[0] + (width / 2),
        pos[1] + height_boxes + height_text / 2,
        upper_label,
        verticalalignment="center",
        horizontalalignment="center",
        fontsize=16,
        weight="bold",
    )

################## Add row labels ##################
pos_row1 = ax[0, 0].get_position()
pos_row2 = ax[1, 0].get_position()
pos_row3 = ax[2, 0].get_position()
pos_row4 = ax[3, 0].get_position()
pos_row5 = ax[4, 0].get_position()
pos_row6 = ax[5, 0].get_position()
# Calculate the y-coordinate that is centered between the first two rows
y_center_row12 = (pos_row1.y0 + pos_row2.y1) / 2
y_center_row13 = (pos_row1.y0 + pos_row3.y1) / 2
y_center_row3 = (pos_row3.y0 + pos_row3.y1) / 2
y_center_row4 = (pos_row4.y0 + pos_row4.y1) / 2
y_center_row5 = (pos_row5.y0 + pos_row5.y1) / 2
y_center_row6 = (pos_row6.y0 + pos_row6.y1) / 2 - 0.02
# Loop over labels
positions_y = [y_center_row12, y_center_row3, y_center_row4, y_center_row5, y_center_row6]
positions_x = [0.02, 0.02, 0.02, 0.02, 0.02]
upper_labels = [
    "First-field\nobservations",
    "Second-field\nobservations",
    "Posterior",
    "Information\ngain",
    "RIIG",
]
for pos_x, pos_y, upper_label in zip(positions_x, positions_y, upper_labels):
    fig.text(
        pos_x,  # x position
        pos_y,  # y position
        upper_label,
        rotation="vertical",
        verticalalignment="center",
        horizontalalignment="center",
        fontsize=16,
        weight="bold",
    )

# Add colorbar of posterior contours
points_last_contour = ax[row_contours, col_obs_dv].get_position().get_points()
height_last_contour = points_last_contour[1, 1] - points_last_contour[0, 1]
cbar_ax = fig.add_axes(
    [
        points_last_contour[1, 0] + 0.035,
        points_last_contour[0, 1] + 0.035,
        0.03,
        height_last_contour - 0.035,
    ]
)
cbar = fig.colorbar(contour, cax=cbar_ax)
cbar.set_label(r"Posterior $p(\boldsymbol{x} | \boldsymbol{y}_\text{obs})$")
# Add legend
ax[row_contours, 2].legend(bbox_to_anchor=(1.15, -0.14), loc="lower left")

##### Add SP column labels #####
# Calculate the centered x-coordinate of each column
x_center_col1 = positions[0][0] + width_left_box / 4
x_center_col2 = positions[0][0] + width_left_box * 3 / 4
y_center = get_axis_bbox(ax[0, 0], fig).y1 + 0.06
# Loop over labels
positions_x = [x_center_col1, x_center_col2]
upper_labels = [
    r"Model with 2 fields:",
    r"Model with 3 fields:",
]
lower_labels = [
    r"tissue and air",
    r"tissue, air, and blood",
]
color_boxes = [colors.MODEL_LIGHT, colors.MULTIPHYSICS_LIGHTER]
for pos_x, upper_label, lower_label, color_box in zip(
    positions_x, upper_labels, lower_labels, color_boxes
):
    labels_per_line = [upper_label, lower_label]
    for label_i, offset in zip(labels_per_line, [0, -0.025]):
        fig.text(
            pos_x,  # x position
            y_center + offset,  # y position
            label_i,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=14,
            weight="bold",
        )
    fig.patches.append(
        plt.Rectangle(
            (pos_x - width_left_box / 4, y_center - 0.04),
            width_left_box / 2,
            0.06,
            fill=True,
            facecolor=color_box,
            edgecolor=None,
            linewidth=linewidth,
            transform=fig.transFigure,
            figure=fig,
            zorder=zorder - 1,
        )
    )

##### Add MP column label #####
x_center_col3 = positions[1][0] + width_right_box / 2
for label_i, offset in zip(labels_per_line, [0, -0.025]):
    fig.text(
        x_center_col3,  # x position
        y_center + offset,  # y position
        label_i,
        verticalalignment="center",
        horizontalalignment="center",
        fontsize=14,
        weight="bold",
    )
fig.patches.append(
    plt.Rectangle(
        (x_center_col3 - width_right_box / 2, y_center - 0.04),
        width_right_box,
        0.06,
        fill=True,
        facecolor=colors.MULTIPHYSICS_LIGHTER,
        edgecolor=None,
        linewidth=linewidth,
        transform=fig.transFigure,
        figure=fig,
        zorder=zorder - 1,
    )
)

plt.savefig(directory / "poro_posteriors_with_observations.png", dpi=300)
