"""Plot the multi-physics information gain on a grid."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import mpebia.plotting.rc_params  # pylint: disable=unused-import
from mpebia.electromechanical_model.parameters_riig_grid import ParametersRIIGGrid
from mpebia.electromechanical_model.parameters_shared import ParametersShared
from mpebia.output import get_directory
from mpebia.plotting import colors
from mpebia.plotting.positioning import get_axis_bbox
from mpebia.plotting.style import arrowprops
from mpebia.truncated_gaussian_prior import TruncatedNormalPrior

# SETUP

directory = get_directory(__file__)

params = ParametersShared()
params_riig = ParametersRIIGGrid()

prior = TruncatedNormalPrior(
    params.mean_prior,
    params.std_prior,
    params.truncations_E,
    params.truncations_nu,
    (params.num_grid_points_E, params.num_grid_points_nu),
    params.offset_ppf,
)

# Load data from npz file
data = np.load(directory / "data.npz", allow_pickle=True)
nums_obs_I = data["nums_obs_I"]
snrs_I = data["snrs_I"]
posteriors_mp_grid = data["posteriors_mp_grid"]
information_gain_sp = data["information_gain_sp"]
relative_information_gain_grid = data["relative_increase_information_gain_grid"]
eval_points = data["eval_points"]
F_gt = data["F_gt"]
d_gt = data["d_gt"]
I_gt = data["I_gt"]
F_obs_d = data["F_obs_d"]
F_obs_I_grid = data["F_obs_I_grid"]
d_obs = data["d_obs"]
I_obs_grid = data["I_obs_grid"]
all_noise = data["all_noise"]
log_prior_grid = data["log_prior_grid"]

data_indicated_posteriors = np.load(directory / "data_indicated_posteriors.npz", allow_pickle=True)
nums_obs_plot_post = data_indicated_posteriors["nums_obs_plot_post"]
snrs_plot_post = data_indicated_posteriors["snrs_plot_post"]
log_likelihood_d_to_plot = data_indicated_posteriors["log_likelihood_d_to_plot"]
log_likelihood_I_to_plot = data_indicated_posteriors["log_likelihood_I_to_plot"]
posteriors_to_plot = data_indicated_posteriors["posteriors_to_plot"]

likelihood_d_to_plot = np.exp(
    log_likelihood_d_to_plot
    - np.max(log_likelihood_d_to_plot, axis=(1, 2))[:, np.newaxis, np.newaxis]
)
likelihood_I_to_plot = np.exp(
    log_likelihood_I_to_plot
    - np.max(log_likelihood_I_to_plot, axis=(1, 2))[:, np.newaxis, np.newaxis]
)

# PLOTTING

fontsize_marker = 14

fig = plt.figure(figsize=(11.9, 14.5))
num_rows = 7
num_cols = 7
width_ratio_gap = 0.53
gs = GridSpec(
    num_rows,
    num_cols,
    height_ratios=[1, 0.35, 0.4, 0.2, 0.4, 0.2, 0.4],
    width_ratios=[1, width_ratio_gap, 1, width_ratio_gap, 1, width_ratio_gap, 1],
    wspace=0.0,
    hspace=0.0,
    left=0.14,
    right=0.978,
    top=0.934,
    bottom=0.05,
)

######### PLOT RIIG GRID #########

# Plot RIIG grid
row_riig = 0
col_riig_start = 0
col_riig_end = num_cols - 1
ax_riig = fig.add_subplot(gs[row_riig, col_riig_start:col_riig_end])
contour_ig = ax_riig.contourf(
    nums_obs_I,
    snrs_I,
    relative_information_gain_grid.T,
    12,
    cmap=colors.CMAP,
)
ax_riig.set_xlabel(
    r"Number of electric current observations $N_{\text{obs}, 2}$ [\hspace{1pt}-\hspace{1pt}]"
)
ax_riig.set_ylabel(
    r"Signal-to-noise ratio of electric current $\text{SNR}_2$ [\hspace{1pt}-\hspace{1pt}]"
)
ax_riig.grid(True)
ax_riig.set_xscale("log")
ax_riig.set_yscale("log")
ax_riig.set_aspect("equal", "box")

# Add colorbar
points_riig = ax_riig.get_position().get_points()
width_riig = points_riig[1, 0] - points_riig[0, 0]
height_cbar_riig = width_riig / 24
cbar_ax = fig.add_axes([points_riig[0, 0], points_riig[1, 1] + 0.01, width_riig, height_cbar_riig])
cbar = fig.colorbar(
    contour_ig,
    cax=cbar_ax,
    location="top",
)
cbar.set_label(
    r"\textbf{Relative increase in information gain} $\text{RIIG}\!\left( \boldsymbol{y}_{1, \text{obs}}, \boldsymbol{y}_{2, \text{obs}} \right)$ [\hspace{1pt}-\hspace{1pt}]",
    fontsize=16,
    labelpad=12,
)
# Indicate points where posterior is plotted on the right
for i, (num_obs, snr) in enumerate(zip(nums_obs_plot_post, snrs_plot_post)):
    ax_riig.scatter(num_obs, snr, color="k", s=250, marker="o", zorder=10)
    ax_riig.scatter(num_obs, snr, color="w", s=200, marker="o", zorder=11)
    ax_riig.annotate(
        r"$\bf{" + str(i + 1) + r"}$",
        (num_obs * 1.025, snr * 0.97),
        color="k",
        fontsize=fontsize_marker,
        ha="center",
        va="center",
        zorder=12,
    )

######### PLOT PRIORS, LIKELIHOODS, AND POSTERIORS #########
col_prior = 0
col_liked = 2
col_likeI = 4
col_post = 6
cols = [col_prior, col_liked, col_likeI, col_post]
rows = [2, 4, 6]
distrs_to_plot = [
    3 * [np.exp(log_prior_grid)],
    likelihood_d_to_plot,
    likelihood_I_to_plot,
    posteriors_to_plot,
]
axes_distr = np.empty((len(rows), len(cols)), dtype=object)
for j, (col, distrs) in enumerate(zip(cols, distrs_to_plot)):
    for i, (row, distr) in enumerate(zip(rows, distrs)):
        ax_distr = fig.add_subplot(gs[row, col])
        axes_distr[i, j] = ax_distr

        # Add contour plot of the distribution
        contour = ax_distr.contourf(
            prior.grid_points_1 / 1000,
            prior.grid_points_2,
            distr.T,
            6,
            cmap=colors.CMAP,
        )
        ax_distr.set_xlabel(r"$x_1 = E$ [kPa]")
        ax_distr.set_ylabel(r"$x_2 = \nu$ [\hspace{1pt}-\hspace{1pt}]")
        ax_distr.grid(True)
        ax_distr.set_xlim(9, 13)
        ax_distr.set_ylim(0.15, np.max(prior.grid_points_2))
        ax_distr.set_xticks(np.arange(9, 14, 1))

        # Add numbers
        if col == col_prior:
            pos_number = np.array([-0.55, 0.36])
            ax_distr.scatter(
                pos_number[0],
                pos_number[1],
                color="k",
                s=250,
                marker="o",
                zorder=10,
                transform=ax_distr.transAxes,
                clip_on=False,
            )
            ax_distr.scatter(
                pos_number[0],
                pos_number[1],
                color="w",
                s=200,
                marker="o",
                zorder=11,
                transform=ax_distr.transAxes,
                clip_on=False,
            )
            ax_distr.annotate(
                r"$\bf{" + str(i + 1) + r"}$",
                (11, 0.5),
                xytext=pos_number + (0.005, -0.007),
                color="k",
                fontsize=fontsize_marker,
                ha="center",
                va="center",
                zorder=12,
                textcoords=ax_distr.transAxes,
            )

        # Add column titles
        if row == rows[0]:
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
            ax_distr.set_title(title, fontsize=16, pad=30)

######### ADD ARROWS #########

# Add arrows to RIIG plot
ax_riig.annotate(
    "",
    xy=(nums_obs_plot_post[2] * 0.7, snrs_plot_post[2]),
    xytext=(nums_obs_plot_post[0] * 1.5, snrs_plot_post[0]),
    arrowprops=arrowprops,
)
ax_riig.annotate(
    "",
    xy=(nums_obs_plot_post[2], snrs_plot_post[2] * 0.7),
    xytext=(nums_obs_plot_post[1], snrs_plot_post[1] * 1.5),
    arrowprops=arrowprops,
)

######## ADD COLORED BOXES ########

pad = 0.01
width_primary_label = 0.03
width_secondary_label = 0.037
linewidth = 5

# Get column positions
pos_row0col0 = get_axis_bbox(axes_distr[0, 0], fig)
pos_row1col0 = get_axis_bbox(axes_distr[1, 0], fig)
pos_row2col0 = get_axis_bbox(axes_distr[2, 0], fig)
pos_row0col3 = get_axis_bbox(axes_distr[0, 3], fig)
pos_row2col3 = get_axis_bbox(axes_distr[2, 3], fig)
y_center_rows01 = (pos_row0col0.y0 + pos_row1col0.y1) / 2
y_center_rows12 = (pos_row1col0.y0 + pos_row2col0.y1) / 2
offset_box_left = pad + width_secondary_label
width_box = pos_row0col3.x1 - pos_row0col0.x0 + pad + offset_box_left
height_box = y_center_rows12 - y_center_rows01

positions = [
    (pos_row0col0.x0 - offset_box_left, y_center_rows01 - height_box),
    (pos_row0col0.x0 - offset_box_left, y_center_rows01),
    (pos_row0col0.x0 - offset_box_left, y_center_rows12),
]
color_frame = colors.MULTIPHYSICS_LIGHT
zorder = -1000

for i, pos in enumerate(positions):
    # frame
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
    positions[1][0] + (width_primary_label / 2),
    positions[1][1] + height_box / 2,
    r"\textbf{Multi-physics BIA}",
    verticalalignment="center",
    horizontalalignment="center",
    fontsize=16,
    weight="bold",
    rotation="vertical",
)

plt.savefig(directory / "riig_grid_with_posteriors.png")


print(f"Max. RIIG: {np.max(relative_information_gain_grid)}")
