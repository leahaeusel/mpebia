"""Plot the multi-physics information gain on a grid."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

import mpebia.plotting.rc_params  # pylint: disable=unused-import
from mpebia.electromechanical_model.parameters_riig_grid import ParametersRIIGGrid
from mpebia.electromechanical_model.parameters_shared import ParametersShared
from mpebia.output import get_directory
from mpebia.plotting import colors
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
log_likelihood_grid_d = data["log_likelihood_grid_d"]
log_prior_grid = data["log_prior_grid"]

data_indicated_posteriors = np.load(directory / "data_indicated_posteriors.npz", allow_pickle=True)
nums_obs_plot_post = data_indicated_posteriors["nums_obs_plot_post"]
snrs_plot_post = data_indicated_posteriors["snrs_plot_post"]
posteriors_to_plot = data_indicated_posteriors["posteriors_to_plot"]

# PLOTTING

fontsize_marker = 14

fig = plt.figure(figsize=(11.9, 6.4))
gs = GridSpec(
    5,
    5,
    height_ratios=[0.08, 0.73, 0.65, 0.8, 0.3],
    width_ratios=[2.25, 0.6, 0.85, 0.6, 0.85],
    wspace=0.0,
    hspace=0.0,
    left=0.05,
    right=0.99,
    top=0.895,
    bottom=0.0,
)

######### PLOT RIIG GRID #########

# Plot RIIG grid
ax_riig = fig.add_subplot(gs[1:4, 0])
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
cbar_ax = fig.add_axes([points_riig[0, 0], points_riig[1, 1] + 0.02, width_riig, 0.04])
fig.colorbar(
    contour_ig,
    cax=cbar_ax,
    label=r"Relative increase in information gain $\text{RIIG}\!\left( \boldsymbol{y}_{1, \text{obs}}, \boldsymbol{y}_{2, \text{obs}} \right)$ [\hspace{1pt}-\hspace{1pt}]",
    location="top",
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

######### PLOT POSTERIORS #########
axes_post = [fig.add_subplot(gs[0:2, 2]), fig.add_subplot(gs[3:4, 4]), fig.add_subplot(gs[0:2, 4])]
for i, (ax_post, posterior) in enumerate(zip(axes_post, posteriors_to_plot)):
    col = i + 1
    row_points = 0
    row_obs = 1
    row_post = 2

    # Add posteriors
    contour = ax_post.contourf(
        prior.grid_points_1 / 1000,
        prior.grid_points_2,
        posterior.T,
        6,
        cmap=colors.CMAP,
    )
    ax_post.set_xlabel(r"$x_1 = E$ [kPa]")
    ax_post.set_ylabel(r"$x_2 = \nu$ [\hspace{1pt}-\hspace{1pt}]")
    ax_post.grid(True)
    ax_post.set_xlim(9, 13)
    ax_post.set_ylim(0.15, np.max(prior.grid_points_2))
    ax_post.set_xticks(np.arange(9, 14, 1))

    # Add numbers
    pos = np.array([0.5, 1.12])
    ax_post.scatter(
        pos[0],
        pos[1],
        color="k",
        s=250,
        marker="o",
        zorder=10,
        transform=ax_post.transAxes,
        clip_on=False,
    )
    ax_post.scatter(
        pos[0],
        pos[1],
        color="w",
        s=200,
        marker="o",
        zorder=11,
        transform=ax_post.transAxes,
        clip_on=False,
    )
    ax_post.annotate(
        r"$\bf{" + str(i + 1) + r"}$",
        (11, 0.5),
        xytext=pos + (0.005, -0.007),
        color="k",
        fontsize=fontsize_marker,
        ha="center",
        va="center",
        zorder=12,
        textcoords=ax_post.transAxes,
    )

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

# Add arrows to posterior plots
# - horizontal arrow
ax_arrow1 = fig.add_subplot(gs[0:2, 3])
ax_arrow1.axis("off")
ax_arrow1.annotate(
    "",
    xy=(1, 0.5),
    xytext=(0, 0.5),
    arrowprops=arrowprops,
)
ax_arrow1.set_xlim(-0.15, 2.15)
# - vertical arrow
ax_arrow2 = fig.add_subplot(gs[2, 4])
ax_arrow2.axis("off")
ax_arrow2.annotate(
    "",
    xy=(0.5, 1),
    xytext=(0.5, 0),
    arrowprops=arrowprops,
)
ax_arrow2.set_ylim(-0.8, 2.0)

# Add posteriors title
axes_post[0].annotate(
    r"Posterior densities $p(\boldsymbol{x} | \boldsymbol{y}_{1, \text{obs}}, \boldsymbol{y}_{2, \text{obs}})$ at indicated $N_{\text{obs}, 2}$"
    r" and $\text{SNR}_2$:",
    xy=(-0.0, 1.35),
    fontsize=12,
    ha="left",
    va="top",
    xycoords=axes_post[0].transAxes,
    textcoords=axes_post[0].transAxes,
)


plt.savefig(directory / "riig_grid_with_posteriors.png")


print(np.max(relative_information_gain_grid))
