"""Evaluate and plot information gains for different 2D distributions."""

import matplotlib.pyplot as plt
import numpy as np
from queens.distributions import Normal

import mpebia.plotting.rc_params  # pylint: disable=unused-import
from mpebia.entropies import kld_2d, trapezoid_2d_constant_dx
from mpebia.logging import get_logger
from mpebia.output import get_directory
from mpebia.plotting.colors import CMAP

# SETUP

logger = get_logger(__file__)
directory = get_directory(__file__)

mean_prior = np.array([0.5, 0.5])
mean_ig1 = np.array([0.5, 0.725])
mean_ig2 = np.array([0.5, 0.8])
variance_prior = 0.045
variance_ig1_shifted = 0.01486
variance_ig1 = 0.00714
variance_ig2 = 0.00236
var_mix_ig1 = 0.0054
var_mix_ig2 = 0.0014
covariance_ig1 = np.ones((2, 2)) * (variance_prior - 0.00315)
covariance_ig2 = np.ones((2, 2)) * (variance_prior - 0.000414)
np.fill_diagonal(covariance_ig1, variance_prior)
np.fill_diagonal(covariance_ig2, variance_prior)

distributions = [
    Normal(mean=mean_prior, covariance=np.eye(2) * variance_prior),
    Normal(mean=mean_ig1, covariance=np.eye(2) * variance_ig1_shifted),
    Normal(mean=mean_prior, covariance=np.eye(2) * variance_ig1),
    Normal(mean=mean_prior, covariance=covariance_ig1),
    Normal(mean=mean_ig2, covariance=np.eye(2) * variance_ig1),
    Normal(mean=mean_prior, covariance=np.eye(2) * variance_ig2),
    Normal(mean=mean_prior, covariance=covariance_ig2),
]

# Generate [0,1]x[0,1] grid for evaluation
grid_size = 1000
x1 = np.linspace(-0.8, 1.8, grid_size)
x2 = np.linspace(-0.8, 1.8, grid_size + 1)
x1_grid, x2_grid = np.meshgrid(x1, x2)
pos = np.dstack((x1_grid, x2_grid))

dx1 = x1[1] - x1[0]
dx2 = x2[1] - x2[0]
dx = [dx1, dx2]


# COMPUTATIONS

logpdf_grids = []
info_gains = []
integrals = []
posteriors = []
for i, distr in enumerate(distributions):
    logpdf = distr.logpdf(pos)
    logpdf_grids.append(logpdf.reshape(x1_grid.shape))

    prior = np.exp(logpdf_grids[0])
    posterior = np.exp(logpdf_grids[-1])
    info_gain = kld_2d(posterior.T, prior.T, x1, x2)

    info_gains.append(info_gain)
    integrals.append(trapezoid_2d_constant_dx(posterior, dx))
    posteriors.append(posterior)

[logger.info("Integral logpdf #%d: %f", i, integral) for i, integral in enumerate(integrals)]
[logger.info("Information gain #%d: %f", i, info_gain) for i, info_gain in enumerate(info_gains)]

# PLOTTING

# IG arrow with multiple posteriors

num_columns = 3
num_rows = 4
fig_height_1 = 5.0
fig, ax = plt.subplots(
    num_rows,
    num_columns,
    figsize=(9, fig_height_1),
    gridspec_kw={"height_ratios": [1, 1, 1, 0.3]},
    dpi=600,
)
xlabelpad = -4.0
ylabelpad = 2.0
aspect = 8

# Plot contours
is_axis_per_posterior = [
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 1),
    (0, 2),
    (1, 2),
    (2, 2),
]
for i, (i_ax, posterior) in enumerate(zip(is_axis_per_posterior, posteriors, strict=True)):
    contour = ax[i_ax].contourf(x1_grid, x2_grid, posterior, cmap=CMAP)
    if i == 0:
        # This is the uniform prior
        cbar = fig.colorbar(contour, label=r"prior \ $p(\bm{x})$", aspect=aspect)
    else:
        cbar = fig.colorbar(contour, aspect=aspect)
        cbar.set_label(r"$p(\bm{x}|\bm{y}_\text{obs})$", labelpad=ylabelpad)
    ticks = cbar.locator.locs
    cbar.locator.locs = [ticks[0], ticks[-1]]
    ax[i_ax].set_xlim(0.0, 1.0)
    ax[i_ax].set_ylim(0.0, 1.0)
    ax[i_ax].grid(True)

# Set axis labels
for i in range(num_rows):
    for j in range(num_columns):
        if ax[i, j].has_data():
            ax[i, j].set_xlabel(r"$x_1$", labelpad=xlabelpad)
            ax[i, j].set_ylabel(r"$x_2$", labelpad=ylabelpad)
            ax[i, j].set_aspect("equal", "box")
            xticks = ax[i, j].get_xticks()
            yticks = ax[i, j].get_xticks()
            ax[i, j].set_xticks(
                [xticks[0], xticks[-1]],
            )
            ax[i, j].set_yticks(
                [yticks[0], yticks[-1]],
            )
        else:
            ax[i, j].axis("off")

plt.tight_layout()
fig.subplots_adjust(left=-0.0, right=0.84, top=0.98, bottom=-0.02)

# Add arrow
height_arrow = 0.055
annotation = plt.annotate(
    text=r"$\text{IG}\!\left(\bm{y}_\text{obs}\right)$ [nats]",
    xy=(0.08, height_arrow),
    xytext=(0.86, height_arrow - 0.0095),
    xycoords="figure fraction",
    arrowprops=dict(arrowstyle="<-", lw=2),
)
annotation.set_fontsize("large")

# Add lines to arrow
for i in range(num_columns):
    # Get column center
    bbox = ax[1, i].get_position()
    x_center = bbox.x0 + (bbox.x1 - bbox.x0) / 2
    annotation = plt.annotate(
        text=r"$\boldsymbol{" + str(i) + r"}$",
        xy=(x_center, height_arrow + 0.015),
        xycoords="figure fraction",
        xytext=(x_center, height_arrow - 0.049),
        # xytext=(-3, -25),
        # textcoords="offset points",
        arrowprops=dict(arrowstyle="-", lw=2),
        ha="center",
    )
    annotation.set_fontsize("large")

plt.draw()  # Needed to redraw colorbars with altered ticks
plt.savefig(directory / "information_gains.png")
plt.close()


# IG arrow with RIIG

num_rows = 2
num_columns = 3
fig_height_2 = 2.3
fig, ax = plt.subplots(
    num_rows,
    num_columns,
    figsize=(9, fig_height_2),
    gridspec_kw={"height_ratios": [1, 0.7]},
    dpi=600,
)

# Plot contours
for i, posterior in enumerate([posteriors[i] for i in [0, 1, 4]]):
    contour = ax[0, i].contourf(x1_grid, x2_grid, posterior, cmap=CMAP)
    cbar = fig.colorbar(contour, aspect=aspect)
    ticks = cbar.locator.locs
    cbar.locator.locs = [ticks[0], ticks[-1]]
    ax[0, i].set_xlim(0.0, 1.0)
    ax[0, i].set_ylim(0.0, 1.0)
    ax[0, i].grid(True)

for j in range(num_columns):
    ax[0, j].set_xlabel(r"$x_1$", labelpad=xlabelpad)
    ax[0, j].set_ylabel(r"$x_2$", labelpad=ylabelpad)
    ax[0, j].set_aspect("equal", "box")
    xticks = ax[0, j].get_xticks()
    yticks = ax[0, j].get_xticks()
    ax[0, j].set_xticks(
        [xticks[0], xticks[-1]],
    )
    ax[0, j].set_yticks(
        [yticks[0], yticks[-1]],
    )

for i in range(num_columns):
    ax[1, i].axis("off")

plt.tight_layout()
fig.subplots_adjust(left=0.04, right=0.78, top=0.96, bottom=-0.02)

# Add arrow for IG
height_arrow = 0.3
size_factor = fig_height_1 / fig_height_2

# Add lines to arrow
for i in range(num_columns):
    # Get column center
    bbox = ax[0, i].get_position()
    x_center = bbox.x0 + (bbox.x1 - bbox.x0) / 2
    annotation = plt.annotate(
        text=r"$\boldsymbol{" + str(i) + r"}$",
        xy=(x_center, height_arrow + 0.015 * size_factor),
        xycoords="figure fraction",
        xytext=(x_center, height_arrow - 0.049 * size_factor),
        arrowprops=dict(arrowstyle="-", lw=2),
        ha="center",
    )
    annotation.set_fontsize("large")

arrow_end = bbox.x1 + 0.06

annotation = plt.annotate(
    text=r"$\text{IG}\!\left(\bm{y}_\text{obs}\right)$ [nats]",
    xy=(0.08, height_arrow),
    xytext=(arrow_end, height_arrow - 0.0095 * size_factor),
    xycoords="figure fraction",
    arrowprops=dict(arrowstyle="<-", lw=2),
)
annotation.set_fontsize("large")

# Add arrow for RIIG
height_arrow = 0.11
annotation = plt.annotate(
    text=r"$\text{RIIG}\!\left(\bm{y}_{1, \text{obs}}, \bm{y}_{2, \text{obs}}\right)$ [\hspace{1pt}-\hspace{1pt}]",
    xy=(0.4, height_arrow),
    xytext=(arrow_end, height_arrow - 0.0095 * size_factor),
    xycoords="figure fraction",
    arrowprops=dict(arrowstyle="<-", lw=2),
)
annotation.set_fontsize("large")

# Add lines to arrow
for i in range(1, num_columns):
    # Get column center
    bbox = ax[0, i].get_position()
    x_center = bbox.x0 + (bbox.x1 - bbox.x0) / 2
    annotation = plt.annotate(
        text=r"$\boldsymbol{" + str(i - 1) + r"}$",
        xy=(x_center, height_arrow + 0.015 * size_factor),
        xycoords="figure fraction",
        xytext=(x_center, height_arrow - 0.049 * size_factor),
        arrowprops=dict(arrowstyle="-", lw=2),
        ha="center",
    )
    annotation.set_fontsize("large")

plt.savefig(directory / "ig_and_riig.png")
