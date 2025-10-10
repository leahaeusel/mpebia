"""Plot the true solution fields with the locations of the observations."""

import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.gridspec import GridSpec

import mpebia.plotting.rc_params  # pylint: disable=unused-import
from mpebia.output import get_directory

# SETUP

directory = get_directory(__file__)
input_dir = (directory / "..").resolve()
results_dir = input_dir / "1_ground_truth"
fourc_input_file = results_dir / "4C_poro_input_with_blood.yaml"


# READ DATA

with open(results_dir / "mpebia_poro_ground_truth_full.pickle", "rb") as f:
    data = pickle.load(f)
with open(results_dir / "mpebia_poro_ground_truth_at_dline.pickle", "rb") as f:
    data_obs = pickle.load(f)

# Read output data
data_d = data["output"]["result"][0][0]  # shape (num_nodes, num_dims_data_point)
data_v = data["output"]["result"][0][1]
data_obs_d = data_obs["output"]["result"][0][0]
data_obs_v = data_obs["output"]["result"][0][1]
d_magnitude = np.linalg.norm(data_d, axis=1)

# Read geometric data
node_coordinates = data["coords"]
node_coordinates_obs = data_obs["coords"]
x = node_coordinates[:, 0]
y = node_coordinates[:, 1]
x_obs_d = node_coordinates_obs[0][:, 0]
y_obs_d = node_coordinates_obs[0][:, 1]
x_obs_v = node_coordinates_obs[1][:, 0]
y_obs_v = node_coordinates_obs[1][:, 1]


# PLOTTING

rc("axes", labelpad=8.0)
fig = plt.figure(figsize=(11.9, 4.2))
gs = GridSpec(
    1,
    2,
    wspace=0.7,
    left=0.06,
    right=0.87,
    top=1.01,
)

# Plot magnitude of displacement over domain
for i, (x_grid, y_grid, data_grid, x_obs, y_obs, label) in enumerate(
    zip(
        [x, x],
        [y, y],
        [d_magnitude, data_v],
        [x_obs_d, x_obs_v],
        [y_obs_d, y_obs_v],
        [
            r"Tissue displacement norm $\| \bm{d}^t \|$ [mm]",
            r"Blood volume fraction $\varepsilon^b$ [-]",
        ],
    )
):
    ax_contour = fig.add_subplot(gs[i])

    # Plot observed field
    im = ax_contour.tripcolor(x_grid, y_grid, data_grid, shading="gouraud")

    # Plot observations
    ax_contour.scatter(x_obs, y_obs, c="#ffffff", s=60, linewidths=2.5, marker="x")
    ax_contour.scatter(x_obs, y_obs, c="#000000", s=50, linewidths=1.5, marker="x", zorder=10)

    for i, (x, y) in enumerate(zip(x_obs, y_obs)):
        ax_contour.annotate(
            f"{i+1}",
            (x - 0.3, y + 0.3),
            ha="center",
            fontsize=12,
        )

    ax_contour.set_xlabel(r"$c_1$ [mm]")
    ax_contour.set_ylabel(r"$c_2$ [mm]")
    ax_contour.set_aspect("equal", "box")

    # Add colorbar
    bbox_ax = ax_contour.get_position()
    height_riig = bbox_ax.y1 - bbox_ax.y0
    cbar_ax = fig.add_axes([bbox_ax.x1 + 0.02, bbox_ax.y0, 0.03, height_riig])
    fig.colorbar(im, cax=cbar_ax, label=label)

plt.savefig(directory / f"observations_locations.png", dpi=300)
