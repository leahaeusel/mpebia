"""Class for a truncated normal distribution used as prior."""

import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from mpebia.entropies import entropy_2d
from mpebia.plotting import colors

logger = logging.getLogger(__name__)


class TruncatedNormalPrior:
    """A two-dimensional truncated normal distribution that is used as prior.

    Attributes:
        prior_1 (sp.stats.truncnorm): Truncated normal distribution for the first dimension.
        prior_2 (sp.stats.truncnorm): Truncated normal distribution for the second dimension.
        grid_points_1 (np.ndarray): Grid points in first dimension of shape(num_grid_points_1,)
        grid_points_2 (np.ndarray): Grid points in second dimension of shape(num_grid_points_2,)
        offset_ppf (float): Offset of the first and last grid point with respect to the percent
            point function.
    """

    def __init__(
        self,
        mean_prior,
        std_prior,
        truncations_1,
        truncations_2,
        grid_size,
        offset_ppf,
    ):
        """Initialize the ProbabilisticCubeModel.

        Args:
            mean_prior (list): List of mean values for the prior distributions. Has length 2.
            std_prior (list): List of standard deviations for the prior distributions. Has length 2.
            truncations_1 (list): List of truncation values for the first dimension. Has length 2.
            truncations_2 (list): List of truncation values for the second dimension. Has length 2.
            grid_size (list): List containing the number of grid points for each dimension. Has
                length 2.
            offset_ppf (float): Offset of the first and last grid point with respect to the percent
                point function.
        """
        self.offset_ppf = offset_ppf

        self.prior_1 = self.get_prior_distribution(mean_prior[0], std_prior[0], truncations_1)
        self.prior_2 = self.get_prior_distribution(mean_prior[1], std_prior[1], truncations_2)

        self.grid_points_1 = self.get_grid_points_according_to_distribution(
            grid_size[0], self.prior_1
        )
        self.grid_points_2 = self.get_grid_points_according_to_distribution(
            grid_size[1], self.prior_2
        )

        # Log grid parameters
        logger.info("Minimal grid point in dimension 1: %f", np.min(self.grid_points_1))
        logger.info("Maximal grid point in dimension 1: %f", np.max(self.grid_points_1))
        logger.info("Minimal grid point in dimension 2: %f", np.min(self.grid_points_2))
        logger.info("Maximal grid point in dimension 2: %f\n", np.max(self.grid_points_2))

    @staticmethod
    def get_prior_distribution(mean, std, truncations):
        """Get a truncated normal distribution object.

        Args:
            mean (float): Mean of the distribution.
            std (float): Standard deviation of the distribution.
            truncations (list): List of truncation values. Has length 2.

        Returns:
            sp.stats.truncnorm: A truncated normal distribution object.
        """
        a, b = (truncations - mean) / std
        prior = sp.stats.truncnorm(a, b, loc=mean, scale=std)
        return prior

    def get_grid_points_according_to_distribution(self, num_grid_points, distribution):
        """Get grid points spaced according to ppf of a distribution.

        Args:
            num_grid_points (int): Number of grid points to generate.
            distribution (np.stats.truncnorm): A truncated normal distribution object.

        Returns:
            np.ndarray: Array of grid points.
        """
        equally_spaced_ppf = np.linspace(self.offset_ppf, 1 - self.offset_ppf, num_grid_points)
        grid_points = distribution.ppf(equally_spaced_ppf)
        return grid_points

    def get_log_prior_on_grid(self):
        """Get the log prior on the grid.

        Returns:
            np.ndarray: Array of log prior values on the grid.
        """
        log_prior_1 = self.prior_1.logpdf(self.grid_points_1)
        log_prior_2 = self.prior_2.logpdf(self.grid_points_2)

        log_prior_grid = np.zeros((len(self.grid_points_1), len(self.grid_points_2)))
        for i_1, _ in enumerate(self.grid_points_1):
            for i_2, _ in enumerate(self.grid_points_2):
                log_prior_grid[i_1, i_2] = log_prior_1[i_1] + log_prior_2[i_2]

        prior_grid = np.exp(log_prior_grid)
        entropy_prior = entropy_2d(prior_grid, self.grid_points_1, self.grid_points_2)
        logger.info("Entropy of prior: %f\n", entropy_prior)

        return log_prior_grid

    def plot_prior_and_grid_points(
        self, prior_grid, true_params, directory, scale_dim1=1.0, scale_dim2=1.0
    ):
        """Plot and save the prior distribution and the grid points.

        Args:
            prior_grid (np.ndarray): Array of prior values on the grid.
            true_params (list): List of true parameter values. Has length 2.
            directory (Path): Directory to save the plot.
            scale_dim1 (float, opt): Scaling factor for the first dimension in the plot.
            scale_dim2 (float, opt): Scaling factor for the second dimension in the plot.
        """
        fig, ax = plt.subplots()
        contour = ax.contourf(
            self.grid_points_1 * scale_dim1,
            self.grid_points_2,
            prior_grid.T,
            cmap=colors.CMAP,
        )
        fig.colorbar(contour)
        mesh1, mesh2 = np.meshgrid(
            self.grid_points_1 * scale_dim1,
            self.grid_points_2,
        )
        ax.scatter(
            mesh1.flatten(),
            mesh2.flatten(),
            color="#000000",
            s=1,
            label="Evaluation points",
        )
        ax.scatter(
            true_params[0] * scale_dim1,
            true_params[1],
            marker="x",
            color=colors.GROUND_TRUTH,
            s=100,
            label="Ground truth",
        )
        ax.set_xlabel(r"input $x_1 \times %.2f$" % scale_dim1)
        ax.set_ylabel(r"input $x_2 \times %.2f$" % scale_dim2)
        ax.legend(loc="lower right")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(
            directory / "prior.png",
            dpi=300,
        )
