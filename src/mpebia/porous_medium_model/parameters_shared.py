"""Parameters that both demonstration examples share."""

import numpy as np


class ParametersShared:
    """Wrapper for parameters that both demonstration examples share."""

    # Ground truth input parameters
    young_h_gt = 6.0
    young_d_gt = 12.0  # -
    k_j = -2 / 3

    # Prior parameters
    truncations_Eh = np.array([0.0, np.inf])
    truncations_Ed = np.array([0.0, np.inf])
    mean_prior = [7.0, 15.0]
    std_prior = [3.0, 10.0]

    # Parameters for the grid on which to evaluate the posterior
    offset_ppf = 0.07
    num_grid_points_Eh = 50
    num_grid_points_Ed = num_grid_points_Eh

    # Observation parameters
    snr_1 = 50
    snr_2 = 50000
    seed_noise = 38
