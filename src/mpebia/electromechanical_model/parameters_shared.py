"""Parameters that both demonstration examples share."""

import numpy as np


class ParametersShared:
    """Wrapper for parameters that both demonstration examples share."""

    # Parameters of the electromechanical cube model
    l0 = 0.01  # m
    U = 10  # V
    rho = 1.0  # Ohm*m

    # Ground truth input parameters
    E_gt = 1.1e4  # Pa,
    nu_gt = 0.35  # -

    # Prior parameters
    truncations_E = np.array([0.0, np.inf])
    truncations_nu = np.array([0.0, 0.5])
    mean_prior = [1.0e4, 0.3]
    std_prior = [2.0e3, 0.15]

    # Parameters for the grid on which to evaluate the posterior
    offset_ppf = 0.00001
    num_grid_points_E = 100
    num_grid_points_nu = num_grid_points_E

    # Parameters for the points of observation
    F_max = 0.4  # N
    num_obs_d = 16
    snr_d = 50
    seed = 42

    # Parameters for the electric current observations
    nums_obs_I = [2, int(2**8)]
    snrs_I = [int(1.2e4), 80]
