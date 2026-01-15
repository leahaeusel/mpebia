"""Utility functions for entropy, KL divergence, and information gain."""

import numpy as np


def entropy_2d(p, x1, x2):
    """Numerical entropy of a two-dimensional probability distribution.

    Args:
        p (np.ndarray): Probability distribution evaluated on x1 and x2.
        x1 (float): Points at which p was evaluated in first dimension.
        x2 (float): Points at which p was evaluated in second dimension.

    Returns:
        float: Numerical entropy of the probability distribution.
    """
    # Avoid computing np.log(0.0) because mathematically,
    # limit for x->0 of log(x)*x is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(p == 0.0, 0.0, -np.log(p) * p)
    entropy_numerical = trapezoid_2d(integrand, x1, x2)
    return entropy_numerical


def kld_2d(p, q, x1, x2):
    """Numerical Kullback-Leibler divergence between two probability distributions: DKL(p||q).

    Args:
        p (np.ndarray): First probability distribution evaluated on x.
        q (np.ndarray): Second probability distribution evaluated on x.
        x1 (float): Points at which all PDFs were evaluated in first dimension.
        x2 (float): Points at which all PDFs were evaluated in second dimension.

    Returns:
        float: Numerical Kullback-Leibler divergence between the two probability distributions.
    """
    # Ensure that if q is 0, p is 0 too
    if np.any(np.logical_and(q == 0.0, p > 0)):
        raise ValueError("KLD is infinite.")

    # Avoid computing np.log(0.0) because mathematically,
    # limit for x->0 of log(x)*x is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(p == 0.0, 0.0, np.log(p / q) * p)

    kld_numerical = trapezoid_2d(integrand, x1, x2)
    return kld_numerical


def trapezoid_2d_constant_dx(x, dx):
    """Integrates the first two dimensions of x based on the trapezoidal rule.

    Args:
        x (np.ndarray): Array to integrate.
        dx (list): Constant distances between x values.

    Returns:
        float: Integral of the first two dimensions of x.
    """
    integral = np.trapz(np.trapz(x, dx=dx[0], axis=0), dx=dx[1], axis=0)
    return integral


def trapezoid_2d(x, grid_points_first_dim, grid_points_second_dim):
    """Integrates the first two dimensions of x based on the trapezoidal rule.

    Args:
        x (np.ndarray): Array to integrate.
        grid_points_first_dim (list): Grid points on which x was evaluated in the first dimension.
        grid_points_second_dim (list): Grid points on which x was evaluated in the second dimension.

    Returns:
        float: Integral of the first two dimensions of x.
    """
    integral = np.trapz(np.trapz(x, grid_points_first_dim, axis=0), grid_points_second_dim, axis=0)
    return integral


def information_gain_2d(
    log_likelihood_grid,
    log_prior_grid,
    grid_points_first_dim,
    grid_points_second_dim,
    return_posterior=False,
):
    """Compute the information gain employing the log-sum-exp trick.

    Args:
        log_likelihood_grid (np.ndarray): Log-likelihood values on the grid.
        log_prior_grid (np.ndarray): Log-prior values on the grid.
        grid_points_first_dim (list): Grid points on which the log-likelihood and log-prior were
            evaluated in the first dimension.
        grid_points_second_dim (list): Grid points on which the log-likelihood and log-prior were
            evaluated in the second dimension.
        return_posterior (bool, optional): Whether to return the posterior distribution as well.
            Default is False.

    Returns:
        float: Information gain.
        np.ndarray (optional): Posterior distribution on the grid, returned if return_posterior is True.
    """
    log_like_plus_log_prior = log_likelihood_grid + log_prior_grid
    c = np.max(log_like_plus_log_prior)
    likelihood_times_prior_c = np.exp(log_like_plus_log_prior - c)
    evidence_c = trapezoid_2d(
        likelihood_times_prior_c, grid_points_first_dim, grid_points_second_dim
    )

    posterior = likelihood_times_prior_c / evidence_c

    # Avoid computing np.log(0.0) because mathematically,
    # limit for x->0 of log(x)*x is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(
            posterior == 0.0, 0.0, posterior * (np.log(posterior) - log_prior_grid)
        )

    info_gain = trapezoid_2d(integrand, grid_points_first_dim, grid_points_second_dim)

    if return_posterior:
        return info_gain, posterior

    return info_gain


def relative_increase_in_information_gain(single_physics_info_gain, multi_physics_info_gain):
    """Compute the relative increase in information gain.

    Args:
        single_physics_info_gain (float): Information gain from single-physics data.
        multi_physics_info_gain (float): Information gain from multi-physics data.

    Returns:
        float: Relative increase in information gain.
    """
    return (multi_physics_info_gain - single_physics_info_gain) / single_physics_info_gain
