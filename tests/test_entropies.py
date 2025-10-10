"""Test entropies."""

import numpy as np
import scipy as sp

from mpebia.entropies import (
    entropy_2d,
    information_gain_2d,
    kld_2d,
    trapezoid_2d,
    trapezoid_2d_constant_dx,
)


def entropy_constant_dx(p, dx):
    """Numerical entropy of a probability distribution.

    Args:
        p (np.ndarray): Probability distribution evaluated on x.
        dx (float): Constant distance between x values.

    Returns:
        float: Numerical entropy of the probability distribution.
    """
    entropy_numerical = np.trapz(-np.log(p) * p, dx=dx)
    return entropy_numerical


def entropy_gaussian(variance):
    """Analytical entropy of a Gaussian distribution.

    Args:
        variance (float): Variance of the Gaussian distribution.

    Returns:
        float: Analytical entropy of the Gaussian distribution.
    """
    entropy_analytical = 0.5 * np.log(2 * np.pi * np.e * variance)
    return entropy_analytical


def kld_1d_constant_dx(p, q, dx):
    """Numerical Kullback-Leibler divergence between two probability distributions: DKL(p||q).

    Args:
        p (np.ndarray): First probability distribution evaluated on x.
        q (np.ndarray): Second probability distribution evaluated on x.
        dx (float): Constant distance between x values.

    Returns:
        float: Numerical Kullback-Leibler divergence between the two probability distributions.
    """
    kld_numerical = np.trapz(np.log(p / q) * p, dx=dx)
    return kld_numerical


def kld_gaussian(mean_p, mean_q, var_p, var_q):
    """Analytical Kullback-Leibler divergence between two Gaussian distributions p and q: DKL(p||q).

    Args:
        mean_p (float): Mean of the first Gaussian distribution.
        mean_q (float): Mean of the second Gaussian distribution.
        var_p (float): Variance of the first Gaussian distribution.
        var_q (float): Variance of the second Gaussian distribution.

    Returns:
        float: Analytical Kullback-Leibler divergence between the two Gaussian distributions.
    """
    kld_analytical = (
        np.log(np.sqrt(var_q) / np.sqrt(var_p))
        + (var_p + (mean_p - mean_q) ** 2) / (2 * var_q)
        - 1 / 2
    )
    return kld_analytical


def information_gain_2d_constant_dx(log_likelihood_grid, log_prior, dx, return_posterior=False):
    """Compute the information gain employing the log-sum-exp trick."""
    log_like_plus_log_prior = log_likelihood_grid + log_prior
    c = np.max(log_like_plus_log_prior)
    likelihood_times_prior_c = np.exp(log_like_plus_log_prior - c)
    evidence_c = trapezoid_2d_constant_dx(likelihood_times_prior_c, dx=dx)

    posterior = likelihood_times_prior_c / evidence_c

    # Avoid computing np.log(0.0) because mathematically,
    # limit for x->0 of log(x)*x is 0
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where(posterior == 0.0, 0.0, posterior * (np.log(posterior) - log_prior))

    info_gain = trapezoid_2d_constant_dx(integrand, dx=dx)

    if return_posterior:
        return info_gain, posterior

    return info_gain


def trapezoid_2d_scipy(x, grid_points_first_dim, grid_points_second_dim):
    """Integrates the first two dimensions of x based on the trapezoidal rule.

    Args:
        x (np.ndarray): Array to integrate.
        grid_points_first_dim (list): Grid points on which x was evaluated in the first dimension.
        grid_points_second_dim (list): Grid points on which x was evaluated in the second dimension.

    Returns:
        float: Integral of the first two dimensions of x.
    """
    integral = sp.integrate.trapezoid(
        sp.integrate.trapezoid(x, grid_points_first_dim, axis=0), grid_points_second_dim, axis=0
    )
    return integral


def test_entropies():
    """Test entropies."""
    x, dx = np.linspace(-6, 6, 10000, retstep=True)
    mean = -2.0
    scale = 0.5
    pdf = sp.stats.norm.pdf(x, loc=mean, scale=scale)
    pdf_with_extra_dim = np.tile(pdf, reps=(2, 1))
    x2 = np.linspace(0, 1, len(x))

    entropy = entropy_2d(pdf_with_extra_dim, x, x2)
    entropy_numerical = entropy_constant_dx(pdf, dx)
    entropy_analytical = entropy_gaussian(scale**2)

    np.testing.assert_almost_equal(entropy, entropy_numerical)
    np.testing.assert_almost_equal(entropy, entropy_analytical)


def test_kld_2d():
    """Test Kullback-Leibler divergences."""
    x, dx = np.linspace(-6, 6, 10000, retstep=True)
    mean1 = -2.0
    scale1 = 0.5
    mean2 = -1.0
    scale2 = 0.5
    pdf1 = sp.stats.norm.pdf(x, loc=mean1, scale=scale1)
    pdf2 = sp.stats.norm.pdf(x, loc=mean2, scale=scale2)
    x2 = np.linspace(0, 1, len(x))
    pdf1_with_extra_dim = np.tile(pdf1, reps=(2, 1))
    pdf2_with_extra_dim = np.tile(pdf2, reps=(2, 1))

    kld = kld_2d(pdf1_with_extra_dim, pdf2_with_extra_dim, x, x2)
    kld_numerical = kld_1d_constant_dx(pdf1, pdf2, dx)
    kld_analytical = kld_gaussian(mean1, mean2, scale1**2, scale2**2)

    np.testing.assert_almost_equal(kld, kld_numerical)
    np.testing.assert_almost_equal(kld, kld_analytical)


def test_trapezoid_2d():
    """Test integration over two dimensions based on the trapezoidal rule."""
    grid_points_first_dim, dx1 = np.linspace(-6e4, 6e4, 100, retstep=True)
    grid_points_second_dim, dx2 = np.linspace(0, 5, 101, retstep=True)

    grid = np.zeros((len(grid_points_first_dim), len(grid_points_second_dim)))
    for i, x1 in enumerate(grid_points_first_dim):
        for j, x2 in enumerate(grid_points_second_dim):
            grid[i, j] = x1 * 1.23e-5 + x2

    integral_numpy = trapezoid_2d(grid, grid_points_first_dim, grid_points_second_dim)
    integral_scipy = trapezoid_2d_scipy(grid, grid_points_first_dim, grid_points_second_dim)
    integral_constant_dx = trapezoid_2d_constant_dx(grid, [dx1, dx2])

    np.testing.assert_almost_equal(integral_numpy, integral_scipy)
    np.testing.assert_almost_equal(integral_numpy, integral_constant_dx)


def test_information_gain_2d_uniform_prior():
    """Test information gain over two dimensions."""
    x, dx = np.linspace(-6, 6, 10000, retstep=True)
    mean_likelihood = -1.0
    scale_likelihood = 0.5
    likelihood = sp.stats.norm.pdf(x, loc=mean_likelihood, scale=scale_likelihood)
    prior = np.ones(likelihood.shape)

    # Add an artificial extra dimension to the pdfs
    log_likelihoods = np.tile(np.log(likelihood), reps=(2, 1))
    log_prior = np.tile(np.log(prior), reps=(2, 1))

    info_gain_2d_constant_dx, posterior_constant_dx = information_gain_2d_constant_dx(
        log_likelihoods, log_prior, [1, dx], return_posterior=True
    )
    info_gain_2d, posterior = information_gain_2d(
        log_likelihoods, log_prior, np.linspace(0, 1, 2), x, return_posterior=True
    )

    np.testing.assert_almost_equal(likelihood, posterior[0])
    np.testing.assert_almost_equal(likelihood, posterior_constant_dx[0])
    np.testing.assert_almost_equal(info_gain_2d, info_gain_2d_constant_dx)


def test_information_gain_2d_gaussian_prior():
    """Test information gain over two dimensions."""
    x, dx = np.linspace(-6, 6, 10000, retstep=True)
    mean_likelihood = 0.5
    scale_likelihood = 1.5
    var_likelihood = scale_likelihood**2
    mean_prior = 0.0
    scale_prior = 2.5
    var_prior = scale_prior**2
    var_posterior = 1 / (1 / var_prior + 1 / var_likelihood)
    mean_posterior = (mean_prior / var_prior + mean_likelihood / var_likelihood) * var_posterior
    likelihood = sp.stats.norm.pdf(x, loc=mean_likelihood, scale=scale_likelihood)
    prior = sp.stats.norm.pdf(x, loc=mean_prior, scale=scale_prior)
    posterior_analytical = sp.stats.norm.pdf(x, loc=mean_posterior, scale=np.sqrt(var_posterior))

    # Add an artificial extra dimension to the pdfs
    log_likelihoods = np.tile(np.log(likelihood), reps=(2, 1))
    log_prior = np.tile(np.log(prior), reps=(2, 1))

    info_gain_2d_constant_dx, posterior_constant_dx = information_gain_2d_constant_dx(
        log_likelihoods, log_prior, [1, dx], return_posterior=True
    )
    info_gain_2d, posterior = information_gain_2d(
        log_likelihoods, log_prior, np.linspace(0, 1, 2), x, return_posterior=True
    )
    info_gain_analytical = kld_gaussian(mean_posterior, mean_prior, var_posterior, var_prior)

    np.testing.assert_almost_equal(posterior[0], posterior_analytical, decimal=5)
    np.testing.assert_almost_equal(posterior[0], posterior_constant_dx[0])
    np.testing.assert_almost_equal(info_gain_2d, info_gain_analytical, decimal=4)
    np.testing.assert_almost_equal(info_gain_2d, info_gain_2d_constant_dx)
