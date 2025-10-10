"""Utility functions to generate observations."""

import numpy as np
import scipy as sp


def get_equidistant_indices(num_indices, max_index):
    """Compute equidistant indices at which observations are collected.

    Args:
        num_indices (int): Number of indices at which the solution is observed.
        max_index (int): Maximum index of the solution vector.

    Returns:
        np.ndarray: Vector of indices.
    """
    start = max_index / (num_indices + 1)
    end = max_index - start
    indices = np.linspace(start, end, num_indices, dtype=int)

    return indices


def snr_to_std(snr, observations):
    """Compute the standard deviation from the medium signal-to-noise ratio.

    Args:
        snr (float): Signal-to-noise ratio.
        observations (np.ndarray): Observations of shape (num_obs,).

    Returns:
        float: Standard deviation.
    """
    dim_observations = len(observations.shape)
    if dim_observations > 1:
        observations = np.linalg.norm(observations, axis=0)
    mean_squared_observations = np.square(observations).mean()
    variance = mean_squared_observations / (snr * dim_observations)
    std = np.sqrt(variance)

    return std


def std_to_snr(std, observations):
    """Compute the medium signal-to-noise ratio from the standard deviation.

    Args:
        std (float): Standard deviation.
        observations (np.ndarray): Observations of shape (num_obs,).

    Returns:
        float: Signal-to-noise ratio.
    """
    dim_observations = len(observations.shape)
    if dim_observations > 1:
        observations = np.linalg.norm(observations, axis=0)

    mean_squared_observations = np.square(observations).mean()
    variance = std**2
    snr = mean_squared_observations / (variance * dim_observations)

    return snr


def sample_sobol(num_samples, seed):
    """Sample noise from a Sobol sequence.

    Args:
        num_samples (int): Number of samples to draw.
        seed (int): Seed for the Sobol sequence.

    Returns:
        np.ndarray: Noise samples.
    """
    sobol_samples = sp.stats.qmc.Sobol(d=1, seed=seed).random(num_samples)
    noise = sp.stats.norm.ppf(sobol_samples).squeeze()

    return noise


def gaussian_log_likelihood(y_obs, y_model, std):
    """Compute the Gaussian log-likelihood of the observations.

    Args:
        y_obs (lst, np.array): Observations.
        y_model (lst, np.array): Model results.
        std (float): Standard deviation.

    Returns:
        float: Gaussian log-likelihood.
    """
    y_obs = np.array(y_obs)
    y_model = np.array(y_model)

    var = std**2
    logpdf_const = -0.5 * np.log(2.0 * np.pi * var)

    log_likelihoods = logpdf_const - 0.5 * ((y_obs - y_model) ** 2) / var
    log_likelihood = log_likelihoods.sum(axis=-1)

    return log_likelihood
