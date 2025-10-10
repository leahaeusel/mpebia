"""Test observation utilities."""

from time import time

import numpy as np
import pytest
import scipy as sp

from mpebia.observations import (
    gaussian_log_likelihood,
    get_equidistant_indices,
    snr_to_std,
    std_to_snr,
)


def test_get_equidistant_indices():
    """Test the computation of equidistant indices."""
    num_indices = 3
    max_index = 7

    indices = get_equidistant_indices(num_indices, max_index)

    np.testing.assert_array_equal(indices, np.array([1, 3, 5]))


def test_snr_to_std_1d():
    """Test getting the standard deviation from the signal-to-noise ratio."""
    snr = 2.0
    observations = np.array([1, 2, 3])

    std = snr_to_std(snr, observations)
    np.testing.assert_almost_equal(snr, (observations**2).mean() / std**2)

    snr_computed = std_to_snr(std, observations)
    np.testing.assert_almost_equal(snr, snr_computed)


def test_snr_to_std_2d():
    """Test getting the standard deviation from the signal-to-noise ratio."""
    snr = 3.0
    observations = np.array([[1, 2], [3, 4], [5, 6]])

    std = snr_to_std(snr, observations)
    np.testing.assert_almost_equal(
        snr, (np.linalg.norm(observations, axis=0) ** 2).mean() / (2 * std**2)
    )

    snr_computed = std_to_snr(std, observations)
    np.testing.assert_almost_equal(snr, snr_computed)


def test_snr_scaling():
    """Test if SNR is invariant to scaling of observations and STD."""
    std = 1.0
    observations = np.array([1, 2, 3])

    snr = std_to_snr(std, observations)
    snr_scale10 = std_to_snr(std * 10, observations * 10)

    np.testing.assert_almost_equal(snr, snr_scale10)


@pytest.mark.parametrize(
    "ys_obs, ys_model",
    [([0], [1]), (np.linspace(1, 2, 1000), np.linspace(2, 4, 1000))],
)
def test_gaussian_log_likelihood(ys_obs, ys_model):
    """Compare the gaussian log-likelihood implementation with scipy."""
    std = 2.0

    start = time()
    log_like_me = gaussian_log_likelihood(ys_obs, ys_model, std)
    time_me = time() - start
    print(f"\nTime for computation me:    {time_me}")

    start = time()
    log_like_scipy = sp.stats.norm.logpdf(ys_model, loc=ys_obs, scale=std).sum()
    time_scipy = time() - start
    print(f"Time for computation scipy: {time_scipy}")

    np.testing.assert_almost_equal(log_like_me, log_like_scipy)

    assert time_me < time_scipy
