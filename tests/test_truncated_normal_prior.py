"""Test cube class for electromechanical model."""

import numpy as np
import pytest

from mpebia.electromechanical_model.parameters_shared import ParametersShared as params
from mpebia.entropies import trapezoid_2d
from mpebia.truncated_gaussian_prior import TruncatedNormalPrior


@pytest.fixture(name="truncated_normal_prior")
def fixture_truncated_normal_prior():
    """Fixture for TruncatedNormalPrior."""
    return TruncatedNormalPrior(
        params.mean_prior,
        params.std_prior,
        params.truncations_E,
        params.truncations_nu,
        (params.num_grid_points_E, params.num_grid_points_nu),
        params.offset_ppf,
    )


def test_grid_points_distributed_according_to_prior(truncated_normal_prior):
    """Test if grid points are distributed according to the prior."""
    percent_points_E = truncated_normal_prior.prior_1.cdf(truncated_normal_prior.grid_points_1)
    percent_points_nu = truncated_normal_prior.prior_2.cdf(truncated_normal_prior.grid_points_2)

    assert np.allclose(
        percent_points_E, params.offset_ppf, 1 - params.offset_ppf, len(percent_points_E)
    )
    assert np.allclose(
        percent_points_nu, params.offset_ppf, 1 - params.offset_ppf, len(percent_points_nu)
    )


def test_get_log_prior_on_grid(truncated_normal_prior):
    """Test if log prior on grid is computed correctly."""
    log_prior_on_grid = truncated_normal_prior.get_log_prior_on_grid()
    integral = trapezoid_2d(
        np.exp(log_prior_on_grid),
        truncated_normal_prior.grid_points_1,
        truncated_normal_prior.grid_points_2,
    )

    assert log_prior_on_grid.shape == (
        params.num_grid_points_E,
        params.num_grid_points_nu,
    )
    assert np.all(np.isfinite(log_prior_on_grid))
    assert np.isclose(integral, 1.0, atol=1e-1)


def test_plot_prior_andplot_prior_and_grid_points(tmp_path, truncated_normal_prior):
    """Test if prior and grid points can be plotted."""
    log_prior_on_grid = truncated_normal_prior.get_log_prior_on_grid()
    truncated_normal_prior.plot_prior_and_grid_points(
        np.exp(log_prior_on_grid), [params.E_gt, params.nu_gt], tmp_path
    )
