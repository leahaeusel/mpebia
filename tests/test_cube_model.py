"""Test cube class for electromechanical model."""

import numpy as np

from mpebia.electromechanical_model.cube_model import CubeModel
from mpebia.electromechanical_model.parameters_shared import ParametersShared as params


def test_solve():
    """Test the solution of the electromechanical model."""
    cube = CubeModel(params.l0, params.U, params.rho)
    cube.set_material_parameters(params.E_gt, params.nu_gt)
    F_test = [0.0, 0.5, 1.0]
    d_test, I_test = cube.solve(F_test)

    assert d_test[0] == 0.0
    assert d_test[1] > d_test[0]
    assert I_test[0] == cube.get_I(0.0)

    for d, I, F in zip(d_test, I_test, F_test):
        residuals = cube.get_rhs(F, [d, I])
        assert abs(np.linalg.norm(residuals)) <= cube.tol


def test_solve_on_grid():
    """Test the solution of the electromechanical model on a grid."""
    cube = CubeModel(params.l0, params.U, params.rho)
    num_grid_points_E = 2
    num_grid_points_nu = 3
    num_force_values = 4
    grid_points_E = np.linspace(0.9 * params.E_gt, 1.1 * params.E_gt, num_grid_points_E)
    grid_points_nu = np.linspace(0.9 * params.nu_gt, 1.1 * params.nu_gt, num_grid_points_nu)
    force_values = np.linspace(0, params.F_max, num_force_values)

    d_on_grid = cube.solve_on_grid(grid_points_E, grid_points_nu, force_values, index_sol=0)
    I_on_grid = cube.solve_on_grid(grid_points_E, grid_points_nu, force_values, index_sol=1)

    assert d_on_grid.shape == (num_grid_points_E, num_grid_points_nu, num_force_values)
    assert I_on_grid.shape == (num_grid_points_E, num_grid_points_nu, num_force_values)

    # Test if the solution is consistent with the single solve method
    cube.set_material_parameters(grid_points_E[0], grid_points_nu[0])
    d, I = cube.solve(force_values)

    assert np.allclose(d_on_grid[0, 0, :], d)
    assert np.allclose(I_on_grid[0, 0, :], I)
