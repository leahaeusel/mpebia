"""Class that models the electromechanical behavior of a cube."""

import numpy as np

from mpebia.observations import gaussian_log_likelihood


class CubeModel:
    """Tensile test of a cube with concurrent electric current measurements.

    The cube is assumed clamped on one side while a force is applied to the opposing side.
    At the same time, these opposing sides are subjected to different potentials and the resulting
    electrical current is measured.
    The lateral surfaces of the cube are free to move in one direction and clamped in the other
    direction.

    Attributes:
        l0 (float): Initial side lengths of the cube.
        U (float): Potential difference between the opposing sides.
        rho (float): Electrical resistivity of the cube.
        E (float): Young's modulus of the cube.
        nu (float): Poisson's ratio of the cube.
        tol (float): Tolerance for Newton scheme.
    """

    def __init__(self, l0, U, rho):
        """Initialize the cube.

        Args:
            l0 (float): Initial side lengths of the cube.
            U (float): Potential difference between the opposing sides.
            rho (float): Electrical resistivity of the cube.
        """
        self.l0 = l0
        self.U = U
        self.rho = rho

        self.E = None
        self.nu = None

        self.tol = 1.0e-10

    def set_material_parameters(self, E, nu):
        """Set the material parameters of the cube.

        Args:
            E (float): Young's modulus of the cube.
            nu (float): Poisson's ratio of the cube.
        """
        self.E = E
        self.nu = nu

    def get_I(self, d):
        """Compute the electric current I of the cube for a given displacement.

        Args:
            d (float): Displacement of the cube.

        Returns:
            float: Electric current I.
        """
        I = (
            self.U
            * self.l0
            * np.sqrt((2 * self.l0 * d + d**2) * self.nu / (self.nu - 1) + self.l0**2)
            / (self.rho * (self.l0 + d))
        )
        return I

    def get_A(self, F, y):
        """Compute the system matrix A.

        Args:
            F (float): Applied force.
            y (np.ndarray): Current solution vector of length 2. Consists of the displacement d and
                the electric current I.

        Returns:
            np.array: 2x2 system matrix A.
        """
        d = y[0]
        I = y[1]

        a11 = 2 * self.l0**2 + 6 * self.l0 * d + 3 * d**2
        a12 = 0
        a21 = self.rho * I - self.U * self.l0 / (
            2 * np.sqrt((2 * self.l0 * d + d**2) * self.nu / (self.nu - 1) + self.l0**2)
        ) * self.nu / (self.nu - 1) * (1 + 2 * d)
        a22 = self.rho * (self.l0 + d)

        if a11 == 0 or a22 == 0:
            raise ValueError("System matrix A has a eigenvalue equal to zero.")

        return np.array([[a11, a12], [a21, a22]])

    def get_rhs(self, F, y):
        """Get the right-hand side of the system of equations.

        Args:
            F (float): Applied force.
            y (np.ndarray): Current solution vector of length 2. Consists of the displacement d and
                the electric current I.

        Returns:
            np.ndarray: Right-hand side vector of length 2.
        """
        d = y[0]
        I = y[1]

        f1 = (
            2 * self.l0**2 * d
            + 3 * self.l0 * d**2
            + d**3
            - 2 * F * self.l0 * (1 - self.nu**2) / self.E
        )
        f2 = self.rho * (self.l0 + d) * I - self.U * self.l0 * np.sqrt(
            (2 * self.l0 * d + d**2) * self.nu / (self.nu - 1) + self.l0**2
        )

        return np.array([-f1, -f2])

    def solve(self, Fs):
        """Compute d and I for the given forces with the Newton method.

        Args:
            d0 (float): Initial displacement.
            I0 (float): Initial electric current.
            Fs (np.ndarray): Force values of shape (num_steps,).

        Returns:
            np.array: Displacements d of shape (num_steps,).
            np.array: Electric currents I of shape (num_steps,).
        """
        # Compute start values for Newton method
        d0 = 0.0
        I0 = self.get_I(d0)
        y0 = [d0, I0]

        num_force_steps = len(Fs)
        ys = np.zeros((num_force_steps + 1, 2))
        ys[0] = y0

        # Iterate over force values
        for step in range(num_force_steps):
            y_n = ys[step]
            y_n1 = y_n.copy()  # Proposal for next time step
            F = Fs[step]
            rhs = self.get_rhs(F, y_n1)

            # Newton iteration
            i = 1
            while abs(np.linalg.norm(rhs)) > self.tol:
                A = self.get_A(F, y_n1)
                rhs = self.get_rhs(F, y_n1)

                dy = np.linalg.solve(A, rhs)

                y_n1 += dy

                rhs = self.get_rhs(F, y_n1)
                i += 1
                if i > 100:
                    raise ValueError(
                        f"Newton method did not converge for step {step} with force {F}. "
                        f"Current solution: {y_n1}, residual: {rhs}"
                    )

            ys[step + 1] = y_n1

        ds = ys[1:, 0]
        Is = ys[1:, 1]

        return ds, Is

    def solve_on_grid(self, grid_points_E, grid_points_nu, F_obs, index_sol):
        """Solve the cube model on a grid.

        Args:
            grid_points_E (np.ndarray): Grid points for Young's modulus.
            grid_points_nu (np.ndarray): Grid points for Poisson's ratio.
            F_obs (np.ndarray): The forces at which observations were made.
            index_sol (int): The index within the solution vector at which to return the solution.

        Returns:
            np.ndarray: The solution on the grid.
        """
        solution_on_grid = np.zeros((len(grid_points_E), len(grid_points_nu), len(F_obs)))
        for i_E, E in enumerate(grid_points_E):
            for i_nu, nu in enumerate(grid_points_nu):
                self.set_material_parameters(E, nu)
                solution = self.solve(F_obs)
                solution_on_grid[i_E, i_nu] = solution[index_sol]

        return solution_on_grid

    def get_log_likelihood_on_grid(self, prior, observations, std, F_obs, index_sol):
        """Evaluate the log likelihood on the grid.

        Args:
            prior (TruncatedNormalPrior): The prior distribution with the grid points.
            observations (np.ndarray): The observed data.
            std (float): The standard deviation of the observations.
            F_obs (np.ndarray): The forces at which observations were made.
            index_sol (int): The index of the solution to evaluate.

        Returns:
            np.ndarray: The log likelihood values on the grid.
        """
        solution = self.solve_on_grid(prior.grid_points_1, prior.grid_points_2, F_obs, index_sol)
        log_likelihood_grid = gaussian_log_likelihood(observations, solution, std)

        return log_likelihood_grid
