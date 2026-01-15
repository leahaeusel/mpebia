"""Utility functions for generating spaced points."""

import numpy as np


def evenly_log_spaced(start, end, num_points):
    """Return evenly spaced points on a logarithmic scale.

    Args:
        start (float): Start of the range.
        end (float): End of the range.
        num_points (int): Number of points to generate.

    Returns:
        np.ndarray: Array of evenly spaced points on a logarithmic scale.
    """
    array = np.exp(np.linspace(np.log(start), np.log(end), num_points))
    return array
