"""Test spacing utilities."""

import numpy as np

from mpebia.spacing import evenly_log_spaced


def test_evenly_log_spaced():
    """Test the evenly_log_spaced function."""
    start = 1
    end = 100
    result = evenly_log_spaced(1, 100, 5)

    assert np.isclose(result[0], start)
    assert np.isclose(result[-1], end)
    assert len(result) == 5

    log_diffs = [np.log(result[i + 1]) - np.log(result[i]) for i in range(len(result) - 1)]

    np.testing.assert_allclose(log_diffs, log_diffs[0], rtol=1e-5)
