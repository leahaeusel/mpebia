"""Truncated normal distribution module."""

import scipy as sp
from queens.distributions._distribution import Continuous
from queens.utils.logger_settings import log_init_args


class TruncatedNormal(Continuous):
    """Univariate Truncated normal distribution class."""

    @log_init_args
    def __init__(
        self,
        mean,
        std,
        lower_bound,
        upper_bound,
    ):
        """Initialize the truncated normal distribution.

        Args:
            mean (array_like): Mean of the distribution
            std (array_like): Standard deviation of the distribution
            lower_bound (array_like): Lower bound(s) of the distribution
            upper_bound (array_like): Upper bound(s) of the distribution
        """
        if (
            not isinstance(mean, (float))
            or not isinstance(std, (float))
            or not isinstance(lower_bound, (float))
            or not isinstance(upper_bound, (float))
        ):
            raise NotImplementedError("Only one-dimensional distributions are supported.")

        dimension = 1
        covariance = std**2
        super().__init__(mean=mean, covariance=covariance, dimension=dimension)

        super().check_bounds(lower_bound, upper_bound)

        a = (lower_bound - mean) / std
        b = (upper_bound - mean) / std
        truncnorm = sp.stats.truncnorm(a, b, loc=mean, scale=std)

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.truncnorm = truncnorm

    def cdf(self, x):
        """Cumulative distribution function.

        Args:
            x (np.ndarray): Positions at which the cdf is evaluated
        """
        raise NotImplementedError

    def draw(self, num_draws=1):
        """Draw samples.

        Args:
            num_draws (int, optional): Number of draws
        """
        raise NotImplementedError

    def logpdf(self, x):
        """Log of the probability density function.

        Args:
            x (np.ndarray): Positions at which the log pdf is evaluated
        """
        raise NotImplementedError

    def grad_logpdf(self, x):
        """Gradient of the log pdf with respect to *x*.

        Args:
            x (np.ndarray): Positions at which the gradient of log pdf is evaluated
        """
        raise NotImplementedError

    def pdf(self, x):
        """Probability density function.

        Args:
            x (np.ndarray): Positions at which the pdf is evaluated
        """
        raise NotImplementedError

    def ppf(self, quantiles):
        """Percent point function (inverse of cdf — quantiles).

        Args:
            quantiles (np.ndarray): Quantiles at which the ppf is evaluated

        Returns:
            ppf (np.ndarray): Positions which correspond to given quantiles
        """
        ppf = self.truncnorm.ppf(quantiles)
        return ppf
