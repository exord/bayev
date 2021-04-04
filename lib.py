import numpy as np
import scipy.linalg
import scipy.stats
import random
from math import log, pi, e


def log_sum(log_summands):
    a = np.inf
    x = log_summands.copy()
    while a == np.inf or a == -np.inf or np.isnan(a):
        a = x[0] + np.log(1 + np.sum(np.exp(x[1:] - x[0])))
        random.shuffle(x)
    return a


def numerical_error(loglike, logweight, logevidence):
    """
    Compute the standard numerical error as defined by Geweke (Econometria 57,
    N. 6, pp. 1317-1339).

    :param array loglike:
        1-D array with log(likelihood)values evaluated in the sample drawn from
        the importance sampling density.

    :param array logweight:
        1-D array with log(weight) values evaluated in the sample drawn from
        the importance sampling density. The weight function is the prior
        density divided the importance sampling density (w = pi/I)

    :param float logevidence:
        log of the marginal likelihood estimation obtained.

    :return:
    """

    log_likeminusevidence = np.zeros_like(loglike)
    for i in range(len(loglike)):
        log_likeminusevidence = log_sum(np.array([loglike[i], logevidence +
                                                  np.log(-1)]))

    return log_sum(log(like - e) + 2*logweight) - 2 * log_sum(logweight)


def multivariate_normal(r, c, method='cholesky'):
    """
    Computes multivariate normal density for "residuals" vector r and
    covariance c.

    :param array r:
        1-D array of k dimensions.

    :param array c:
        2-D array or matrix of (k x k).

    :param string method:
        Method used to compute multivariate density.
        Possible values are:
            * "cholesky": uses the Cholesky decomposition of the covariance c,
              implemented in scipy.linalg.cho_factor and
              scipy.linalg.cho_solve.
            * "solve": uses the numpy.linalg functions solve() and slogdet().

    :return array: multivariate density at vector position r.
    """

    # Compute normalization factor used for all methods.
    kk = len(r) * log(2*pi)

    if method == 'cholesky':
        # Use Cholesky decomposition of covariance.
        cho, lower = scipy.linalg.cho_factor(c)
        alpha = scipy.linalg.cho_solve((cho, lower), r)
        return -0.5 * (kk + np.dot(r, alpha) +
                       2 * np.sum(np.log(np.diag(cho)))
                       )

    elif method == 'solve':
        # Use slogdet and solve
        (s, d) = np.linalg.slogdet(c)
        alpha = np.linalg.solve(c, r)
        return -0.5 * (kk + np.dot(r, alpha) + d)


class MultivariateGaussian(scipy.stats.rv_continuous):
    def __init__(self, mu, cov):
        self.mu = mu
        self.covariance = cov + 1e-10
        self.dimensions = len(cov)

    # CHANGE THIS TO COMPUTE ON MULTI DIMENSIONAL x....
    def pdf(self, x, method='cholesky'):
        if 1 < len(x.shape) < 3:
            # Input array is multi-dimensional
            # Check that input array is well aligned with covariance.
            if x.T.shape[0] != len(self.covariance):
                raise ValueError('Input array not aligned with covariance. '
                                 'It must have dimensions (n x k), where k is '
                                 'the dimension of the multivariate Gaussian.')

            # If ok, create array to contain results
            mvg = np.zeros(len(x))
            for s, rr in enumerate(x):
                mvg[s] = multivariate_normal(rr - self.mu, self.covariance,
                                             method)
            return mvg

        elif len(x.shape) == 1:
            return multivariate_normal(x - self.mu, self.covariance, method)

        else:
            raise ValueError('Input array must be 1- or 2-D.')

    def rvs(self, nsamples):
        return np.random.multivariate_normal(self.mu, self.covariance,
                                             nsamples)
