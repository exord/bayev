import numpy as np
import scipy.linalg
import scipy.stats
import random
import math


def log_sum(log_summands):
    a = np.inf
    while a == np.inf:
        a = log_summands[0] + np.log(1 + np.sum(np.exp(log_summands[1:] -
                                                       log_summands[0])))
        random.shuffle(log_summands)
    return a


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
              implemented in scipy.linalg.cho_factor and scipy.linalg.cho_solve.
            * "solve": uses the numpy.linalg functions solve() and slogdet().

    :return array: multivariate density at vector position r.
    """

    # Compute normalization factor used for all methods.
    kk = len(r) * math.log(2*math.pi)

    if method == 'cholesky':
        # Use Cholesky decomposition of covariance.
        cho, lower = scipy.linalg.cho_factor(c)
        alpha = scipy.linalg.cho_solve((cho, lower), r)
        return -0.5 * (kk + np.dot(r, alpha) + 2 * np.sum(np.log(np.diag(cho))))

    elif method == 'solve':
        # Use slogdet and solve
        (s, d) = np.linalg.slogdet(c)
        alpha = np.linalg.solve(c, r)
        return -0.5 * (kk + np.dot(r, alpha) + d)


class MultivariateGaussian(scipy.stats.rv_continuous):
    def __init__(self, mu, cov):
        self.mu = mu
        self.covariance = cov

    def pdf(self, x, method='cholesky'):
        return multivariate_normal(x - self.mu, self.covariance, method)

    def rvs(self, nsamples):
        return np.random.multivariate_normal(self.mu, self.covariance, nsamples)
