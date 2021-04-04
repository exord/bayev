import numpy as np
import scipy.stats
from math import sqrt, log

from . import lib

def compute_perrakis_estimate(marginal_sample, lnlikefunc, lnpriorfunc,
                              lnlikeargs=(), lnpriorargs=(),
                              densityestimation='histogram', **kwargs):
    """
    Computes the Perrakis estimate of the bayesian evidence.

    The estimation is based on n marginal posterior samples
    (indexed by s, with s = 0, ..., n-1).

    :param array marginal_sample:
        A sample from the parameter marginal posterior distribution.
        Dimensions are (n x k), where k is the number of parameters.

    :param callable lnlikefunc:
        Function to compute ln(likelihood) on the marginal samples.

    :param callable lnpriorfunc:
        Function to compute ln(prior density) on the marginal samples.

    :param tuple lnlikeargs:
        Extra arguments passed to the likelihood function.

    :param tuple lnpriorargs:
        Extra arguments passed to the lnprior function.

    :param str densityestimation:
        The method used to estimate the marginal posterior density of each
        model parameter ("normal", "kde", or "histogram").


    Other parameters
    ----------------
    :param kwargs:
        Additional arguments passed to estimate_density function.

    :return:

    References
    ----------
    Perrakis et al. (2014; arXiv:1311.0674)
    """

    if not isinstance(marginal_sample, np.ndarray):
        marginal_sample = np.array(marginal_sample)

    number_parameters = marginal_sample.shape[1]

    ##
    # Estimate marginal posterior density for each parameter.
    log_marginal_posterior_density = np.zeros(marginal_sample.shape)

    for parameter_index in range(number_parameters):

        # Extract samples for this parameter.
        x = marginal_sample[:, parameter_index]

        # Estimate density with method "densityestimation".
        log_marginal_posterior_density[:, parameter_index] = \
            estimate_logdensity(x, method=densityestimation, **kwargs)

    # Compute produt of marginal posterior densities for all parameters
    log_marginal_densities = log_marginal_posterior_density.sum(axis=1)
    ##

    # Compute log likelihood in marginal sample.
    log_likelihood = lnlikefunc(marginal_sample, *lnlikeargs)

    # Compute weights (i.e. prior over marginal density)
    w = weight(marginal_sample, lnpriorfunc, lnpriorargs, log_marginal_densities)

    # Mask values with zero likelihood (a problem in lnlike)
    cond = log_likelihood != 0

    # Use identity for summation
    # http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction
    # ln(sum(x)) = ln(x[0]) + ln(1 + sum( exp( ln(x[1:]) - ln(x[0]) ) ) )
    # log_summands = log_likelihood[cond] + np.log(prior_probability[cond])
    #  - np.log(marginal_densities[cond])
    perr = lib.log_sum(w[cond] + log_likelihood[cond]) - log(len(w[cond]))

    return perr


def weight(marginal_sample, lnpriorfunc, lnpriorargs, log_marginal_densities):
    """
    Compute the weights for the Perrakis estimator.

    :param marginal_sample:
        A sample from the parameter marginal posterior distribution.
        Dimensions are (n x k), where k is the number of parameters.

    :param callable lnpriorfunc:
        Function to compute ln(prior density) on the marginal samples.

    :param tuple lnpriorargs:
        Extra arguments passed to the lnprior function.

    :param array log_marginal_densities:
        1-D array containing the value of the sum of the log marginal densities
        evaluated in the marginal_sample.

    :return: array with the wegiht values
    """
    log_prior = lnpriorfunc(marginal_sample, *lnpriorargs)
    return log_prior - log_marginal_densities


def estimate_logdensity(x, method='histogram', **kwargs):
    """
    Estimate log probability density based on a sample. Return value of density at
    sample points.

    :param array_like x: sample.

    :param str method:
        Method used for the estimation. 'histogram' estimates the density based
        on a normalised histogram of nbins bins; 'kde' uses a 1D non-parametric
        gaussian kernel; 'normal approximates the distribution by a normal
        distribution.

    Additional parameters

    :param int nbins:
        Number of bins used in "histogram method".

    :return: density estimation at the sample points.
    """

    nbins = kwargs.pop('nbins', 100)

    if method == 'normal':
        # Approximate each parameter distribution by a normal.
        return scipy.stats.norm.logpdf(x, loc=x.mean(), scale=sqrt(x.var()))

    elif method == 'kde':
        # Approximate each parameter distribution using a gaussian
        # kernel estimation
        return scipy.stats.gaussian_kde(x).logpdf(x)

    elif method == 'histogram':
        # Approximate each parameter distribution based on the histogram
        # Uses nbins keyword parameter.
        density, bin_edges = np.histogram(x, nbins, density=True)

        # Find to which bin each element corresponds
        density_indexes = np.searchsorted(bin_edges, x, side='left')

        # Correct to avoid index zero from being assiged to last element
        density_indexes = np.where(density_indexes > 0, density_indexes,
                                   density_indexes + 1)

        return np.log(density[density_indexes - 1])


def make_marginal_samples(joint_sample, nsamples=None, seed=None):
    """
    Reshuffles samples from joint distribution of k parameters to obtain samples
    from the _marginal_ distribution of each parameter.

    :param np.array joint_sample:
        Sample from the parameter joint distribution. Dimensions are (n x k),
        where k is the number of parameters.

    :param nsamples:
        Number of samples to produce. If 0, use number of joint samples.
    :type nsamples:
        int or None

    :param seed:
        If not None, use as seed for random number generator.
    :type seed:
        int or None

    :return: shuffled version of joint distribution sample.
    """
    if seed is not None:
        np.random.seed(seed)

    if nsamples > len(joint_sample) or nsamples is None:
        nsamples = len(joint_sample)

    # Randomly select parameter vector elements from different joint dist.
    # samples
    number_parameters = joint_sample.shape[-1]

    ind = np.random.randint(joint_sample.shape[0],
                            size=(nsamples, number_parameters))

    return joint_sample[ind, range(number_parameters)]
