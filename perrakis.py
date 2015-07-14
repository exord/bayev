import numpy as np
import random
import scipy.stats
from math import sqrt, log

import importlib

import time
import lib


def compute_perrakis_estimate(marginal_samples, loglike, prior,
                              likeargs=(), priorargs=(),
                              densityestimation='histogram', **kwargs):
    """
    Computes the Perrakis estimate of the bayesian evidence.

    The estimation is based on n marginal posterior samples
    (indexed by s, with s = 0, ..., n-1).

    :param array marginal_samples: samples from the parameter marginal\
    posterior distribution. Dimensions are (n x k), where k is the number of parameters.

    :param callable loglike: function to compute ln(likelihood) on the \
    marginal samples.
    :param callable prior: function to compute ln(prior density) on the \
    marginal samples.

    :param tuple likeargs: extra arguments passed to the likelihood function.
    :param tuple priorargs: extra arguments passed to the prior function.

    :param str densityestimation: the method used to estimate the marginal \
    posterior density of each model parameter ("normal", "kde", or "histogram").


    Other parameters
    ----------------
    :param kwargs: additional arguments passed to estimate_density function.
        :param nbins: number of bins used in "histogram method".
        :type nbins: int

    :return:

    References
    ----------
    Perrakis et al. (2014; arXiv:1311.0674)
    """

    if not isinstance(marginal_samples, np.ndarray):
        marginal_samples = np.array(marginal_samples)

    number_parameters = marginal_samples.shape[1]

    ##
    # Estimate marginal posterior density for each parameter.
    marginal_posterior_density = np.zeros(marginal_samples.shape)

    for parameter_index in range(number_parameters):

        # Extract samples for this parameter.
        x = marginal_samples[:, parameter_index]

        # Estimate density with method "densityestimation".
        marginal_posterior_density[parameter_index] = \
            estimate_density(x, method=densityestimation, **kwargs)

    # Compute produt of marginal posterior densities for all parameters
    prod_marginal_densities = marginal_posterior_density.prod(axis=0)
    ##

    ##
    # Compute prior and likelihood in marginal sample.
    log_prior = prior(marginal_samples, *priorargs)
    log_likelihood = loglike(marginal_samples, *likeargs)
    ##

    # Mask values with zero likelihood (a problem in loglike)
    cond = log_likelihood != 0

    # Use identity for summation
    # http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction
    # ln(sum(x)) = ln(x[0]) + ln(1 + sum( exp( ln(x[1:]) - ln(x[0]) ) ) )
    # log_summands = log_likelihood[cond] + np.log(prior_probability[cond])
    #  - np.log(prod_marginal_densities[cond])
    log_summands = log_likelihood[cond] + log_prior[cond] - \
                    np.log(prod_marginal_densities[cond])

    perr = lib.log_sum(log_summands) - log(len(log_summands))

    return perr


def estimate_density(x, method='histogram', **kwargs):
    """
    Estimate probability density based on a sample. Return value of density at
    sample points.

    :param array_like x: sample
    :param str method: method used for the estimation. 'histogram' estimates \
    the density based on a normalised histogram of nbins bins; 'kde' uses a 1D \
    non-parametric gaussian kernel; 'normal approximates the distribution by a \
    normal distribution

    Additional parameters

    :param int nbins: number of bins used in "histogram method".

    :return: density estimation at the sample points.
    """

    nbins = kwargs.pop('nbins', 100)

    if method == 'normal':
        # Approximate each parameter distribution by a normal.
        return scipy.stats.norm.pdf(x, loc=x.mean(), scale=sqrt(x.var()))

    elif method == 'kde':
        # Approximate each parameter distribution using a gaussian
        # kernel estimation
        return scipy.stats.gaussian_kde(x)(x)

    elif method == 'histogram':
        # Approximate each parameter distribution based on the histogram
        # Uses nbins keyword parameter.
        density, bin_edges = np.histogram(x, nbins, density=True)

        # Find to which bin each element corresponds
        density_indexes = np.searchsorted(bin_edges, x, side='left')

        # Correct to avoid index zero from being assiged to last element
        density_indexes = np.where(density_indexes > 0, density_indexes,
                                   density_indexes + 1)

        return density[density_indexes - 1]


def make_marginal_samples(joint_samples, nsamples=None):
    """
    Reshuffles samples from joint distribution of k parameters to obtain samples
    from the _marginal_ distribution of each parameter.

    :param joint_samples: samples from the parameter joint distribution.
    :type joint_samples: array_like (n x k), where k is the number of paramters
    :param nsamples: number of samples to produce. If 0, use number of joint
    samples.
    :type nsamples: int or None
    """

    # Copy joint samples before reshuffling in place.
    # Keep only last nsamples
    # WARNING! Always taking the last nsamples. This is not changed
    # with MonteCarlo
    if nsamples > len(joint_samples) or nsamples is None:
        nsamples = len(joint_samples)

    marginal_samples = joint_samples[-nsamples:, :].copy()

    number_parameters = marginal_samples.shape[-1]
    # Reshuffle joint posterior samples to obtain _marginal_ posterior
    # samples
    for parameter_index in range(number_parameters):
        random.shuffle(marginal_samples[:, parameter_index])

    return marginal_samples


def compute_perrakis_estimate_old(target, simul, nsamples=1e3, **kwargs):
    """
    Compute the evidence for simulation simul of target using estimator by
    Perrakis et al. (2014; arXiv:1311.0674)

    Parameters
    ----------
    target, str
        Name of the target.

    simul, str
        String identifiying the simulation for which the computation is to
        be done.

    Other parameters
    ----------------
    nsamples, int
        Number of samples from marginal posteriors. If None, or if larger than
        samples from joint posterior use the same number of samples as posterior
        samples.

    nmc, int
        Number of times the computation is repeated with different samples
        from the posterior (not from the proposal density!) to compute error.

    datadict, dict
        Datadict obtained using DataReader. If None, datadict will be
        constructed from input_dict. Useful to save time when performing a
        series of computation involving the same datadict.

    mergefile, str
        Name (including path) of merged chain file. If None, it will be
        constructed from target and simulation name.

    pastisfile, str
        Name (including path) of .pastis configuration file. If None, it will
        be constructed from simulation name.

    method, str
        The method used to estimate the marginal posterior densities of the
        parameters.
        Options are:
            - "normal": approximate by a normal distribution with same mean
                and variance.

            - "kde": use gaussian kernel method implemented in
                scipy.stats.gaussian_kde

    nmc, int
        Number of times the computation is performed to estimate dispersion.

    Returns
    -------
    logE, array
        Estimate of the log(n)(Evidence)
    """

    number_marginal_samples = marginal_posterior_samples.shape[1]
    number_parameters = joint_posterior_samples.shape[0]

    # Perform computation number_montecarlo_runs times
    perrakis_evidence = np.zeros(number_montecarlo_runs)

    timei = time.time()
    for iteration in range(number_montecarlo_runs):

        # Print progress
        if (iteration + 1) % 10 == 0:
            print('Iteration {} out of {}'.format(iteration + 1,
                                                  number_montecarlo_runs))


    # Estimate _marginal_ posterior probability for each parameter.

    # Use identity for summation
    # http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction
    # ln(sum(x)) = ln(x[0]) + ln(1 + sum( exp( ln(x[1:]) - ln(x[0]) ) ) )
    # log_summands = log_likelihood[cond] + np.log(prior_probability[cond])
    #  - np.log(prod_marginal_densities[cond])
    log_summands = log_likelihood[cond] + np.log(prior_probability[cond]) - \
                np.log(prod_marginal_densities[cond])

    perr = lib.log_sum(log_summands) - log(len(log_summands))

    perrakis_evidence[iteration] = perr

    print('Running time without initialisation: '
          '{:.2f} minutes'.format((time.time() - timei)/60.0))
    return perrakis_evidence
