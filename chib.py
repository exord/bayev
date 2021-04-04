"""
Module to compute the estimation of the evidence using the method by
Chib and Jeliazkov (2001).
"""
from math import log
import numpy as np
from . import lib

__all__ = ['compute_cj_estimate', ]


def compute_cj_estimate(posterior_sample, lnlikefunc, lnpriorfunc,
                        param_post, nsamples, qprob=None, lnlikeargs=(),
                        lnpriorargs=(), lnlike_post=None, lnprior_post=None):

    """
    Computes the Chib & Jeliazkov estimate of the bayesian evidence.

    The estimation is based on an posterior sample with n elements
    (indexed by s, with s = 0, ..., n-1), and a sample from the proposal
    distribution used in MCMC (qprob) of size nsample. Note that if qprob is
    None, it is estimated as a multivariate Gaussian.

    :param array posterior_sample:
        A sample from the parameter posterior distribution. Dimensions are
        (n x k), where k is the number of parameters.

    :param callable lnlikefunc:
        Function to compute ln(likelihood) on the marginal samples.

    :param callable lnpriorfunc:
        Function to compute ln(prior density) on the marginal samples.

    :param array param_post:
        Posterior parameter sample used to obtained fixed point needed by the
        algorithm.

    :param int nsamples:
        Size of sample drawn from proposal distribution.

    :param object or None qprob:
        Proposal distribution function. If None, it will be estimated as a
        multivariate Gaussian. If not None, it must possess the methods pdf and
        rvs. See scipy.stats.rv_continuous.

    :param tuple lnlikeargs:
        Extra arguments passed to the likelihood function.

    :param tuple lnpriorargs:
        Extra arguments passed to the lnprior function.

    :param array lnlike_post:
        log(likelihood) computed over a posterior sample. 1-D array of length n.

    :param array lnprior_post:
        log(prior) computed over a posterior sample. 1-D array of length n.

    :raises AttributeError:
        if instace qprob does not have method 'pdf' or 'rvs'.

    :raises TypeError:
        if methods 'pdf' or 'rvs' from instance qprob are not callable.

    :returns: Natural logarithm of estimated Bayesian evidence.

    References
    ----------
    Chib & Jeliazkov (2001): Journal of the Am. Stat. Assoc.; Mar 2001; 96, 453
    """

    # Find fixed point on which to estimate posterior ordinate.
    if lnlike_post is not None:
        # Pass values of log(likelihood) in posterior sample.
        arg_fp = [lnlike_post, ]
    else:
        # Pass function that computes log(likelihood).
        arg_fp = [lnlikefunc, ]

    if lnlike_post is not None:
        # Pass values of log(prior) in posterior sample.
        arg_fp.append(lnprior_post)
    else:
        # Pass function that computes log(prior).
        arg_fp.append(lnpriorfunc)

    fp, lnpost0 = get_fixed_point(posterior_sample, param_post, arg_fp,
                                  lnlikeargs=lnlikeargs,
                                  lnpriorargs=lnpriorargs)

    # If proposal distribution is not given, define as multivariate Gaussian.
    if qprob is None:
        # Get covariance from posterior sample
        k = np.cov(posterior_sample.T)

        qprob = lib.MultivariateGaussian(fp, k)

    else:
        # Check that qprob has the necessary attributes
        for method in ('pdf', 'rvs'):
            try:
                att = getattr(qprob, method)
            except AttributeError:
                raise AttributeError('qprob does not have method '
                                     '\'{}\''.format(method))

            if not callable(att):
                raise TypeError('{} method of qprob is not '
                                'callable'.format(method))

    # Compute proposal density in posterior sample
    q_post = qprob.pdf(posterior_sample)

    # If likelihood over posterior sample is not given, compute it
    if lnlike_post is None:
        lnlike_post = lnlikefunc(posterior_sample, *lnlikeargs)
    # Idem for prior
    if lnprior_post is None:
        lnprior_post = lnpriorfunc(posterior_sample, *lnpriorargs)

    # Compute Metropolis ratio with respect to fixed point over posterior
    # sample.
    lnalpha_post = metropolis_ratio(lnprior_post + lnlike_post, lnpost0)

    #
    # Sample from the proposal distribution with respect to fixed point
    #
    proposal_sample = qprob.rvs(nsamples)

    # Compute likelihood and prior on proposal_sample
    lnprior_prop = lnpriorfunc(proposal_sample, *lnpriorargs)

    # Note that elements from the proposal distribution sample outside of the
    # prior support will get lnprior_prop = -np.inf. These elements should be
    # included in the denominator average below as zero (see reference), which
    # is naturally obtained thanks to the ability of numpy to treat inf.
    # We check, however, that not all samples have zero prior probability
    if np.all(lnprior_prop == -np.inf):
        raise ValueError('All samples from proposal density have zero prior'
                         'probability. Increase nsample.')

    # Now compute likelihood only on the samples where prior != 0.
    # This is to avoid computing on sample elements that will nevertheless
    # give zero (i.e. speed the code).
    lnlike_prop = np.full_like(lnprior_prop, -np.inf)
    ind = lnprior_prop != -np.inf
    lnlike_prop[ind] = lnlikefunc(proposal_sample[ind, :], *lnlikeargs)

    # Get Metropolis ratio with respect to fixed point over proposal sample
    lnalpha_prop = metropolis_ratio(lnpost0, lnprior_prop + lnlike_prop)

    # Compute estimate of posterior ordinate (see Eq. 9 from reference)
    num = lib.log_sum(lnalpha_post + q_post) - log(len(posterior_sample))
    den = lib.log_sum(lnalpha_prop) - log(len(proposal_sample))
    lnpostord = num - den

    # Return log(Evidence) estimation
    return lnpost0 - lnpostord


def get_fixed_point(posterior_samples, param_post, funcs,
                    lnlikeargs=(), lnpriorargs=()):
    """
    Find the posterior point closest to the model of the lnlike distribution.

    :param array posterior_samples:
        A sample from the parameters posterior distribution. Array dimensions
        must be (n x k), where n is the number of elements in the sample and
        k is the number of parameters.

    :param array or None param_post:
        A sample from the marginal posterior distribution of the parameter
        chosen to identify the high-density point to use as fixed point. This 
        is typically one of the columns of posterior_samples, but could be any
        1-D array of size n. If None, then a multivariate Gaussian kernel
        estimate of the joint posterior distribution is used.

    :param iterable of len 2 funcs:
        A list containing to callables or arrays, as follows:
            
    :param array or callable lnlike:
        Function to compute log(likelihood). If an array is given, this is
        simply the log(likelihood) values at the posterior samples, and the 
        best value will be chosen from this array.

    :param array or callable lnprior:
        Function to compute log(prior). If an array is given, this is
        simply the log(prior) values at the posterior samples, and the
        best value will be chosen from this array.

    :param tuple lnlikeargs:
        Extra arguments passed to lnlike functions.

    :param tuple lnpriorargs:
        Extra arguments passed to lnprior functions.

    :raises IndexError: if either lnlike or lnprior are arrays with length not
        matching the number of posterior samples.

    :return:
        the fixed point in parameter space and the value of
        log(prior * likelihood) evaluated at this point.
    """
    lnlike = funcs[0]
    lnprior = funcs[1]
    
    if param_post is not None:

        # Use median of param_post as fixed point.
        param0 = np.median(param_post)

        # Find argument closest to median.
        ind0 = np.argmin(np.abs(param_post - param0))

        fixed_point = posterior_samples[ind0, :]

        # Compute log(likelihood) at fixed_point
        if hasattr(lnlike, '__iter__'):
            if len(lnlike) != len(posterior_samples):
                raise IndexError('Number of elements in lnlike array and in '
                                 'posterior sample must match.')

            lnlike0 = lnlike[ind0]

        else:
            # Evaluate lnlike function at fixed point.
            lnlike0 = lnlike(fixed_point, *lnlikeargs)

        # Compute log(prior) at fixed_point
        if hasattr(lnprior, '__iter__'):
            if len(lnprior) != len(posterior_samples):
                raise IndexError('Number of elements in lnprior array and in '
                                 'posterior sample must match.')

            lnprior0 = lnprior[ind0]

        else:
            # Evaluate lnlike function at fixed point.
            lnprior0 = lnprior(fixed_point, *lnpriorargs)

        return fixed_point, lnlike0 + lnprior0

    else:
        raise NotImplementedError
        pass


def metropolis_ratio(lnpost0, lnpost1):
    """
    Compute Metropolis ratio for two states.

    :param float or array lnpost0:
        Value of ln(likelihood * prior) for inital state.

    :param float or array lnpost1:
        Value of ln(likelihood * prior) for proposal state.

    :raises ValueError: if lnpost0 and lnpost1 have different lengths.

    :return: log(Metropolis ratio)
    """
    if (hasattr(lnpost0, '__iter__') and hasattr(lnpost1, '__iter__') and
            len(lnpost0) != len(lnpost1)):
        raise ValueError('lnpost0 and lnpost1 have different lenghts.')

    return np.minimum(lnpost1 - lnpost0, 0.0)


def proposaldensity_posterior(posterior_samples, fixed_point, qprob=None,
                              cov=None, method='solve'):
    """
    Compute the proposal density with respect to a fixed point in parameter
    space over a sample drew from the posterior distribution.

    :param array posterior_samples:
          Samples from the parameter posterior distribution. Dimensions are
        (n x k), where k is the number of parameters.

    :param array fixed_point:
        1-D array containing a fixed point in the parameter space.

    :param function or None qprob:
        Function to compute proposal density. If None, qprob is estimated as
        a multivariate normal with covariance matrix taken from the cov
        parameter.

    :param array cov:
        If qprob is None, then use covariance for multivariate

    :return:
    """

    # If not given estimate qprob as multivariate normal.
    if qprob is None:

        # Distance vector in parameter space
        r = posterior_samples - fixed_point

        return lib.multivariate_normal(r, cov, method)

    else:
        return qprob(posterior_samples, fixed_point)


def sample_proposal_distribution(n, fixed_point, cov):
    """
    Draw n-sized sample from the proposal density distribution (assumed to be
    multivariate) with respect to a fixed point.

    :param int n:
        number of samples to draw.

    :param array fixed_point:
        point with respect to wich the proposal sample is obtained.

    :param array cov:
        covariance of multinormal distribution.

    :raises ValueError: if cov is not square or if its dimensions do not match
        those of fixed_point.

    :return: the proposal distribution sample. Dimensions (n x k).
    """

    if not all(len(row) == len(cov) for row in cov):
        raise ValueError('Covariance matrix must be square.')

    if len(fixed_point) != len(cov):
        raise ValueError('Covariance and fixed point dimensions do not match.')

    return (np.random.multivariate_normal(np.zeros_like(fixed_point), cov, n) +
            fixed_point)
