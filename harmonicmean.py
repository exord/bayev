import numpy.random
import numpy as np
import lib
from math import log


def compute_harmonicmean(lnlike_post, posterior_sample=None, lnlikefunc=None,
                         lnlikeargs=(), **kwargs):
    """
    Computes the harmonic mean estimate of the marginal likelihood.

    The estimation is based on n posterior samples
    (indexed by s, with s = 0, ..., n-1), but can be done directly if the
    log(likelihood) in this sample is passed.

    :param array lnlike_post:
        log(likelihood) computed over a posterior sample. 1-D array of
        length n. If an emply array is given, then compute from posterior
        sample.

    :param array posterior_sample:
        A sample from the parameter posterior distribution.
        Dimensions are (n x k), where k is the number of parameters. If None
        the computation is done using the log(likelihood) obtained from the
        posterior sample.

    :param callable lnlikefunc:
        Function to compute ln(likelihood) on the marginal samples.

    :param tuple lnlikeargs:
        Extra arguments passed to the likelihood function.

    Other parameters
    ----------------
    :param int size:
        Size of sample to use for computation. If none is given, use size of
        given array or posterior sample.

    References
    ----------
    Kass & Raftery (1995), JASA vol. 90, N. 430, pp. 773-795
    """

    if len(lnlike_post) == 0 and posterior_sample is not None:

        samplesize = kwargs.pop('size', len(posterior_sample))

        if samplesize < len(posterior_sample):
            posterior_subsample = numpy.random.choice(posterior_sample,
                                                      size=samplesize,
                                                      replace=False)
        else:
            posterior_subsample = posterior_sample.copy()

        # Compute log likelihood in posterior sample.
        log_likelihood = lnlikefunc(posterior_subsample, *lnlikeargs)

    elif len(lnlike_post) > 0:
        samplesize = kwargs.pop('size', len(lnlike_post))
        log_likelihood = numpy.random.choice(lnlike_post, size=samplesize,
                                             replace=False)

    # Use identity for summation
    # http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction
    # ln(sum(x)) = ln(x[0]) + ln(1 + sum( exp( ln(x[1:]) - ln(x[0]) ) ) )

    hme = -lib.log_sum(-log_likelihood) + log(len(log_likelihood))

    return hme


def run_hme_mc(log_likelihood, nmc, samplesize):
    hme = np.zeros(nmc)
    for i in range(nmc):
        hme[i] = compute_harmonicmean(log_likelihood, size=samplesize)

    return hme


__author__ = 'Rodrigo F. Diaz'
