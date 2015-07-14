"""
Module to test the estimation of the evidence using the method by
Chib and Jeliazkov (2001).
"""
from math import *
import numpy as n
import scipy.stats as st
import random
import time

import lib
import PASTIS_NM
import PASTIS_NM.MCMC.tools as tools
import PASTIS_NM.MCMC.PASTIS_MCMC as MCMC
import PASTIS_NM.MCMC.priors as priors

__all__ = ['compute_cj_estimate', ]


def compute_cj_estimate(target, simul, nsamples=1e3, **kwargs):
    """
    Compute the evidence for simulation simul of target.

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
        Number of samples from proposal density used for computation. If None,
        use the same number of samples as posterior samples. Otherwise, the
        same number of posterior samples are used.

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

    verbose, bool
        Print result to standard output.
    """

    datadict = kwargs.pop('datadict', None)
    mergefile = kwargs.pop('mergefile', None)
    pastisfile = kwargs.pop('pastisfile', None)
    number_montecarlo_runs = kwargs.pop('nmc', 1)
    multivariate = kwargs.pop('multivariate', True)

    configdicts = lib.read_pastis_file(target, simul, pastisfile=pastisfile)
    infodict, input_dict = configdicts[0].copy(), configdicts[1].copy()
    if datadict is None:
        datadict = lib.get_datadict(target, simul, pastisfile=pastisfile)
    priordict = lib.get_priordict(target, simul, pastisfile=pastisfile)
    pastisname, vd = lib.get_posterior_samples(target, simul,
                                               mergefile=mergefile)
    ###

    # To deal with potential drifts, we need initialize to fix TrefRV.
    PASTIS_NM.initialize(infodict, datadict, input_dict)
    reload(PASTIS_NM.AstroClasses)
    reload(PASTIS_NM.ObjectBuilder)
    reload(PASTIS_NM.models.RV)
    reload(MCMC)

    """
    # INITIALISATION
    # Initialize only if needed
    checkblend = False
    checktarget = False
    for oo in input_dict.keys():
        if 'Blend' in oo:
            checkblend = True

        elif 'Target' in oo or 'PlanetHost' in oo:
            checktarget = True

    # Construct initialization condition
    if checkblend:
        init_blend = 'verticesY' not in PASTIS_NM.isochrones.__dict__
    else:
        init_blend = False

    if checktarget:
        init_target = 'verticesT' not in PASTIS_NM.isochrones.__dict__
    else:
        init_target = False

    # init_atmosphericmodels = not 'AMz' in PASTIS_NM.photometry.__dict__
    init_atmosphericmodels = False

    if init_blend:
        print('Will initialize blend tracks.')
    if init_target:
        print('Will initialize target tracks.')
    if init_atmosphericmodels:
        print('Will initialize atm models.')

    if init_blend or init_target or init_atmosphericmodels:
        PASTIS_NM.initialize(infodict, datadict, input_dict)
    reload(PASTIS_NM.AstroClasses)
    reload(PASTIS_NM.ObjectBuilder)
    reload(MCMC)
    reload(PASTIS_NM.models.PHOT)
    reload(PASTIS_NM.models.RV)
    reload(PASTIS_NM.models.SED)
    """

    if 'TrefRV' in infodict:
        PASTIS_NM.AstroClasses.TrefRV = infodict['TrefRV']
    ###
    ###

    # ## Get prior and likelihood values for this element

    # Find mode of logL and posterior
    nbins = n.max([n.min([len(vd['logL']) / 10, 100]), 10])

    print('Computing posterior mode using {} bins.'.format(nbins))

    pdf, binedges = n.histogram(vd['logL'], nbins, normed=True)
    maxind = n.argmax(pdf)
    mode_loglikelihood = 0.5 * (binedges[maxind] + binedges[maxind + 1.0])

    # Index closest to logL mode.
    mode_loglikelihoodindex = n.argmin(n.abs(vd['logL'] - mode_loglikelihood))
    post0 = vd['posterior'][mode_loglikelihoodindex]
    loglike0 = vd['logL'][mode_loglikelihoodindex]

    prior0 = 10 ** (post0 - loglike0 / log(10))

    # Copy value dictionary
    vvd = vd.copy()
    vvd.pop('logL')
    vvd.pop('posterior')

    # Get values from value dictionary
    keys = vvd.keys()
    joint_posterior_samples = n.array(vvd.values())

    # Maximum posterior index:
    # Find best peak of posterior, that we will use as q*
    best_values = joint_posterior_samples[:, mode_loglikelihoodindex]

    # Other methods of finding "best point" (deactivated)
    """
    binedges, pdf = n.histogram(vd['posterior'], nbins, normed = True)
    maxind = n.argmax(pdf)
    post0 = 0.5*(binedges[maxind] + binedges[maxind + 1.0])
    """

    """
    ##Find maximum maximum value of likelihood and logL.
    maxind = n.argmax(vd['logL'])
    loglike0 = vd['logL'][maxind]
    post0 = vd['posterior'][maxind]
    """

    # Set nsamples if not given
    if nsamples is None:
        nsamples = len(vd['logL'])
        print('Using nsamples = %d' % nsamples)

    # Select randomly nsamples from the posterior samples
    indices_posterior = n.arange(joint_posterior_samples.shape[1])
    random.shuffle(indices_posterior)
    indices_posterior = indices_posterior[: nsamples]

    ###
    # Compute transition probability alpha over posterior sample
    ###
    vd['priors'] = 10 ** (vd['posterior'] - vd['logL'] / log(10))

    r_posterior = prior0 / vd['priors'][indices_posterior] * \
        n.exp(loglike0 - vd['logL'][indices_posterior])
    alpha_posterior = n.where(r_posterior > 1.0, 1.0, r_posterior)

    # Compute value of transition kernel on posterior sample.
    xx = joint_posterior_samples[:, indices_posterior[: nsamples]].T - \
        best_values

    if multivariate:

        # Compute covariance matrix
        # (actually correlation coefficient matrix, to avoid issue in the
        # generation of correlated normal samples).

        correlation_matrix = n.corrcoef(joint_posterior_samples)
        covariance = n.cov(joint_posterior_samples)

        print(covariance.shape, xx.shape)
        alpha = n.linalg.solve(covariance, xx.T)
        # Inverse correlation coefficient matrix and determinant.

        # CHANGE THIS! USE n.linalg.solve
        # cov_inv = n.linalg.inv(n.cov(joint_posterior_samples))

        # Compute log determinant of the covariance matrix.
        sign_det_cov_inv, logdet_cov = n.linalg.slogdet(covariance)
        if sign_det_cov_inv != 1:
            raise ArithmeticError('Covariance matrix has negative determinant.')

        # Compute exponent q of the multinormal (for each sample of posterior)
        # q = n.diag(n.dot(xx, n.dot(cov_inv, xx.T)))
        q = n.diag(n.dot(xx, alpha))

        # Evaluate log of transition kernel PDF on posterior samples
        # log( (2*pi)**N * detC)**(1/2) * exp(-0.5 * q) )
        log_transkernel_posterior = -0.5 * (q +
                                            covariance.shape[0] * log(2 * pi) +
                                            logdet_cov)

    else:
        # Use gaussians with deviation equal to the std of each parameter
        # Construct (diagonal) correlation matrix
        correlation_matrix = n.identity(len(vvd))

        qpar = n.zeros(joint_posterior_samples.shape)
        stdpar = n.std(joint_posterior_samples, axis=1)
        for parindex in range(joint_posterior_samples.shape[0]):
            qpar[parindex] = st.norm.pdf(xx[:, parindex],
                                         scale=stdpar[parindex], loc=0)

        # Get q(x' | x) for all parameters. As we consider them independent
        # this is just the product.
        log_transkernel_posterior = n.sum(n.log(qpar), axis=0)

    # Mmmmmmm, we don't have a fixed transition kernel, q(x', x)
    # But we need to sample from q(x', x).

    # Perform computation number_montecarlo_runs times
    chib_evidence = n.zeros(number_montecarlo_runs)

    timei = time.time()
    # Iterate for monte carlo estimation of error.
    for iteration in range(number_montecarlo_runs):

        # Get nsamples from q(x', x), estimated as a multinormal with the
        # covariance of the final posterior samples

        # Get normal samples centred in zero; multiply by variance and add best.
        qsample = n.random.multivariate_normal(mean=n.zeros(len(
            correlation_matrix)),
            cov=correlation_matrix,
            size=nsamples)

        # Re-scale and shift sample
        qsample = qsample * joint_posterior_samples.std(axis=1)
        qsample = qsample + best_values

        # Tweak for cyclic variables.
        for param_index, parameter in enumerate(keys):
            if 'omega' in parameter or 'L0' in parameter or 'M0' in parameter:
                qsample[:, param_index] = qsample[:, param_index] % 360.0

        # Now we need to compute  alpha, the transition probability from the
        # M-H algorithm over this sample.
        loglike = n.zeros(nsamples)
        prior = n.zeros(nsamples)

        # Compute prior and likelihood for each sample.
        for i in range(nsamples):

            for parindex, par in enumerate(vvd.keys()):

                obj, ppar = par.split('_')

                ppar = ppar.replace('AR', 'ar')
                # Fix values on input_dict
                input_dict[obj][ppar][0] = qsample[i, parindex]

            # Construct state
            chainstate, labeldict = tools.state_constructor(input_dict)

            # Compute priors
            prior[i], priorprob = priors.compute_priors(priordict, labeldict)

            if prior[i] == 0:
                loglike[i] = n.nan
                continue

            try:
                PASTIS_NM.ObjectBuilder.ObjectBuilder(input_dict)
            except ValueError as err:
                print(err.message)
                loglike[i] = n.nan
                prior[i] = 0.0
                continue

            # Compute likelihood
            try:
                likelihood, loglike[i], likedict = \
                    MCMC.get_likelihood(chainstate, input_dict, datadict,
                                        labeldict, False, False)

            except:
                loglike[i] = n.nan
                prior[i] = 0.0
                continue

        # Compute Metropolis ratio and transition probability alpha over sample
        # of q(x'|x)
        r_q = prior / prior0 * n.exp(loglike - loglike0)

        # Replace samples outside priors with 0 (as suggested by C&J)
        r_q = n.where(n.isnan(r_q), 0, r_q)
        alpha_q = n.where(r_q > 1.0, 1.0, r_q)

        if n.all(alpha_q == 0.0):
            print alpha_q
            raise ValueError(
                'All samples from proposal density have zero transition '
                'probability from best-value. Increase Nsample.')

        ###
        # Compute estimate of the log posterior ordinate
        ###
        log_summands_num = n.log(alpha_posterior) + log_transkernel_posterior
        # log_summands_den = n.log(alpha_q)

        # Compute log of posterior ordinate
        # Use http://en.wikipedia.org/wiki/List_of_logarithmic_identities#Summation.2Fsubtraction
        """
        log_post_ordinate = lib.log_sum(log_summands_num) - log(
            len(log_summands_num)) - lib.log_sum(log_summands_den) + log(
                len(log_summands_den))
        """

        log_post_ordinate = lib.log_sum(log_summands_num) - log(
            len(log_summands_num)) - log(alpha_q.mean())

        ###
        # Compute estimate evidence
        ###
        chib_evidence[iteration] = loglike0 + log(prior0) - log_post_ordinate

    print('Running time without initialisation: '
          '{:.2f} minutes'.format((time.time() - timei)/60.0))
    return chib_evidence