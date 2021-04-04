"""
Module containing front-end functions to compute Bayesian evidence estimators.
"""
import os
import pickle
import numpy as np
import importlib

from . import perrakis as perr
from . import chib as c

__author__ = 'Rodrigo F. Diaz'


def run_montecarlo(posterior_sample, lnlikefunc, lnpriorfunc, lnlikeargs,
                   lnpriorargs, methodargs, estimator='perrakis',
                   nmc=500):

    ev = np.zeros(nmc)

    if estimator == 'perrakis':

        # Retrieve parameters needed for Perrakis.
        n = methodargs.pop('nsamples', len(posterior_sample))
        densest = methodargs.pop('densityestimation', 'histogram')
        nbins = methodargs.pop('nbins', 200)

        for i in range(nmc):
            marginal_samples = perr.make_marginal_samples(posterior_sample,
                                                          nsamples=n)

            ev[i] = perr.compute_perrakis_estimate(marginal_samples,
                                                   lnlikefunc,
                                                   lnpriorfunc,
                                                   lnlikeargs=lnlikeargs,
                                                   lnpriorargs=lnpriorargs,
                                                   densityestimation=densest,
                                                   nbins=nbins)

    elif estimator == 'chib':

        # Retrieve parameters needed for Chib & Jeliazkov.
        try:
            lnlike_post = methodargs['lnlike_post']
        except KeyError:
            print('log(likelihood) in posterior sample not given. It will '
                  'be computed')
            lnlike_post = lnlikefunc(posterior_sample, *lnlikeargs)

        try:
            lnprior_post = methodargs['lnprior_post']
        except KeyError:
            print('log(prior) in posterior sample not given. It will'
                  'be computed')
            lnprior_post = lnpriorfunc(posterior_sample, *lnpriorargs)

        param_post = methodargs.pop('param_post', lnlike_post)
        n_prop = methodargs.pop('nsamples', len(posterior_sample))
        qprob = methodargs.pop('qprob', None)

        for i in range(nmc):
            ev[i] = c.compute_cj_estimate(posterior_sample, lnlikefunc,
                                          lnpriorfunc, param_post, n_prop,
                                          qprob, lnlikeargs=lnlikeargs,
                                          lnpriorargs=lnpriorargs,
                                          lnlike_post=lnlike_post,
                                          lnprior_post=lnprior_post)

    return ev


def run_montecarlo_pastis(target, simul, estimator, methodargs, nmc,
                          pastisversion='PASTIS_NM', save=True):
    """
    Function to estimate the Bayesian evidence given a posterior sample
    obtained with the PASTIS MCMC algorithm.

    :param str target:
        Name of the target used to retrieve the data and posterior simulations.

    :param str simul:
        String identifying the simulation and model for which the evidence is
        to be estimated.

    :param str estimator:
        Name of the estimator used. Options are 'chib' or 'perrakis'.

    :param methodargs:
        Dictionary containing the arguments for the estimator.

    :param int nmc:
        Number of Monte Carlo simulations to run.

    :param str pastisversion:
        Version of PASTIS to import initially. This serves only to define the
         pastispaths. For consistency, all the computations are performed using
         the version that produced the posterior sample.

    :param bool save:
        Controls if results are saved to a file.

    :return:
    """
    import pastislib as pl

    pastis = importlib.import_module(pastisversion)

    # Read data
    f = open(os.path.join(pastis.resultpath, target,
                          '{0}_{1}_Beta1.000000_mergedchain'
                          '.dat'.format(target, simul)))
    vd = pickle.load(f)
    vd = vd.get_value_dict()
    f.close()

    # Remove unnecessary items
    lnlike_post = vd.pop('logL')
    post = vd.pop('posterior')
    lnpost_post = post/np.log10(np.e)
    lnprior_post = lnpost_post - lnlike_post

    # Produce marginal sample from joint sample
    posterior_sample = np.array(vd.values()).T

    # Complete methodargs if needed
    if estimator == 'chib':
        methodargs['lnprior_post'] = lnprior_post
        methodargs['lnlike_post'] = lnlike_post
        if 'param_post' not in methodargs:
            methodargs['param_post'] = lnlike_post

    # Initialise pastis
    pl.pastis_init(target, simul)

    ev = run_montecarlo(posterior_sample, pl.pastis_loglike,
                        pl.pastis_logprior,
                        (vd.keys(), target, simul), (vd.keys(), target, simul),
                        methodargs.copy(), estimator=estimator, nmc=nmc)

    if save:
        if estimator == 'chib':
            filename = os.path.join(pastis.resultpath, target,
                                    '{0}_{1}_evidence_{2}_{3}samples.dat'
                                    ''.format(target, simul, estimator,
                                              methodargs['nsamples']))

        elif estimator == 'perrakis':
            densityestimation = methodargs.pop('densityestimation',
                                               'histogram')
            nbins = methodargs.pop('nbins', 200)

            filename = os.path.join(pastis.resultpath, target,
                                    '{0}_{1}_evidence_{2}_{3}_{4}bins.dat'
                                    ''.format(target, simul, estimator,
                                              densityestimation,
                                              nbins))

        else:
            raise NameError

        print('Saving results to file {0}'.format(filename))
        fout = open(filename, 'w')

        for i in range(len(ev)):
            fout.write('{:.6f}\n'.format(ev[i]))

        fout.close()

    return ev
