__author__ = 'Rodrigo F. Diaz'

import pickle
import bayev.pastislib as pl
import numpy as n
from math import e


def test_pastis_logprior(nsamples=300):

    # Read test data
    f = open('/Users/rodrigo/homeobs/PASTIS/resultfiles/PlSystem2/'
             'PlSystem2_k0d3_activityterm_noisemodel1_Beta1.000000_'
             'mergedchain.dat')
    vd = pickle.load(f)
    vd = vd.get_value_dict()
    f.close()

    # Compute prior values from test data
    lnlike0 = vd.pop('logL')[:nsamples]
    log10post = vd.pop('posterior')[:nsamples]
    lnprior0 = log10post/n.log10(e) - lnlike0

    lnprior = pl.pastis_logprior(n.array(vd.values()).T[:nsamples], vd.keys(),
                                 'PlSystem2', 'k0d3_activityterm_noisemodel1')

    print('pastis_logprior works: '+n.allclose(lnprior, lnprior0))
    return


def test_pastis_loglike(nsamples=300):

    # Read test data
    f = open('/Users/rodrigo/homeobs/PASTIS/resultfiles/PlSystem2/'
             'PlSystem2_k0d3_activityterm_noisemodel1_Beta1.000000_'
             'mergedchain.dat')
    vd = pickle.load(f)
    vd = vd.get_value_dict()
    f.close()

    lnlike0 = vd.pop('logL')[:nsamples]
    vd.pop('posterior')

    lnlike = pl.pastis_loglike(n.array(vd.values()).T[:nsamples], vd.keys(),
                               'PlSystem2', 'k0d3_activityterm_noisemodel1')

    print('pastis_lnlike works: '+n.allclose(lnlike, lnlike0))
    return


def test_perrakis(nsamples=300):

    target = 'PlSystem1'
    simul = 'k5d3_activityterm_noisemodel1'

    # Read test data
    f = open('/Users/rodrigo/homeobs/PASTIS/resultfiles/{0}/'
             '{0}_{1}_Beta1.000000_mergedchain.dat'.format(target, simul))
    vd = pickle.load(f)
    vd = vd.get_value_dict()
    f.close()

    # Remove unnecessary items
    vd.pop('logL')
    vd.pop('posterior')

    import bayev.perrakis as perr

    # Produce marginal sample from joint sample
    joint_samples = n.array(vd.values()).T
    marg_samples = perr.make_marginal_samples(joint_samples, nsamples)

    # Compute Perrakis estimate
    pe = perr.compute_perrakis_estimate(marg_samples, pl.pastis_loglike,
                                        pl.pastis_logprior,
                                        likeargs=(vd.keys(), target, simul),
                                        priorargs=(vd.keys(), target, simul),
                                        nbins=200)

    print('Perrakis estimate is {:.4f}'.format(pe))
    return


if __name__ == '__main__':
    test_perrakis(nsamples=500)