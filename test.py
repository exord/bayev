import pickle
import time
import bayev.pastislib as pl
import bayev.chib as chib
import bayev.perrakis as perr
import bayev.lib
import numpy as n
import matplotlib.pylab as plt

from math import e, log10

__author__ = 'Rodrigo F. Diaz'


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

    # Produce marginal sample from joint sample
    joint_samples = n.array(vd.values()).T
    marg_samples = perr.make_marginal_samples(joint_samples, nsamples)

    # Compute Perrakis estimate
    pe = perr.compute_perrakis_estimate(marg_samples, pl.pastis_loglike,
                                        pl.pastis_logprior,
                                        lnlikeargs=(vd.keys(), target, simul),
                                        lnpriorargs=(vd.keys(), target, simul),
                                        nbins=200)

    print('Perrakis estimate is {:.4f}'.format(pe))
    return


def test_chib_pastis(target, simul, nsamples=300):

    # Read test data
    f = open('/Users/rodrigo/homeobs/PASTIS/resultfiles/{0}/'
             '{0}_{1}_Beta1.000000_mergedchain.dat'.format(target, simul))
    vd = pickle.load(f)
    vd = vd.get_value_dict()
    f.close()

    # Remove unnecessary items
    lnlike_post = vd.pop('logL')
    post = vd.pop('posterior')
    lnpost_post = post/log10(e)
    lnprior_post = lnpost_post - lnlike_post

    # Produce marginal sample from joint sample
    posterior_sample = n.array(vd.values()).T

    # Compute Perrakis estimate
    pe = chib.compute_cj_estimate(posterior_sample, pl.pastis_loglike,
                                  pl.pastis_logprior, lnlike_post, nsamples,
                                  lnlikeargs=(vd.keys(), target, simul),
                                  lnpriorargs=(vd.keys(), target, simul),
                                  lnlike_post=lnlike_post,
                                  lnprior_post=lnprior_post)

    print('Chib estimate is {:.4f}'.format(pe))
    return


def benchmark_multivariate(nsamples=2 ** n.arange(6, 14),
                           kdim=(2**n.arange(4, 10))):

    dt = [('cholesky', n.float64),
          ('solve', n.float64)
          ]

    times = n.empty(len(kdim), dtype=dt)
    works = n.empty(len(kdim), dtype=bool)

    for i, k in enumerate(kdim):
        r = n.random.rand(nsamples, k)
        c = n.cov(r, rowvar=0)
        c[n.diag_indices_from(c)] += 1e-10

        ti = time.time()

        mvsolve = n.empty(r.shape[0])
        mvcho = n.empty(r.shape[0])

        for j in range(len(r)):
            mvcho[j] = bayev.lib.multivariate_normal(r[j], c,
                                                     method='cholesky')

        tint = time.time()
        times['cholesky'][i] = tint - ti

        for j in range(len(r)):
            mvsolve[j] = bayev.lib.multivariate_normal(r[j], c, method='solve')

        times['solve'][i] = time.time() - tint

        works[i] = n.allclose(mvsolve, mvcho)

    plt.ion()
    fig = plt.figure()
    factor = n.polyfit(n.log10(kdim), n.log10(times['cholesky']), 1)
    ax = fig.add_subplot(111)
    ax.loglog(kdim, times['cholesky']/nsamples, '-o',
              label='cholesky [{:.2f}]'.format(factor[0]))
    factor = n.polyfit(n.log10(kdim), n.log10(times['solve']), 1)
    ax.loglog(kdim, times['solve']/nsamples, '-o',
              label='solve [{:.2f}]'.format(factor[0]))
    ax.legend(loc="lower right")
    ax.set_xlabel('Vector dimension')
    ax.set_ylabel('Time [s]')
    plt.show()
    print(works)
    return times


def check_fixed_point(posterior_samples, fixed_point, labels=None):

    ax = plt.figure().add_subplot(111)
    for i in range(len(fixed_point)):
        ax.hist(posterior_samples[:, i], 100)
        ax.axvline(fixed_point[i], color='r', lw=2)
        if labels is not None:
            ax.set_title(labels[i])
        plt.draw()
        input('Hit any key to see next parameter.')
        plt.clf()
    return


if __name__ == '__main__':
    test_chib_pastis(target='PlSystem1', simul='k4d3_activityterm_noisemodel1',
                     nsamples=500)
    pass
    # benchmark_multivariate(nsamples=512)
