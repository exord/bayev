#! /usr/bin/env python
import pickle
import bayev.perrakis as perr
import bayev.pastislib as pl
import numpy as n

f = open('/Users/rodrigo/homeobs/PASTIS/resultfiles/PlSystem2/PlSystem2_k0d3_'
         'activityterm_noisemodel1_Beta1.000000_mergedchain.dat')
vd = pickle.load(f)
f.close()

vd = vd.get_value_dict()
log1 = vd.pop('logL')[:300]
#post = vd.pop('posterior)[:300]
vd.pop('posterior')

samples = n.array(vd.values()).T[:300]
marginal_samples = perr.make_marginal_samples(samples)

logL = pl.pastis_loglike(marginal_samples, vd.keys(), 'PlSystem2',
                         'k0d3_activityterm_noisemodel1')

print(n.allclose(log1, logL))
