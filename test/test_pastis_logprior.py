#! /usr/bin/env python
import pickle
import bayev.pastislib as pl
import numpy as n
from math import e

f = open('/Users/rodrigo/homeobs/PASTIS/resultfiles/PlSystem2/PlSystem2_k0d3_'
         'activityterm_noisemodel1_Beta1.000000_mergedchain.dat')
vd = pickle.load(f)
f.close()

vd = vd.get_value_dict()
log1 = vd.pop('logL')[:300]
log10post = vd.pop('posterior')[:300]

logPi1 = log10post/n.log10(e) - log1

logPi = pl.pastis_logprior(n.array(vd.values()).T[:300], vd.keys(), 'PlSystem2',
                           'k0d3_activityterm_noisemodel1')

print(n.allclose(logPi1, logPi))