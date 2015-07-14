import numpy as np
import random


def log_sum(log_summands):
    a = np.inf
    while a == np.inf:
        a = log_summands[0] + np.log(1 + np.sum(np.exp(log_summands[1:] -
                                                       log_summands[0])))
        random.shuffle(log_summands)
    return a
