from ts_tests.deep_mmd_image import deep_mmd_image
from ts_tests.deep_mmd_not_image import deep_mmd_not_image
from ts_tests.mmdfuse import mmdfuse 
from ts_tests.mmd_median import mmd_median
from ts_tests.mmdagg import mmdagg 
from ts_tests.agginc import agginc
import numpy as np
from utils import HiddenPrints

def deep_mmd_test(X, Y, key, seed, use_1sample_U, complete, n_epochs=1000):
    return deep_mmd_not_image(X, Y, use_1sample_U, complete, n_epochs=n_epochs)

def deep_mmd_image_test(X, Y, key, seed, use_1sample_U, complete, n_epochs=1000):
    X = X.reshape((X.shape[0], 3, 64, 64))
    Y = Y.reshape((Y.shape[0], 3, 64, 64))
    return deep_mmd_image(X, Y, use_1sample_U, complete, n_epochs=n_epochs)

def mmdfuse_test(X, Y, key, seed):
    return int(mmdfuse(X, Y, key)) 

def mmdagg_test(X, Y, key, seed):
    return int(mmdagg(X, Y)) 

def mmdagg_test_permutation(X, Y, key, seed):
    return int(mmdagg(X, Y, permutations_same_sample_size = True)) 

def mmd_median_test(X, Y, key, seed):
    return int(mmd_median(X, Y, key))

def mmdagginc_test(X, Y, key, seed):
    return int(agginc('mmd', X, Y))