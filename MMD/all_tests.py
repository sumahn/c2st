from ts_tests.deep_mmd_image import deep_mmd_image
from ts_tests.deep_mmd_not_image import deep_mmd_not_image
import numpy as np

def deep_mmd_test(X, Y, key, seed, use_1sample_U, complete, n_epochs=1000):
    return deep_mmd_not_image(X, Y, use_1sample_U, complete, n_epochs=n_epochs)

def deep_mmd_image_test(X, Y, key, seed, use_1sample_U, complete, n_epochs=1000):
    X = X.reshape((X.shape[0], 3, 64, 64))
    Y = Y.reshape((Y.shape[0], 3, 64, 64))
    return deep_mmd_image(X, Y, use_1sample_U, complete, n_epochs=n_epochs)

