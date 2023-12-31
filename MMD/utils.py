import os, sys
import numpy as np
import torch
import jax.numpy as jnp 
from jax.config import config 
config.update("jax_enable_x64", True) 

class HiddenPrints:
    """
    Hide prints and warnings.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr.close()
        sys.stderr = self._original_stderr

def remove_diag(A):
    """Remove diagonal elements from a matrix."""
    return A - jnp.diag(jnp.diag(A)) 

def Pdist2(x, y=None):
    """Compute the paired distance between X and Y"""
    x_norm = jnp.sum(x**2, axis=1, keepdims=True)
    
    if y is not None:
        y_norm = jnp.sum(y**2, axis=1, keepdims=True).T
    else:
        y = x
        y_norm = x_norm.T
    
    Pdist = x_norm + y_norm - 2.0 * jnp.dot(x, y.T)
    Pdist = jnp.where(Pdist < 0, 0, Pdist)
    
    return Pdist


def compute_K(X, Y, sigma0, bias=True):
    """Compute Kernel Matrices."""
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y) 
    
    Kxx = jnp.exp(-Dxx/sigma0)
    Kyy = jnp.exp(-Dyy/sigma0)
    Kxy = jnp.exp(-Dxy/sigma0)
    
    if bias:
        return Kxx, Kyy, Kxy
    else:
        tKxx = remove_diag(Kxx)
        tKyy = remove_diag(Kyy)
        return tKxx, tKyy, Kxy    
    

# Unbiased MMD Estimate
def Ustat_MMD(tKxx, tKyy, Kxy, m, n):
    term1 = jnp.sum(tKxx) / (m * (m - 1))
    term2 = jnp.sum(tKyy) / (n * (n - 1))
    term3 = -2 * jnp.sum(Kxy) / (m * n)

    return term1 + term2 + term3


def center_kernel_matrix(K):
    """Center a kernel matrix using the centering matrix H."""
    n = K.shape[0]
    H = jnp.eye(n) - (1.0 / n) * jnp.ones((n, n))
    return jnp.dot(jnp.dot(H, K), H)

def HSIC(Kxx, Kyy):
    """Compute the Hilbert-Schmidt Independence Criterion."""
    m = Kxx.shape[0]
    n = Kyy.shape[0]
    centered_Kxx = center_kernel_matrix(Kxx)
    centered_Kyy = center_kernel_matrix(Kyy)

    hsic = jnp.trace(jnp.dot(centered_Kxx, centered_Kyy)) / (m * (n - 1))
    return hsic

def jnp_to_tensor(X):
    X_np = np.array(X)
    X_tensor = torch.from_numpy(X_np)
    return X_tensor

