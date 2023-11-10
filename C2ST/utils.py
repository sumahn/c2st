import jax.numpy as jnp 
from jax.config import config 
config.update("jax_enable_x64", True) 

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
