import jax.numpy as jnp
from jax.config import config
config.update("jax_enable_x64", True)

def remove_diag(A):
    """Remove diagonal elements from a square matrix."""
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

# Kernel Matrices
def compute_K_matrices(X, Y,sigma0):
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)

    Kxx = jnp.exp(-Dxx / sigma0)
    Kyy = jnp.exp(-Dyy / sigma0)
    Kxy = jnp.exp(-Dxy / sigma0)

    return Kxx, Kyy, Kxy

def xi_10(Kxx, Kyy, Kxy):
    m, _ = Kxx.shape
    n, _ = Kyy.shape
    
    # Remove diagonal elements
    Kxx_tilde = remove_diag(Kxx)
    Kyy_tilde = remove_diag(Kyy)
    
    one_m = jnp.ones(m)
    one_n = jnp.ones(n)
    
    m_3 = m * (m - 1) * (m - 2)
    m_4 = m_3 * (m - 3)
    
    n_3 = n * (n - 1) * (n - 2)
    n_4 = n_3 * (n - 3)
    
    # Compute terms as per your equation
    term1 = (1 / m_3) * (jnp.linalg.norm(jnp.dot(Kxx_tilde, one_m)) ** 2 - jnp.linalg.norm(Kxx_tilde, 'fro') ** 2)
    term2 = (1 / m_4) * ((one_m.T @ jnp.dot(Kxx_tilde, one_m)) ** 2 - 4 * (jnp.linalg.norm(jnp.dot(Kxx_tilde, one_m)) ** 2) + 2 * jnp.linalg.norm(Kxx_tilde, 'fro') ** 2)

    term3 = (1 / (n * (n - 1))) * jnp.linalg.norm(Kyy_tilde, 'fro') ** 2
    term4 = - (1 / n_4) * ((one_n.T @ jnp.dot(Kyy_tilde, one_n)) ** 2 - 4 * (jnp.linalg.norm(jnp.dot(Kyy_tilde, one_n)) ** 2) + 2 * jnp.linalg.norm(Kyy_tilde, 'fro') ** 2)

    term5 = (1 / n) * (1 / (m * (m - 1))) * (jnp.linalg.norm(jnp.dot(Kxy.T, one_m)) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)
    term6 = - (1 / (m * (m - 1) * n * (n - 1))) * ((one_m.T @ jnp.dot(Kxy, one_n)) ** 2 - jnp.linalg.norm(jnp.dot(Kxy.T, one_m)) ** 2 - jnp.linalg.norm(jnp.dot(Kxy, one_n)) ** 2 + jnp.linalg.norm(Kxy, 'fro') ** 2)

    # term7 = (1 / 2) * (1 / (m * n)) * jnp.linalg.norm(Kxy, 'fro') ** 2
    term7 = (1) * (1 / (m * n)) * jnp.linalg.norm(Kxy, 'fro') ** 2
    # term8 = - (1 / 2) * (1 / (m * (m - 1) * n * (n - 1))) * ((one_m.T @ jnp.dot(Kxy, one_n)) ** 2 - jnp.linalg.norm(jnp.dot(Kxy.T, one_m)) ** 2 - jnp.linalg.norm(jnp.dot(Kxy, one_n)) ** 2 + jnp.linalg.norm(Kxy, 'fro') ** 2)
    term8 = - (1) * (1 / (m * (m - 1) * n * (n - 1))) * ((one_m.T @ jnp.dot(Kxy, one_n)) ** 2 - jnp.linalg.norm(jnp.dot(Kxy.T, one_m)) ** 2 - jnp.linalg.norm(jnp.dot(Kxy, one_n)) ** 2 + jnp.linalg.norm(Kxy, 'fro') ** 2)

    term9 = - (2 / n) * (1 / (m * (m - 1))) * one_m.T @ jnp.dot(Kxx_tilde, jnp.dot(Kxy, one_n))
    term10 = (2 / (n * m_3)) * (one_m.T @ jnp.dot(Kxx_tilde, one_m) * one_m.T @ jnp.dot(Kxy, one_n) - 2 * one_m.T @ jnp.dot(Kxx_tilde, jnp.dot(Kxy, one_n)))

    term11 = - (2 / m) * (1 / (n * (n - 1))) * one_n.T @ jnp.dot(Kyy_tilde, jnp.dot(Kxy.T, one_m))
    term12 = (2 / (n * m_3)) * (one_m.T @ jnp.dot(Kxx_tilde, one_m) * one_m.T @ jnp.dot(Kxy, one_n) - 2 * one_m.T @ jnp.dot(Kxx_tilde, jnp.dot(Kxy, one_n)))

    term13 = - (2 / m) * (1 / (n * (n - 1))) * one_n.T @ jnp.dot(Kyy_tilde, jnp.dot(Kxy.T, one_m))
    term14 = (2 / (m * n_3)) * (one_n.T @ jnp.dot(Kyy_tilde, one_n) * one_n.T @ jnp.dot(Kxy.T, one_m) - 2 * one_n.T @ jnp.dot(Kyy_tilde, jnp.dot(Kxy.T, one_m)))

    term15 = (2 / n) * (1 / (m * (m - 1))) * (jnp.linalg.norm(jnp.dot(Kxy.T, one_m)) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)
    term16 = - (2 / (m * (m - 1) * n * (n - 1))) * ((one_m.T @ jnp.dot(Kxy, one_n)) ** 2 - jnp.linalg.norm(jnp.dot(Kxy.T, one_m)) ** 2 - jnp.linalg.norm(jnp.dot(Kxy, one_n)) ** 2 + jnp.linalg.norm(Kxy, 'fro') ** 2)
    
    
    xi_1_0 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13 + term14 + term15 + term16
    
    return xi_1_0

def xi_01(Kxx, Kyy, Kxy):
    m, _ = Kxx.shape
    n, _ = Kyy.shape
    
    # Remove diagonal elements
    Kxx_tilde = remove_diag(Kxx)
    Kyy_tilde = remove_diag(Kyy)
    
    ones_m = jnp.ones(m)
    ones_n = jnp.ones(n)
    
    m_4 = m * (m - 1) * (m - 2) * (m - 3)
    n_3 = n * (n - 1) * (n - 2)
    n_4 = n_3 * (n - 3)
    mn_3 = m * n * (n - 1)
    mn_4 = m * (m - 1) * n * (n - 1)
    
    term1 = jnp.linalg.norm(Kxx_tilde, "fro") ** 2 / (m * (m - 1))
    term2 = ((ones_m.T @ Kxx_tilde @ ones_m) ** 2 - 4 * jnp.linalg.norm(Kxx_tilde @ ones_m) ** 2 + 2 * jnp.linalg.norm(Kxx_tilde, "fro") ** 2) / (-m_4)
    
    term3 = ((jnp.linalg.norm(Kyy_tilde @ ones_n) ** 2 - jnp.linalg.norm(Kyy_tilde, "fro") ** 2) / n_3)
    term4 = ((ones_n.T @ Kyy_tilde @ ones_n) ** 2 - 4 * jnp.linalg.norm(Kyy_tilde @ ones_n) ** 2 + 2 * jnp.linalg.norm(Kyy_tilde, "fro") ** 2) / (-n_4)
    
    term5 = (jnp.linalg.norm(Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy, "fro") ** 2) / mn_3
    term6 = ((ones_m.T @ Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 - jnp.linalg.norm(Kxy @ ones_n) ** 2 + jnp.linalg.norm(Kxy, "fro") ** 2) / (-mn_4)
    
    # term7 = 0.5 * jnp.linalg.norm(Kxy, "fro") ** 2 / (m * n)
    term7 = 1 * jnp.linalg.norm(Kxy, "fro") ** 2 / (m * n)
    # term8 = ((ones_m.T @ Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 - jnp.linalg.norm(Kxy @ ones_n) ** 2 + jnp.linalg.norm(Kxy, "fro") ** 2) / (-2 * mn_4)
    term8 = ((ones_m.T @ Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 - jnp.linalg.norm(Kxy @ ones_n) ** 2 + jnp.linalg.norm(Kxy, "fro") ** 2) / (-1 * mn_4)
    
    term9 = -2 * ones_m.T @ Kxx_tilde @ Kxy @ ones_n / (n * (m * (m - 1)))
    term10 = 2 * (ones_m.T @ Kxx_tilde @ ones_m * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ Kxx_tilde @ Kxy @ ones_n) / (n * m_4)
    
    term11 = -2 * ones_m.T @ Kxx_tilde @ Kxy @ ones_n / (n * (m * (m - 1)))
    term12 = 2 * (ones_m.T @ Kxx_tilde @ ones_m * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ Kxx_tilde @ Kxy @ ones_n) / (n * m_4)

    term13 = -2 * ones_n.T @ Kyy_tilde @ Kxy.T @ ones_m / (m * n_3)
    term14 = 2 * (ones_n.T @ Kyy_tilde @ ones_n * ones_n.T @ Kxy.T @ ones_m - 2 * ones_n.T @ Kyy_tilde @ Kxy.T @ ones_m) / (m * n_4)
    
    term15 = 2 * (jnp.linalg.norm(Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy, "fro") ** 2) / (m * n_3)
    term16 = ((ones_m.T @ Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 - jnp.linalg.norm(Kxy @ ones_n) ** 2 + jnp.linalg.norm(Kxy, "fro") ** 2) / (-2 * mn_4)
    
    xi_01 = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12 + term13 + term14 + term15 + term16
    return xi_01

# Unbiased MMD Estimate
def compute_mmd_sq(Kxx, Kyy, Kxy, m, n):
    term1 = jnp.sum(Kxx - jnp.diag(jnp.diag(Kxx))) / (m * (m - 1))
    term2 = jnp.sum(Kyy - jnp.diag(jnp.diag(Kyy))) / (n * (n - 1))
    term3 = -2 * jnp.sum(Kxy) / (m * n)

    return term1 + term2 + term3

# Biased MMD Estimate
def Vstat_MMD(Kxx, Kyy, Kxy, m, n) :
    return jnp.mean(Kxx) + jnp.mean(Kyy) - 2 * jnp.mean(Kxy)

def compute_var(Xi, m, n, complete = True):
    denom = m * (m - 1) * n * (n - 1)
    term_coefficients = [
        (4 * (m - 2) * (m - 3) * (n - 2)),
        (2 * (n - 2) * (n - 3)),
        (4 * (m - 2) * (n - 2) * (n - 3)),
        (16 * (n - 2) * (m - 2)),
        (8 * (m - 2)),
        (2 * (n - 2) * (n - 3)),
        (2 * (n - 2)),
        (4)
    ]
    
    if complete:
        terms = [coeff * Xi[i] / denom for i, coeff in enumerate(term_coefficients)]
    else:
        terms = [(term_coefficients[0] * Xi[0] / denom), (term_coefficients[2] * Xi[1] / denom)]
    return sum(terms)

# @partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def MMDVar(X, Y, sigma0, complete=True, bias = True):
    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X
        
    m, n = X.shape[0], Y.shape[0]

    assert n <= m
    assert n >= 2 and m >= 2  
    
    Kxx, Kyy, Kxy = compute_K_matrices(X, Y, sigma0)
    
    if bias == True:
        mmd_sq = Vstat_MMD(Kxx, Kyy, Kxy, m, n) # Compute all of component as V statistics
    else:
        mmd_sq = compute_mmd_sq(Kxx, Kyy, Kxy, m, n)
    Xi = [xi_01(Kxx, Kyy, Kxy), xi_10(Kxx, Kyy, Kxy)]
    var = compute_var(Xi, m, n, complete)

    return var

def MMDu_var(Kxx, Kyy, Kxy) :
    nx = Kxx.shape[0]
    ny = Kyy.shape[0]

    mmd2 = jnp.mean(Kxx)+ jnp.mean(Kyy) - 2 * jnp.mean(Kxy)

    one_x = jnp.ones(nx)
    one_y = jnp.ones(ny)

    h1 = ((one_x.T @ Kxx)/nx) + ((one_y.T @ Kyy @ one_y)/(ny**2)) - ((Kxy @ one_y)/(ny)) - ((one_x.T @ Kxy @ one_y)/(nx*ny))
    h2 = ((one_y.T @ Kyy)/ny) + ((one_x.T @ Kxx @ one_x)/(nx**2)) - ((one_x.T @ Kxy)/(nx)) - ((one_x.T @ Kxy @ one_y)/(nx*ny))

    xi_1 = (h1.T @ h1)/nx - (mmd2 ** 2)
    xi_2 = (h2.T @ h2)/ny - (mmd2 ** 2)

    var = (4*xi_1/nx) + (4*xi_2/ny)

    return var


def h1_mean_var_gram(Kxx, Kyy, Kxy, is_var_computed, use_1sample_U=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = jnp.concatenate((Kxx, Kxy), axis=1)
    Kyxy = jnp.concatenate((jnp.transpose(Kxy), Kyy), axis=1)
    Kxyxy = jnp.concatenate((Kxxy, Kyxy), axis=0)
    nx = Kxx.shape[0]
    ny = Kyy.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = (jnp.sum(Kxx) - jnp.sum(jnp.diag(Kxx))) / (nx * (nx - 1))
        yy = (jnp.sum(Kyy) - jnp.sum(jnp.diag(Kyy))) / (ny * (ny - 1))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = (jnp.sum(Kxy) - jnp.sum(jnp.diag(Kxy))) / (nx * (ny - 1))
        else:
            xy = jnp.sum(Kxy) / (nx * ny)
        mmd2 = xx - 2 * xy + yy
    else:
        xx = jnp.sum(Kxx) / (nx * nx)
        yy = jnp.sum(Kyy) / (ny * ny)
        # one-sample U-statistic.
        if use_1sample_U:
            xy = jnp.sum(Kxy) / (nx * ny)
        else:
            xy = jnp.sum(Kxy) / (nx * ny)
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None

    hh = Kxx + Kyy - Kxy - jnp.transpose(Kxy)
    V1 = jnp.dot(hh.sum(axis=1) / ny, hh.sum(axis=1) / ny) / ny
    V2 = hh.sum() / nx / nx
    varEst = 4 * (V1 - V2**2)
    if varEst == 0.0:
        print('error!!' + str(V1))
    return mmd2, varEst/nx, Kxyxy

