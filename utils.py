import os, sys
import jax.numpy as jnp
from jax import jit

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

# def Pdist2(X, Y):
#     # Assuming this function computes pairwise squared distances.
#     XX = jnp.sum(X*X, axis=1)[:, jnp.newaxis]
#     YY = jnp.sum(Y*Y, axis=1)[jnp.newaxis, :]
#     distances = XX + YY - 2.0 * jnp.dot(X, Y.T)
#     return distances

def compute_K_matrices(X, Y, sigma0):
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)

    Kxx = jnp.exp(-Dxx / sigma0)
    Kyy = jnp.exp(-Dyy / sigma0)
    Kxy = jnp.exp(-Dxy / sigma0)
    return Kxx, Kyy, Kxy


def compute_mmd_sq(Kxx, Kyy, Kxy, m, n):
    term1 = jnp.sum(Kxx - jnp.diag(jnp.diag(Kxx))) / (m * (m - 1))
    term2 = jnp.sum(Kyy - jnp.diag(jnp.diag(Kyy))) / (n * (n - 1))
    term3 = -2 * jnp.sum(Kxy) / (m * n)

    return term1 + term2 + term3

def Vstat_MMD(Kxx, Kyy, Kxy, m, n):
    return jnp.mean(Kxx) + jnp.mean(Kyy) - 2 * jnp.mean(Kxy)


def compute_moments(Kxx, Kyy, Kxy):
    return [
        0,
        jnp.trace(Kxx.T @ Kxx),
        jnp.sum(Kxx.T @ Kxx),
        jnp.sum(Kxx) * jnp.sum(Kxx), 
        jnp.sum(Kxx) * jnp.sum(Kyy),
        jnp.sum(Kxx @ Kxy),
        jnp.sum(Kxx) * jnp.sum(Kxy),
        jnp.sum(Kxy @ Kyy),
        jnp.sum(Kxy) * jnp.sum(Kyy),
        jnp.trace(Kxy.T @ Kxy), 
        jnp.sum(Kxy.T @ Kxy),
        jnp.sum(Kxy @ Kxy.T),
        jnp.sum(Kxy) * jnp.sum(Kxy),
        jnp.trace(Kyy.T @ Kyy),
        jnp.sum(Kyy @ Kyy),
        jnp.sum(Kyy) * jnp.sum(Kyy)
    ]

def compute_Xi_values(C, m, n, mmd_sq, complete=True):
    powers_m = [m ** i for i in range(5)]
    powers_n = [n ** i for i in range(5)]
    m1, m2, m3, m4 = powers_m[1], powers_m[2], powers_m[3], powers_m[4]
    n1, n2, n3, n4 = powers_n[1], powers_n[2], powers_n[3], powers_n[4]

    mmd2 = mmd_sq ** 2

    # def calc_xi(coefficients, denominator_power):
    #     xi_value = sum(coefficients)
    #     return xi_value / ((m1 ** denominator_power[0]) * (n1 ** denominator_power[1])) - mmd2
    
    def calc_xi(coefficients, denominator_power):
        # print(len(coefficients))
        log_numerator = jnp.log(sum(coefficients))
        log_denominator = denominator_power[0] * jnp.log(m1) + denominator_power[1] * jnp.log(n1)
        xi_value = jnp.exp(log_numerator - log_denominator)
        return xi_value - mmd2
    
    Xi = [
        # xi_01
        calc_xi([n3 * C[3], 2 * m2 * n1 * C[4], -4 * m1 * n2 * C[6], -2 * m3 * n1 * C[7], -2 * m3 * C[8], m2 * n2 * C[11], 
                 3 * m2 * n1 * C[12], m4 * C[14]], [4, 3]),
        
        # xi_02
        calc_xi([n2 * C[3], 2 * m2 * C[4], -4 * m1 * n1 * C[6], -4 * m3 * C[7], 2 * m2 * n1 * C[11], 2 * m2 * C[12], 
                 m4 * C[13]], [4, 2]),
        
        # xi_10
        calc_xi([n4 * C[2], 2 * m1 * n2 * C[4], -2 * m1 * n3 * C[5], -2 * n3 * C[6], -4 * m2 * n1 * C[8], 
                 m2 * n2 * C[10], 3 * m1 * n2 * C[12], m3 * C[15]], [3, 4]),
        
        # xi_11
        calc_xi([n3 * C[2], 2 * m1 * n1 * C[4], -2 * m1 * n2 * C[5], -2 * n2 * C[6], -2 * m2 * n1 * C[7], -2 * m2 * C[8], 
                 0.25 * m2 * n2 * C[9], 0.75 * m2 * n1 * C[10], 0.75 * m1 * n2 * C[11], 2.25 * m1 * n1 * C[12], m3 * C[14]], [3, 3]),
        
        # xi_12
        calc_xi([m1 * n2 * C[2], 2 * m1 * C[4], -2 * m1 * n1 * C[5], -2 * n1 * C[6], -4 * m2 * C[7], 0.5 * m2 * n1 * C[9], 
                 0.5 * m2 * C[10], 1.5 * m1 * n1 * C[11], 1.5 * m * C[12], m3 * C[13]], [3, 2]),
        
        # xi_20
        calc_xi([n4 * C[1], 2 * n2 * C[4], -4 * n3 * C[5], -4 * m1 * n1 * C[8], 2 * m1 * n2 * C[10], 2 * n2 * C[12], 
                 m2 * C[15]], [2, 4]),
        
        # xi_21
        calc_xi([n3 * C[1], 2 * n1 * C[4], -4 * n2 * C[5], -2 * m1 * n1 * C[7], -2 * m1 * C[8], 0.5 * m1 * n2 * C[9], 
                 1.5 * m1 * n1 * C[10], 0.5 * n2 * C[11], 1.5 * n1 * C[12], m2 * C[14]], [2, 3]),
        
        # xi_22
        calc_xi([n2 * C[1], 2 * C[4], -4 * n1 * C[5], -4 * m1 * C[7], m1 * n1 * C[9], m1 * C[10], n1 * C[11], 
                 C[12], m2 * C[13]], [2, 2])
    ]

    if complete == False:
        Xi = [Xi[0], Xi[2]]
        
    return Xi

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

# @partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def MMDVar(X, Y, sigma0, complete=True):
    m, n = len(X), len(Y)
    
    Kxx, Kyy, Kxy = compute_K_matrices(X, Y, sigma0)
    mmd_sq = Vstat_MMD(Kxx, Kyy, Kxy, m, n)
    C = compute_moments(Kxx, Kyy, Kxy)
    Xi = compute_Xi_values(C, m, n, mmd_sq, complete)
    # print(Xi)
    var = compute_var(Xi, m, n, complete)
    
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