import os, sys
import jax.numpy as jnp
from jax import jit
from mpmath import mp

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


def compute_K_matrices(X, Y,sigma0):
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)


    Kxx = jnp.exp(-Dxx / sigma0)
    Kyy = jnp.exp(-Dyy / sigma0)
    Kxy = jnp.exp(-Dxy / sigma0)
    
    # Kxx = Kxx - jnp.diag(jnp.diag(Kxx))
    # Kyy = Kyy - jnp.diag(jnp.diag(Kyy))
    
    return Kxx, Kyy, Kxy

def compute_mmd_sq(Kxx, Kyy, Kxy, m, n):
    term1 = jnp.sum(Kxx - jnp.diag(jnp.diag(Kxx))) / (m * (m - 1))
    term2 = jnp.sum(Kyy - jnp.diag(jnp.diag(Kyy))) / (n * (n - 1))
    term3 = -2 * jnp.sum(Kxy) / (m * n)

    return term1 + term2 + term3

def Vstat_MMD(Kxx, Kyy, Kxy, m, n) :
    return jnp.mean(Kxx) + jnp.mean(Kyy) - 2 * jnp.mean(Kxy)


# @jit
def compute_moments(Kxx, Kyy, Kxy):
    Kxx = Kxx - jnp.diag(jnp.diag(Kxx))
    Kyy = Kyy - jnp.diag(jnp.diag(Kyy))
    return [
        0,
        jnp.trace(Kxx.T @ Kxx),
        jnp.sum(Kxx.T @ Kxx),
        jnp.sum(Kxx * jnp.sum(Kxx)),
        jnp.sum(Kxx * jnp.sum(Kyy)),
        jnp.sum(Kxx @ Kxy),
        jnp.sum(Kxx*jnp.sum(Kxy)),
        jnp.sum(Kxy @ Kyy),
        jnp.sum(Kxy*jnp.sum(Kyy)),
        jnp.trace(Kxy.T @ Kxy),
        jnp.sum(Kxy.T @ Kxy),
        jnp.sum(Kxy @ Kxy.T),
        jnp.sum(Kxy*jnp.sum(Kxy)),
        jnp.trace(Kyy.T @ Kyy),
        jnp.sum(Kyy.T @ Kyy),
        jnp.sum(Kyy*jnp.sum(Kyy))
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
    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X
        
    m, n = X.shape[0], Y.shape[0]

    assert n <= m
    assert n >= 2 and m >= 2  
    
    Kxx, Kyy, Kxy = compute_K_matrices(X, Y, sigma0)
    # mmd_sq = compute_mmd_sq(Kxx, Kyy, Kxy, m, n)
    mmd_sq = Vstat_MMD(Kxx, Kyy, Kxy, m, n) # Compute all of component as V statistics
    C = compute_moments(Kxx, Kyy, Kxy)
    Xi = compute_Xi_values(C, m, n, mmd_sq, complete)
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

def falling_factorial(n, k):
    """Computes the falling factorial (n)_k."""
    return jnp.prod(jnp.arange(n - k + 1, n + 1)) 

# def V_hat(Kxx, Kyy, Kxy, m, n):
#     n = Kxx.shape[0]  # Assuming K_xx, K_yy, K_xy are square matrices of shape (n, n)

#     ones = jnp.ones((n,))

#     # Off-diagonal versions of K_xx, K_yy, K_xy
#     K_tilde_xx = Kxx - jnp.diag(jnp.diag(Kxx))
#     K_tilde_yy = Kyy - jnp.diag(jnp.diag(Kyy))
    
#     # Sub-expressions based on the provided approximations
#     mu_X_mu_X = (jnp.dot(ones, jnp.dot(K_tilde_xx, ones))) / (n * (n - 1))
#     mu_Y_mu_Y = (jnp.dot(ones, jnp.dot(K_tilde_yy, ones))) / (n * (n - 1))
#     mu_X_mu_Y = (jnp.dot(ones, jnp.dot(Kxy, ones))) / (n * n)
    
#     mu_X_C_X_mu_X = jnp.square(jnp.linalg.norm(jnp.dot(K_tilde_xx, ones))) / ((n)*(n-1)*(n-2)) - jnp.linalg.norm(K_tilde_xx)**2 / ((n)*(n-1)*(n-2))
#     mu_Y_C_Y_mu_Y = jnp.square(jnp.linalg.norm(jnp.dot(K_tilde_yy, ones))) / ((n)*(n-1)*(n-2)) - jnp.linalg.norm(K_tilde_yy)**2 / ((n)*(n-1)*(n-2))
#     mu_Y_C_X_mu_Y = jnp.square(jnp.linalg.norm(jnp.dot(Kxy, ones))) / (n * n * (n - 1)) - jnp.linalg.norm(Kxy)**2 / (n * n * (n - 1))
#     mu_X_C_X_mu_Y = (jnp.dot(ones, jnp.dot(K_tilde_xx, jnp.dot(Kxy, ones)))) / (n * n * (n - 1))
#     mu_Y_C_Y_mu_X = (jnp.dot(ones, jnp.dot(K_tilde_yy, jnp.dot(Kxy, ones)))) / (n * n * (n - 1))
    
#     E_k_X1_X2 = jnp.linalg.norm(Kxy)**2 / (n * n)
    
#     # Calculating V_m based on the final expression
#     V_m = (
#         (4*(m-2)/(m*(m-1))) * (mu_X_C_X_mu_X + mu_Y_C_Y_mu_Y + mu_Y_C_X_mu_Y) -
#         (2*(2*m-3)/(m*(m-1))) * (mu_X_mu_X**2 + mu_Y_mu_Y**2) -
#         (4*(2*m-3)/(m*(m-1))) * (mu_X_mu_Y ** 2) -
#         (8/m) * (mu_X_C_X_mu_Y) +
#         (8/m) * mu_X_mu_X * mu_X_mu_Y -
#         (8/m) * mu_Y_C_Y_mu_X +
#         (8/m) * mu_Y_mu_Y * mu_X_mu_Y +
#         (2/(m*(m-1))) * (2 * E_k_X1_X2)
#     )
    
#     return V_m



def V_hat(Kxx, Kyy, Kxy, m, n):
    """Computes V_hat based on the given equation."""
    
    tKxx = Kxx - jnp.diag(jnp.diag(Kxx))
    tKyy = Kyy - jnp.diag(jnp.diag(Kyy))

    ones = jnp.ones(n)  # Assuming Kxx, Kyy, and Kxy are square matrices with shape (n, n)

    # Calculate each term in the equation
    term1 = 4 * (m * n + m - 2 * n) / (falling_factorial(m, 2) * falling_factorial(n, 4)) * (
        jnp.linalg.norm(tKxx @ ones) ** 2 + jnp.linalg.norm(tKyy @ ones) ** 2
    )

    term2 = -2 * (2 * m - n) / (m * n * (m - 1) * (n - 2) * (n - 3)) * (
        jnp.linalg.norm(tKxx, 'fro') ** 2 + jnp.linalg.norm(tKyy, 'fro') ** 2
    )

    term3 = 4 * (m * n + m - 2 * n - 1) / (falling_factorial(m, 2) * n ** 2 * (n - 1) ** 2) * (
        jnp.linalg.norm(Kxy @ ones) ** 2 + jnp.linalg.norm(Kxy.T @ ones) ** 2
    )

    term4 = -4 * (2 * m - n - 2) / (falling_factorial(m, 2) * n * (n - 1) ** 2) * (
        jnp.linalg.norm(Kxy, 'fro') ** 2
    )

    term5 = -2 * (2 * m - 3) / (falling_factorial(m, 2) * falling_factorial(n, 4)) * (
        (ones.T @ tKxx @ ones) ** 2 + (ones.T @ tKyy @ ones) ** 2
    )

    term6 = -4 * (2 * m - 3) / (falling_factorial(m, 2) * n ** 2 * (n - 1) ** 2) * (
        (ones.T @ Kxy @ ones) ** 2
    )

    term7 = -8 / (m * falling_factorial(n, 3)) * (
        ones.T @ tKxx @ Kxy @ ones + ones.T @ tKyy @ Kxy.T @ ones
    )

    term8 = 8 / (m * n * falling_factorial(n, 3)) * (
        (ones.T @ tKxx @ ones + ones.T @ tKyy @ ones) * (ones.T @ Kxy @ ones)
    )

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8

# Update for unbiased estimator


def Xi_10(Kxx, Kyy, Kxy, m ,n) :
    tKxx = Kxx - jnp.diag(jnp.diag(Kxx))
    tKyy = Kyy - jnp.diag(jnp.diag(Kyy))

    ones_n = jnp.ones(n) 
    ones_m = jnp.ones(m)
    
    term1 =  (jnp.linalg.norm(tKxx @ ones_m) ** 2 - jnp.linalg.norm(tKxx, 'fro') ** 2) / (falling_factorial(m, 3))
    
    term2 = ((ones_m.T @ tKxx @ ones_m) **2 - 4 * jnp.linalg.norm(tKxx @ ones_m) ** 2 + 2* jnp.linalg.norm(tKxx, 'fro') ** 2)/(falling_factorial(m, 4))
    
    term3 = (jnp.linalg.norm(Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(m * falling_factorial(n,2))
    
    term4 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 -jnp.linalg.norm(Kxy @ ones_n) ** 2+jnp.linalg.norm(Kxy, 'fro') ** 2)/(
        falling_factorial(m, 2) * falling_factorial(n, 2)
        )
    
    term5 = (ones_m @ tKxx @ Kxy @ ones_n) /(n * falling_factorial(m,2))
    
    term6 = (ones_m.T @ tKxx @ ones_m * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ tKxx @ Kxy @ ones_n) / (falling_factorial(m,3) * n)
    
    return term1 - term2 + term3 - term4 -(2 * term5) + (2 * term6)
    

def Xi_01(Kxx, Kyy, Kxy, m ,n) : 
    tKxx = Kxx - jnp.diag(jnp.diag(Kxx))
    tKyy = Kyy - jnp.diag(jnp.diag(Kyy))

    ones_n = jnp.ones(n) 
    ones_m = jnp.ones(m)
    
    term1 =  (jnp.linalg.norm(tKyy @ ones_n) ** 2 - jnp.linalg.norm(tKyy, 'fro') ** 2) / (falling_factorial(n, 3))
    
    term2 = ((ones_n.T @ tKyy @ ones_n) **2 - 4 * jnp.linalg.norm(tKyy @ ones_n) ** 2 + 2* jnp.linalg.norm(tKyy, 'fro') ** 2)/(falling_factorial(n, 4))
    
    term3 = (jnp.linalg.norm(Kxy.T @ ones_m) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(n * falling_factorial(m,2))
    
    term4 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 -jnp.linalg.norm(Kxy @ ones_n) ** 2+jnp.linalg.norm(Kxy, 'fro') ** 2)/(
        falling_factorial(m, 2) * falling_factorial(n, 2)
        )
    
    term5 = (ones_m @ Kxy @ tKyy @ ones_n) /(m * falling_factorial(n,2))
    
    term6 = (ones_n.T @ tKyy @ ones_n * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ Kxy @ tKyy @ ones_n) / (falling_factorial(n,3) * m)
    
    return term1 - term2 + term3 - term4 -(2 * term5) + (2 * term6)

def Xi_11(Kxx, Kyy, Kxy, m ,n) : 
    tKxx = Kxx - jnp.diag(jnp.diag(Kxx))
    tKyy = Kyy - jnp.diag(jnp.diag(Kyy))

    ones_n = jnp.ones(n) 
    ones_m = jnp.ones(m)
    
    term1 = (jnp.linalg.norm(tKxx @ ones_m) ** 2 - jnp.linalg.norm(tKxx, 'fro') ** 2) / (falling_factorial(m, 3))
    term2 = ((ones_m.T @ tKxx @ ones_m) **2 - 4 * jnp.linalg.norm(tKxx @ ones_m) ** 2 + 2* jnp.linalg.norm(tKxx, 'fro') ** 2)/(falling_factorial(m, 4))
    term3 = (jnp.linalg.norm(tKyy @ ones_n) ** 2 - jnp.linalg.norm(tKyy, 'fro') ** 2) / (falling_factorial(n, 3))
    term4 = ((ones_n.T @ tKyy @ ones_n) **2 - 4 * jnp.linalg.norm(tKyy @ ones_n) ** 2 + 2* jnp.linalg.norm(tKyy, 'fro') ** 2)/(falling_factorial(n, 4))
    term5 = (jnp.linalg.norm(Kxy, 'fro') ** 2)/(m*n)
    term6 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m)**2- jnp.linalg.norm(Kxy @ ones_n)**2 + jnp.linalg.norm(Kxy, 'fro') ** 2 )/(
        falling_factorial(m,2) * falling_factorial(n,2)
    )
    term7 = (ones_m @ tKxx @ Kxy @ ones_n) /(n * falling_factorial(m,2))
    term8 = (ones_m.T @ tKxx @ ones_m * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ tKxx @ Kxy @ ones_n) / (falling_factorial(m,3) * n)
    term9 = (ones_m @ Kxy @ tKyy @ ones_n) /(m * falling_factorial(n,2))
    term10 = (ones_n.T @ tKyy @ ones_n * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ Kxy @ tKyy @ ones_n) / (falling_factorial(n,3) * m)
    term11= (jnp.linalg.norm(Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(m * falling_factorial(n,2))
    term12 = (jnp.linalg.norm(Kxy.T @ ones_m) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(n * falling_factorial(m,2))

    # Print each term
    # print("term1:", term1)
    # print("term2:", term2)
    # print("term3:", term3)
    # print("term4:", term4)
    # print("term5:", term5)
    # print("term6:", term6)
    # print("term7:", term7)
    # print("term8:", term8)
    # print("term9:", term9)
    # print("term10:", term10)
    # print("term11:", term11)
    # print("term12:", term12)
    
    return term1 - term2 + term3 -term4 + (0.25 * term5) - (1.25 * term6) - 3 * term7 + 3 * term8 -2*term9 + 2 * term10 + (0.5 * term11) + (0.5 * term12)

def Xi_20(Kxx, Kyy, Kxy, m ,n) : 
    tKxx = Kxx - jnp.diag(jnp.diag(Kxx))
    tKyy = Kyy - jnp.diag(jnp.diag(Kyy))

    ones_n = jnp.ones(n) 
    ones_m = jnp.ones(m)

    term1 = jnp.linalg.norm(tKxx, 'fro') ** 2 / (falling_factorial(m,2))
    
    term2 = ((ones_m.T @ tKxx @ ones_m) **2 - 4 * jnp.linalg.norm(tKxx @ ones_m) ** 2 + 2* jnp.linalg.norm(tKxx, 'fro') ** 2)/(falling_factorial(m, 4))
    
    term3 = (jnp.linalg.norm(Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(m * falling_factorial(n,2))
    
    term4 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 -jnp.linalg.norm(Kxy @ ones_n) ** 2+jnp.linalg.norm(Kxy, 'fro') ** 2)/(
        falling_factorial(m, 2) * falling_factorial(n, 2)
        )
    
    term5 = (ones_m @ tKxx @ Kxy @ ones_n) /(n * falling_factorial(m,2))
    
    term6 = (ones_m.T @ tKxx @ ones_m * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ tKxx @ Kxy @ ones_n) / (falling_factorial(m,3) * n)
    
    return term1 - term2 + 4 * term3 - 4 * term4 - 4 * term5 + 4 * term6


def Xi_02(Kxx, Kyy, Kxy, m ,n) : 
    tKxx = Kxx - jnp.diag(jnp.diag(Kxx))
    tKyy = Kyy - jnp.diag(jnp.diag(Kyy))

    ones_n = jnp.ones(n) 
    ones_m = jnp.ones(m)

    term1 = jnp.linalg.norm(tKyy, 'fro') ** 2 / (falling_factorial(n,2))
    
    term2 = ((ones_n.T @ tKyy @ ones_n) **2 - 4 * jnp.linalg.norm(tKyy @ ones_n) ** 2 + 2* jnp.linalg.norm(tKyy, 'fro') ** 2)/(falling_factorial(n, 4))
    
    term3 = (jnp.linalg.norm(Kxy.T @ ones_m) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(n * falling_factorial(m,2))
    
    term4 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 -jnp.linalg.norm(Kxy @ ones_n) ** 2+jnp.linalg.norm(Kxy, 'fro') ** 2)/(
        falling_factorial(m, 2) * falling_factorial(n, 2)
        )
    
    term5 = (ones_m @ Kxy @ tKyy @ ones_n) /(m * falling_factorial(n,2))
    
    term6 = (ones_n.T @ tKyy @ ones_n * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ Kxy @ tKyy @ ones_n) / (falling_factorial(n,3) * m)
    
    return term1 - term2 + 4 * term3 - 4 * term4 - 4 * term5 + 4 * term6


def Xi_21(Kxx, Kyy, Kxy, m ,n) : 
    tKxx = Kxx - jnp.diag(jnp.diag(Kxx))
    tKyy = Kyy - jnp.diag(jnp.diag(Kyy))

    ones_n = jnp.ones(n) 
    ones_m = jnp.ones(m)

    term1 = jnp.linalg.norm(tKxx, 'fro') ** 2 / (falling_factorial(m,2))
    
    term2 = ((ones_m.T @ tKxx @ ones_m) **2 - 4 * jnp.linalg.norm(tKxx @ ones_m) ** 2 + 2* jnp.linalg.norm(tKxx, 'fro') ** 2)/(falling_factorial(m, 4))
    
    term3 = (jnp.linalg.norm(tKyy @ ones_n) ** 2 - jnp.linalg.norm(tKyy, 'fro') ** 2) / (falling_factorial(n, 3))
    
    term4 = ((ones_n.T @ tKyy @ ones_n) **2 - 4 * jnp.linalg.norm(tKyy @ ones_n) ** 2 + 2* jnp.linalg.norm(tKyy, 'fro') ** 2)/(falling_factorial(n, 4))
    
    term5 = (jnp.linalg.norm(Kxy, 'fro') ** 2)/(m*n)
    
    term6 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 -jnp.linalg.norm(Kxy @ ones_n) ** 2+jnp.linalg.norm(Kxy, 'fro') ** 2)/(
        falling_factorial(m, 2) * falling_factorial(n, 2)
        )
    
    term7 = (jnp.linalg.norm(Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(m * falling_factorial(n,2))
    
    term8 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 -jnp.linalg.norm(Kxy @ ones_n) ** 2+jnp.linalg.norm(Kxy, 'fro') ** 2)/(
        falling_factorial(m, 2) * falling_factorial(n, 2)
        )
    
    term9 = (ones_m @ tKxx @ Kxy @ ones_n) /(n * falling_factorial(m,2))
    
    term10 = (ones_m.T @ tKxx @ ones_m * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ tKxx @ Kxy @ ones_n) / (falling_factorial(m,3) * n)
    
    term11 = (ones_m @ Kxy @ tKyy @ ones_n) /(m * falling_factorial(n,2))
    
    term12 = (ones_n.T @ tKyy @ ones_n * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ Kxy @ tKyy @ ones_n) / (falling_factorial(n,3) * m)
    
    return term1 - term2 + term3 -term4 + term5 - term6 + 3*term7 -3* term8 -4 * term9 + 4 * term10 -2 * term11 + 2 * term12  


def Xi_12(Kxx, Kyy, Kxy, m ,n) : 
    tKxx = Kxx - jnp.diag(jnp.diag(Kxx))
    tKyy = Kyy - jnp.diag(jnp.diag(Kyy))

    ones_n = jnp.ones(n) 
    ones_m = jnp.ones(m)

    term1 = (jnp.linalg.norm(tKxx @ ones_m) ** 2 - jnp.linalg.norm(tKxx, 'fro') ** 2) / (falling_factorial(m, 3))
    
    term2 = ((ones_m.T @ tKxx @ ones_m) **2 - 4 * jnp.linalg.norm(tKxx @ ones_m) ** 2 + 2* jnp.linalg.norm(tKxx, 'fro') ** 2)/(falling_factorial(m, 4))
    
    term3 = jnp.linalg.norm(tKyy, 'fro') ** 2 / (falling_factorial(n,2))
    
    term4 = ((ones_n.T @ tKyy @ ones_n) **2 - 4 * jnp.linalg.norm(tKyy @ ones_n) ** 2 + 2* jnp.linalg.norm(tKyy, 'fro') ** 2)/(falling_factorial(n, 4))
    
    term5 = (jnp.linalg.norm(Kxy, 'fro') ** 2)/(m*n)
    
    term6 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 -jnp.linalg.norm(Kxy @ ones_n) ** 2+jnp.linalg.norm(Kxy, 'fro') ** 2)/(
        falling_factorial(m, 2) * falling_factorial(n, 2)
        )
    
    term7 = (jnp.linalg.norm(Kxy.T @ ones_m) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(n * falling_factorial(m,2))
    
    term8 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 -jnp.linalg.norm(Kxy @ ones_n) ** 2+jnp.linalg.norm(Kxy, 'fro') ** 2)/(
        falling_factorial(m, 2) * falling_factorial(n, 2)
        )
    
    term9 = (ones_m @ tKxx @ Kxy @ ones_n) /(n * falling_factorial(m,2))
    
    term10 = (ones_m.T @ tKxx @ ones_m * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ tKxx @ Kxy @ ones_n) / (falling_factorial(m,3) * n)
    
    term11 = (ones_m @ Kxy @ tKyy @ ones_n) /(m * falling_factorial(n,2))
    
    term12 = (ones_n.T @ tKyy @ ones_n * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ Kxy @ tKyy @ ones_n) / (falling_factorial(n,3) * m)
    
    return term1 - term2 + term3 -term4 + term5 - term6 + 3*term7 -3* term8 -4 * term9 + 4 * term10 -2 * term11 + 2 * term12  
    
    

def Xi_22(Kxx, Kyy, Kxy, m ,n) : 
    tKxx = Kxx - jnp.diag(jnp.diag(Kxx))
    tKyy = Kyy - jnp.diag(jnp.diag(Kyy))

    ones_n = jnp.ones(n) 
    ones_m = jnp.ones(m)

    term1 = jnp.linalg.norm(tKxx, 'fro') ** 2 / (falling_factorial(m,2))
    
    term2 = ((ones_m.T @ tKxx @ ones_m) **2 - 4 * jnp.linalg.norm(tKxx @ ones_m) ** 2 + 2* jnp.linalg.norm(tKxx, 'fro') ** 2)/(falling_factorial(m, 4))
    
    term3 = jnp.linalg.norm(tKyy, 'fro') ** 2 / (falling_factorial(n,2))
    
    term4 = ((ones_n.T @ tKyy @ ones_n) **2 - 4 * jnp.linalg.norm(tKyy @ ones_n) ** 2 + 2* jnp.linalg.norm(tKyy, 'fro') ** 2)/(falling_factorial(n, 4))
    
    term5 = (jnp.linalg.norm(Kxy, 'fro') ** 2)/(m*n)
    
    term6 = ((ones_m.T @ Kxy @ ones_n)**2 - jnp.linalg.norm(Kxy.T @ ones_m) ** 2 -jnp.linalg.norm(Kxy @ ones_n) ** 2+jnp.linalg.norm(Kxy, 'fro') ** 2)/(
        falling_factorial(m, 2) * falling_factorial(n, 2)
        )
    
    term7 = (ones_m @ tKxx @ Kxy @ ones_n) /(n * falling_factorial(m,2))
    
    term8 = (ones_m.T @ tKxx @ ones_m * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ tKxx @ Kxy @ ones_n) / (falling_factorial(m,3) * n)
    
    term9 = (ones_m @ Kxy @ tKyy @ ones_n) /(m * falling_factorial(n,2))
    
    term10 = (ones_n.T @ tKyy @ ones_n * ones_m.T @ Kxy @ ones_n - 2 * ones_m.T @ Kxy @ tKyy @ ones_n) / (falling_factorial(n,3) * m)
    
    term11 = (jnp.linalg.norm(Kxy.T @ ones_m) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(n * falling_factorial(m,2))
    
    term12 = (jnp.linalg.norm(Kxy @ ones_n) ** 2 - jnp.linalg.norm(Kxy, 'fro') ** 2)/(m * falling_factorial(n,2))
    
    return term1 - term2 + term3 - term4 + term5 - 2 * term6 - 4*term7 + 4*term8 -4*term9 +4*term10 + (0.5)*term11 + (0.5)*term12



def compute_unbiased_var (Kxx, Kyy, Kxy, m ,n, complete = True) :
    denom = m * (m - 1) * n * (n - 1)
    Xi = [
        # xi_01
        Xi_01(Kxx, Kyy, Kxy, m ,n),
        
        # xi_02
        Xi_02(Kxx, Kyy, Kxy, m ,n),
        
        # xi_10
        Xi_10(Kxx, Kyy, Kxy, m ,n),
        
        # xi_11
        Xi_11(Kxx, Kyy, Kxy, m ,n),
        
        # xi_12
        Xi_12(Kxx, Kyy, Kxy, m ,n),
        
        # xi_20
        Xi_20(Kxx, Kyy, Kxy, m ,n),
        
        # xi_21
        Xi_21(Kxx, Kyy, Kxy, m ,n),
        
        # xi_22
        Xi_22(Kxx, Kyy, Kxy, m ,n)        
    ]
    
    if complete == False:
        Xi = [Xi[0], Xi[2]]
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