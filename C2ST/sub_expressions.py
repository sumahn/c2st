import jax.numpy as jnp 
from jax.scipy.special import gammaln
import jax 
jax.config.update("jax_enable_x64", True)

# (m)_k
def factorial(x, n):
    value = 1.0
    for i in range(n):
        value *= (x - i)
    return value

# <\mu_x, \mu_y>^2 
def mxmy2(Kxy, complete=False):
    m, n = Kxy.shape
    one_m = jnp.ones(m)
    term2 = jnp.linalg.norm(Kxy.T @ one_m)**2
    term4 = jnp.linalg.norm(Kxy, "fro")**2
    if complete:
        one_n = jnp.ones(n)
        term1 = (one_m.T @ Kxy @ one_n)**2
        term3 = jnp.linalg.norm(Kxy @ one_n) ** 2 
        res = (term1 - term2 - term3 + term4) / (factorial(m, 2) * factorial(n, 2))
    else: 
        term1 = (one_m.T @ Kxy @ one_m)**2
        term3 = jnp.linalg.norm(Kxy @ one_m)**2
        res = (term1 - term2 - term3 + term4) / (m**2 * (m-1)**2)
    return res 

# <\mu_x, \mu_x>^2
def mxmx2(tKxx):
    m = tKxx.shape[0]
    one_m = jnp.ones(m)
    term1 = (one_m.T @ tKxx @ one_m)**2
    term2 = jnp.linalg.norm(tKxx @ one_m)**2
    term3 = jnp.linalg.norm(tKxx, "fro")**2
    res = (term1 - 4*term2 + 2*term3) / factorial(m, 4)
    return res 

def mymy2(tKyy):
    n = tKyy.shape[0]
    one_n = jnp.ones(n)
    term1 = (one_n.T @ tKyy @ one_n) ** 2
    term2 = jnp.linalg.norm(tKyy @ one_n)**2
    term3 = jnp.linalg.norm(tKyy, "fro")**2
    res = (term1 - 4*term2 + 2*term3) / factorial(n, 4)
    return res 

# <\mu_x, \mu_x><\mu_x, \mu_y>
def mxmxmxmy(tKxx, Kxy, complete=False):
    """Compute the term <\mu_x, \mu_x><\mu_x, \mu_y>."""
    m = tKxx.shape[0]
    n = Kxy.shape[1]
    one_m = jnp.ones(m)
    term1 = jnp.dot(one_m, tKxx @ one_m) * jnp.dot(one_m, Kxy @ one_m)  # Scalar result
    term2 = one_m.T @ tKxx @ Kxy @ one_m  # Scalar result
    
    if not complete:
        res = (term1 - 2 * term2) / (m * factorial(m, 3))
    else:
        one_n = jnp.ones(n)
        term1 = (one_m.T @ tKxx @ one_m) * (one_m.T @ Kxy @ one_n)
        term2 = one_m.T @ tKxx @ Kxy @ one_n  # Scalar result
        res = (term1 - 2*term2) / (factorial(m, 3) * n)
    return res

def mymymxmy(tKyy, Kxy, complete=False):
    m, n = Kxy.shape
    """Compute the term <\mu_y, \mu_y><\mu_x, \mu_y>."""
    one_m = jnp.ones(m)
    term1 = one_m.T @ tKyy @ one_m * (one_m.T @ Kxy @ one_m)
    term2 = one_m.T @ tKyy @ Kxy @ one_m  # Scalar result

    if not complete:
        res = (term1 - 2 * term2) / (m * factorial(m, 3))
    else:
        one_n = jnp.ones(n)
        # Adjusting calculations for the 'complete' scenario.
        # Assuming tKyy is n x n and Kxy is m x n for the 'complete' flag being True.
        term1 =  (one_n.T @ tKyy @ one_n) * (one_m.T @ Kxy @ one_n)  # Scalar result
        term2 = one_m.T @ Kxy @ tKyy @ one_n  # Scalar result
        res = (term1 - 2 * term2) / (m * factorial(n, 3))

    return res

# <\mu_x, \mu_y><\mu_x, \mu_z>
def mxmymxmz(Kxy, Kxz):
    m = Kxy.shape[0]
    one_m = jnp.ones(m)
    term1 = one_m.T @ Kxy @ one_m @ one_m.T @ Kxz @ one_m
    term2 = one_m.T @ Kxy.T @ Kxz @ one_m 
    res = (term1 - term2) / (m**3 * (m-1))
    return res 

# <\mu_x, Cx\mu_x>
def mxCxmx(tKxx):
    m = tKxx.shape[0]
    one_m = jnp.ones(m)
    term1 = jnp.linalg.norm(tKxx @ one_m)**2
    term2 = jnp.linalg.norm(tKxx, "fro")**2
    res = (term1 - term2) / factorial(m, 3)
    return res

# <\mu_y, Cy\mu_y>
def myCymy(tKyy):
    n = tKyy.shape[0]
    one_m = jnp.ones(n)
    term1 = jnp.linalg.norm(tKyy @ one_m)**2
    term2 = jnp.linalg.norm(tKyy, "fro")**2
    res = (term1 - term2) / factorial(n , 3)
    return res

# <\mu_y, Cx\mu_y>
def myCxmy(Kxy, complete=False):
    m, n = Kxy.shape
    one_m = jnp.ones(m)
    term1 = jnp.linalg.norm(Kxy @ one_m)**2
    term2 = jnp.linalg.norm(Kxy, "fro")**2
    res = (term1 - term2) / (m**2 * (m-1))
    if complete:
        one_n = jnp.ones(n)
        term1 = jnp.linalg.norm(Kxy @ one_n)**2
        res = (term1 - term2) / (m*factorial(n, 2))
    return res 

def mxCymx(Kxy, complete=False):
    m, n = Kxy.shape
    one_m = jnp.ones(m)
    term1 = jnp.linalg.norm(Kxy.T @ one_m)**2
    term2 = jnp.linalg.norm(Kxy, "fro")**2
    res = (term1 - term2) / (m**2 * (m-1))
    if complete == True:
        res = (term1 - term2) / (factorial(m, 2) * n)
    return res 

# <\mu_x, Cx\mu_y>
def mxCxmy(tKxx, Kxy, complete=False):
    m, n = Kxy.shape
    one_m = jnp.ones(m)
    res = (one_m.T @ tKxx @ Kxy @ one_m) / (m**2 * (m-1))
    if complete:
        one_n = jnp.ones(n) 
        res = (one_m.T @ tKxx @ Kxy @ one_n) / (n*factorial(m,2))
    return res 

# <\mu_x, Cx\mu_z>
def mxCxmz(Kxy, Kxz):
    m = Kxy.shape[0]
    one_m = jnp.ones(m)
    res = (one_m.T @ Kxy.T @ Kxz @ one_m) / (m**3)
    return res

def myCymx(tKyy, Kxy, complete=False):
    m, n = Kxy.shape
    one_m = jnp.ones(m)
    res = (one_m.T @ tKyy @ Kxy.T @ one_m) / (m**2 * (m-1))
    if complete:
        one_n = jnp.ones(n) 
        res = (one_m.T @ Kxy @ tKyy @ one_n) / (m * factorial(n, 2))
    return res




    
    

    
    