import jax.numpy as jnp
from sub_expressions import * 
import jax 
import torch 
from utils import HSIC, jnp_to_tensor
jax.config.update("jax_enable_x64", True)


# 1. 
def IncomMMDVar(tKxx, tKyy, Kxy):
    """Incomplete MMD Variance Estimates(Sutherland et al. 2019) """
    m, _ = tKxx.shape
    n, _ = tKyy.shape
    ones_m = jnp.ones(m)
    ones_n = jnp.ones(n)
    
    term1 = 4 * (m * n + m - 2 * n) / (factorial(m, 2) * factorial(n, 4)) * (jnp.linalg.norm(tKxx @ ones_m) ** 2 + jnp.linalg.norm(tKyy @ ones_n) ** 2)
    term2 = -2 * (2 * m - n) / (m * n * (m - 1) * (n - 2) * (n - 3)) * (jnp.linalg.norm(tKxx, 'fro') ** 2 + jnp.linalg.norm(tKyy, 'fro') ** 2)
    term3 = 4 * (m * n + m - 2 * n - 1) / (factorial(m, 2) * n**2 * (n - 1)**2) * (jnp.linalg.norm(Kxy @ ones_n) ** 2 + jnp.linalg.norm(Kxy.T @ ones_m) ** 2)
    term4 = -4 * (2 * m - n - 2) / (factorial(m, 2) * n * (n - 1)**2) * jnp.linalg.norm(Kxy, 'fro') ** 2
    term5 = -2 * (2 * m - 3) / (factorial(m, 2) * factorial(n, 4)) * ((ones_m.T @ tKxx @ ones_m) ** 2 + (ones_n.T @ tKyy @ ones_n) ** 2)
    term6 = -4 * (2 * m - 3) / (factorial(m, 2) * n**2 * (n - 1)**2) * (ones_m.T @ Kxy @ ones_n) ** 2
    term7 = -8 / (m * n**2 * (n - 1)) * (ones_m.T @ tKxx @ Kxy @ ones_n + ones_n.T @ tKyy @ Kxy.T @ ones_m)
    term8 = 8 / (m * n * factorial(n, 3)) * ((ones_m.T @ tKxx @ ones_m + ones_n.T @ tKyy @ ones_n) * (ones_m.T @ Kxy @ ones_n))
    term9 = -16 / (m * n * factorial(n, 3)) * (ones_m.T @ tKxx @ Kxy @ ones_n + ones_n.T @ tKyy @ Kxy.T @ ones_m)
    
    V_hat = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9
    return V_hat
    
# 2. 
def ComMMDVar(tKxx, tKyy, Kxy):
    """Complete MMD Variance Estimate(Ours) """
    m, _ = tKxx.shape
    n, _ = tKyy.shape
    Xi10 = mxCxmx(tKxx) - mxmx2(tKxx) + myCxmy(Kxy, complete=True) - mxmy2(Kxy, complete=True) - 2*mxCxmy(tKxx, Kxy, complete=True) + 2*mxmxmxmy(tKxx, Kxy, complete=True)
    Xi01 = myCymy(tKyy) - mymy2(tKyy) + mxCymx(Kxy, complete=True) - mxmy2(Kxy, complete=True) - 2*myCymx(tKyy, Kxy, complete=True) + 2*mymymxmy(tKyy, Kxy, complete=True)
    Xi11 = mxCxmx(tKxx) - mxmx2(tKxx) + myCymy(tKyy) - mymy2(tKyy) + 0.25*jnp.linalg.norm(Kxy, "fro")**2/m/n - 1.25*mxmy2(Kxy, complete=True) \
        - 3*mxCxmy(tKxx, Kxy, complete=True) + 3*mxmxmxmy(tKxx, Kxy, complete=True) - 2*myCymx(tKyy, Kxy, complete=True) + 2*mymymxmy(tKyy, Kxy, complete=True) \
            + 0.5*myCxmy(Kxy, complete=True) + 0.5*mxCymx(Kxy, complete=True)
    Xi20 = jnp.linalg.norm(tKxx, "fro")**2/factorial(m, 2) - mxmx2(tKxx) + 4*myCxmy(Kxy, complete=True) - 4*mxmy2(Kxy, complete=True) \
        - 4*mxCxmy(tKxx, Kxy, complete=True) + 4*mxmxmxmy(tKxx, Kxy, complete=True)
    Xi02 = jnp.linalg.norm(tKyy, "fro")**2/factorial(n, 2) - mymy2(tKyy) + 4*mxCymx(Kxy, complete=True) - 4*mxmy2(Kxy, complete=True) \
        - 4*myCymx(tKyy, Kxy, complete=True) + 4*mymymxmy(tKyy, Kxy, complete=True)
    Xi21 = jnp.linalg.norm(tKxx, "fro")**2/factorial(m, 2) - mxmx2(tKxx) + myCymy(tKyy) - mymy2(tKyy) + jnp.linalg.norm(Kxy, "fro")**2/m/n \
        - mxmy2(Kxy, complete=True) + 3*myCxmy(Kxy, complete=True) - 3*mxmy2(Kxy, complete=True) - 4*mxCxmy(tKxx, Kxy, complete=True) \
            + 4*mxmxmxmy(tKxx, Kxy, complete=True) - 2*myCymx(tKyy, Kxy, complete=True) + 2*mymymxmy(tKyy, Kxy, complete=True)
    Xi12 = mxCxmx(tKxx) - mxmx2(tKxx) + jnp.linalg.norm(tKyy, "fro")**2/factorial(n, 2) - mymy2(tKyy) + jnp.linalg.norm(Kxy, "fro")**2/m/n \
        - mxmy2(Kxy, complete=True) + 3*mxCymx(Kxy, complete=True) - 3*mxmy2(Kxy, complete=True) - 2*mxCxmy(tKxx, Kxy, complete=True) \
            + 2*mxmxmxmy(tKxx, Kxy, complete=True) - 4*myCymx(tKyy, Kxy, complete=True) + 4*mymymxmy(tKyy, Kxy, complete=True)
    Xi22 = jnp.linalg.norm(tKxx, "fro")**2/factorial(m, 2) - mxmx2(tKxx) + jnp.linalg.norm(tKyy, "fro")**2/factorial(n, 2) - mymy2(tKyy) \
        + jnp.linalg.norm(Kxy, "fro")**2/m/n - 2*mxmy2(Kxy, complete=True) - 4*mxCxmy(tKxx, Kxy, complete=True) \
            + 4*mxmxmxmy(tKxx, Kxy, complete=True) - 4*myCymx(tKyy, Kxy, complete=True) + 4*mymymxmy(tKyy, Kxy, complete=True) \
                + 0.5*mxCymx(Kxy, complete=True) + 0.5*myCxmy(Kxy, complete=True)
    
    res = sum([4*(m-2)*(m-3)*(n-2)*Xi01,
           2*(n-2)*(n-3)*Xi02,
           4*(m-2)*(n-2)*(n-3)*Xi10,
           16*(n-2)*(m-3)*Xi11,
           8*(m-2)*Xi12,
           2*(n-2)*(n-3)*Xi20,
           2*(n-2)*Xi21,
           4*Xi22           
    ])/factorial(m,2)/factorial(n,2)
    return res 

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True, complete=True):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    
    # 우리가 수정할 부분
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None, Kxyxy
    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    hsic_xx = HSIC(jnp.array(Kx.cpu().detach().numpy()), jnp.array(Kx.cpu().detach().numpy()))
    hsic_yy = HSIC(jnp.array(Ky.cpu().detach().numpy()), jnp.array(Ky.cpu().detach().numpy()))
    hsic_xy = HSIC(jnp.array(Kx.cpu().detach().numpy()), jnp.array(Ky.cpu().detach().numpy())) 
    
    if complete:
        tKxx = Kx - torch.diag(torch.diag(Kx)) 
        tKyy = Ky - torch.diag(torch.diag(Ky))
        tKxx = jnp.array(tKxx.cpu().detach().numpy())
        tKyy = jnp.array(tKyy.cpu().detach().numpy())
        Kxy = jnp.array(Kxy.cpu().detach().numpy())
        varEst = jnp_to_tensor(ComMMDVar(tKxx, tKyy, Kxy))
    else: 
        V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
        V2 = (hh).sum() / (nx) / nx
        varEst = 4*(V1 - V2**2)
    
    # if varEst == 0.0:
    #     raise ValueError("error var")
    return mmd2, varEst, Kxyxy, hsic_xx, hsic_yy, hsic_xy