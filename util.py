import numpy as np
import torch
import scipy.spatial
import jax.numpy as jnp

# def Pdist2(x, y):
#     """compute the paired distance between x and y."""
#     x_norm = (x ** 2).sum(1).view(-1, 1)
#     if y is not None:
#         y_norm = (y ** 2).sum(1).view(1, -1)
#     else:
#         y = x
#         y_norm = x_norm.view(1, -1)
#     Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
#     Pdist[Pdist<0]=0
#     return Pdist

def get_item(x, is_cuda):
    """Get the numpy value from a torch tensor."""
    if is_cuda:
        x = x.cpu().detach().numpy() 
    else:
        x = x.detach().numpy() 
    
    return x 

def MatConvert(x, device, dtype):
    """Convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype) 
    
    return x 

def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U = True):
    """Compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat([Kx, Kxy], dim=1)
    Kyxy = torch.cat([Kxy.transpose(0, 1), Ky], dim=1)
    Kxyxy = torch.cat([Kxxy, Kyxy], dim=0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True 
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (nx - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy 
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        
        # one-sample U-statsitic. 
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (nx - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy 
    
    if not is_var_computed:
        return mmd2, None 
    
    hh = Kx + Ky - Kxy - Kxy.transpose(0,1) 
    V1 = torch.dot(hh.sum(1)/ny, hh.sum(1)/ny) / ny 
    V2 = (hh).sum() / (nx) / nx 
    varEst = 4 * (V1 - V2 ** 2)
    if varEst == 0:
        print('Error!! ' + str(V1))
    return mmd2, varEst, Kxyxy



    

# DeepMMD
def deep_mmd_permutation(X, Y,params,model, n_perm) :
    mmd = MMDu_var(X, Y, params, model)[0]
    perm_stat = torch.zeros(n_perm)
    count = 0
    N = X.shape[0]
    for i in range(n_perm):
      idx = torch.randperm(N)
      perm_Y = Y[idx,:]
      perm_mmd = MMDu_var(X, perm_Y, params, model)[0]
      if perm_mmd >= mmd :
          count += 1
      else :
          count += 0
    # Compute p-value
    p_value = (count + 1) / (n_perm + 1)

    return p_value, mmd

def MMDu_var(X, Y, params, model, device):
    """Compute the value and std of deep-kernel MMD using merged data."""
    epsilon = params.get("epsilon")
    sigma0 = params.get("sigma0")
    sigma = params.get("sigma")
    
    X_org = X.clone().detach()
    Y_org = Y.clone().detach() 
    
    X_fea = model(X) 
    Y_fea = model(Y) 
    
    Dxx = Pdist2(X_fea, X_fea) 
    Dyy = Pdist2(Y_fea, Y_fea)
    Dxy = Pdist2(X_fea, Y_fea)
    Dxx_org = Pdist2(X_org, Y_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    
    Kx = (1-epsilon) * torch.exp(-Dxx / 2*sigma0) + epsilon * torch.exp(-Dxx_org / 2*sigma)
    Ky = (1-epsilon) * torch.exp(-Dyy / 2*sigma0) + epsilon * torch.exp(-Dyy_org / 2*sigma)
    Kxy = (1-epsilon) * torch.exp(-Dxy / 2*sigma0) + epsilon * torch.exp(-Dxy_org / 2*sigma)
    
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    nxy = Kxy.shape[0]
    
    xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
    yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
    xy = torch.div(torch.sum(Kxy), (nx * ny))
    
    mmd2 = xx - 2 * xy + yy 
    
    one_x = torch.ones(nx).to(device) 
    one_y = torch.ones(ny).to(device)
    one_xy = torch.ones(nxy).to(device) 
    
    h1 = ((one_x.T @ Kx)/nx) + ((one_y.T @ Ky @ one_y)/(ny**2)) - ((Kxy @ one_y)/(ny)) - ((one_x.T @ Kxy @ one_y)/(nx*ny))
    h2 = ((one_y.T @ Ky)/ny) + ((one_x.T @ Kx @ one_x)/(nx**2)) - ((one_x.T @ Kxy)/(nx)) - ((one_x.T @ Kxy @ one_y)/(nx*ny))

    xi_1 = (h1.T @ h1)/nx
    xi_2 = (h2.T @ h2)/ny

    var = (4*xi_1/nx) + (4*xi_2/ny)

    return mmd2, var**0.5
    

# C2ST 
def calc_stats(tau0: float, tau1: float, y_true):
    test_n0 = len(y_true[y_true == 0])
    test_n1 = len(y_true[y_true == 1])
    
    nom = 1 - tau0 - tau1 
    denom = np.sqrt((tau0*(1-tau0)/test_n0) + (tau1*(1-tau1)/test_n1))
    stats = nom / denom 
    
    return stats

def find_optimal_cutoff(output, y_true, cutvals):
    pwrs = [] 
    
    for i in range(len(cutvals)): 
        pred = (output[:, 1] >= cutvals[i]).int() 
        tau0 = np.mean((pred[y_true == 0] == 1))
        tau1 = np.mean((pred[y_true == 1] == 0))
        
        pwr = calc_stats(tau0, tau1, y_true)
        pwrs.append(pwr)
    
    opt_cut = cutvals[pwrs.index(max(pwrs))]
    
    return opt_cut

# MMD-Agg 
def compute_pairwise_matrix(X, Y, l):
    """
    Compute the pairwise distance matrix between all the points in X and Y,
    in L1 norm or L2 norm.

    inputs: X: (m,d) array of samples
            Y: (m,d) array of samples
            l: "l1" or "l2" or "l2sq"
    output: (2m,2m) pairwise distance matrix
    """
    Z = np.concatenate((X, Y))
    if l == "l1":
        return scipy.spatial.distance.cdist(Z, Z, 'cityblock')
    elif l == "l2":
        return scipy.spatial.distance.cdist(Z, Z, 'euclidean')
    else:
        raise ValueError("Third input should either be 'l1' or 'l2'.")

def kernel_matrix(pairwise_matrix, l, kernel_type, bandwidth):
    """
    Compute kernel matrix for a given kernel_type and bandwidth. 

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel_type: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel_type must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel_type == "gaussian" and l == "l2":
        return  np.exp(-d ** 2)
    elif kernel_type == "imq" and l == "l2":
        return (1 + d ** 2) ** (-0.5)
    elif (kernel_type == "matern_0.5_l1" and l == "l1") or (kernel_type == "matern_0.5_l2" and l == "l2") or (kernel_type == "laplace" and l == "l1"):
        return  np.exp(-d)
    elif (kernel_type == "matern_1.5_l1" and l == "l1") or (kernel_type == "matern_1.5_l2" and l == "l2"):
        return (1 + np.sqrt(3) * d) * np.exp(- np.sqrt(3) * d)
    elif (kernel_type == "matern_2.5_l1" and l == "l1") or (kernel_type == "matern_2.5_l2" and l == "l2"):
        return (1 + np.sqrt(5) * d + 5 / 3 * d ** 2) * np.exp(- np.sqrt(5) * d)
    elif (kernel_type == "matern_3.5_l1" and l == "l1") or (kernel_type == "matern_3.5_l2" and l == "l2"):
        return (1 + np.sqrt(7) * d + 2 * 7 / 5 * d ** 2 + 7 * np.sqrt(7) / 3 / 5 * d ** 3) * np.exp(- np.sqrt(7) * d)
    elif (kernel_type == "matern_4.5_l1" and l == "l1") or (kernel_type == "matern_4.5_l2" and l == "l2"):
        return (1 + 3 * d + 3 * (6 ** 2) / 28 * d ** 2 + (6 ** 3) / 84 * d ** 3 + (6 ** 4) / 1680 * d ** 4) * np.exp(- 3 * d)
    else:
        raise ValueError(
            'The values of l and kernel_type are not valid.'
        )

def create_weights(N, weights_type):
    """
    Create weights.
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centred"
    output: (N,) array of weights
    """
    if weights_type == "uniform":
        weights = np.array([1 / N,] * N)
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centred":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = np.array(
                [1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)]
            )
        else:
            normaliser = sum(
                [1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)]
            )
            weights = np.array(
                [
                    1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser)
                    for i in range(1, N + 1)
                ]
            )
    else:
        raise ValueError(
            'The value of weights_type should be "uniform" or'
            '"decreasing" or "increasing" or "centred".'
        )
    return weights


import jax
jax.config.update("jax_enable_x64", True)

# complete MMD 
def Pdist2(X, Y):
    # Assuming this function computes pairwise squared distances.
    # If there's another implementation detail, please provide.
    XX = jnp.sum(X*X, axis=1)[:, jnp.newaxis]
    YY = jnp.sum(Y*Y, axis=1)[jnp.newaxis, :]
    distances = XX + YY - 2.0 * jnp.dot(X, Y.T)
    return distances

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


def compute_moments(Kxx, Kyy, Kxy):
    return [
        0,
        jnp.trace(Kxx.T @ Kxx),
        jnp.sum(Kxx @ Kxx),
        jnp.sum(jnp.kron(Kxx, Kxx)),
        jnp.sum(jnp.kron(Kxx, Kyy)),
        jnp.sum(Kxx @ Kxy),
        jnp.sum(jnp.kron(Kxx, Kxy)),
        jnp.sum(Kxy @ Kyy),
        jnp.sum(jnp.kron(Kxy, Kyy)),
        jnp.trace(Kxy.T @ Kxy),
        jnp.sum(Kxy.T @ Kxy),
        jnp.sum(Kxy @ Kxy.T),
        jnp.sum(jnp.kron(Kxy, Kxy)),
        jnp.trace(Kyy.T @ Kyy),
        jnp.sum(Kyy @ Kyy),
        jnp.sum(jnp.kron(Kyy, Kyy))
    ]

def compute_Xi_values(C, m, n, mmd_sq):
    powers_m = [m ** i for i in range(5)]
    powers_n = [n ** i for i in range(5)]
    m1, m2, m3, m4 = powers_m[1], powers_m[2], powers_m[3], powers_m[4]
    n1, n2, n3, n4 = powers_n[1], powers_n[2], powers_n[3], powers_n[4]

    mmd2 = mmd_sq ** 2

    def calc_xi(coefficients, denominator_power):
        xi_value = sum(coefficients)
        return xi_value / ((m1 ** denominator_power[0]) * (n1 ** denominator_power[1])) - mmd2

    Xi = [
        # xi_01
        calc_xi([n3 * C[3], 2 * m2 * n1 * C[4], -4 * m * n2 * C[6], -2 * m3 * n1 * C[7], -2 * m3 * C[8], m2 * n2 * C[11], 
                 3 * m2 * n1 * C[12], m4 * C[14]], [4, 3]),
        
        # xi_02
        calc_xi([n2 * C[3], 2 * m2 * C[4], -4 * m * n1 * C[6], -4 * m3 * C[7], 2 * m2 * n1 * C[11], 2 * m2 * C[12], 
                 m4 * C[13]], [4, 2]),
        
        # xi_10
        calc_xi([n4 * C[2], 2 * m * n1 * n2 * C[4], -2 * m * n1 * n3 * C[5], -2 * n3 * C[6], -4 * m2 * n1 * C[8], 
                 m2 * n1 * n2 * C[10], 3 * m * n1 * n2 * C[12], m3 * C[15]], [3, 4]),
        
        # xi_11
        calc_xi([n3 * C[2], 2 * m * n1 * C[4], -2 * m * n2 * C[5], -2 * n2 * C[6], -2 * m2 * n1 * C[7], -2 * m2 * C[8], 
                 0.25 * m2 * n2 * C[9], 0.75 * m2 * n1 * C[10], 2.25 * m * n1 * C[12], m3 * C[14]], [3, 3]),
        
        # xi_12
        calc_xi([m * n2 * C[2], 2 * m * C[4], -2 * m * n1 * C[5], -2 * n1 * C[6], -4 * m2 * C[7], 0.5 * m2 * n1 * C[9], 
                 0.5 * m2 * C[10], 1.5 * m * n1 * C[11], 1.5 * m * C[12], m3 * C[13]], [3, 2]),
        
        # xi_20
        calc_xi([n4 * C[1], 2 * n2 * C[4], -4 * n3 * C[5], -4 * m * n1 * C[8], 2 * m * n2 * C[10], 2 * n2 * C[12], 
                 m2 * C[15]], [2, 4]),
        
        # xi_21
        calc_xi([n3 * C[1], 2 * n1 * C[4], -4 * n2 * C[5], -2 * m * n1 * C[7], -2 * m * C[8], 0.5 * m * n2 * C[9], 
                 1.5 * m * n1 * C[10], 0.5 * n2 * C[11], 1.5 * n1 * C[12], m2 * C[14]], [2, 3]),
        
        # xi_22
        calc_xi([n2 * C[1], 2 * C[4], -4 * n1 * C[5], -4 * m * C[7], m * n1 * C[9], m * C[10], n1 * C[11], 
                 C[12], m2 * C[13]], [2, 2])
    ]

    return Xi



def compute_var(Xi, m, n):
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
    denom = m * (m - 1) * n * (n - 1)
    terms = [coeff * Xi[i] / denom for i, coeff in enumerate(term_coefficients)]
    
    return sum(terms)


def completeMMDVar(X, Y, sigma0):
    m, n = len(X), len(Y)
    
    Kxx, Kyy, Kxy = compute_K_matrices(X, Y, sigma0)
    mmd_sq = compute_mmd_sq(Kxx, Kyy, Kxy, m, n)
    C = compute_moments(Kxx, Kyy, Kxy)
    Xi = compute_Xi_values(C, m, n, mmd_sq)
    var = compute_var(Xi, m, n)
    
    return var
