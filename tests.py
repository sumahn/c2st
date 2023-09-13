import numpy as np 
import torch
import torch.nn as nn
from util import MMDu_var, compute_pairwise_matrix, kernel_matrix, create_weights
from models import MyModel
from scipy.stats import norm
import scipy.spatial
from kernel import kernel_matrices, mutate_K
from median import compute_median_bandwidth_subset

# MMD-D
def deep_mmd_permutation(X, Y, params, model, n_perm, device):
    mmd = MMDu_var(X, Y, params, model, device)[0]
    perm_stat = torch.zeros(n_perm)
    count = 0 
    N = X.shape[0]
    
    for i in range(n_perm):
        idx = torch.randperm(N)
        perm_Y = Y[idx, :]
        perm_mmd = MMDu_var(X, perm_Y, params, model, device)[0]
        
        if perm_mmd >= mmd:
            count += 1
        else:
            count += 0 
        
        # Compute p-value 
        p_value = (count + 1) / (n_perm + 1)
        
        return p_value, mmd
    

def TST_MMD(X, Y, params, model, n_perm, alpha, device):
    tmp = MMDu_var(X, Y, params, model, device)[0]
    p_value, mmd = deep_mmd_permutation(X, Y, params, model, n_perm, device)
    
    if p_value > alpha: 
        h = 0           # Do not reject H0 
    else:
        h = 1           # Reject H0 
    
    return h, tmp, p_value

# MMD-Agg 
def TST_MMD_AGG(X, Y, alpha, seed, kernel='laplace_gaussian', number_bandwidths=10, 
                weights_type='uniform', B1=2000, B2=2000, B3=50, 
                return_dictionary = False, 
                permutation_same_sample_size = False):
    
    # Assertion
    m = X.shape[0]
    n = Y.shape[0]
    mn = m + n 
    assert n >=2 and m >=2 
    if m != n or permutation_same_sample_size:
        approx_type = "permutations"
    else:
        approx_type = "wild bootstrap"
    assert 0 < alpha and alpha < 1
    assert kernel in (
        "gaussian", 
        "laplace", 
        "imq", 
        "matern_0.5_l1", 
        "matern_1.5_l1", 
        "matern_2.5_l1", 
        "matern_3.5_l1", 
        "matern_4.5_l1", 
        "matern_0.5_l2", 
        "matern_1.5_l2", 
        "matern_2.5_l2", 
        "matern_3.5_l2", 
        "matern_4.5_l2", 
        "all_matern_l1", 
        "all_matern_l2", 
        "all_matern_l1_l2", 
        "all", 
        "laplace_gaussian", 
        "gaussian_laplace", 
    )
    assert number_bandwidths > 1 and type(number_bandwidths) == int
    assert weights_type in ("uniform", "decreasing", "increasing", "centred")
    assert B1 > 0 and type(B1) == int
    assert B2 > 0 and type(B2) == int
    assert B3 > 0 and type(B3) == int

    # Collection of bandwidths 
    # lambda_min / 2 * C^r for r = 0, ..., number_bandwidths -1
    # where C is such that lambda_max * 2 = lambda_min / 2 * C^(number_bandwidths - 1)
    def compute_bandwidths(distances, number_bandwidths):    
        if np.min(distances) < 10 ** (-1):
            d = np.sort(distances)
            lambda_min = np.maximum(d[int(np.floor(len(d) * 0.05))], 10 ** (-1))
        else:
            lambda_min = np.min(distances)
        lambda_min = lambda_min / 2
        lambda_max = np.maximum(np.max(distances), 3 * 10 ** (-1))
        lambda_max = lambda_max * 2
        power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
        bandwidths = np.array([power ** i * lambda_min for i in range(number_bandwidths)])
        return bandwidths
    max_samples = 500
    # bandwidths L1 for laplace, matern_0.5_l1, matern_1.5_l1, matern_2.5_l1, matern_3.5_l1, matern_4.5_l1
    distances_l1 = scipy.spatial.distance.cdist(X[:max_samples], Y[:max_samples], "cityblock").reshape(-1)
    bandwidths_l1 = compute_bandwidths(distances_l1, number_bandwidths)
    # bandwidths L2 for gaussian, imq, matern_0.5_l2, matern_1.5_l2, matern_2.5_l2, matern_3.5_l2, matern_4.5_l2
    distances_l2 = scipy.spatial.distance.cdist(X[:max_samples], Y[:max_samples], "euclidean").reshape(-1)
    bandwidths_l2 = compute_bandwidths(distances_l2, number_bandwidths)
    
    # Kernel and bandwidths list (order: "l1" first, "l2" second)
    if kernel in ( 
        "laplace", 
        "matern_0.5_l1", 
        "matern_1.5_l1", 
        "matern_2.5_l1", 
        "matern_3.5_l1", 
        "matern_4.5_l1", 
    ):
        kernel_bandwidths_l_list = [(kernel, bandwidths_l1, "l1"), ]
    elif kernel in (
        "gaussian", 
        "imq", 
        "matern_0.5_l2", 
        "matern_1.5_l2", 
        "matern_2.5_l2", 
        "matern_3.5_l2", 
        "matern_4.5_l2", 
    ):
        kernel_bandwidths_l_list = [(kernel, bandwidths_l2, "l2"), ]
    elif kernel in ("laplace_gaussian", "gaussian_laplace"):
        kernel_bandwidths_l_list = [("laplace", bandwidths_l1, "l1"), ("gaussian", bandwidths_l2, "l2")]
    elif kernel == "all_matern_l1":
        kernel_list = ["matern_" + str(i) + ".5_l1" for i in range(5)]
        kernel_bandwidths_l_list = [(kernel, bandwidths_l1, "l1") for kernel in kernel_list]
    elif kernel == "all_matern_l2":
        kernel_list = ["matern_" + str(i) + ".5_l2" for i in range(5)]
        kernel_bandwidths_l_list = [(kernel, bandwidths_l2, "l2") for kernel in kernel_list]
    elif kernel == "all_matern_l1_l2":
        kernel_list = [
            "matern_" + str(i) + ".5_l" + str(j) for j in (1, 2) for i in range(5) 
        ]
        bandwidths_list = [bandwidths_l1, ] * 5 + [bandwidths_l2, ] * 5
        l_list = ["l1", ] * 5 + ["l2", ] * 5
        kernel_bandwidths_l_list = [
            (kernel_list[i], bandwidths_list[i], l_list[i]) for i in range(10)
        ]
    elif kernel == "all":
        kernel_list = [
            "matern_" + str(i) + ".5_l" + str(j) for j in (1, 2) for i in range(5) 
        ] + ["gaussian", "imq"] 
        bandwidths_list = [] + [bandwidths_l1, ] * 5 + [bandwidths_l2, ] * 7
        l_list = ["l1", ] * 5 + ["l2", ] * 7
        kernel_bandwidths_l_list = [
            (kernel_list[i], bandwidths_list[i], l_list[i]) for i in range(12)
        ]
    else:
        raise ValueError("Kernel not defined.")
    
    # Weights 
    weights = create_weights(number_bandwidths, weights_type) / len(
        kernel_bandwidths_l_list
    )
    
    # Setup for wild bootstrap or permutations (efficient as in Appendix C in our paper)
    rs = np.random.RandomState(seed)
    if approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + B2 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R))  # (2n, B1+B2+1) 
    elif approx_type == "permutations":
        idx = rs.rand(B1 + B2 + 1, m + n).argsort(axis=1)  # (B1+B2+1, m+n): rows of permuted indices
        #11
        v11 = np.concatenate((np.ones(m), -np.ones(n)))  # (m+n, )
        V11i = np.tile(v11, (B1 + B2 + 1, 1))  # (B1+B2+1, m+n)
        V11 = np.take_along_axis(V11i, idx, axis=1)  # (B1+B2+1, m+n): permute the entries of the rows
        V11[B1] = v11  # (B1+1)th entry is the original MMD (no permutation)
        V11 = V11.transpose()  # (m+n, B1+B2+1)
        #10
        v10 = np.concatenate((np.ones(m), np.zeros(n)))
        V10i = np.tile(v10, (B1 + B2 + 1, 1))
        V10 = np.take_along_axis(V10i, idx, axis=1)
        V10[B1] = v10
        V10 = V10.transpose() 
        #01
        v01 = np.concatenate((np.zeros(m), -np.ones(n)))
        V01i = np.tile(v01, (B1 + B2 + 1, 1))
        V01 = np.take_along_axis(V01i, idx, axis=1)
        V01[B1] = v01
        V01 = V01.transpose() 
    else:
        raise ValueError("Approximation type not defined.")
        
    # Step 1: compute all simulated MMD estimates (efficient as in Appendix C in our paper)
    N = number_bandwidths * len(kernel_bandwidths_l_list)
    M = np.zeros((N, B1 + B2 + 1))  
    last_l_pairwise_matrix_computed = ""
    for j in range(len(kernel_bandwidths_l_list)):
        kernel, bandwidths, l = kernel_bandwidths_l_list[j]
        # since kernel_bandwidths_l_list is ordered "l1" first, "l2" second
        # compute pairwise matrices the minimum amount of time
        # store only one pairwise matrix at once
        if l != last_l_pairwise_matrix_computed:
            pairwise_matrix = compute_pairwise_matrix(X, Y, l)
            last_l_pairwise_matrix_computed = l
        for i in range(number_bandwidths):
            bandwidth = bandwidths[i]
            K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
            if approx_type == "wild bootstrap": 
                # set diagonal elements of all four submatrices to zero
                np.fill_diagonal(K, 0)
                np.fill_diagonal(K[:n, n:], 0)
                np.fill_diagonal(K[n:, :n], 0)
                # compute MMD bootstrapped values
                M[number_bandwidths * j + i] = np.sum(R * (K @ R), 0)
            elif approx_type == "permutations": 
                # set diagonal elements to zero
                np.fill_diagonal(K, 0)
                # compute MMD permuted values
                M[number_bandwidths * j + i] = (
                    np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
                    + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
                    + np.sum(V11 * (K @ V11), 0) / (m * n)
                )  
            else:
                raise ValueError("Approximation type not defined.")           
    MMD_original = M[:, B1]
    M1_sorted = np.sort(M[:, :B1 + 1])  # (N, B1+1)
    M2 = M[:, B1 + 1:]  # (N, B2)

    # Step 2: compute u_alpha_hat using the bisection method
    quantiles = np.zeros((N, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0 # or alpha
    u_max = np.min(1 / weights)
    for _ in range(B3): 
        u = (u_max + u_min) / 2
        for j in range(len(kernel_bandwidths_l_list)):
            for i in range(number_bandwidths):
                quantiles[number_bandwidths * j + i] = M1_sorted[
                    number_bandwidths * j + i, 
                    int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
                ]
        P_u = np.sum(np.max(M2 - quantiles, 0) > 0) / B2
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min

    # Step 3: output test result
    p_vals = np.mean((M1_sorted - MMD_original.reshape(-1, 1) >= 0), -1)
    all_weights = np.zeros(p_vals.shape)
    for j in range(len(kernel_bandwidths_l_list)):
         for i in range(number_bandwidths):
            all_weights[number_bandwidths * j + i] = weights[i]
    thresholds = u * all_weights
    # reject if p_val <= threshold
    reject_p_vals = p_vals <= thresholds

    mmd_vals = MMD_original
    quantiles = quantiles.reshape(-1)
    # reject if mmd_val > quantile
    reject_mmd_vals = mmd_vals > quantiles
    
    # create rejection dictionary 
    reject_dictionary = {}
    reject_dictionary["MMDAgg test reject"] = False
    for j in range(len(kernel_bandwidths_l_list)):
        kernel, bandwidths, l = kernel_bandwidths_l_list[j]
        for i in range(number_bandwidths):
            index = "Single test " + str(j + 1) + "." + str(i + 1)
            idx = number_bandwidths * j + i
            reject_dictionary[index] = {}
            reject_dictionary[index]["Reject"] = reject_mmd_vals[idx]
            reject_dictionary[index]["Kernel " + kernel] = True
            reject_dictionary[index]["Bandwidth"] = bandwidths[i]
            reject_dictionary[index]["MMD"] = mmd_vals[idx]
            reject_dictionary[index]["MMD quantile"] = quantiles[idx]
            reject_dictionary[index]["p-value"] = p_vals[i]
            reject_dictionary[index]["p-value threshold"] = thresholds[idx]
            # Aggregated test rejects if one single test rejects
            reject_dictionary["MMDAgg test reject"] = any((
                reject_dictionary["MMDAgg test reject"], 
                reject_p_vals[idx]
            ))

    if return_dictionary:
        return int(reject_dictionary["MMDAgg test reject"]), reject_dictionary
    else:
        return int(reject_dictionary["MMDAgg test reject"])

# This is not the mmdagg test we recommend using in practice.
# We recommend using the parameter-free mmdagg test of mmdagg/jax.py (or mmdagg/np.py)
# Those functions can easily be used using our mmdagg package at:
# https://github.com/antoninschrab/mmdagg
def mmdagg(
    seed, X, Y, alpha, kernel_type, approx_type, weights_type, l_minus, l_plus, B1, B2, B3
):
    """
    Compute MMDAgg as defined in Algorithm 1 in our paper using the collection of
    bandwidths defined in Eq. (16) and the weighting strategies proposed in Section 5.1.
    inputs: seed: integer random seed
            X: (m,d) array (m d-dimensional points)
            Y: (n,d) array (n d-dimensional points)
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            weights_type: "uniform", "decreasing", "increasing" or "centred" (Section 5.1 of our paper)
            l_minus: integer (for collection of bandwidths Eq. (16) in our paper)
            l_plus: integer (for collection of bandwidths Eq. (16) in our paper)
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the probability in Eq. (13) in our paper
            B3: number of iterations for the bisection method
    output: result of MMDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    m = X.shape[0]
    n = Y.shape[0]
    mn = m + n
    assert n >= 2 and m >= 2
    assert X.shape[1] == Y.shape[1]
    assert 0 < alpha  and alpha < 1
    assert kernel_type in ["gaussian", "laplace"]
    assert approx_type in ["permutation", "wild bootstrap"]
    if m != n:
        assert approx_type == "permutation"
    assert weights_type in ["uniform", "decreasing", "increasing", "centred"]
    assert l_plus >= l_minus

    # compute median bandwidth
    median_bandwidth = compute_median_bandwidth_subset(seed, X, Y)
    
    # define bandwidth_multipliers and weights
    bandwidth_multipliers = np.array([2 ** i for i in range(l_minus, l_plus + 1)])
    N = bandwidth_multipliers.shape[0]  # N = 1 + l_plus - l_minus
    weights = create_weights(N, weights_type)
    
    # compute the kernel matrices
    kernel_matrices_list = kernel_matrices(
        X, Y, kernel_type, median_bandwidth, bandwidth_multipliers
    ) 

    return mmdagg_custom(
        seed, 
        kernel_matrices_list, 
        weights, 
        m, 
        alpha, 
        approx_type, 
        B1, 
        B2, 
        B3,
    )


def mmdagg_custom(
    seed, kernel_matrices_list, weights, m, alpha, approx_type, B1, B2, B3
):
    """
    Compute MMDAgg as defined in Algorithm 1 in our paper with custom kernel matrices
    and weights.
    inputs: seed: integer random seed
            kernel_matrices_list: list of N kernel matrices
                these can correspond to kernel matrices obtained by considering
                different bandwidths of a fixed kernel as we consider in our paper
                but one can also use N fundamentally different kernels.
                It is assumed that the kernel matrices are of shape (m+n,m+n) with
                the top left (m,m) submatrix corresponding to samples from X and 
                the bottom right (n,n) submatrix corresponding to samples from Y
            weights: array of shape (N,) consisting of positive entries summing to 1
            m: the number of samples from X used to create the kernel matrices
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            B1: number of simulated test statistics to estimate the quantiles
            B2: number of simulated test statistics to estimate the probability in Eq. (13) in our paper
            B3: number of iterations for the bisection method
    output: result of MMDAgg (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    n = kernel_matrices_list[0].shape[0] - m
    mn = m + n
    N = len(kernel_matrices_list)
    assert len(kernel_matrices_list) == weights.shape[0]
    assert n >= 2 and m >= 2
    assert 0 < alpha  and alpha < 1
    assert approx_type in ["permutation", "wild bootstrap"]
    if m != n:
        assert approx_type == "permutation"
    
    # Step 1: compute all simulated MMD estimates (efficient as in Appendix C in our paper)
    M  = np.zeros((N, B1 + B2 + 1))  
    rs = np.random.RandomState(seed)
    if approx_type == "permutation":
        idx = rs.rand(B1 + B2 + 1, m + n).argsort(axis=1)  # (B1+B2+1, m+n): rows of permuted indices
        #11
        v11 = np.concatenate((np.ones(m), -np.ones(n)))  # (m+n, )
        V11i = np.tile(v11, (B1 + B2 + 1, 1))  # (B1+B2+1, m+n)
        V11 = np.take_along_axis(V11i, idx, axis=1)  # (B1+B2+1, m+n): permute the entries of the rows
        V11[B1] = v11  # (B1+1)th entry is the original MMD (no permutation)
        V11 = V11.transpose()  # (m+n, B1+B2+1)
        #10
        v10 = np.concatenate((np.ones(m), np.zeros(n)))
        V10i = np.tile(v10, (B1 + B2 + 1, 1))
        V10 = np.take_along_axis(V10i, idx, axis=1)
        V10[B1] = v10
        V10 = V10.transpose() 
        #01
        v01 = np.concatenate((np.zeros(m), -np.ones(n)))
        V01i = np.tile(v01, (B1 + B2 + 1, 1))
        V01 = np.take_along_axis(V01i, idx, axis=1)
        V01[B1] = v01
        V01 = V01.transpose() 
        for i in range(N):
            K = kernel_matrices_list[i]
            mutate_K(K, approx_type)
            M[i] = (
                np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
                + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
                + np.sum(V11 * (K @ V11), 0) / (m * n)
            )  # (B1+B2+1, ) permuted MMD estimates
    elif approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + B2 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R))  # (2n, B1+B2+1) 
        for i in range(N):
            K = kernel_matrices_list[i]
            mutate_K(K, approx_type)
            M[i] = np.sum(R * (K @ R) , 0) /(n * (n - 1))
    else:
        raise ValueError(
            'The value of approx_type should be either "permutation" or "wild bootstrap".'
        )
    MMD_original = M[:, B1]
    M1_sorted = np.sort(M[:, :B1 + 1])  # (N, B1+1)
    M2 = M[:, B1 + 1:]  # (N, B2)
    
    # Step 2: compute u_alpha_hat using the bisection method
    quantiles = np.zeros((N, 1))  # (1-u*w_lambda)-quantiles for the N bandwidths
    u_min = 0
    u_max = np.min(1 / weights)
    for _ in range(B3): 
        u = (u_max + u_min) / 2
        for i in range(N):
            quantiles[i] = M1_sorted[
                i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1
            ]
        P_u = np.sum(np.max(M2 - quantiles, 0) > 0) / B2
        if P_u <= alpha:
            u_min = u
        else:
            u_max = u
    u = u_min
        
    # Step 3: output test result
    for i in range(N):
        if ( MMD_original[i] 
            > M1_sorted[i, int(np.ceil((B1 + 1) * (1 - u * weights[i]))) - 1]
        ):
            return 1
    return 0 


def mmd_median_test(
    seed, X, Y, alpha, kernel_type, approx_type, B1, bandwidth_multiplier=1
):
    """
    Compute MMD test using kernel with bandwidth the median bandwidth multiplied by bandwidth_multiplier.
    This test has been proposed by 
        Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf and Alexander Smola
        A Kernel Two-Sample Test
        Journal of Machine Learning Research 2012
        https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf
    inputs: seed: random seed
            X: (m,d) array (m d-dimensional points)
            Y: (n,d) array (n d-dimensional points)
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            B1: number of simulated test statistics to estimate the quantiles
            bandwidth_multiplier: multiplicative factor for the median bandwidth 
    output: result of the MMD test with median bandwidth multiplied by bandwidth_multiplier
            (1 for "REJECT H_0" and 0 for "FAIL TO REJECT H_0")
    """
    m = X.shape[0]
    n = Y.shape[0]
    mn = m + n
    assert n >= 2 and m >= 2
    assert X.shape[1] == Y.shape[1]
    assert 0 < alpha  and alpha < 1
    assert kernel_type in ["gaussian", "laplace"]
    assert approx_type in ["permutation", "wild bootstrap"]
    if m != n:
        assert approx_type == "permutation"

    # compute median bandwidth
    median_bandwidth = compute_median_bandwidth_subset(seed, X, Y)
                           
    # compute all simulated MMD estimates (efficient)
    K = kernel_matrices(
        X, Y, kernel_type, median_bandwidth, np.array([bandwidth_multiplier])
    )[0]
    mutate_K(K, approx_type)  
    rs = np.random.RandomState(seed)
    if approx_type == "permutation":
        v11 = np.concatenate((np.ones(m), -np.ones(n)))  # (m+n, )
        v10 = np.concatenate((np.ones(m), np.zeros(n)))
        v01 = np.concatenate((np.zeros(m), -np.ones(n)))
        V11 = np.tile(v11, (B1 + 1, 1))  # (B1+1, m+n)
        V10 = np.tile(v10, (B1 + 1, 1))
        V01 = np.tile(v01, (B1 + 1, 1))
        idx = rs.rand(*V11.shape).argsort(axis=1)  # (B1+1, m+n): rows of permuted indices
        V11 = np.take_along_axis(V11, idx, axis=1)  # (B1+1, m+n): permute the entries of the rows
        V10 = np.take_along_axis(V10, idx, axis=1)
        V01 = np.take_along_axis(V01, idx, axis=1)
        V11[B1] = v11  # (B1+1)th entry is the original MMD (no permutation)
        V10[B1] = v10
        V01[B1] = v01
        V11 = V11.transpose()  # (m+n, B1+1)
        V10 = V10.transpose() 
        V01 = V01.transpose() 
        M1 = (
            np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
            + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
            + np.sum(V11 * (K @ V11), 0) / (m * n)
        )  # (B1+1, ) permuted MMD estimates
    elif approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R))  # (2n, B1+1) 
        M1 = np.sum(R * (K @ R) , 0) /(n * (n - 1))
    else:
        raise ValueError(
            'The value of approx_type should be either "permutation" or "wild bootstrap".'
        )
    MMD_original = M1[B1]
    M1_sorted = np.sort(M1) 
    
    # output test result
    if MMD_original > M1_sorted[int(np.ceil((B1 + 1) * (1 - alpha))) - 1]:
        return 1
    return 0 


def ratio_mmd_stdev(K, approx_type, regulariser=10**(-8)):
    """
    Compute the estimated ratio of the MMD to the asymptotic standard deviation under the alternative.
    This is stated in Eq. (15) in our paper, it originally comes from Eq. (3) in:
        F. Liu, W. Xu, J. Lu, G. Zhang, A. Gretton, and D. J. Sutherland
        Learning deep kernels for non-parametric two-sample tests
        International Conference on Machine Learning, 2020
        http://proceedings.mlr.press/v119/liu20m/liu20m.pdf
    assumption: m = n: equal number of samples in X and Y
    inputs: K: (m+n, m+n) kernel matrix for pooled sample WITH diagonal 
               (K has NOT been mutated by mutate_K function)
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            regulariser: small positive number (we use 10**(-8) as done by Liu et al.)
            m: number of samples (d-dimensional points) in X 
            K: (m+n, m+n) kernel matrix for pooled sample WITH diagonal        
    output: estimate of the ratio of MMD^2 and of the variance under the H_a
    warning: this function mutates K using the mutate_K function
             there is no approximation but approx_type is required to determine whether to use
             MMD_a estimate as in Eq. (3) or MMD_b estimate as in Eq. (6)
    """ 
    n = int(K.shape[0]/2)

    # compute variance
    Kxx = K[:n, :n]
    Kxy = K[:n, n:]
    Kyx = K[n:, :n]
    Kyy = K[n:, n:]
    H_column_sum = (
        np.sum(Kxx, axis=1)
        + np.sum(Kyy, axis=1)
        - np.sum(Kxy, axis=1)
        - np.sum(Kyx, axis=1)
    )
    var = (
        4 / n ** 3 * np.sum(H_column_sum ** 2)
        - 4 / n ** 4 * np.sum(H_column_sum) ** 2
        + regulariser
    )
    # we should obtain var > 0, if var <= 0 then we discard the corresponding
    # bandwidth by returning a large negative value so that we do not select
    # the corresponding bandwidth when selecting the maximum of the outputs
    # of ratio_mmd_stdev for bandwidths in the collection
    if not var > 0:
        return -1e10 

    # compute original MMD estimate
    mutate_K(K, approx_type)
    if approx_type == "permutation":
        # compute MMD_a estimate
        Kxx = K[:n, :n]
        Kxy = K[:n, n:]
        Kyy = K[n:, n:]
        s = np.ones(n)
        mmd = (
            s @ Kxx @ s / (n * (n - 1))
            + s @ Kyy @ s / (n * (n - 1))
            - 2 * s @ Kxy @ s / (n ** 2)
        )
    elif approx_type == "wild bootstrap":
        # compute MMD_b estimate
        v = np.concatenate((np.ones(n), -np.ones(n)))
        mmd = v @ K @ v / (n * (n - 1))
    else:
        raise ValueError(
            'The value of approx_type should be either "permutation" or "wild bootstrap".'
        )
    return mmd / np.sqrt(var)

# MMD split (maximize std of mmd using split data)
def mmd_split_test(
    seed, X, Y, alpha, kernel_type, approx_type, B1, bandwidth_multipliers, proportion=0.5
):
    """
    Split data in equal halves. Select 'optimal' bandwidth using first half (in the sense 
    that it maximizes ratio_mmd_stdev) and run the MMD test with the selected bandwidth on 
    the second half. This was first proposed by Gretton et al. (2012) for the linear-time 
    MMD estimate and generalised by Liu et al. (2020) to the quadratic-time MMD estimate.
            Arthur Gretton, Bharath Sriperumbudur, Dino Sejdinovic, Heiko Strathmann,
        Sivaraman Balakrishnan, Massimiliano Pontil and Kenji Fukumizu
        Optimal kernel choice for large-scale two-sample tests
        Advances in Neural Information Processing Systems 2012
        https://papers.nips.cc/paper/2012/file/dbe272bab69f8e13f14b405e038deb64-Paper.pdf
            F. Liu, W. Xu, J. Lu, G. Zhang, A. Gretton, and D. J. Sutherland
        Learning deep kernels for non-parametric two-sample tests
        International Conference on Machine Learning, 2020
        http://proceedings.mlr.press/v119/liu20m/liu20m.pdf
    inputs: seed: integer random seed
            X: (m,d) array (m d-dimensional points)
            Y: (n,d) array (n d-dimensional points)
            alpha: real number in (0,1) (level of the test)
            kernel_type: "gaussian" or "laplace"
            approx_type: "permutation" (for MMD_a estimate Eq. (3)) 
                         or "wild bootstrap" (for MMD_b estimate Eq. (6))
            B1: number of simulated test statistics to estimate the quantiles
            bandwidth_multipliers: array such that the 'optimal' bandwidth is selected from
                                   collection_bandwidths = [c*median_bandwidth for c in bandwidth_multipliers]
            proportion: proportion of data used to select the bandwidth 
    output: result of MMD test run on half the data with the bandwidth from collection_bandwidths which is 
            'optimal' in the sense that it maximizes ratio_mmd_stdev on the other half of the data
            (REJECT H_0 = 1, FAIL TO REJECT H_0 = 0)
    """
    assert X.shape == Y.shape
    n, d = X.shape 
    
    split_size = int(n * proportion) 
    
    rs = np.random.RandomState(seed)
    pX = rs.permutation(n)
    pY = rs.permutation(n)
    X1 = X[pX][:split_size]
    X2 = X[pX][split_size:]
    Y1 = Y[pY][:split_size]
    Y2 = Y[pY][split_size:]  
    
    # compute median bandwidth
    median_bandwidth = compute_median_bandwidth_subset(seed, X, Y)

    # using X1 and Y1 select bandwidth which maximizes 
    # the ratio of MMD^2 and of the variance under the H_a 
    kernel_matrices_list = kernel_matrices(
        X1, Y1, kernel_type, median_bandwidth, bandwidth_multipliers
    )
    ratio_values = []
    for i in range(len(kernel_matrices_list)):
        K = kernel_matrices_list[i]
        ratio_values.append(ratio_mmd_stdev(K, approx_type))
    selected_multiplier = bandwidth_multipliers[np.argmax(ratio_values)]
    
    # run MMD test on X2 and Y2 with the selected bandwidth
    return mmd_median_test(
        seed, X2, Y2, alpha, kernel_type, approx_type, B1, selected_multiplier
    )


# C2ST 
def TST_C2ST(X, Y, model, alpha, learning_rate, n_epochs, seed, loss_fn, device):
    
    from sklearn.model_selection import train_test_split
    from split_data import split_data
    
    labels = (torch.cat((torch.zeros(len(X), 1), torch.ones(len(Y), 1)), 0)).squeeze(1).to(device).long()
    labels_np = labels.to('cpu').numpy() 
    
    # Merge two dataset
    dataset = torch.cat([X, Y], dim=0).to('cpu')
    
    n = len(dataset)
    # print(f"Total sample size : {n}")
    # Split to training and test dataset
    
    # Imbalanced training set / Imbalanced validation and test dataset
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels_np, test_size = int(0.4 * n), random_state=seed)
    # X_te, X_val, y_te, y_val = train_test_split(X_test, y_test, test_size=int(0.2 * len(X_test)), random_state=seed)
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=int(0.2 * len(X_train)), random_state=seed)
    
    
    # Imbalanced training set / Balanced validation and test dataset
    # X_train, X_val, X_test, y_train, y_val, y_test = split_data(dataset, labels_np, 0.5, 0.1)
        
    # Dataloader 
    train_dataset = torch.utils.data.TensorDataset(X_train.to(device), torch.tensor(y_train).to(device))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = len(train_dataset), shuffle=True)
    # val_dataset = torch.utils.data.TensorDataset(X_val.to(device), torch.tensor(y_val).to(device)) 
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = len(val_dataset), shuffle=True) 
    test_dataset = torch.utils.data.TensorDataset(X_test.to(device), torch.tensor(y_test).to(device)) 
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    
    # Train model(classifier)
    model.fit(train_dataloader, lr = learning_rate, n_epochs = n_epochs, loss_fn = loss_fn, valloader=None)
    
    # Calculate test statistics
    _, stats, tau0, tau1, correct = model.compute_objective(test_dataloader)

    # Decision 
    if stats >= 1 - norm.ppf(alpha):
        h = 1
    else:
        h = 0 
    
    print(f"stats: {np.round(stats, 4)}, tau0: {np.round(tau0, 4)}, tau1: {np.round(tau1, 4)}, accuracy: {correct / len(test_dataset)}")
    return h



    