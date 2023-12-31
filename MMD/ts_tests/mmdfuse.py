import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
from jax.scipy.special import logsumexp
from kernel import jax_distances, kernel_matrix


@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
def mmdfuse(
    X,
    Y,
    key,
    alpha=0.05,
    kernels=("laplace", "gaussian"),
    lambda_multiplier=1,
    number_bandwidths=10,
    number_permutations=2000,
    return_p_val=False,
):
    """
    Two-Sample MMD-FUSE test.

    Given data from one distribution and data from another distribution,
    return 0 if the test fails to reject the null
    (i.e. data comes from the same distribution),
    or return 1 if the test rejects the null
    (i.e. data comes from different distributions).

    Fixing the two sample sizes and the dimension, the first time the function is
    run it is getting compiled. After that, the function can fastly be evaluated on
    any data with the same sample sizes and dimension (with the same other parameters).

    Parameters
    ----------
    X : array_like
        The shape of X must be of the form (m, d) where m is the number
        of samples and d is the dimension.
    Y: array_like
        The shape of X must be of the form (n, d) where m is the number
        of samples and d is the dimension.
    key:
        Jax random key (can be generated by jax.random.PRNGKey(seed) for an integer seed).
    alpha: scalar
        The value of alpha (level of the test) must be between 0 and 1.
    kernels: str or list
        The list should contain strings.
        The value of the strings must be: "gaussian", "laplace", "imq", "matern_0.5_l1",
        "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1",
        "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2",
        "matern_4.5_l2".
    lambda_multiplier: scalar
        The value of lambda_multiplier must be positive.
        The regulariser lambda is taken to be jnp.sqrt(minimum_m_n * (minimum_m_n - 1)) * lambda_multiplier
        where minimum_m_n is the minimum of the sample sizes of X and Y.
    number_bandwidths: int
        The number of bandwidths per kernel to include in the collection.
    number_permutations: int
        Number of permuted test statistics to approximate the quantiles.
    return_p_val: bool
        If true, the p-value is returned.
        If false, the test output Indicator(p_val <= alpha) is returned.

    Returns
    -------
    output : int
        0 if the aggregated MMD-FUSE test fails to reject the null
            (i.e. data comes from the same distribution)
        1 if the aggregated MMD-FUSE test rejects the null
            (i.e. data comes from different distributions)
    """
    # Assertions
    if Y.shape[0] > X.shape[0]:
        X, Y = Y, X
    m = X.shape[0]
    n = Y.shape[0]
    assert n <= m
    assert n >= 2 and m >= 2
    assert 0 < alpha and alpha < 1
    assert lambda_multiplier > 0
    assert number_bandwidths > 1 and type(number_bandwidths) == int
    assert number_permutations > 0 and type(number_permutations) == int
    if type(kernels) is str:
        # convert to list
        kernels = (kernels,)
    for kernel in kernels:
        assert kernel in (
            "imq",
            "rq",
            "gaussian",
            "matern_0.5_l2",
            "matern_1.5_l2",
            "matern_2.5_l2",
            "matern_3.5_l2",
            "matern_4.5_l2",
            "laplace",
            "matern_0.5_l1",
            "matern_1.5_l1",
            "matern_2.5_l1",
            "matern_3.5_l1",
            "matern_4.5_l1",
        )

    # Lists of kernels for l1 and l2
    all_kernels_l1 = (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    )
    all_kernels_l2 = (
        "imq",
        "rq",
        "gaussian",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
    )
    number_kernels = len(kernels)
    kernels_l1 = [k for k in kernels if k in all_kernels_l1]
    kernels_l2 = [k for k in kernels if k in all_kernels_l2]

    # Setup for permutations
    key, subkey = random.split(key)
    B = number_permutations
    # (B, m+n): rows of permuted indices
    idx = random.permutation(
        subkey,
        jnp.array([[i for i in range(m + n)]] * (B + 1)),
        axis=1,
        independent=True,
    )
    # 11
    v11 = jnp.concatenate((jnp.ones(m), -jnp.ones(n)))  # (m+n, )
    V11i = jnp.tile(v11, (B + 1, 1))  # (B, m+n)
    V11 = jnp.take_along_axis(
        V11i, idx, axis=1
    )  # (B, m+n): permute the entries of the rows
    V11 = V11.at[B].set(v11)  # (B+1)th entry is the original MMD (no permutation)
    V11 = V11.transpose()  # (m+n, B+1)
    # 10
    v10 = jnp.concatenate((jnp.ones(m), jnp.zeros(n)))
    V10i = jnp.tile(v10, (B + 1, 1))
    V10 = jnp.take_along_axis(V10i, idx, axis=1)
    V10 = V10.at[B].set(v10)
    V10 = V10.transpose()
    # 01
    v01 = jnp.concatenate((jnp.zeros(m), -jnp.ones(n)))
    V01i = jnp.tile(v01, (B + 1, 1))
    V01 = jnp.take_along_axis(V01i, idx, axis=1)
    V01 = V01.at[B].set(v01)
    V01 = V01.transpose()

    # Compute all permuted MMD estimates
    N = number_bandwidths * number_kernels
    M = jnp.zeros((N, B + 1))
    kernel_count = -1  # first kernel will have kernel_count = 0
    for r in range(2):
        kernels_l = (kernels_l1, kernels_l2)[r]
        l = ("l1", "l2")[r]
        if len(kernels_l) > 0:
            # Pairwise distance matrix
            Z = jnp.concatenate((X, Y))
            pairwise_matrix = jax_distances(Z, Z, l, matrix=True)

            # Collection of bandwidths
            def compute_bandwidths(distances, number_bandwidths):
                median = jnp.median(distances)
                distances = distances + (distances == 0) * median
                dd = jnp.sort(distances)
                lambda_min = dd[(jnp.floor(len(dd) * 0.05).astype(int))] / 2
                lambda_max = dd[(jnp.floor(len(dd) * 0.95).astype(int))] * 2
                bandwidths = jnp.linspace(lambda_min, lambda_max, number_bandwidths)
                return bandwidths

            distances = pairwise_matrix[jnp.triu_indices(pairwise_matrix.shape[0])]
            bandwidths = compute_bandwidths(distances, number_bandwidths)

            # Compute all permuted MMD estimates for either l1 or l2
            for j in range(len(kernels_l)):
                kernel = kernels_l[j]
                kernel_count += 1
                for i in range(number_bandwidths):
                    # compute kernel matrix and set diagonal to zero
                    bandwidth = bandwidths[i]
                    K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
                    K = K.at[jnp.diag_indices(K.shape[0])].set(0)
                    # compute standard deviation
                    unscaled_std = jnp.sqrt(jnp.sum(K**2))
                    # compute MMD permuted values 
                    # with lambda = jnp.sqrt(n * (n - 1))
                    M = M.at[kernel_count * number_bandwidths + i].set(
                        # following the reasoning of
                        # Schrab et al. MMDAgg Appendix C
                        (
                            jnp.sum(V10 * (K @ V10), 0)
                            * (n - m + 1) 
                            * (n - 1)
                            / (m * (m - 1))
                            + jnp.sum(V01 * (K @ V01), 0)
                            * (m - n + 1)
                            / m
                            + jnp.sum(V11 * (K @ V11), 0) 
                            * (n - 1)
                            / m
                        )
                        / unscaled_std
                        * jnp.sqrt(n * (n - 1))
                    )

    # Compute permuted and original statistics
    all_statistics = logsumexp(lambda_multiplier * M, axis=0, b=1 / N)  # (B1+1,)
    original_statistic = all_statistics[-1]  # (1,)

    # Compute statistics and test output
    p_val = jnp.mean(all_statistics >= original_statistic)
    output = p_val <= alpha

    # Return output
    if return_p_val:
        return output.astype(int), p_val
    else:
        return output.astype(int)