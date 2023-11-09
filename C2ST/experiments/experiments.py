import jax
import jax.numpy as jnp
import jax.random as random
from jax.random import multivariate_normal
from utils import MMDVar, compute_mmd_sq, compute_K_matrices, h1_mean_var_gram, MMDu_var
import os
import json
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--n_samples', type=int, default=1000, help='number of samples in one set')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def simulate_mmd(mean1, mean2, cov1, cov2, ratio, num_samples, sigma0):
    mmd_samples = jnp.zeros(100)
    full_vars, complete_vars, incomplete_vars = [], [], []

    key = random.PRNGKey(42) # Set random seed
    for i in range(100):
        key, subkey1, subkey2 = random.split(key, 3)
        X = multivariate_normal(mean=mean1, cov=cov1, shape=(num_samples * ratio,), key=subkey1)
        Y = multivariate_normal(mean=mean2, cov=cov2, shape=(num_samples * ratio,), key=subkey2)

        Kxx, Kyy, Kxy = compute_K_matrices(X, Y, sigma0)
        mmd_value = compute_mmd_sq(Kxx, Kyy, Kxy, len(X), len(Y))
        mmd_samples = mmd_samples.at[i].set(mmd_value)

        full_vars.append(MMDVar(X, Y, sigma0))
        complete_vars.append(MMDVar(X, Y, sigma0, complete=False))
        incomplete_vars.append(h1_mean_var_gram(Kxx, Kyy, Kxy, is_var_computed=True, use_1sample_U=True)[1])

    return mmd_samples, full_vars, complete_vars, incomplete_vars

def save_results_to_json(results, filename="mmd_results.json"):
    with open(filename, 'w') as f:
        # Convert JAX arrays to native Python lists for serialization
        json_results = jax.tree_map(lambda x: x.tolist() if isinstance(x, jnp.ndarray) else x, results)
        json.dump(json_results, f, indent=4)


def main():
    sigma0 = 1.0
    num_samples = args.n_samples
    dimensions = [5, 10, 15, 20, 25]  # for instance; adjust as needed
    ratio = 1  # Fixed ratio
    mean_differences = jnp.linspace(0.1, 5.0, num=5)

    results = {}
    print("Running Mean Difference Experiment with Varying Dimensions...")
    
    for mean_diff in mean_differences:
        for dim in dimensions:
            mean1 = jnp.zeros(dim)
            mean2 = jnp.array([mean_diff] * dim)
            
            cov1 = jnp.eye(dim)
            cov2 = jnp.eye(dim)
            
            mmd_samples, full_vars, complete_vars, incomplete_vars = simulate_mmd(mean1, mean2, cov1, cov2, ratio, num_samples, sigma0)
            true_variance = jnp.var(mmd_samples, ddof=1)
            
            results_key = f"{float(mean_diff.item())}, {dim}"  # Using dimension instead of ratio
            results[results_key] = {
                'MMDu': jnp.mean(mmd_samples),
                'true_variance': true_variance,
                'full_variance_estimate': jnp.mean(jnp.array(full_vars)),
                'complete_variance_estimate': jnp.mean(jnp.array(complete_vars)),
                'incomplete_variance_estimate': jnp.mean(jnp.array(incomplete_vars))
            }

    for key, vals in results.items():
        mean_diff, dim = map(float, key.split(', '))
        print(f"Mean Difference: {mean_diff}, Dimension: {dim}")
        for metric, value in vals.items():
            print(f"{metric}: {value}")
        print("-------")

    save_results_to_json(results, f"mmd_results_dim_{num_samples}.json")
    print("Results saved to mmd_results_dim.json")

if __name__ == "__main__":
    main()

