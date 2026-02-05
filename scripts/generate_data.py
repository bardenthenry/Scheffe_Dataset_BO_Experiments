import torch
import numpy as np
import os
import sys
import argparse
from scipy.optimize import minimize, Bounds, LinearConstraint

# Import generator
try:
    from scheffe_generator import ScheffeGenerator
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from scheffe_generator import ScheffeGenerator

def find_ground_truth(gen, restarts=20):
    """
    Uses Scipy SLSQP to find the true global maximum of the generator.
    """
    D = gen.D
    bounds = Bounds([0]*D, [1]*D)
    cons = LinearConstraint(np.ones(D), [1], [1])
    
    best_val = -np.inf
    best_x = None
    
    # Silence scipy output for bulk generation
    for i in range(restarts):
        x0 = np.random.rand(D)
        x0 /= x0.sum()
        
        # Maximize gen.oracle (Minimize negative)
        res = minimize(
            lambda x: -gen.oracle(x[None, :], noiseless=True)[0],
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'ftol': 1e-9}
        )
        
        val = -res.fun
        if val > best_val:
            best_val = val
            best_x = res.x
            
    return best_x, best_val

def generate_suite(D, N_datasets, k, N_init_samples=10):
    """
    Generates a full benchmark suite for a given Dimension D.
    Creates N_datasets for EACH variant (A, B, C).
    """
    variants = ['A', 'B', 'C']
    base_seed = 1000  # Offset to ensure non-overlapping seeds between runs
    
    # 1. Setup Directory Structure
    root_dir = os.path.join(os.path.dirname(__file__), '..')
    suite_dir_name = f"D={D}_N={N_datasets}_K={k}"
    data_dir = os.path.join(root_dir, "datasets", suite_dir_name)
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"\n=== Generating Benchmark Suite ===")
    print(f"  Output Directory: {data_dir}")
    print(f"  Dimension (D): {D}")
    print(f"  Datasets per Variant: {N_datasets}")
    
    total_files = len(variants) * N_datasets
    count = 0

    for variant in variants:
        print(f"\n--- Processing Variant {variant} ---")
        
        for i in range(N_datasets):
            # Unique seed for every single file
            # variant_offset: A=0, B=10000, C=20000 to keep distinct
            variant_offset = {'A': 0, 'B': 10000, 'C': 20000}
            seed = base_seed + variant_offset[variant] + i
            
            # Instantiate
            # Note: noise_std is set to 0.01 as per standard benchmarks
            '''
            D: int = 20, 
            k_active: Union[int, Tuple[int, int]] = (6, 12), 
            variant: str = 'B', 
            noise_std: float = 0.01,
            seed: int = 42
            '''
            gen = ScheffeGenerator(
                D=D, k_active=k, variant=variant, noise_std=0.01, seed=seed
            )
            
            # Initial Data
            X_init = gen.sample_inputs(N_init_samples)
            y_init = gen.oracle(X_init, noiseless=False)
            
            # Ground Truth
            true_x, true_val = find_ground_truth(gen)
            
            dataset = {
                "config": {"D": D, "variant": variant, "seed": seed, "k_active": k},
                "initial_data": {
                    "X": torch.tensor(X_init, dtype=torch.float64),
                    "Y": torch.tensor(y_init, dtype=torch.float64).unsqueeze(-1)
                },
                "ground_truth": {
                    "x_star": torch.tensor(true_x, dtype=torch.float64),
                    "f_star": float(true_val),
                    "generator_seed": seed
                }
            }
            
            # Print sample for the first file of the variant
            if i == 0:
                print(f"  Example Data Point:")
                # Just show the active components for brevity + Y
                active_sample = X_init[0][gen.active_indices]
                y_sample = y_init[0]
                # Format sparse vector for display
                active_str = ", ".join([f"{val:.3f}" for val in active_sample])
                print(f"    X (Active Only): [{active_str}]")
                print(f"    Y: {y_sample:.4f}")
                print(f"    Active Indices: {gen.active_indices}")
            
            # Save
            filename = f"oracle_data_D{D}_{variant}_{i:03d}.pt"
            save_path = os.path.join(data_dir, filename)
            torch.save(dataset, save_path)
            
            count += 1
            sys.stdout.write(f"\r  > Generated {count}/{total_files}: {filename}")
            sys.stdout.flush()
    
    print(f"\n\n✓ Generation Complete. {total_files} files saved to {data_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Scheffe Benchmark Suite")
    parser.add_argument("--d", type=int, default=20, help="Dimension of the problem")
    parser.add_argument("--n", type=int, default=5, help="Number of datasets per variant")
    parser.add_argument("--k", type=int, default=5, help="Number of active components")
    parser.add_argument("--n_int", type=int, default=5, help="Number of initial samples")
    
    args = parser.parse_args()
    generate_suite(D=args.d, N_datasets=args.n, k=args.k, N_init_samples=args.n_int)