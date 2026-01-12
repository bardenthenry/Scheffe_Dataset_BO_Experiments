import os
import sys
import torch
import glob
import re
import matplotlib.pyplot as plt
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import LogExpectedImprovement, PosteriorMean
from botorch.optim import optimize_acqf
from botorch.models.transforms import Standardize, Normalize

# --- Import Generator for Ground Truth Evaluation ---
# Assumes scheffe_generator.py is in the parent/src folder or same directory
try:
    from scheffe_generator import ScheffeGenerator
except ImportError:
    # Adjust this path to where your generator script actually lives
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    try:
        from scheffe_generator import ScheffeGenerator
    except ImportError:
        print("CRITICAL ERROR: Could not import 'ScheffeGenerator'.")
        print("Please ensure scheffe_generator.py is in the python path.")
        sys.exit(1)

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {"dtype": torch.float64, "device": device}

def optimize_acq_function(acq_func, D):
    """
    Optimizes the acquisition function over the simplex constraints.
    Constraint: sum(x) = 1, x >= 0
    """
    # Define linear constraint: sum(x) = 1
    # BoTorch format: (indices, coefficients, rhs) -> sum(x[indices] * coeffs) = rhs
    # But optimize_acqf uses inequality constraints by default or equality via passing options.
    # For the simplex, we often use specific inequality constraints or fixed_features.
    # However, BoTorch's `optimize_acqf` supports equality_constraints directly.
    
    # Equality: sum(x) - 1 = 0
    constraints = [(torch.tensor(range(D), device=device), 
                   torch.ones(D, device=device, dtype=torch.float64), 
                   1.0)]

    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.stack([torch.zeros(D, **tkwargs), torch.ones(D, **tkwargs)]),
        q=1,
        num_restarts=10,
        raw_samples=100,  # Initialization samples
        equality_constraints=constraints,
        options={"batch_limit": 5, "maxiter": 200}
    )
    return candidates

def run_single_benchmark(filepath, n_steps=10):
    """
    Runs a BO loop for a single dataset file.
    """
    print(f"Processing: {os.path.basename(filepath)}")
    
    # 1. Load Data
    data = torch.load(filepath)
    D = data['config']['D']
    variant = data['config']['variant']
    seed = data['config']['seed']
    
    # 2. Re-instantiate the Ground Truth Oracle
    # We need the generator to evaluate NEW points selected by BO
    gen = ScheffeGenerator(D=D, variant=variant, seed=seed, noise_std=0.01)
    
    # 3. Setup Initial Data
    train_x = data['initial_data']['X'].to(**tkwargs)
    train_y = data['initial_data']['Y'].to(**tkwargs) # Shape (N, 1)
    
    # Ground Truth Optimal Value (for Regret)
    f_star = data['ground_truth']['f_star']
    
    inference_regrets = []
    
    # --- BO Loop ---
    for t in range(n_steps):
        # A. Fit GP Model
        # Normalize inputs [0,1] and Standardize outputs (mean 0, std 1)
        gp = SingleTaskGP(
            train_x, 
            train_y, 
            input_transform=Normalize(d=D),
            outcome_transform=Standardize(m=1)
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # B. Calculate Inference Regret (Current best guess)
        # We ask the model: "Where do you think the peak is?" (Maximize Posterior Mean)
        post_mean_acq = PosteriorMean(model=gp)
        rec_x = optimize_acq_function(post_mean_acq, D)
        
        # Evaluate recommendation on TRUE noiseless oracle
        f_rec_val = gen.oracle(rec_x.cpu().numpy(), noiseless=True).item()
        inf_regret = f_star - f_rec_val
        inference_regrets.append(max(0, inf_regret)) # Clamp at 0 for numerical noise
        
        # C. Select Next Query Point (Exploration)
        # Use LogExpectedImprovement for better numerical stability than EI
        EI = LogExpectedImprovement(model=gp, best_f=train_y.max())
        new_x = optimize_acq_function(EI, D)
        
        # D. Query Oracle (with noise)
        new_y_val = gen.oracle(new_x.cpu().numpy(), noiseless=False).item()
        new_y = torch.tensor([[new_y_val]], **tkwargs)
        
        # E. Update Training Data
        train_x = torch.cat([train_x, new_x])
        train_y = torch.cat([train_y, new_y])
        
    return inference_regrets

def benchmark_suite(suite_folder, n_steps=15):
    """
    Runs benchmark for all files in the folder and aggregates results.
    """
    # Find all files
    pattern = os.path.join(suite_folder, "oracle_data_*.pt")
    files = sorted(glob.glob(pattern))
    
    if not files:
        print(f"No files found in {suite_folder}")
        return

    results = {'A': [], 'B': [], 'C': []}
    
    # Run loop
    for f in files:
        # Extract variant from filename (e.g., ..._A_000.pt)
        match = re.search(r'_([ABC])_\d+\.pt', f)
        if match:
            variant = match.group(1)
            regrets = run_single_benchmark(f, n_steps)
            results[variant].append(regrets)
            
    return results

def plot_results(results):
    """
    Plots the Average Log Inference Regret with Std Error.
    """
    plt.figure(figsize=(10, 6))
    
    colors = {'A': 'blue', 'B': 'red', 'C': 'green'}
    labels = {
        'A': 'Variant A (Linear)', 
        'B': 'Variant B (Sparse Synergism)', 
        'C': 'Variant C (Antagonism)'
    }
    
    for variant, runs in results.items():
        if not runs:
            continue
            
        # Shape: (N_datasets, N_steps)
        data = np.array(runs)
        
        # Compute Mean and Std Error
        mean = np.mean(data, axis=0)
        std_err = np.std(data, axis=0) / np.sqrt(data.shape[0])
        x = np.arange(1, len(mean) + 1)
        
        # Log Plot handling
        # We plot log10(regret + epsilon) to handle perfect 0 regret
        epsilon = 1e-6
        
        plt.plot(x, mean, label=labels[variant], color=colors[variant], linewidth=2)
        plt.fill_between(x, mean - std_err, mean + std_err, color=colors[variant], alpha=0.1)
        
    plt.title("Benchmark Performance: Log Inference Regret", fontsize=14)
    plt.xlabel("BO Iterations", fontsize=12)
    plt.ylabel("Inference Regret (Lower is Better)", fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.tight_layout()
    output_png = "benchmark_results.png"
    plt.savefig(output_png, format='png', dpi=300)
    print(f"Plot saved to {output_png}")
    # plt.show() # Disabled to prevent backend errors on some systems

    # Save raw results
    output_pt = "benchmark_results.pt"
    torch.save(results, output_pt)
    print(f"Raw results saved to {output_pt}")

if __name__ == "__main__":
    # Point this to your generated folder
    # Default: Look in ../datasets/D=5_N=5 relative to THIS script
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_dataset_dir = os.path.join(script_dir, "..", "datasets", "D=5_N=5")
    
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = default_dataset_dir
        
    print(f"Running Benchmark on: {folder}")
    if not os.path.exists(folder):
        print(f"Error: Folder not found at {folder}")
        print("Did you run 'python scripts/generate_data.py --d 5 --n 5'?")
        sys.exit(1)
        
    results = benchmark_suite(folder, n_steps=15)
    
    if results:
        plot_results(results)
    else:
        print("No results to plot.")