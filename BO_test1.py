import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import imageio.v2 as imageio
import os
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from botorch.acquisition import ExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood

# Import your generator
from scheffe_generator import ScheffeGenerator

# --- 1. Helper: Analytical Optimum Finder (for D=3) ---
def find_analytical_optimum(gen, n_grid=100):
    """Brute-force grid search over ternary plot to find 'True Max'."""
    best_val = -np.inf
    best_x = None
    x = np.linspace(0, 1, n_grid)
    for x1 in x:
        for x2 in x:
            if x1 + x2 <= 1.0:
                x3 = 1.0 - x1 - x2
                x_vec = np.zeros(gen.D)
                idx = gen.active_indices
                x_vec[idx[0]], x_vec[idx[1]], x_vec[idx[2]] = x1, x2, x3
                
                val = gen.oracle(x_vec[None, :], noiseless=True)[0]
                if val > best_val:
                    best_val = val
                    best_x = x_vec
    return best_x, best_val

# --- 2. Enhanced Visualization Class ---
class TernaryAnimator:
    def __init__(self, gen, optimum_x, optimum_val):
        self.gen = gen
        self.optimum_x = optimum_x
        self.optimum_val = optimum_val
        self.idx = gen.active_indices
        self.frames = []
        
        # History for Bottom Plots
        self.history_best = []
        self.history_uncertainty = []
        
        # Precompute Grid for plotting contours
        self.n_grid = 60
        x = np.linspace(0, 1, self.n_grid)
        y = np.linspace(0, 1, self.n_grid)
        self.T_X, self.T_Y, self.Z_True = [], [], []
        self.grid_bary = [] 
        
        for x1 in x:
            for x2 in y:
                if x1 + x2 <= 1.0:
                    x3 = 1.0 - x1 - x2
                    # Cartesian Coords
                    cart_x = 0.5 * (2 * x2 + x3)
                    cart_y = (np.sqrt(3) / 2) * x3
                    self.T_X.append(cart_x)
                    self.T_Y.append(cart_y)
                    
                    # True Oracle Value
                    full_vec = np.zeros(gen.D)
                    full_vec[self.idx[0]], full_vec[self.idx[1]], full_vec[self.idx[2]] = x1, x2, x3
                    self.grid_bary.append(full_vec)
                    self.Z_True.append(gen.oracle(full_vec[None, :], noiseless=True)[0])
        
        self.triang = tri.Triangulation(self.T_X, self.T_Y)
        self.grid_tensor = torch.tensor(np.array(self.grid_bary), dtype=torch.float32)

    def capture_frame(self, model, train_x, train_y, iteration):
        # Create 2x2 Grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        plt.subplots_adjust(hspace=0.3)
        
        # --- Top Left: Ground Truth & Observations ---
        ax1 = axes[0, 0]
        ax1.set_title(f"Ground Truth (Iter {iteration})")
        ax1.tricontourf(self.triang, self.Z_True, levels=14, cmap='viridis', alpha=0.6)
        
        # Plot Optimum
        opt_bary = self.optimum_x[self.idx]
        opt_cx = 0.5 * (2 * opt_bary[1] + opt_bary[2])
        opt_cy = (np.sqrt(3) / 2) * opt_bary[2]
        ax1.scatter([opt_cx], [opt_cy], marker='*', s=200, c='gold', edgecolors='black', label="True Opt", zorder=10)
        
        # Plot Observations
        obs_x_np = train_x.numpy()
        obs_cx, obs_cy = [], []
        for row in obs_x_np:
            b1, b2, b3 = row[self.idx[0]], row[self.idx[1]], row[self.idx[2]]
            obs_cx.append(0.5 * (2 * b2 + b3))
            obs_cy.append((np.sqrt(3) / 2) * b3)
            
        n_init = 5
        if len(obs_cx) <= n_init:
            ax1.scatter(obs_cx, obs_cy, c='white', edgecolors='black')
        else:
            ax1.scatter(obs_cx[:n_init], obs_cy[:n_init], c='white', edgecolors='black', label="Initial")
            ax1.scatter(obs_cx[n_init:-1], obs_cy[n_init:-1], c='red', alpha=0.5, s=30)
            ax1.scatter(obs_cx[-1], obs_cy[-1], c='red', edgecolors='white', s=100, label="Newest")
        ax1.legend(loc='upper right', fontsize=8)
        ax1.axis('off')

        # --- Top Right: Acquisition Function (EI) ---
        ax2 = axes[0, 1]
        ax2.set_title("Acquisition (Expected Improvement)")
        
        # Calculate EI on the grid
        with torch.no_grad():
            posterior = model.posterior(self.grid_tensor)
            mu = posterior.mean.squeeze()
            sigma = posterior.variance.sqrt().squeeze()
            best_f = train_y.max() # Standardized Y used in model, but we just need shape
            # Manual EI calc for plotting speed
            # Note: We need to respect the standardization of the model
            # For visualization shape, (mu - best) is sufficient proxy for "desirability"
            z = (mu - best_f) / (sigma + 1e-9)
            norm = torch.distributions.Normal(0, 1)
            ei_vals = (mu - best_f) * norm.cdf(z) + sigma * norm.log_prob(z).exp()
            
            # Store mean variance for bottom plot
            self.history_uncertainty.append(sigma.mean().item())
            
        cntr = ax2.tricontourf(self.triang, ei_vals.numpy(), levels=14, cmap='plasma')
        ax2.axis('off')

        # --- Bottom Left: Optimization Trace (Regret) ---
        ax3 = axes[1, 0]
        ax3.set_title("Optimization Trace")
        
        current_best = train_y.max().item()
        self.history_best.append(current_best)
        
        ax3.plot(self.history_best, marker='o', color='blue', label="Best Found")
        ax3.axhline(self.optimum_val, color='gold', linestyle='--', label="True Max")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Response y")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # --- Bottom Right: Global Uncertainty ---
        ax4 = axes[1, 1]
        ax4.set_title("Model Uncertainty (Avg Variance)")
        ax4.plot(self.history_uncertainty, marker='x', color='purple')
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("Average Sigma")
        ax4.grid(True, alpha=0.3)

        # Save frame
        plt.tight_layout()
        filename = f"temp_frame_{iteration:03d}.png"
        plt.savefig(filename)
        self.frames.append(filename)
        plt.close()

    def save_animation(self, filename):
        # Increased duration to 7.5s per frame (0.2x speed)
        with imageio.get_writer(filename, mode='I', duration=7.5) as writer:
            for frame_file in self.frames:
                image = imageio.imread(frame_file)
                writer.append_data(image)
                os.remove(frame_file)
        print(f"Animation saved: {filename}")


# --- 3. Main BO Loop ---
def run_bo_experiment(variant='B', seed=42):
    print(f"\n--- Starting BO Experiment: Variant {variant} ---")
    
    # 1. Setup Oracle (D=3 for visualization)
    gen = ScheffeGenerator(D=3, k_active=(3,3), variant=variant, seed=seed)
    true_opt_x, true_opt_val = find_analytical_optimum(gen)
    print(f"True Optimum Value: {true_opt_val:.4f}")

    # 2. Initial Dataset (5 Random Points)
    n_init = 5
    train_X = torch.tensor(gen.sample_inputs(n_init), dtype=torch.float32)
    train_Y = torch.tensor(gen.oracle(train_X.numpy()), dtype=torch.float32).unsqueeze(-1)

    animator = TernaryAnimator(gen, true_opt_x, true_opt_val)
    
    # 3. BO Loop (20 Iterations)
    n_iters = 20
    
    for i in range(n_iters):
        # A. Fit GP Model
        train_Y_std = standardize(train_Y)
        model = SingleTaskGP(train_X, train_Y_std)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        # B. Visualize current state BEFORE optimizing next point
        # Pass raw Y for tracking best value
        animator.capture_frame(model, train_X, train_Y, i)
        
        # C. Define & Optimize Acquisition Function
        EI = ExpectedImprovement(model, best_f=train_Y_std.max())
        
        # Grid Search Optimization on Ternary Mesh
        with torch.no_grad():
            # Must evaluate on the same scale as model (standardized)
            grid_evals = EI(animator.grid_tensor.unsqueeze(1))
            best_idx = torch.argmax(grid_evals)
            new_x = animator.grid_tensor[best_idx].unsqueeze(0)
            
        # D. Get Oracle Value
        new_y = torch.tensor(gen.oracle(new_x.numpy()), dtype=torch.float32).unsqueeze(-1)
        
        # E. Update Data
        train_X = torch.cat([train_X, new_x])
        train_Y = torch.cat([train_Y, new_y])
        
        print(f"Iter {i+1}/{n_iters}: Best Found = {train_Y.max().item():.4f}")

    # Save Results
    animator.save_animation(f"BO_dynamics_variant_{variant}.gif")

if __name__ == "__main__":
    # Run Synergism Case
    run_bo_experiment(variant='B', seed=101)