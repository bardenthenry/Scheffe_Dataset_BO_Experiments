import numpy as np
import pandas as pd
from typing import Union, Tuple, List

class ScheffeGenerator:
    """
    Generates synthetic datasets based on Scheffe's Quadratic Mixture Models.
    Supports variable sparsity (k sampled from range) and low-dim debugging.
    """
    def __init__(self, 
                 D: int = 20, 
                 k_active: Union[int, Tuple[int, int]] = (6, 12), 
                 variant: str = 'B', 
                 noise_std: float = 0.01,
                 seed: int = 42):
        """
        Args:
            D: Total dimensionality.
            k_active: Integer (fixed k) or Tuple (min_k, max_k) to sample from.
            variant: 'A' (Linear), 'B' (Synergistic), 'C' (Antagonistic).
            noise_std: Standard deviation of observation noise.
            seed: Random seed.
        """
        self.D = D
        self.variant = variant
        self.noise = noise_std
        self.rng = np.random.default_rng(seed)
        
        # 1. Determine actual k for this instance
        if isinstance(k_active, tuple) or isinstance(k_active, list):
            self.k = self.rng.integers(k_active[0], k_active[1] + 1)
        else:
            self.k = int(k_active)
            
        # Safety clamp: k cannot exceed D
        self.k = min(self.k, self.D)
            
        # State to store the 'Ground Truth' coefficients
        self.active_indices = None
        self.beta_linear = np.zeros(D)
        self.beta_interaction = np.zeros((D, D))  # Upper triangular
        
        self._initialize_ground_truth()

    def _initialize_ground_truth(self):
        """Defines coefficients based on the chosen variant."""
        # 1. Select Active Subspace
        self.active_indices = self.rng.choice(self.D, size=self.k, replace=False)
        # Sort indices for cleaner debugging/printing
        self.active_indices.sort()
        active = self.active_indices
        
        # 2. Define Coefficients
        if self.variant == 'A':  # Dominant Main Effects
            self.beta_linear[active] = self.rng.uniform(0.8, 1.2, size=self.k)
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    idx_i, idx_j = active[i], active[j]
                    self.beta_interaction[idx_i, idx_j] = self.rng.normal(0, 0.05)

        elif self.variant == 'B':  # Sparse Synergism (PBT Case)
            self.beta_linear[active] = self.rng.uniform(0.2, 0.4, size=self.k)
            
            # Generate all possible pairs within the active set
            pairs = [(active[i], active[j]) for i in range(self.k) for j in range(i+1, self.k)]
            
            if len(pairs) > 0:
                # Robustly pick 3-5 pairs (or fewer if k is small)
                desired_pairs = self.rng.integers(3, 6)
                num_pairs = min(desired_pairs, len(pairs))
                
                chosen_pairs = self.rng.choice(pairs, size=num_pairs, replace=False)
                
                for (i, j) in chosen_pairs:
                    # Synergism: Mixture performs better than sum of parts
                    self.beta_interaction[i, j] = self.rng.uniform(2.0, 5.0)

        elif self.variant == 'C':  # Antagonism
            self.beta_linear[active] = self.rng.uniform(1.0, 1.2, size=self.k)
            for i in range(self.k):
                for j in range(i + 1, self.k):
                    idx_i, idx_j = active[i], active[j]
                    self.beta_interaction[idx_i, idx_j] = self.rng.uniform(-5.0, -2.0)

    def sample_inputs(self, N: int) -> np.ndarray:
        """Generates N sample mixtures (Simplex) on active indices."""
        X = np.zeros((N, self.D))
        alpha = np.ones(self.k) # Uniform on the active face
        active_mixtures = self.rng.dirichlet(alpha, size=N)
        X[:, self.active_indices] = active_mixtures
        return X

    def oracle(self, X: np.ndarray, noiseless: bool = False) -> np.ndarray:
        """Evaluates y = Beta*x + x'Beta_int*x"""
        N = X.shape[0]
        y = np.zeros(N)
        
        # Linear Term
        y += X @ self.beta_linear
        
        # Interaction Term (Efficient Sparse Calculation)
        # Only iterate over non-zero interaction terms to save time
        rows, cols = np.nonzero(self.beta_interaction)
        for i, j in zip(rows, cols):
            y += self.beta_interaction[i, j] * X[:, i] * X[:, j]
        
        if not noiseless:
            y += self.rng.normal(0, self.noise, size=N)
        return y

    def print_ground_truth(self):
        """Helper to inspect the generated function (for debugging)."""
        print(f"--- Generator State (D={self.D}, k={self.k}) ---")
        print(f"Active Indices: {self.active_indices}")
        print("Linear Coeffs (Active only):")
        with np.printoptions(precision=3, suppress=True):
            print(self.beta_linear[self.active_indices])
        
        print("\nNon-Zero Interactions (Synergism):")
        rows, cols = np.nonzero(self.beta_interaction)
        if len(rows) == 0:
            print("  None")
        else:
            for i, j in zip(rows, cols):
                print(f"  Idx {i} & {j}: {self.beta_interaction[i, j]:.3f}")
        print("---------------------------------------------")