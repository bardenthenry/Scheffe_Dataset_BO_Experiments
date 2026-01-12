import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scheffe_generator import ScheffeGenerator

def get_equation_string(gen, active_slice=None):
    """
    Returns a LaTeX-formatted equation string. 
    If active_slice is provided (for high-dim), only shows terms relevant to those 3 dims.
    """
    terms = []
    
    # Decide which indices to display
    if active_slice is None:
        indices = gen.active_indices
    else:
        indices = active_slice

    # Helper to format term with sign
    def add_term(coeff, var_str):
        if abs(coeff) < 1e-2:
            return
        
        # If it's the very first term, allow negative but no leading "+"
        if not terms:
            if coeff < 0:
                terms.append(f"- {abs(coeff):.1f}{var_str}")
            else:
                terms.append(f"{coeff:.1f}{var_str}")
        else:
            # Subsequent terms get explicit " + " or " - "
            op = " + " if coeff >= 0 else " - "
            terms.append(f"{op}{abs(coeff):.1f}{var_str}")

    # Linear Terms
    for i in indices:
        add_term(gen.beta_linear[i], f"x_{{{i}}}")
            
    # Interaction Terms
    for i in indices:
        for j in indices:
            if i < j:
                add_term(gen.beta_interaction[i, j], f"x_{{{i}}}x_{{{j}}}")
    
    full_str = "".join(terms)
    
    # Prepend "y = "
    # Note: "y =" should usually be part of the first math block or separate
    # Let's keep it simple: "$y = ...$"
    
    # Crude split if too long
    if len(full_str) > 40:
        # Find a split point (operator) near the middle or after some length
        # We want to split *before* an operator " + " or " - "
        # Let's search for " + " or " - " after char 30
        
        candidates = [full_str.find(" + ", 30), full_str.find(" - ", 30)]
        # Filter -1
        candidates = [c for c in candidates if c != -1]
        
        if candidates:
            split_idx = min(candidates)
            part1 = full_str[:split_idx]
            part2 = full_str[split_idx:] # Includes the operator
            
            return f"$y = {part1}$ \n ${part2}$"
            
    return f"$y = {full_str}$"

def plot_ternary_on_ax(ax, gen, active_subset=None):
    """
    Plots a ternary landscape on a specific matplotlib axis `ax`.
    active_subset: A list of exactly 3 indices to visualize.
    """
    # Determine which 3 dimensions to plot
    if active_subset is None:
        if len(gen.active_indices) != 3:
            raise ValueError("Generator must have k=3 for auto-plot, or provide active_subset.")
        idx_1, idx_2, idx_3 = gen.active_indices
    else:
        idx_1, idx_2, idx_3 = active_subset

    # Create Barycentric Grid
    n_grid = 60
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    
    Ternary_X = []
    Ternary_Y = []
    Values = []
    
    for x1 in x:
        for x2 in y:
            if x1 + x2 <= 1.0 + 1e-9:
                x3 = 1.0 - x1 - x2
                
                # Construct input vector
                full_vec = np.zeros(gen.D)
                full_vec[idx_1] = x1
                full_vec[idx_2] = x2
                full_vec[idx_3] = x3
                
                # Oracle Call
                val = gen.oracle(full_vec[None, :], noiseless=True)[0]
                
                # Cartesian Projection for Plotting
                # V1=(0,0), V2=(1,0), V3=(0.5, sqrt(3)/2)
                cart_x = 0.5 * (2 * x2 + x3)
                cart_y = (np.sqrt(3) / 2) * x3
                
                Ternary_X.append(cart_x)
                Ternary_Y.append(cart_y)
                Values.append(val)

    # Triangulation for contour plot
    triang = tri.Triangulation(Ternary_X, Ternary_Y)
    
    # Contour Plot
    cntr = ax.tricontourf(triang, Values, levels=14, cmap='viridis')
    
    # Labels (using math mode)
    ax.text(-0.05, -0.05, f"$x_{{{idx_1}}}$", ha='right', va='top', fontsize=12, fontweight='bold')
    ax.text(1.05, -0.05, f"$x_{{{idx_2}}}$", ha='left', va='top', fontsize=12, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, f"$x_{{{idx_3}}}$", ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Title (Equation)
    eqn_str = get_equation_string(gen, active_subset)
    ax.set_title(eqn_str, fontsize=10, pad=20)
    
    ax.axis('equal')
    ax.axis('off')
    return cntr

def run_suite():
    # Create Figure: 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    axes = axes.flatten()
    
    print("Generating 4 Low-Dimensional Plots...")
    # --- Generate 4 Random Low-Dim Examples ---
    seeds = [101, 202, 303, 404]
    variants = ['B', 'A', 'B', 'C'] # Mix of scenarios
    
    for i in range(4):
        gen = ScheffeGenerator(D=3, k_active=(3,3), variant=variants[i], seed=seeds[i])
        plot_ternary_on_ax(axes[i], gen)
        axes[i].text(0.5, -0.2, f"Low-Dim {variants[i]} (D=3)", ha='center', transform=axes[i].transAxes)

    print("Generating 2 High-Dimensional Slices...")
    # --- Generate 2 High-Dimensional Examples (Slices) ---
    
    # High-Dim 1: Variant B (Synergism)
    gen_hd1 = ScheffeGenerator(D=20, k_active=(6, 12), variant='B', seed=42)
    # Find top 3 interacting components to define the slice
    rows, cols = np.nonzero(gen_hd1.beta_interaction)
    # Get indices of the pair with max interaction + one other active dim
    best_idx = np.argmax(gen_hd1.beta_interaction[rows, cols])
    i1, i2 = rows[best_idx], cols[best_idx]
    # Pick a 3rd active dimension that is not i1 or i2
    remaining = [x for x in gen_hd1.active_indices if x not in [i1, i2]]
    i3 = remaining[0] if remaining else i1 # Fallback
    
    plot_ternary_on_ax(axes[4], gen_hd1, active_subset=[i1, i2, i3])
    axes[4].text(0.5, -0.2, f"High-Dim Slice (D=20)\nActive: [{i1}, {i2}, {i3}]", ha='center', transform=axes[4].transAxes)

    # High-Dim 2: Variant C (Antagonism)
    gen_hd2 = ScheffeGenerator(D=20, k_active=(6, 12), variant='C', seed=99)
    # Just pick first 3 active indices for a random slice
    slice_indices = gen_hd2.active_indices[:3]
    plot_ternary_on_ax(axes[5], gen_hd2, active_subset=slice_indices)
    
    # Format slice_indices as a comma-separated list in brackets to be consistent
    indices_str = ", ".join(map(str, slice_indices))
    axes[5].text(0.5, -0.2, f"High-Dim Slice (D=20)\nActive: [{indices_str}]", ha='center', transform=axes[5].transAxes)

    # Global Title
    fig.suptitle("Scheffe Mixture Landscapes: Synthetic Benchmark Suite", fontsize=16)
    
    # Save
    save_path = 'mixture_landscapes.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Suite saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    run_suite()