import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.widgets import Slider

def barycentric_to_cartesian(x1, x2, x3):
    """Project 3D barycentric to 2D Cartesian."""
    x = 0.5 * (2 * x2 + x3)
    y = (np.sqrt(3) / 2) * x3
    return x, y

def generate_grid(n=50):
    """Pre-compute the grid points to speed up interactive rendering."""
    x_lin = np.linspace(0, 1, n)
    y_lin = np.linspace(0, 1, n)
    
    points_bary = [] # (x1, x2, x3)
    points_cart_x = []
    points_cart_y = []
    
    for x1 in x_lin:
        for x2 in y_lin:
            if x1 + x2 <= 1.0 + 1e-9:
                x3 = 1.0 - x1 - x2
                points_bary.append([x1, x2, x3])
                cx, cy = barycentric_to_cartesian(x1, x2, x3)
                points_cart_x.append(cx)
                points_cart_y.append(cy)
                
    return np.array(points_bary), points_cart_x, points_cart_y

def run_interactive():
    # 1. Setup Data
    Bary, CartX, CartY = generate_grid(n=60)
    triang = tri.Triangulation(CartX, CartY)
    
    # Initial Coefficients
    # Linear: b1, b2, b3
    # Interaction: b12, b13, b23
    init_params = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

    # 2. Setup Figure
    # Increase height to accommodate legend at bottom
    fig, ax = plt.subplots(figsize=(12, 11))
    # Adjust margins: top for title/eq, bottom for sliders + legend
    # CHANGED: top=0.80 to give more room for titles
    plt.subplots_adjust(left=0.1, bottom=0.45, top=0.80) 
    
    # Instructions Title
    fig.suptitle("Interactive Scheffe Mixture Explorer", fontsize=16, fontweight='bold', y=0.98)
    fig.text(0.5, 0.95, "Adjust sliders to verify how coefficients reshape the landscape", 
             ha='center', fontsize=12, style='italic')

    # 3. Initial Plot function
    def calculate_z(params):
        b1, b2, b3, b12, b13, b23 = params
        x1, x2, x3 = Bary[:, 0], Bary[:, 1], Bary[:, 2]
        
        linear = b1*x1 + b2*x2 + b3*x3
        interaction = b12*x1*x2 + b13*x1*x3 + b23*x2*x3
        return linear + interaction

    z = calculate_z(init_params)
    contour = ax.tricontourf(triang, z, levels=14, cmap='viridis')
    ax.axis('equal')
    ax.axis('off')
    
    # Labels
    # Vertices: x1 (bottom-left), x2 (bottom-right), x3 (top-center)
    ax.text(-0.05, -0.05, "$x_1$", fontsize=14, fontweight='bold')
    ax.text(1.05, -0.05, "$x_2$", fontsize=14, fontweight='bold')
    # x3 label: ensure it is clearly separate from the equation title above
    # CHANGED: +0.06 instead of +0.08, slightly closer to tip but title is moved up
    ax.text(0.5, np.sqrt(3)/2 + 0.06, "$x_3$", fontsize=14, fontweight='bold', ha='center')

    # Equation Title
    # We use ax.set_title with padding to place it above x3.
    eqn = (f"$y = {init_params[0]:.1f}x_1 + {init_params[1]:.1f}x_2 + {init_params[2]:.1f}x_3 "
           f"+ {init_params[3]:.1f}x_1x_2 + {init_params[4]:.1f}x_1x_3 + {init_params[5]:.1f}x_2x_3$")
    # CHANGED: pad=40 to push title higher
    title_obj = ax.set_title(eqn, fontsize=11, pad=40)

    # 4. Create Sliders
    # Position: [left, bottom, width, height]
    # Move them down to make room for plot
    slider_y_start = 0.30
    ax_b1 = plt.axes([0.1, slider_y_start, 0.2, 0.03])
    ax_b2 = plt.axes([0.4, slider_y_start, 0.2, 0.03])
    ax_b3 = plt.axes([0.7, slider_y_start, 0.2, 0.03])
    
    ax_b12 = plt.axes([0.1, slider_y_start - 0.08, 0.2, 0.03])
    ax_b13 = plt.axes([0.4, slider_y_start - 0.08, 0.2, 0.03])
    ax_b23 = plt.axes([0.7, slider_y_start - 0.08, 0.2, 0.03])

    s_b1 = Slider(ax_b1, r'$\beta_1$', 0.0, 2.0, valinit=0.5)
    s_b2 = Slider(ax_b2, r'$\beta_2$', 0.0, 2.0, valinit=0.5)
    s_b3 = Slider(ax_b3, r'$\beta_3$', 0.0, 2.0, valinit=0.5)
    
    s_b12 = Slider(ax_b12, r'$\beta_{12}$', -5.0, 10.0, valinit=0.0)
    s_b13 = Slider(ax_b13, r'$\beta_{13}$', -5.0, 10.0, valinit=0.0)
    s_b23 = Slider(ax_b23, r'$\beta_{23}$', -5.0, 10.0, valinit=0.0)

    # 5. Legend / Explanation Text (Bottom)
    legend_text = (
        r"$\bf{Model}: \eta(x) = \sum \beta_i x_i + \sum \beta_{ij} x_i x_j + \epsilon$" + "\n"
        r"$\bullet$ $\bf{Linear\ Blending}$ ($\beta_i$): Response of pure component $i$. High $\beta_i \approx 1.0$ means high quality raw material." + "\n"
        r"$\bullet$ $\bf{Interaction}$ ($\beta_{ij}$): Non-linear effect when mixing $i$ and $j$." + "\n"
        r"   - $\bf{Synergism}$ ($\beta_{ij} > 0$): Mixture is better than sum of parts (e.g. Resin+Fiber)." + "\n"
        r"   - $\bf{Antagonism}$ ($\beta_{ij} < 0$): Mixing degrades performance (e.g. Precipitation)." + "\n"
        r"$\bullet$ $\bf{Scenarios}$:" + "\n"
        r"   1. $\bf{Dominant\ Main\ Effects}$: High $\beta_i$, low/zero $\beta_{ij}$. Quality driven by ingredients." + "\n"
        r"   2. $\bf{Synergism\ Trap}$: Low $\beta_i$, high positive $\beta_{ij}$. Optimum is a specific mixture ratio."
    )
    
    # Place text box at the bottom
    fig.text(0.05, 0.02, legend_text, fontsize=9, ha='left', va='bottom', 
             bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

    # Colorbar Axis
    # Add an axis for colorbar on the right side of the contour plot
    # [left, bottom, width, height]
    cbar_ax = fig.add_axes([0.85, 0.45, 0.03, 0.35])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label('Response Value $y$', fontsize=10)

    # 6. Update Function
    def update(val):
        params = [s_b1.val, s_b2.val, s_b3.val, 
                  s_b12.val, s_b13.val, s_b23.val]
        
        new_z = calculate_z(params)
        
        # Update contour
        ax.clear()
        ax.axis('equal')
        ax.axis('off')
        
        # Re-plot contour
        # Use vmin/vmax based on current data range or fixed range if desired
        # Here letting it auto-scale to show local contrast
        cntr = ax.tricontourf(triang, new_z, levels=14, cmap='viridis')
        
        # Update Colorbar
        # We need to clear the colorbar axis and draw a new one
        cbar_ax.clear()
        cbar_new = fig.colorbar(cntr, cax=cbar_ax)
        cbar_new.set_label('Response Value $y$', fontsize=10)
        
        # Redraw labels
        ax.text(-0.05, -0.05, "$x_1$", fontsize=14, fontweight='bold')
        ax.text(1.05, -0.05, "$x_2$", fontsize=14, fontweight='bold')
        # CHANGED: +0.06
        ax.text(0.5, np.sqrt(3)/2 + 0.06, "$x_3$", fontsize=14, fontweight='bold', ha='center')
        
        # Update Equation Title
        eqn = (f"$y = {params[0]:.1f}x_1 + {params[1]:.1f}x_2 + {params[2]:.1f}x_3 "
               f"+ {params[3]:.1f}x_1x_2 + {params[4]:.1f}x_1x_3 + {params[5]:.1f}x_2x_3$")
        # Split if too long just in case
        if len(eqn) > 80:
             # simplistic split
             mid = len(eqn)//2
             split_plus = eqn.find('+', mid)
             if split_plus != -1:
                 eqn = eqn[:split_plus] + "$\n$" + eqn[split_plus:]
                 
        # CHANGED: pad=40
        ax.set_title(eqn, fontsize=11, pad=40)
        
        fig.canvas.draw_idle()

    # Register updaters
    s_b1.on_changed(update)
    s_b2.on_changed(update)
    s_b3.on_changed(update)
    s_b12.on_changed(update)
    s_b13.on_changed(update)
    s_b23.on_changed(update)

    print("Interactive Interface Loaded. Adjust sliders to explore.")
    plt.show()

if __name__ == "__main__":
    run_interactive()