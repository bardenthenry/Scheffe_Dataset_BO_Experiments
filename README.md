# Sparse Compositional Bayesian Optimization Benchmark

A physics-informed benchmarking suite for evaluating Bayesian Optimization (BO) algorithms on high-dimensional, sparse chemical formulation problems.

## 1. Overview
This repository implements a synthetic data generator based on **Scheffé's Quadratic Mixture Models** [Scheffé, 1958]. Unlike standard BBOB benchmarks, this suite respects the geometric constraints of the simplex ($\sum x_i = 1$) and models chemical phenomena like synergism and antagonism.

### Mathematical Formulation
The ground truth oracle is defined as:
$$\eta(x) = \sum_{i \in \mathcal{A}} \beta_i x_i + \sum_{i,j \in \mathcal{A}, i<j} \beta_{ij} x_i x_j + \epsilon$$
Where $\mathcal{A}$ is a sparse subset of active ingredients ($|\mathcal{A}| \ll D$).

## 2. File Structure

| File | Description |
| :--- | :--- |
| `src/scheffe_generator.py` | **The Oracle.** Generates sparse mixture functions with $D$ dimensions. |
| `scripts/benchmark_bo.py` | **The Loop.** Runs a BO experiment using BoTorch. |
| `scripts/visualize_interactive.py` | **Visualization.** GUI with sliders to adjust $\beta$ coefficients. |
| `scripts/check_validity.py` | **Unit Tests.** Verifies constraints and sparsity. |
| `results/` | **Outputs.** All plots and GIFs are saved here. |

## 3. Usage

### 1. Prerequisite (One-time)
If you (or your collaborators) haven't done it yet, install the package in editable mode:
```bash
pip install -e .
```

### 2. Running the Scripts
Run these commands from the **root directory**:

*   **Benchmark Loop (creates GIF in `results/`):**
    ```bash
    python scripts/benchmark_bo.py
    ```

*   **Sanity Check (creates PNG in `results/`):**
    ```bash
    python scripts/check_validity.py
    ```

*   **Interactive Visualization:**
    ```bash
    python scripts/visualize_interactive.py
    ```