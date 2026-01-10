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
| `scheffe_generator.py` | **The Oracle.** Generates sparse mixture functions with $D$ dimensions. Supports 'Synergism', 'Linear', and 'Antagonism' variants. |
| `BO_test1.py` | **The Loop.** Runs a BO experiment using BoTorch. Includes a `TernaryAnimator` class that generates GIFs of the optimization dynamics (Truth vs. Acquisition). |
| `interactive_mixture.py` | **Visualization.** A Matplotlib GUI with sliders to manually adjust $\beta$ coefficients and see the resulting Ternary landscape in real-time. |
| `generator_check.py` | **Unit Tests.** Verifies that generated samples satisfy simplex constraints and sparsity requirements. |

## 3. Usage

### A. Verify the Generator
Check that the math holds and constraints are respected:
```bash
python generator_check.py