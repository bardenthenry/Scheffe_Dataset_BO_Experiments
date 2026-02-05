import math
import os
import warnings
import argparse

import gpytorch
import torch

from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from torch.quasirandom import SobolEngine

from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement, LogExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize

from dataclasses import dataclass
from typing import Optional

from src.scheffe_generator import ScheffeGenerator

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

# Maintain the TuRBO state
@dataclass
class TurboState:
    """Turbo state used to track the recent history of the trust region."""
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        """Post-initialize the state of the trust region."""
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )

def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    """Update the state of the trust region based on the new function values."""
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

# Generate new batch
def generate_batch(
    state: TurboState,
    model: SingleTaskGP,  # GP model
    X: torch.Tensor,  # Evaluated points on the domain [0, 1]^d
    Y: torch.Tensor,  # Function values
    batch_size: int,
    n_candidates: Optional[int] = None,  # Number of candidates for Thompson sampling
    num_restarts: int = 10,
    raw_samples: int = 512,
    acqf: str = "ts",  # "ei" or "ts"
) -> torch.Tensor:
    """Generate a new batch of points."""
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0
    assert X.max() <= 1.0
    assert torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = LogExpectedImprovement(model, Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Scheffe Benchmark Suite")
    parser.add_argument("--result_path", type=str, default='result.json', help="output bo result path")
    parser.add_argument("--data_dir", type=str, default='./datasets/D=5_N=5', help="dataset directory")
    parser.add_argument("--const_method", type=str, default='SLSQP', help="constraints method = PGA or SLSQP")
    parser.add_argument("--acqu", type=str, default='EI', help="acqu fun ['EI', 'UCB']")
    parser.add_argument("--niter", type=int, default=20, help="Number of iter")

    args = parser.parse_args()

    # 讀取參數
    result_path = args.result_path
    n_iterations = args.niter
    dataset_dir = args.data_dir
    constraints_method=args.const_method
    acqu = args.acqu

    # 讀取 datasets
    dataset_names = [n for n in os.listdir(dataset_dir) if '.pt' in n]
    dataset_names.sort()
    dataset_paths = [os.path.join(dataset_dir, n) for n in dataset_names]

    # 建立結果儲存位置
    result_dir = os.path.split(result_path)[0]
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

