import sys
import os
import torch
import matplotlib.pyplot as plt
import math
import numpy as np
import json
import argparse
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, PosteriorMean, UpperConfidenceBound
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from pprint import pprint
from typing import Optional
from datetime import datetime
from src.scheffe_generator import ScheffeGenerator

parser = argparse.ArgumentParser(description="Generate Scheffe Benchmark Suite")
parser.add_argument("--result_dir", type=str, default='./results/jsonfile', help="output bo result path")
parser.add_argument("--data_dir", type=str, default='./datasets/D=5_N=5', help="dataset directory")
parser.add_argument("--ninit", type=int, default=5, help="number of initial points")
parser.add_argument("--acqu", type=str, default='LogEI', help="acqu fun ['LogEI', 'UCB']")
parser.add_argument("--kernel", type=str, default='RBF', help="kernel type ['RBF', 'Matern']")
parser.add_argument("--ard", type=int, default=0, help="Kernel ARD setting (0: False, 1: True)")
parser.add_argument("--niter", type=int, default=20, help="Number of iter")
parser.add_argument("--use_gpu", type=int, default=0, help="Use GPU if available (0: False, 1: True)")
parser.add_argument("--model_type", type=str, default="gp", help="Model type ['gp', 'saasbo']")

args = parser.parse_args()

# 讀取參數
result_dir = args.result_dir
n_iterations = args.niter
dataset_dir = args.data_dir
ninit = args.ninit
acqu = args.acqu
kernel_type = args.kernel
ard = False if args.ard == 0 else True
use_gpu = False if args.use_gpu == 0 else True
model_type = args.model_type

# 設定設備與型別
if use_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
dtype = torch.double
torch.set_default_dtype(dtype)

def set_single_task_gp(train_x:torch.Tensor, train_y:torch.Tensor, kernel_type:str='RBF', ard:bool=False) -> SingleTaskGP:
    '''
    This function sets up a SingleTaskGP surrogate model with the specified kernel type (RBF or Matern).
    Parameters:
    train_x: Training input data as a torch.Tensor.
    train_y: Training output data as a torch.Tensor.
    kernel_type: The type of kernel to use for the GP model ('RBF' or 'Matern').
    Returns:
    surrogate_model: The configured SingleTaskGP model.
    '''
    if kernel_type == 'RBF':
        surrogate_model = SingleTaskGP(
            train_x,
            train_y,
            outcome_transform=Standardize(m=train_y.shape[-1])
        )
    elif kernel_type == 'Matern':
        if ard:
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=train_x.shape[-1])
            )
        else:
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5)
            )
        surrogate_model = SingleTaskGP(
            train_x,
            train_y,
            covar_module=covar_module,
            outcome_transform=Standardize(m=train_y.shape[-1])
        )
    else:
        raise ValueError('Unsupported kernel type, kernel_type should be "RBF" or "Matern".')

    return surrogate_model

def set_single_task_saasbo_gp(train_x:torch.Tensor, train_y:torch.Tensor) -> SaasFullyBayesianSingleTaskGP:
    surrogate_model = SaasFullyBayesianSingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    return surrogate_model

def one_bo_step_with_gp_ei(
    train_x:torch.Tensor,
    train_y:torch.Tensor,
    test_x:torch.Tensor,
    test_y:torch.Tensor,
    kernel_type:str,
    ard:bool,
    best_f:torch.Tensor,
    bounds:torch.Tensor,
    constraints=None,
    acquisition_type:str='LogEI'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, SingleTaskGP]:
    '''
    This function performs one step of Bayesian Optimization using a Gaussian Process surrogate model and an acquisition function (LogEI or UCB).
    The output of this function is:
    candidate: The next evaluation point suggested by the acquisition function.
    acq_value: The acquisition function value at the candidate point.
    best_predicted_x: The point that the surrogate model predicts to be the best.
    mse: The mean squared error on the test set.
    surrogate_model: The trained surrogate model (Gaussian Process).
    '''
    # build surrogate model with GP
    surrogate_model = set_single_task_gp(train_x, train_y, kernel_type=kernel_type, ard=ard)
    with gpytorch.settings.cholesky_jitter(1e-3):
        mll = ExactMarginalLogLikelihood(surrogate_model.likelihood, surrogate_model)
        fit_gpytorch_mll(mll, max_retries=50)

    # Find Surrogate Model Best Point
    post_mean_func = PosteriorMean(model=surrogate_model)
    if constraints is not None:
        best_predicted_x, _ = optimize_acqf(
            acq_function=post_mean_func,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=256,
            equality_constraints=constraints,
        )
    else:
        best_predicted_x, _ = optimize_acqf(
            acq_function=post_mean_func,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=256
        )

    # 計算MSE
    with torch.no_grad():
        # 取得後驗分佈
        posterior = surrogate_model.posterior(test_x)
        mean = posterior.mean
        mse = torch.mean((test_y - mean)**2)

    # 定義獲取函數
    if acquisition_type == 'LogEI':
        acqu_fun = LogExpectedImprovement(model=surrogate_model, best_f=best_f)
    elif acquisition_type == 'UCB':
        acqu_fun = UpperConfidenceBound(model=surrogate_model)
    else:
        raise ValueError('Unsupported acquisition type, the acquisition_type should be "LogEI" or "UCB".')
    
    # 最適化獲取函數以找到下一個評估點
    if constraints is not None:
        candidate, acq_value = optimize_acqf(
            acq_function=acqu_fun,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=256,
            equality_constraints=constraints,
        )
    else:
        candidate, acq_value = optimize_acqf(
            acq_function=acqu_fun,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=256,
        )
    
    return candidate, acq_value, best_predicted_x, mse, surrogate_model

def one_saasbo_step_with_gp_ei(
    train_x:torch.Tensor,
    train_y:torch.Tensor,
    test_x:torch.Tensor,
    test_y:torch.Tensor,
    kernel_type:str,
    best_f:torch.Tensor,
    bounds:torch.Tensor,
    constraints=None,
    acquisition_type:str='LogEI'
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, SingleTaskGP]:
    '''
    This function performs one step of Bayesian Optimization using a Gaussian Process surrogate model and an acquisition function (LogEI or UCB).
    The output of this function is:
    candidate: The next evaluation point suggested by the acquisition function.
    acq_value: The acquisition function value at the candidate point.
    best_predicted_x: The point that the surrogate model predicts to be the best.
    mse: The mean squared error on the test set.
    surrogate_model: The trained surrogate model (Gaussian Process).
    '''
    # build surrogate model with GP
    surrogate_model = SaasFullyBayesianSingleTaskGP(train_x, train_y, outcome_transform=Standardize(m=1))
    with gpytorch.settings.cholesky_jitter(1e-3):
        fit_fully_bayesian_model_nuts(surrogate_model, warmup_steps=128, num_samples=128, thinning=8)

    # Find Surrogate Model Best Point
    post_mean_func = PosteriorMean(model=surrogate_model)
    if constraints is not None:
        best_predicted_x, _ = optimize_acqf(
            acq_function=post_mean_func,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=256,
            equality_constraints=constraints,
        )
    else:
        best_predicted_x, _ = optimize_acqf(
            acq_function=post_mean_func,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=256
        )
    
    # 計算MSE
    with torch.no_grad():
        # 取得後驗分佈
        posterior = surrogate_model.posterior(test_x)
        mean = posterior.mean
        mse = torch.mean((test_y - mean)**2)

    # 定義獲取函數
    if acquisition_type == 'LogEI':
        acqu_fun = LogExpectedImprovement(model=surrogate_model, best_f=best_f)
    elif acquisition_type == 'UCB':
        acqu_fun = UpperConfidenceBound(model=surrogate_model)
    else:
        raise ValueError('Unsupported acquisition type, the acquisition_type should be "LogEI" or "UCB".')
    
    # 最適化獲取函數以找到下一個評估點
    if constraints is not None:
        candidate, acq_value = optimize_acqf(
            acq_function=acqu_fun,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=256,
            equality_constraints=constraints,
        )
    else:
        candidate, acq_value = optimize_acqf(
            acq_function=acqu_fun,
            bounds=bounds,
            q=1,
            num_restarts=20,
            raw_samples=256,
        )
    
    # return candidate, acq_value, best_predicted_x, surrogate_model
    return candidate, acq_value, best_predicted_x, mse, surrogate_model


def optimization_loop(
    train_x:torch.Tensor,
    train_obj:torch.Tensor,
    test_x:torch.Tensor,
    test_obj:torch.Tensor,
    gt_x:torch.Tensor,
    gt_y:float,
    gt_func,
    bounds:torch.Tensor,
    constraints:Optional[list]=None,
    n_iterations:int=20,
    kernel_type:str='RBF',
    acquisition_type:str='LogEI',
    gt_func_noiseless: bool = False,
    model_type:str='gp'
):
    '''
    This function performs the Bayesian Optimization loop for a specified number of iterations.
    Parameters:
    train_x: Initial training input data as a torch.Tensor.
    train_obj: Initial training output data as a torch.Tensor.
    gt_x: Ground truth optimal input data as a torch.Tensor.
    gt_y: Ground truth optimal output value as a float.
    gt_func: The oracle function to evaluate.
    bounds: The bounds for the input space as a torch.Tensor.
    constraints: Optional list of constraints for the optimization.
    n_iterations: Number of BO iterations to perform.
    kernel_type: The type of kernel to use for the GP model ('RBF' or 'Matern').
    acquisition_type: The type of acquisition function to use ('LogEI' or 'UCB').
    '''
    best_f = train_obj.max()

    inference_regrets = [] # 用來存 inference regret 的數值
    simple_regrets = [] # 用來儲存 simple regret 的數值
    best_predicted_xs = [] # 用來儲存每次迭代中 surrogate model 預測的最佳解
    best_predicted_ys = [] # 用來儲存每次迭代中 surrogate model 預測的最佳解對應到 oracle function 的目標值
    mse_ls = [] # 用來存每次迭代的 MSE 數值

    for iteration in range(n_iterations):
        start_time = datetime.now()
        if model_type == 'gp':
            candidate, acq_value, best_predicted_x, mse, surrogate_model = one_bo_step_with_gp_ei(
                train_x, train_obj, test_x, test_obj, kernel_type, ard, best_f, bounds, constraints, acquisition_type
            )
        elif model_type == 'saasbo':
            candidate, acq_value, best_predicted_x, mse, surrogate_model = one_saasbo_step_with_gp_ei(
                train_x, train_obj, test_x, test_obj, kernel_type, best_f, bounds, constraints, acquisition_type
            )
        else:
            raise ValueError('Unsupported model type, the model_type should be "gp" or "saasbo".')

        # Evaluate the oracle function at the candidate point
        new_y = gt_func( X=candidate.cpu().numpy(), noiseless=gt_func_noiseless)
        new_y = torch.tensor(new_y, device=device).unsqueeze(0)

        # Update training data
        train_x = torch.cat([train_x, candidate], dim=0)
        train_obj = torch.cat([train_obj, new_y], dim=0)

        # Update best observed value
        if new_y > best_f:
            best_f = new_y

        # Calculate regrets
        inferred_y = gt_func(best_predicted_x.cpu().numpy(), noiseless=gt_func_noiseless)
        inference_regret = float(gt_y - inferred_y[0])
        simple_regret = float(gt_y - best_f.cpu().item())
        inference_regrets.append(inference_regret)
        simple_regrets.append(simple_regret)
        best_predicted_xs.append(best_predicted_x.cpu().numpy().tolist()[0])
        best_predicted_ys.append(inferred_y[0])
        mse_ls.append(mse.item())

        end_time = datetime.now()
        iteration_time = (end_time - start_time).total_seconds()

        # Print iteration results
        print(f'Iter {iteration+1}/{n_iterations}, SpendTime: {iteration_time:.2f}s' + '-'*50)
        print(f'Ground True: {gt_y:.3f} New Value: {new_y.item():.3f}, Model Best Value: {inferred_y[0]:.3f}, Best Value: {best_f.item():.3f}')
        print(f'Ground True X: {np.array2string(gt_x.cpu().numpy(), precision=3, suppress_small=True)}')
        print(f'New Candidate: {np.array2string(candidate.cpu().numpy(), precision=3, suppress_small=True)}')
        print(f'New Model Best: {np.array2string(best_predicted_x.cpu().numpy(), precision=3, suppress_small=True)}')
        

    return train_x, train_obj, surrogate_model, inference_regrets, simple_regrets, best_predicted_xs, best_predicted_ys, mse_ls

# 出現以下 warning 代表優化器在滿足 等式約束 (Equality Constraints) 的超平面上移動時，遇到了嚴重的數值梯度不穩定。
# /home/appuser/.local/lib/python3.12/site-packages/botorch/optim/optimize.py:789: RuntimeWarning: Optimization failed in `gen_candidates_scipy` with the following warning(s):
#[OptimizationWarning('Optimization failed within `scipy.optimize.minimize` with status 8 and message Positive directional derivative for linesearch.')]
#Trying again with a new set of initial conditions.
#  return _optimize_acqf_batch(opt_inputs=opt_inputs)

if __name__ == '__main__':
    # 讀取資料集位置
    dataset_names = [n for n in os.listdir(dataset_dir) if '.pt' in n]
    dataset_names.sort()
    dataset_paths = [os.path.join(dataset_dir, n) for n in dataset_names]

    # 創建結果儲存位置
    os.makedirs(result_dir, exist_ok=True)

    # 執行 BO 實驗
    for dataset_path in dataset_paths:
        start_time = datetime.now()
        # load dataset
        print(f'Processing dataset: {dataset_path}')
        dataset = torch.load(dataset_path)
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]

        # Set Oracle Function
        D = dataset['config']['D']
        variant = dataset['config']['variant']
        k_active = dataset['config']['k_active']
        seed = dataset['config']['seed']
        gen = ScheffeGenerator( D=D, k_active=k_active, variant=variant, seed=seed )
        gt_func = gen.oracle

        # initial dataset
        train_x = dataset['initial_data']['X'][:ninit].to(device)
        train_obj = dataset['initial_data']['Y'][:ninit].to(device)

        test_x = dataset['initial_data']['X'][ninit:].to(device)
        test_obj = dataset['initial_data']['Y'][ninit:].to(device)

        gt_x = dataset['ground_truth']['x_star'].to(device)
        gt_y = dataset['ground_truth']['f_star']

        # Set X bound
        bounds = torch.stack([torch.zeros(D), torch.ones(D)]).to(device, dtype=dtype)

        # Set constraints
        constraints = [
            (
                torch.arange(D, device=device), # indices: X 的哪些維度要參與計算
                torch.ones(D, dtype=dtype, device=device), # coefficients: 這些維度的係數
                1.0 # rhs: 等號右邊的值 (Sum = 1.0)
            )
        ]

        # Run Optimization Loop
        final_train_x, final_train_obj, final_surrogate_model, inference_regrets, simple_regrets, best_predicted_xs, best_predicted_ys, mse_ls = optimization_loop(
            train_x=train_x,
            train_obj=train_obj,
            test_x=test_x,
            test_obj=test_obj,
            gt_x=gt_x,
            gt_y=gt_y,  
            gt_func=gt_func,
            bounds=bounds,
            constraints=constraints,
            n_iterations=n_iterations,
            kernel_type=kernel_type,
            acquisition_type=acqu,
            gt_func_noiseless=True,
            model_type=model_type
        )

        output = {
            'dataset_path': dataset_path,
            'config': { 'variant': variant, 'D': D, 'k_active': k_active },
            'gt_x': gt_x.cpu().numpy().tolist(),
            'gt_y': gt_y,
            'inference_regrets': inference_regrets,
            'simple_regrets': simple_regrets,
            'final_train_x': final_train_x.cpu().numpy().tolist(),
            'final_train_obj': final_train_obj.cpu().numpy().tolist(),
            'best_predicted_xs': best_predicted_xs,
            'best_predicted_ys': best_predicted_ys,
            'mse_ls': mse_ls
        }

        # 儲存 model
        state = {
            'model_state_dict': final_surrogate_model.state_dict(),
            'covar_module': final_surrogate_model.covar_module,
            'train_X': final_surrogate_model.train_inputs[0],
            'train_Y': final_surrogate_model.train_targets,
        }
        model_save_dir = os.path.join(result_dir, 'models')
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(
            model_save_dir, dataset_name + '_botorch_model.pth'
        )
        torch.save(state, model_save_path)

        # 輸出成 json
        json_dir = os.path.join(result_dir, 'jsonfiles')
        os.makedirs(json_dir, exist_ok=True)
        result_path = os.path.join(
            json_dir, dataset_name + '.json'
        )
        with open(result_path, 'w') as json_file:
            json_file.write(json.dumps(output, indent=4))

        end_time = datetime.now()
        experiment_time = (end_time - start_time).total_seconds()
        print(f'Experiment Spend Time: {experiment_time:.2f}s' + '='*100)