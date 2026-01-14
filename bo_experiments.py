
import sys
import os
import torch
import matplotlib.pyplot as plt
import math
import json
import argparse
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, LogExpectedImprovement, PosteriorMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from pprint import pprint

# sys.path.append('/workspaces/BO_EXPERIMENTS/src')
from src.scheffe_generator import ScheffeGenerator

# 設定設備與型別
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

def find_factors_x_ge_y(z):
    z = abs(z)
    if z == 0: return 0, 0
    
    # 從平方根開始向下尋找第一個因數
    candidate = math.isqrt(z)
    
    for i in range(candidate, 0, -1):
        if z % i == 0:
            small_factor = i
            large_factor = z // i
            
            # 因為我們是從平方根往下找，large_factor 一定會 >= small_factor
            # 依照題目要求 X >= Y
            x, y = large_factor, small_factor
            return x, y

# 1. Simplex 投影函數 (將 X 投影回 sum(X)=1 且 0<=X<=1)
def project_to_simplex(v, z=1.0):
    n_features = v.shape[-1]
    # 使用 sorting-based algorithm 進行歐幾里得投影
    u, _ = torch.sort(v, descending=True, dim=-1)
    cssv = torch.cumsum(u, dim=-1) - z
    ind = torch.arange(1, n_features + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    # rho = torch.count_nonzero(cond, dim=-1, keepdim=True) - 1

    # rho = cond.to(dtype=v.dtype).sum(dim=-1, keepdim=True) - 1
    rho = (cond.sum(dim=-1, keepdim=True) - 1).long()
    
    # 取得對應索引的累積和值
    theta = cssv.gather(-1, rho) / (rho + 1)
    w = torch.clamp(v - theta, min=0.0)
    return w

# 2. 手寫 Projected Gradient Ascent 優化採集函數
def optimize_acqf_pga(acq_func, n_features, n_restarts=10, steps=50, lr=0.01):
    # 隨機初始化多個點 (符合 sum=1)
    dist = torch.distributions.Dirichlet(torch.ones(n_features, device=device, dtype=dtype))
    x = dist.sample([n_restarts]).requires_grad_(True)
    
    optimizer = torch.optim.Adam([x], lr=lr)
    
    for _ in range(steps):
        optimizer.zero_grad()
        # 因為是 Ascent (求最大值)，所以取負號給 optimizer 最小化
        loss = -acq_func(x.unsqueeze(1)).sum() 
        loss.backward()
        optimizer.step()
        
        # 投影步驟 (Projected Step)
        with torch.no_grad():
            x.copy_(project_to_simplex(x))
            
    # 從重啟的多個點中選出 EI 最大的
    with torch.no_grad():
        ei_values = acq_func(x.unsqueeze(1))
        best_idx = torch.argmax(ei_values)
        output = x[best_idx].unsqueeze(0).detach()
        return output
    
def BO_with_GP_EI_and_PGA(
    train_x, train_obj, gt_x, gt_y, gt_func, n_iterations, 
):
    inference_regrets = [] # 用來存 inference regret 的數值
    simple_regrets = [] # 用來儲存 simple regret 的數值
    n_features = train_x.shape[-1]

    for i in range(n_iterations):
        # Fit GP surrogate model
        gp = SingleTaskGP(train_x, train_obj)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Set Acquisition Function
        best_f = train_obj.max().item()
        EI = LogExpectedImprovement(model=gp, best_f=best_f)

        # Optimize
        # 使用自定義 PGA 尋找下一個點
        candidate = optimize_acqf_pga(EI, n_features=n_features)
        
        # Get New Observe Data
        new_y = gt_func( X=candidate.cpu().numpy(), noiseless=True)

        # Compute Simple Regret
        max_train_obj = train_obj.max().item()
        simple_regret = gt_y - max_train_obj
        simple_regrets.append(simple_regret)

        # Compute Inference Regrets
        infer_regret = float((gt_y - new_y)[0])
        inference_regrets.append(infer_regret)

        # Combine Old and New Observe Data
        train_x = torch.cat([train_x, candidate])
        train_obj = torch.cat([train_obj, torch.tensor(new_y, device=device).unsqueeze(0) ])

        print(f'Epoch {i+1}: Real Best Value: {gt_y:.2f}, Max Train Obj = {max_train_obj:.2f}, Current Train Obj = {float(new_y[0]):.2f}, Simple Regret = {simple_regret:.2f}, Infer Regret = {infer_regret:.2f}, SumX = {candidate.sum().item():.1f}')
    
    output = {
        'inference_regrets': inference_regrets,
        'simple_regrets': simple_regrets,
        'opt_x': candidate.cpu().numpy().tolist(),
        'opt_y': float(new_y[0])
    }

    return output
    
def BO_with_GP_EI_and_SLSQP(
    train_x, train_obj, gt_x, gt_y, gt_func, bounds, constraints, n_iterations
):
    inference_regrets = [] # 用來存 inference regret 的數值
    simple_regrets = [] # 用來儲存 simple regret 的數值

    for i in range(n_iterations):
        # Fit GP surrogate model
        gp = SingleTaskGP(train_x, train_obj)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Find Surrogate Model Best Point
        post_mean_func = PosteriorMean(model=gp)
        best_predicted_x, _ = optimize_acqf(
            acq_function=post_mean_func,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=100,
            equality_constraints=constraints,
        )

        # Get New Observe Data
        new_y = gt_func( X=best_predicted_x.cpu().numpy(), noiseless=True)

        # Set Acquisition Function
        best_f = train_obj.max().item()
        EI = LogExpectedImprovement(model=gp, best_f=best_f)

        # Optimize
        candidate, acq_value = optimize_acqf(
            acq_function=EI,
            bounds=bounds,
            q=1,                     # 每次推薦 1 個點
            num_restarts=10,         # 隨機重啟次數（類似 SLSQP 的重啟）
            raw_samples=100,         # 初始採樣點數量
            equality_constraints=constraints,
        )

        # Combine Old and New Observe Data
        train_x = torch.cat([train_x, candidate])
        train_obj = torch.cat([train_obj, torch.tensor(new_y, device=device).unsqueeze(0) ])
        
        # Compute Simple Regret
        max_train_obj = train_obj.max().item()
        simple_regret = gt_y - max_train_obj
        simple_regrets.append(simple_regret)

        # Compute Inference Regrets
        infer_regret = float((gt_y - new_y)[0])
        inference_regrets.append(infer_regret)
        
        print(f'Epoch {i+1}: Real Best Value: {gt_y:.2f}, Max Train Obj = {max_train_obj:.2f}, Current Train Obj = {float(new_y[0]):.2f}, Simple Regret = {simple_regret:.2f}, Infer Regret = {infer_regret:.2f}, SumX = {candidate.sum().item():.1f}')
    
    output = {
        'inference_regrets': inference_regrets,
        'simple_regrets': simple_regrets,
        'opt_x': candidate.cpu().numpy().tolist(),
        'opt_y': float(new_y[0])
    }

    return output

def run_single_bo_experiments(dataset_path, n_iterations, constraints_method='SLSQP'):
    dataset = torch.load(dataset_path)
    print(dataset['initial_data']['X'].shape)
    print(torch.sum(dataset['ground_truth']['x_star']))
    print(dataset['config'])

    # 設定 Oracle Function
    variant = dataset['config']['variant']
    D = dataset['config']['D']
    seed = dataset['config']['seed']
    gen = ScheffeGenerator( D=D, variant=variant, seed=seed )

    # initial dataset
    train_x = dataset['initial_data']['X'].to(device)
    train_obj = dataset['initial_data']['Y'].to(device)
    gt_x = dataset['ground_truth']['x_star'].to(device)
    gt_y = dataset['ground_truth']['f_star']

    if constraints_method == 'SLSQP':
        # Set X bound
        bounds = torch.stack([torch.zeros(D), torch.ones(D)]).to(device, dtype=dtype)

        # Set constraints
        constraints = [
            (torch.arange(D, device=device), torch.ones(D, dtype=dtype, device=device), 1.0)
        ]

        # 優化過程
        result = BO_with_GP_EI_and_SLSQP(
            train_x, train_obj, gt_x, gt_y, gen.oracle, bounds, constraints, n_iterations
        )
    elif constraints_method == 'PGA':
        result = BO_with_GP_EI_and_PGA(
            train_x, train_obj, gt_x, gt_y, gen.oracle, n_iterations, 
        )
        
    output = {
        'dataset_path': dataset_path,
        'config': { 'variant': variant, 'D': D },
        'result': result
    }

    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Scheffe Benchmark Suite")
    parser.add_argument("--result_path", type=str, default='result.json', help="output bo result path")
    parser.add_argument("--data_dir", type=str, default='./datasets/D=5_N=5', help="dataset directory")
    parser.add_argument("--const_method", type=str, default='SLSQP', help="constraints method = PGA or SLSQP")
    parser.add_argument("--niter", type=int, default=20, help="Number of iter")

    args = parser.parse_args()

    result_path = args.result_path
    n_iterations = args.niter
    dataset_dir = args.data_dir
    constraints_method=args.const_method

    dataset_names = [n for n in os.listdir(dataset_dir) if '.pt' in n]
    dataset_names.sort()
    dataset_paths = [os.path.join(dataset_dir, n) for n in dataset_names]

    # 執行 BO
    results = []
    for dataset_path in dataset_paths:
        print(dataset_path)
        result = run_single_bo_experiments(dataset_path, n_iterations=n_iterations, constraints_method=constraints_method)
        results.append(result)

    # 輸出成 json
    with open(result_path, 'w') as json_file:
        json_file.write(json.dumps(results, indent=4))

    # # 畫圖
    # num_exps = len(dataset_paths)  # 這裡設定為 15

    # # 設定網格
    # rows, cols = find_factors_x_ge_y(num_exps)

    # # 建立畫布，figsize 寬度 18, 高度 20 (可以根據螢幕大小調整)
    # fig, axes = plt.subplots(rows, cols, figsize=(18, 20))
    # axes = axes.flatten()  # 將 5x3 的二維陣列拉平，方便用 for 迴圈遍歷

    # for i in range(num_exps):
    #     ax = axes[i]
        
    #     # 這裡提取資料 (請確保 results[i] 存在)
    #     inf_reg = results[i]['result']['inference_regrets']
    #     sim_reg = results[i]['result']['simple_regrets']
    #     iters = range(1, len(inf_reg) + 1)
        
    #     # 繪製 Inference Regret (藍色小點+虛線)
    #     ax.plot(
    #         iters, inf_reg, 
    #         label='Inference Regret', 
    #         color='midnightblue',    # 改用深藍色
    #         marker='o',              # 加上圓點點
    #         markersize=3, 
    #         linestyle='-',           # 改為實線（若要更明顯）
    #         linewidth=1.2, 
    #         alpha=0.5               # 提高不透明度
    #     )
        
    #     # 繪製 Simple Regret (紅色階梯線)
    #     ax.step(
    #         iters, sim_reg, 
    #         label='Simple Regret (Best)', 
    #         color='firebrick',       # 深紅色
    #         where='post', 
    #         linewidth=2.5
    #     )
        
    #     # 取得檔名後 15 個字元作為標題，避免太長
    #     file_info = os.path.basename(results[i].get('dataset_path', f'Exp_{i+1}'))
    #     ax.set_title(f"Trial {i+1}\n{file_info[-20:]}", fontsize=10)
        
    #     # 圖表細節優化
    #     ax.axhline(0, color='black', linewidth=0.8, linestyle='--') # 零線
    #     ax.grid(True, which='both', linestyle='--', alpha=0.3)
        
    #     # 只有最左邊的圖顯示 Y 軸標籤，最下方的圖顯示 X 軸標籤
    #     if i % cols == 0:
    #         ax.set_ylabel('Regret Value')
    #     if i >= num_exps - cols:
    #         ax.set_xlabel('Iterations')
            
    #     # 在第一張圖顯示圖例
    #     if i == 0:
    #         ax.legend(loc='upper right', fontsize='small')

    # # 如果實驗不足 15 次，隱藏多餘的空白子圖
    # for j in range(num_exps, len(axes)):
    #     axes[j].axis('off')

    # # 自動調整佈局，避免標題與座標軸重疊
    # plt.tight_layout()

    # plt.savefig(output_plot_file_name, dpi=300, bbox_inches='tight')

    # # # 顯示圖片
    # # plt.show()

    # # plt.close()

