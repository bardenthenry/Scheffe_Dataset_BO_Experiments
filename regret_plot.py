import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

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
        
def plot_candidate_data(results_list, title_prefix, filename):
    # 2. 設定繪圖矩陣 (5列 x 3欄 = 15張子圖)
    num_exps = len(results_list)  # 這裡設定為 15
    rows, cols = find_factors_x_ge_y(num_exps)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 22))
    axes = axes.flatten()

    # 使用高區隔度配色
    x_colors = ['#1A5276', '#BA4A00', '#1D8348'] # 亮橘、天藍、翠綠

    for i in range(num_exps):
        ax = axes[i]
        x_history = np.array(results_list[i]['result']['candidate_x'])
        iterations = np.arange(1, len(x_history) + 1)
        
        for d in range(x_history.shape[1]):
            ax.plot(iterations, x_history[:, d], label=f'X{d+1}', 
                    color=x_colors[d], linewidth=2.0, alpha=0.85)
        
        ax.set_title(os.path.basename(results_list[i]['dataset_path']), fontsize=10, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.2)
        
        if i % cols == 0: ax.set_ylabel('Coordinate X')
        if i >= (rows - 1) * cols: ax.set_xlabel('Iteration')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=12, bbox_to_anchor=(0.5, 0.98))
    plt.suptitle('Candidate X Progression for {}'.format(title_prefix), fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"成功儲存圖表：{filename}")
    # plt.show()
    plt.close(fig)
        
def plot_length_and_outside_scale(results_list, title_prefix, filename):
    # 2. 設定繪圖矩陣 (5列 x 3欄 = 15張子圖)
    num_exps = len(results_list)  # 這裡設定為 15
    rows, cols = find_factors_x_ge_y(num_exps)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 22))
    axes = axes.flatten()

    # 設定顏色與標籤
    # Dim 1-3 使用原色系，outside_scales 使用顯眼的紅色虛線或粗線
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    dim_labels = ['Dim 1', 'Dim 2', 'Dim 3', 'Outside Scale']

    # 3. 遍歷 15 次實驗進行繪圖
    for i in range(num_exps):
        ax = axes[i]
        
        # 提取檔名
        full_path = results_list[i]['dataset_path']
        pt_filename = os.path.basename(full_path)
        
        # 提取資料
        res = results_list[i]['result']
        ls_history = np.array(res['length_scales'])
        os_history = np.array(res['outside_scales'])
        iterations = np.arange(1, len(ls_history) + 1)
        
        # 繪製三個 Length Scales (實線)
        for d in range(ls_history.shape[1]):
            ax.plot(iterations, ls_history[:, d], label=dim_labels[d], color=colors[d], linewidth=1.2, alpha=0.8)
        
        # 繪製第四條線：Outside Scales (粗虛線以便區分)
        ax.plot(iterations, os_history, label=dim_labels[3], color=colors[3], linewidth=2, linestyle='--')
        
        # 圖表細節設定
        ax.set_title(pt_filename, fontsize=10, fontweight='bold', pad=10)
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # 軸標籤優化
        if i % cols == 0:
            ax.set_ylabel('Scale Value', fontsize=9)
        if i >= (rows - 1) * cols:
            ax.set_xlabel('Iteration', fontsize=9)
        
        ax.tick_params(axis='both', which='major', labelsize=8)

    # 4. 在圖表最上方添加圖例 (Legend) - 包含第四條線
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=12, bbox_to_anchor=(0.5, 0.98))

    # 5. 整體佈局調整
    plt.suptitle('GP Kernel Parameters (Length Scales & Outside Scales) for {}'.format(title_prefix), fontsize=20, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"成功儲存圖表：{filename}")
    # plt.show()
    plt.close(fig)
    
        
def plot_regret_grid(results_list, regret_key, title_prefix, filename, color, plot_type='line', is_log=False):
    """
    繪製 5x3 的 Regret 網格圖，支援對數轉換。
    """
    num_exps = len(results_list)  # 這裡設定為 15
    rows, cols = find_factors_x_ge_y(num_exps)
    fig, axes = plt.subplots(rows, cols, figsize=(18, 22))
    axes = axes.flatten()

    for i in range(num_exps):
        ax = axes[i]
        try:
            # 取得原始數據
            raw_data = np.array(results_list[i]['result'][regret_key])
            
            # --- 核心邏輯：對數轉換 ---
            if is_log:
                # 使用 np.maximum 確保數值不低於 1e-10，再取 log10
                plot_data = np.log10(np.maximum(raw_data, 1e-10))
                ylabel = "Log10(Regret)"
            else:
                plot_data = raw_data
                ylabel = "Regret Value"
            
            iters = range(1, len(plot_data) + 1)
            
            # 繪圖樣式
            if plot_type == 'line':
                ax.plot(iters, plot_data, color=color, marker='o', 
                        markersize=3, linewidth=1.2, alpha=0.8)
            else:
                ax.step(iters, plot_data, color=color, where='post', linewidth=2.5)

            # 標題優化
            file_name = os.path.basename(results_list[i].get('dataset_path', f'Exp_{i+1}'))
            ax.set_title(f"Trial {i+1}: {file_name[-20:]}", fontsize=10, fontweight='bold')
            
            # 只有在非對數模式下才畫 y=0 的線 (對數模式下 y=0 代表原始值為 1)
            if not is_log:
                ax.axhline(0, color='black', linewidth=1, linestyle='--')
            
            ax.grid(True, which='both', linestyle=':', alpha=0.6)
            
            # 座標軸標籤
            if i % cols == 0: ax.set_ylabel(ylabel)
            if i >= (rows - 1) * cols: ax.set_xlabel('Iterations')

        except Exception as e:
            ax.set_title(f"Error in Trial {i+1}")
            print(f"Error: {e}")

    # 隱藏多餘子圖
    for j in range(num_exps, len(axes)):
        axes[j].axis('off')

    # 總標題調整
    log_status = "(Log Scale)" if is_log else "(Linear Scale)"
    fig.suptitle(f"{title_prefix} {log_status}", fontsize=18, y=1.02)
    # fig.suptitle(f"Comparison of {title_prefix} across {num_exps} Trials", fontsize=16, y=1.02)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"成功儲存圖表：{filename}")
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate Scheffe Benchmark Suite")
    parser.add_argument("--result_path", type=str, default='result.json', help="output bo result path")
    parser.add_argument("--output_plot_dir", type=str, default='./datasets/D=5_N=5', help="dataset directory")
    parser.add_argument("--is_log", type=int, default=0, help="0 or 1. If regret need to be take log.")

    args = parser.parse_args()

    result_path = args.result_path
    output_plot_dir = args.output_plot_dir
    is_log = False if args.is_log == 0 else True

    result_name = os.path.splitext(os.path.split(result_path)[-1])[0]

    if not os.path.isdir(output_plot_dir):
        os.makedirs(output_plot_dir)

    with open(result_path, 'r') as file:
        all_results = json.load(file)

    simple_regrets_plot_name = result_name
    title_prefix = 'Simple Regret (Best So Far)'
    if is_log:
        simple_regrets_plot_name += '_Log'
        # title_prefix = 'Log Simple Regret (Best So Far)'
    simple_regrets_plot_name += '_Simple_Regret.png'
    simple_regrets_plot_path = os.path.join(output_plot_dir, simple_regrets_plot_name)

    plot_regret_grid(
        results_list=all_results,
        regret_key='simple_regrets',
        title_prefix=title_prefix,
        filename=simple_regrets_plot_path,
        color='firebrick',    # 用深紅色來區分
        plot_type='step',
        is_log=is_log
    )

    inference_regrets_plot_name = result_name
    title_prefix='Inference Regret'
    if is_log:
        inference_regrets_plot_name += '_Log'
        # title_prefix='Log Inference Regret',
    inference_regrets_plot_name += '_Infer_Regret.png'
    inference_regrets_plot_path = os.path.join(output_plot_dir, inference_regrets_plot_name)

    plot_regret_grid(
        results_list=all_results,
        regret_key='inference_regrets',
        title_prefix=title_prefix,
        filename=inference_regrets_plot_path,
        color='midnightblue',  # 你要求的深一點的顏色
        plot_type='line',
        is_log=is_log
    )
    scale_plot_name = '{}_Kernel_Param.png'.format(result_name)
    scale_plot_path = os.path.join(output_plot_dir, scale_plot_name)
    plot_length_and_outside_scale(
        all_results,
        title_prefix=result_name,
        filename=scale_plot_path
    )

    candidate_plot_name = '{}_Candidate_X.png'.format(result_name)
    candidate_plot_path = os.path.join(output_plot_dir, candidate_plot_name)
    plot_candidate_data(
        all_results,
        title_prefix=result_name,
        filename=candidate_plot_path
    )