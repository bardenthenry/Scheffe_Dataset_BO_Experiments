# Scheffe Dataset Generate

```bash
# D=3 k=3 N_int=10 N_Dataset=5
python3 scripts/generate_data.py --d=3 --n=3 --k=3 --n_int=100

# D=5 k=3 N_int=2k+10=20 N_Dataset=5
python3 scripts/generate_data.py --d=5 --n=3 --k=3 --n_int=200

# D=10 k=6 N_int=2k+10=30 N_Dataset=5
python3 scripts/generate_data.py --d=10 --n=3 --k=6 --n_int=300

# D=20 k=6 N_int=2k+10=50 N_Dataset=5
python3 scripts/generate_data.py --d=20 --n=3 --k=6 --n_int=500
```

```bash
# D=10 k=6 N_int=2k+10=30 N_Dataset=5
python3 scripts/generate_data.py --d=10 --n=1 --k=5 --n_int=5000

# D=20 k=6 N_int=2k+10=50 N_Dataset=5
python3 scripts/generate_data.py --d=20 --n=1 --k=5 --n_int=5000
```

# How to Use RBF BO Experiments

```bash
# D=3 k=3 N_int=10 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=3_N=3_K=3/rbf_bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=3_N=3_K=3 \
--acqu=LogEI \
--kernel=RBF \
--niter=3 \
--use_gpu=1

# D=5 k=3 N_int=2k+10=20 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=5_N=3_K=3/bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=5_N=3_K=3 \
--acqu=LogEI \
--kernel=RBF \
--niter=50 \
--use_gpu=1

# D=10 k=3 N_int=2k+10=20 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=10_N=3_K=6/bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=10_N=3_K=6 \
--acqu=LogEI \
--kernel=RBF \
--niter=100 \
--use_gpu=1

# D=20 k=6 N_int=2k+10=50 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=20_N=3_K=6/bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=20_N=3_K=6 \
--acqu=LogEI \
--kernel=RBF \
--niter=200 \
--use_gpu=1
```

# How to Use Marten BO Experiments

```bash
# D=3 k=3 N_int=10 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=3_N=3_K=3/matern_bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=3_N=3_K=3 \
--acqu=LogEI \
--kernel=Matern \
--niter=30 \
--use_gpu=1

# D=5 k=3 N_int=2k+10=20 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=5_N=3_K=3/matern_bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=5_N=3_K=3 \
--acqu=LogEI \
--kernel=Matern \
--niter=50 \
--use_gpu=1

# D=10 k=3 N_int=2k+10=20 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=10_N=3_K=6/matern_bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=10_N=3_K=6 \
--acqu=LogEI \
--kernel=Matern \
--niter=100 \
--use_gpu=1

# D=20 k=6 N_int=2k+10=50 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=20_N=3_K=6/matern_bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=20_N=3_K=6 \
--acqu=LogEI \
--kernel=Matern \
--niter=200 \
--use_gpu=1
```

# How to Use ARD BO Experiments

```bash
# D=3 k=3 N_int=10 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=3_N=3_K=3/matern_ard_bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=3_N=3_K=3 \
--acqu=LogEI \
--kernel=Matern \
--ard 1 \
--niter=30 \
--use_gpu=1

# D=5 k=3 N_int=2k+10=20 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=5_N=3_K=3/matern_ard_bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=5_N=3_K=3 \
--acqu=LogEI \
--kernel=Matern \
--ard 1 \
--niter=50 \
--use_gpu=1

# D=10 k=3 N_int=2k+10=20 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=10_N=3_K=6/matern_ard_bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=10_N=3_K=6_SUBSET \
--acqu=LogEI \
--kernel=Matern \
--ard 1 \
--niter=100 \
--use_gpu=1

# D=20 k=6 N_int=2k+10=50 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=20_N=3_K=6/matern_ard_bo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=20_N=3_K=6 \
--acqu=LogEI \
--kernel=Matern \
--ard 1 \
--niter=200 \
--use_gpu=1
```

# How to Use SAASBO Experiments
```bash
# D=3 k=3 N_int=10 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=3_N=3_K=3/saasbo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=3_N=3_K=3 \
--acqu=LogEI \
--kernel=Matern \
--niter=30 \
--use_gpu=1 \
--model_type=saasbo


# D=5 k=3 N_int=2k+10=20 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=5_N=3_K=3/saasbo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=5_N=3_K=3 \
--acqu=LogEI \
--kernel=Matern \
--niter=50 \
--use_gpu=0 \
--model_type=saasbo

# D=10 k=3 N_int=2k+10=20 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=10_N=3_K=6/saasbo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=10_N=3_K=6 \
--acqu=LogEI \
--kernel=Matern \
--niter=100 \
--use_gpu=1 \
--model_type=saasbo

# D=20 k=6 N_int=2k+10=50 N_Dataset=5
python3 bo_experiments.py \
--result_dir=./results/20260123/D=20_N=3_K=6/saasbo \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=20_N=3_K=6 \
--acqu=LogEI \
--kernel=Matern \
--niter=200 \
--use_gpu=1 \
--model_type=saasbo
```

# Drow BO Result

## D=3
###  matern_ard_bo
```bash
# D=3_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260123/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_A_000.json \
results/20260123/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_A_001.json \
results/20260123/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_A_002.json \
--model_paths \
results/20260123/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_A_000_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_A_001_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=3_N=3_K=3/matern_ard_bo/plot/type_A_result.png

# D=3_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260123/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_B_000.json \
results/20260123/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_B_001.json \
results/20260123/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_B_002.json \
--model_paths \
results/20260123/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_B_000_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_B_001_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=3_N=3_K=3/matern_ard_bo/plot/type_B_result.png

# D=3_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260123/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_C_000.json \
results/20260123/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_C_001.json \
results/20260123/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_C_002.json \
--model_paths \
results/20260123/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_C_000_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_C_001_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=3_N=3_K=3/matern_ard_bo/plot/type_C_result.png
```

### matern_bo
```bash
# D=3_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260123/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_A_000.json \
results/20260123/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_A_001.json \
results/20260123/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_A_002.json \
--model_paths \
results/20260123/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_A_000_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_A_001_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=3_N=3_K=3/matern_bo/plot/type_A_result.png

# D=3_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260123/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_B_000.json \
results/20260123/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_B_001.json \
results/20260123/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_B_002.json \
--model_paths \
results/20260123/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_B_000_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_B_001_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=3_N=3_K=3/matern_bo/plot/type_B_result.png
# D=3_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260123/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_C_000.json \
results/20260123/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_C_001.json \
results/20260123/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_C_002.json \
--model_paths \
results/20260123/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_C_000_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_C_001_botorch_model.pth \
results/20260123/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=3_N=3_K=3/matern_bo/plot/type_C_result.png
```
<!-- 
# How to use Regret Plot

```
python3 regret_plot.py \
--result_path=/workspaces/BO_EXPERIMENTS/src/results/20260115/BO_GP_EI_SLSQP_ITER30.json \
--output_plot_dir=./results/20260115 \
--is_log=0

python3 regret_plot.py \
--result_path=/workspaces/BO_EXPERIMENTS/src/results/20260115/BO_GP_EI_SLSQP_ITER100_A.json \
--output_plot_dir=./results/20260115 \
--is_log=1

python3 regret_plot.py \
--result_path=/workspaces/BO_EXPERIMENTS/src/results/20260115/BO_GP_EI_SLSQP_ITER50.json \
--output_plot_dir=./results/20260115 \
--is_log=0

python3 regret_plot.py \
--result_path=/workspaces/BO_EXPERIMENTS/src/results/20260115/BO_GP_EI_SLSQP_ITER50.json \
--output_plot_dir=./results/20260115 \
--is_log=1

python3 regret_plot.py \
--result_path=/workspaces/BO_EXPERIMENTS/src/results/20260115_2/BO_GP_EI_SLSQP_ITER100.json \
--output_plot_dir=./results/20260115_2 \
--is_log=1

python3 regret_plot.py \
--result_path=/workspaces/BO_EXPERIMENTS/src/results/20260115_2/BO_GP_UCB_SLSQP_ITER100.json \
--output_plot_dir=./results/20260115_2 \
--is_log=1
``` -->
