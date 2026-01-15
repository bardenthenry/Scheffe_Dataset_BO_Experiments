# How to Use BO Experiments

```bash
python3 bo_experiments.py \
--result_path=[BO_GP_EI_SLSQP_ITER200]_D20_regret_plot.png \
--data_dir=./datasets/D=20_N=5 \
--niter=200

python3 bo_experiments.py \
--result_path=./results/20260115/BO_GP_EI_SLSQP_ITER50.json \
--data_dir=./datasets/D=5_N=5 \
--const_method=SLSQP \
--niter=50

python3 bo_experiments.py \
--result_path=./results/BO_GP_EI_PGA_ITER50.json \
--data_dir=./datasets/D=5_N=5 \
--const_method=PGA \
--niter=50

python3 bo_experiments.py \
--result_path=./results/20260115/BO_GP_EI_SLSQP_ITER30.json \
--data_dir=./datasets/D=3_N=5 \
--const_method=SLSQP \
--niter=30

python3 bo_experiments.py \
--result_path=./results/BO_GP_EI_PGA_ITER30.json \
--data_dir=./datasets/D=3_N=5 \
--const_method=PGA \
--niter=30


python3 bo_experiments.py \
--result_path=[BO_GP_EI_PGA_ITER400]_D20_regret_plot.png \
--data_dir=./datasets/D=20_N=5 \
--const_method=PGA \
--niter=400
```

# How to use Regret Plot

```
python3 regret_plot.py \
--result_path=/workspaces/BO_EXPERIMENTS/src/results/20260115/BO_GP_EI_SLSQP_ITER30.json \
--output_plot_dir=./results/20260115 \
--is_log=0

python3 regret_plot.py \
--result_path=/workspaces/BO_EXPERIMENTS/src/results/20260115/BO_GP_EI_SLSQP_ITER30.json \
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
```
