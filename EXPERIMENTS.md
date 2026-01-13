# How to use

```bash
How to Use
python3 bo_experiments.py \
--plot_path=[BO_GP_EI_SLSQP_ITER400]_D20_regret_plot.png \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=20_N=5 \
--niter=400

python3 bo_experiments.py \
--plot_path=[BO_GP_EI_PGA_ITER100]_D5_regret_plot.png \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=5_N=5 \
--const_method=PGA \
--niter=100

python3 bo_experiments.py \
--plot_path=[BO_GP_EI_PGA_ITER400]_D20_regret_plot.png \
--data_dir=/workspaces/BO_EXPERIMENTS/src/datasets/D=20_N=5 \
--const_method=PGA \
--niter=400
```