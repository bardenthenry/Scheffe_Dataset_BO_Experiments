## D=3
###  matern_ard_bo
```bash
# D=3_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_A_000.json \
results/20260122/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_A_001.json \
results/20260122/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_A_002.json \
--model_paths \
results/20260122/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_A_000_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_A_001_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=3_N=3_K=3/matern_ard_bo/plot/type_A_result.png

# D=3_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_B_000.json \
results/20260122/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_B_001.json \
results/20260122/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_B_002.json \
--model_paths \
results/20260122/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_B_000_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_B_001_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=3_N=3_K=3/matern_ard_bo/plot/type_B_result.png

# D=3_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_C_000.json \
results/20260122/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_C_001.json \
results/20260122/D=3_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D3_C_002.json \
--model_paths \
results/20260122/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_C_000_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_C_001_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_ard_bo/models/oracle_data_D3_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=3_N=3_K=3/matern_ard_bo/plot/type_C_result.png
```

### matern_bo
```bash
# D=3_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_A_000.json \
results/20260122/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_A_001.json \
results/20260122/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_A_002.json \
--model_paths \
results/20260122/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_A_000_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_A_001_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=3_N=3_K=3/matern_bo/plot/type_A_result.png

# D=3_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_B_000.json \
results/20260122/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_B_001.json \
results/20260122/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_B_002.json \
--model_paths \
results/20260122/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_B_000_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_B_001_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=3_N=3_K=3/matern_bo/plot/type_B_result.png
# D=3_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_C_000.json \
results/20260122/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_C_001.json \
results/20260122/D=3_N=3_K=3/matern_bo/jsonfiles/oracle_data_D3_C_002.json \
--model_paths \
results/20260122/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_C_000_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_C_001_botorch_model.pth \
results/20260122/D=3_N=3_K=3/matern_bo/models/oracle_data_D3_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=3_N=3_K=3/matern_bo/plot/type_C_result.png
```

## D=5
###  matern_ard_bo
```bash
# D=5_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=5_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D5_A_000.json \
results/20260122/D=5_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D5_A_001.json \
results/20260122/D=5_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D5_A_002.json \
--model_paths \
results/20260122/D=5_N=3_K=3/matern_ard_bo/models/oracle_data_D5_A_000_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_ard_bo/models/oracle_data_D5_A_001_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_ard_bo/models/oracle_data_D5_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=5_N=3_K=3/matern_ard_bo/plot/type_A_result.png

# D=5_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=5_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D5_B_000.json \
results/20260122/D=5_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D5_B_001.json \
results/20260122/D=5_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D5_B_002.json \
--model_paths \
results/20260122/D=5_N=3_K=3/matern_ard_bo/models/oracle_data_D5_B_000_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_ard_bo/models/oracle_data_D5_B_001_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_ard_bo/models/oracle_data_D5_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=5_N=3_K=3/matern_ard_bo/plot/type_B_result.png

# D=5_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=5_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D5_C_000.json \
results/20260122/D=5_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D5_C_001.json \
results/20260122/D=5_N=3_K=3/matern_ard_bo/jsonfiles/oracle_data_D5_C_002.json \
--model_paths \
results/20260122/D=5_N=3_K=3/matern_ard_bo/models/oracle_data_D5_C_000_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_ard_bo/models/oracle_data_D5_C_001_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_ard_bo/models/oracle_data_D5_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=5_N=3_K=3/matern_ard_bo/plot/type_C_result.png
```

### matern_bo
```bash
# D=5_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=5_N=3_K=3/matern_bo/jsonfiles/oracle_data_D5_A_000.json \
results/20260122/D=5_N=3_K=3/matern_bo/jsonfiles/oracle_data_D5_A_001.json \
results/20260122/D=5_N=3_K=3/matern_bo/jsonfiles/oracle_data_D5_A_002.json \
--model_paths \
results/20260122/D=5_N=3_K=3/matern_bo/models/oracle_data_D5_A_000_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_bo/models/oracle_data_D5_A_001_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_bo/models/oracle_data_D5_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=5_N=3_K=3/matern_bo/plot/type_A_result.png

# D=5_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=5_N=3_K=3/matern_bo/jsonfiles/oracle_data_D5_B_000.json \
results/20260122/D=5_N=3_K=3/matern_bo/jsonfiles/oracle_data_D5_B_001.json \
results/20260122/D=5_N=3_K=3/matern_bo/jsonfiles/oracle_data_D5_B_002.json \
--model_paths \
results/20260122/D=5_N=3_K=3/matern_bo/models/oracle_data_D5_B_000_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_bo/models/oracle_data_D5_B_001_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_bo/models/oracle_data_D5_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=5_N=3_K=3/matern_bo/plot/type_B_result.png
# D=5_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=5_N=3_K=3/matern_bo/jsonfiles/oracle_data_D5_C_000.json \
results/20260122/D=5_N=3_K=3/matern_bo/jsonfiles/oracle_data_D5_C_001.json \
results/20260122/D=5_N=3_K=3/matern_bo/jsonfiles/oracle_data_D5_C_002.json \
--model_paths \
results/20260122/D=5_N=3_K=3/matern_bo/models/oracle_data_D5_C_000_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_bo/models/oracle_data_D5_C_001_botorch_model.pth \
results/20260122/D=5_N=3_K=3/matern_bo/models/oracle_data_D5_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=5_N=3_K=3/matern_bo/plot/type_C_result.png
```

## D=10
###  matern_ard_bo
```bash
# D=10_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=10_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D10_A_000.json \
results/20260122/D=10_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D10_A_001.json \
results/20260122/D=10_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D10_A_002.json \
--model_paths \
results/20260122/D=10_N=3_K=6/matern_ard_bo/models/oracle_data_D10_A_000_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_ard_bo/models/oracle_data_D10_A_001_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_ard_bo/models/oracle_data_D10_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=10_N=3_K=6/matern_ard_bo/plot/type_A_result.png

# D=10_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=10_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D10_B_000.json \
results/20260122/D=10_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D10_B_001.json \
results/20260122/D=10_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D10_B_002.json \
--model_paths \
results/20260122/D=10_N=3_K=6/matern_ard_bo/models/oracle_data_D10_B_000_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_ard_bo/models/oracle_data_D10_B_001_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_ard_bo/models/oracle_data_D10_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=10_N=3_K=6/matern_ard_bo/plot/type_B_result.png

# D=10_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=10_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D10_C_000.json \
results/20260122/D=10_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D10_C_001.json \
results/20260122/D=10_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D10_C_002.json \
--model_paths \
results/20260122/D=10_N=3_K=6/matern_ard_bo/models/oracle_data_D10_C_000_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_ard_bo/models/oracle_data_D10_C_001_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_ard_bo/models/oracle_data_D10_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=10_N=3_K=6/matern_ard_bo/plot/type_C_result.png
```

### matern_bo
```bash
# D=10_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=10_N=3_K=6/matern_bo/jsonfiles/oracle_data_D10_A_000.json \
results/20260122/D=10_N=3_K=6/matern_bo/jsonfiles/oracle_data_D10_A_001.json \
results/20260122/D=10_N=3_K=6/matern_bo/jsonfiles/oracle_data_D10_A_002.json \
--model_paths \
results/20260122/D=10_N=3_K=6/matern_bo/models/oracle_data_D10_A_000_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_bo/models/oracle_data_D10_A_001_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_bo/models/oracle_data_D10_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=10_N=3_K=6/matern_bo/plot/type_A_result.png

# D=10_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=10_N=3_K=6/matern_bo/jsonfiles/oracle_data_D10_B_000.json \
results/20260122/D=10_N=3_K=6/matern_bo/jsonfiles/oracle_data_D10_B_001.json \
results/20260122/D=10_N=3_K=6/matern_bo/jsonfiles/oracle_data_D10_B_002.json \
--model_paths \
results/20260122/D=10_N=3_K=6/matern_bo/models/oracle_data_D10_B_000_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_bo/models/oracle_data_D10_B_001_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_bo/models/oracle_data_D10_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=10_N=3_K=6/matern_bo/plot/type_B_result.png
# D=10_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=10_N=3_K=6/matern_bo/jsonfiles/oracle_data_D10_C_000.json \
results/20260122/D=10_N=3_K=6/matern_bo/jsonfiles/oracle_data_D10_C_001.json \
results/20260122/D=10_N=3_K=6/matern_bo/jsonfiles/oracle_data_D10_C_002.json \
--model_paths \
results/20260122/D=10_N=3_K=6/matern_bo/models/oracle_data_D10_C_000_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_bo/models/oracle_data_D10_C_001_botorch_model.pth \
results/20260122/D=10_N=3_K=6/matern_bo/models/oracle_data_D10_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=10_N=3_K=6/matern_bo/plot/type_C_result.png
```

### SAASBO
```bash
# D=10_V=A
python3 bo_result_plot.py \
--json_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/jsonfiles/oracle_data_D10_A_000.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/jsonfiles/oracle_data_D10_A_001.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/jsonfiles/oracle_data_D10_A_002.json \
--model_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/models/oracle_data_D10_A_000_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/models/oracle_data_D10_A_001_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/models/oracle_data_D10_A_002_botorch_model.pth \
--model_type=saasbo \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/plot/type_A_result.png

# D=10_V=B
python3 bo_result_plot.py \
--json_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/jsonfiles/oracle_data_D10_B_000.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/jsonfiles/oracle_data_D10_B_001.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/jsonfiles/oracle_data_D10_B_002.json \
--model_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/models/oracle_data_D10_B_000_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/models/oracle_data_D10_B_001_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/models/oracle_data_D10_B_002_botorch_model.pth \
--model_type=saasbo \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/plot/type_B_result.png

# D=10_V=C
python3 bo_result_plot.py \
--json_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/jsonfiles/oracle_data_D10_C_000.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/jsonfiles/oracle_data_D10_C_001.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/jsonfiles/oracle_data_D10_C_002.json \
--model_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/models/oracle_data_D10_C_000_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/models/oracle_data_D10_C_001_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/models/oracle_data_D10_C_002_botorch_model.pth \
--model_type=saasbo \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=10_N=3_K=6/saasbo/plot/type_C_result.png
```

## D=20
###  matern_ard_bo
```bash
# D=20_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=20_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D20_A_000.json \
results/20260122/D=20_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D20_A_001.json \
results/20260122/D=20_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D20_A_002.json \
--model_paths \
results/20260122/D=20_N=3_K=6/matern_ard_bo/models/oracle_data_D20_A_000_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_ard_bo/models/oracle_data_D20_A_001_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_ard_bo/models/oracle_data_D20_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=20_N=3_K=6/matern_ard_bo/plot/type_A_result.png

# D=20_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=20_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D20_B_000.json \
results/20260122/D=20_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D20_B_001.json \
results/20260122/D=20_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D20_B_002.json \
--model_paths \
results/20260122/D=20_N=3_K=6/matern_ard_bo/models/oracle_data_D20_B_000_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_ard_bo/models/oracle_data_D20_B_001_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_ard_bo/models/oracle_data_D20_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=20_N=3_K=6/matern_ard_bo/plot/type_B_result.png

# D=20_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=20_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D20_C_000.json \
results/20260122/D=20_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D20_C_001.json \
results/20260122/D=20_N=3_K=6/matern_ard_bo/jsonfiles/oracle_data_D20_C_002.json \
--model_paths \
results/20260122/D=20_N=3_K=6/matern_ard_bo/models/oracle_data_D20_C_000_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_ard_bo/models/oracle_data_D20_C_001_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_ard_bo/models/oracle_data_D20_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=20_N=3_K=6/matern_ard_bo/plot/type_C_result.png
```

### matern_b
```bash
# D=20_V=A
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=20_N=3_K=6/matern_bo/jsonfiles/oracle_data_D20_A_000.json \
results/20260122/D=20_N=3_K=6/matern_bo/jsonfiles/oracle_data_D20_A_001.json \
results/20260122/D=20_N=3_K=6/matern_bo/jsonfiles/oracle_data_D20_A_002.json \
--model_paths \
results/20260122/D=20_N=3_K=6/matern_bo/models/oracle_data_D20_A_000_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_bo/models/oracle_data_D20_A_001_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_bo/models/oracle_data_D20_A_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=20_N=3_K=6/matern_bo/plot/type_A_result.png

# D=20_V=B
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=20_N=3_K=6/matern_bo/jsonfiles/oracle_data_D20_B_000.json \
results/20260122/D=20_N=3_K=6/matern_bo/jsonfiles/oracle_data_D20_B_001.json \
results/20260122/D=20_N=3_K=6/matern_bo/jsonfiles/oracle_data_D20_B_002.json \
--model_paths \
results/20260122/D=20_N=3_K=6/matern_bo/models/oracle_data_D20_B_000_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_bo/models/oracle_data_D20_B_001_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_bo/models/oracle_data_D20_B_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=20_N=3_K=6/matern_bo/plot/type_B_result.png
# D=20_V=C
python3 bo_result_plot.py \
--json_paths \
results/20260122/D=20_N=3_K=6/matern_bo/jsonfiles/oracle_data_D20_C_000.json \
results/20260122/D=20_N=3_K=6/matern_bo/jsonfiles/oracle_data_D20_C_001.json \
results/20260122/D=20_N=3_K=6/matern_bo/jsonfiles/oracle_data_D20_C_002.json \
--model_paths \
results/20260122/D=20_N=3_K=6/matern_bo/models/oracle_data_D20_C_000_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_bo/models/oracle_data_D20_C_001_botorch_model.pth \
results/20260122/D=20_N=3_K=6/matern_bo/models/oracle_data_D20_C_002_botorch_model.pth \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260122/D=20_N=3_K=6/matern_bo/plot/type_C_result.png
```

### SAASBO
```bash
# D=20_V=A
python3 bo_result_plot.py \
--json_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/jsonfiles/oracle_data_D20_A_000.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/jsonfiles/oracle_data_D20_A_001.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/jsonfiles/oracle_data_D20_A_002.json \
--model_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/models/oracle_data_D20_A_000_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/models/oracle_data_D20_A_001_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/models/oracle_data_D20_A_002_botorch_model.pth \
--model_type=saasbo \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/plot/type_A_result.png

# D=20_V=B
python3 bo_result_plot.py \
--json_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/jsonfiles/oracle_data_D20_B_000.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/jsonfiles/oracle_data_D20_B_001.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/jsonfiles/oracle_data_D20_B_002.json \
--model_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/models/oracle_data_D20_B_000_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/models/oracle_data_D20_B_001_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/models/oracle_data_D20_B_002_botorch_model.pth \
--model_type=saasbo \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/plot/type_B_result.png

# D=20_V=C
python3 bo_result_plot.py \
--json_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/jsonfiles/oracle_data_D20_C_000.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/jsonfiles/oracle_data_D20_C_001.json \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/jsonfiles/oracle_data_D20_C_002.json \
--model_paths \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/models/oracle_data_D20_C_000_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/models/oracle_data_D20_C_001_botorch_model.pth \
/workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/models/oracle_data_D20_C_002_botorch_model.pth \
--model_type=saasbo \
--output_plot_path /workspaces/BO_EXPERIMENTS/src/results/20260123/D=20_N=3_K=6/saasbo/plot/type_C_result.png
```