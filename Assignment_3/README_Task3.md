# Task 3 — GCN from Scratch on PPI (NumPy)

This repository contains the **Task 3** solution for Assignment 3: a **Graph Convolutional Network (GCN)** implemented **from scratch using NumPy** for **multi-label node classification** on the **PPI dataset**.

## Assignment requirements covered

Task 3 requires:
- a NumPy-only GCN with manual forward pass, backward pass, and weight updates
- Binary Cross-Entropy for the multi-label problem
- Micro-F1 evaluation with threshold `0.5`
- `train.py` containing training and testing/evaluation logic
- `model.py` containing the model architecture
- saved model weights in `.npz` or `.pkl` format
- a report with architecture, training details, curves, results, and ablations

## Final selected model

The final selected model is the **best validation checkpoint** from the final training run.

### Best validation result
- **Best validation Micro-F1:** `0.5503`
- **Best epoch:** `111`
- **Final training loss:** `0.2249`
- **Final validation loss:** `0.1807`
- **Final training Micro-F1:** `0.5464`
- **Final validation Micro-F1:** `0.5317`

These values are taken from the final saved artifacts:
- `run_config.json`
- `summary.json`
- `training_history.json`

## Final best configuration

```text
learning rate        = 0.0015
final learning rate  = 0.0005
lr decay factor      = 0.5
lr decay patience    = 12
min lr               = 0.0005
hidden_dim           = 256
num_layers           = 4
dropout              = 0.1
input_dropout        = 0.0
weight_decay         = 1e-4
normalization        = symmetric
activation           = relu
seed                 = 42
patience             = 45
feature standardization = true
use residual         = true
use pos weight       = true
max_pos_weight       = 8.0
pos_weight_power     = 0.5
grad_clip_value      = 5.0
batch size           = 1 graph/component per optimization step
```

## Dataset structure

Place the PPI files inside `./data/ppi/`:

```text
Tarun/
├── data/
│   └── ppi/
│       ├── ppi-G.json
│       ├── ppi-feats.npy
│       ├── ppi-class_map.json
│       └── ppi-id_map.json
├── train.py
├── model.py
├── report.pdf
└── outputs/
```

## Dependencies

Recommended: **Python 3.10+**

Install dependencies:

### Mac / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy matplotlib
```

## Final training command

Use this command to reproduce the final best model run:

```bash
python3 train.py \
  --mode train \
  --dataset_path ./data/ppi \
  --model_save_path ./gcn_ppi_model.npz \
  --artifacts_dir ./outputs/final_submission_run \
  --epochs 180 \
  --patience 45 \
  --lr 0.0015 \
  --lr_decay_factor 0.5 \
  --lr_decay_patience 12 \
  --min_lr 0.0005 \
  --hidden_dim 256 \
  --num_layers 4 \
  --dropout 0.1 \
  --input_dropout 0.0 \
  --weight_decay 1e-4 \
  --normalization symmetric \
  --max_pos_weight 8 \
  --pos_weight_power 0.5 \
  --seed 42
```

## Final evaluation command

```bash
python3 train.py \
  --mode eval \
  --dataset_path ./data/ppi \
  --model_path ./gcn_ppi_model.npz \
  --artifacts_dir ./outputs/final_submission_eval \
  --eval_split all
```

## Generated output files

### Training run outputs

```text
outputs/final_submission_run/
├── logs/
│   ├── run_config.json
│   ├── summary.json
│   └── training_history.json
└── plots/
    ├── loss_curve.png
    └── micro_f1_curve.png
```

### Evaluation outputs

```text
outputs/final_submission_eval/
└── logs/
    └── summary.json
```

### Model file

```text
gcn_ppi_model.npz
```

## Interpretation of the final plots

### Loss curve
- Training loss drops quickly at the start and then stabilizes.
- Validation loss also drops quickly and stays consistently low.
- Validation loss being lower than training loss is acceptable here because training uses stronger regularization/training-time noise while evaluation is deterministic.
- No severe overfitting pattern is visible in the final run.

### Micro-F1 curve
- Micro-F1 increases rapidly in the first phase of training.
- It then stabilizes around the `0.50–0.55` range.
- The **best validation Micro-F1** occurs at **epoch 111**, so saving the best checkpoint is important.
- Later epochs fluctuate slightly, which is normal for this task.

## Notes about the dataset split

The provided dataset contains **training** and **validation** graphs. The official **test graphs are hidden** and are not included in the released files. If evaluation on `all` shows `num_test_graphs = 0`, that is expected for the provided dataset.

## Final submission files

Inside `ENTRYNUMBER.zip`, include:
- `train.py`
- `model.py`
- `gcn_ppi_model.npz`
- `report.pdf`

## Suggested final workflow

1. Keep the exact `train.py` and `model.py` that produced the `0.5503` run.
2. Copy or regenerate the final best checkpoint as `gcn_ppi_model.npz`.
3. Run the final evaluation command.
4. Verify that `run_config.json`, `summary.json`, `training_history.json`, and both plots are present.
5. Zip the required files:

```bash
zip -r ENTRYNUMBER.zip train.py model.py gcn_ppi_model.npz report.pdf
```

## Final recommendation

The strongest final model for submission is:
- **4-layer GCN**
- **hidden dimension 256**
- **best validation Micro-F1 = 0.5503**
- checkpoint: **`gcn_ppi_model.npz`**
