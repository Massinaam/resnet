# ResNet18 on CIFAR-10 — Compression Playground (Pruning / Quantization / Factorization)

This repository contains a set of PyTorch scripts to train a **ResNet18** on **CIFAR‑10** and then apply several model-compression techniques:
- **Hyperparameter tuning** with Optuna.
- **Quantization** (FP16 “half” and BinaryConnect-style weight binarization).
- **Pruning** (structured and unstructured) with sparsity analysis and simple scoring.
- **Factorization** (SVD decomposition of Conv layers) with plots comparing accuracy vs parameters/compression.

Most scripts assume a local ResNet implementation available as `from models import resnet` (e.g., `models/resnet.py`). 

---

## What was done (high-level)

- Trained a baseline ResNet18 on CIFAR‑10 with standard normalization and data augmentation (random crop + horizontal flip). 
- Ran (or reconstructed) Optuna studies to pick hyperparameters such as batch size, LR, optimizer, weight decay, scheduler parameters, and epochs. 
- Implemented and tested:
  - **BinaryConnect** training loop (binarize weights for forward/backward, restore FP weights for updates, then clip weights). 
  - **FP16 (half precision)** inference/training patterns (casting inputs to `half` and using `autocast` on CUDA in several pruning/combination scripts). 
  - **Structured pruning** via `prune.ln_structured(..., dim=0)` on conv/linear layers + optional “dense/compact” saving. 
  - **Unstructured pruning** via `prune.l1_unstructured` + sparsity per layer plots. 
  - **SVD factorization** for Conv2d layers (especially 3×3 convs) and comparison across ranks with multiple plots. [file:34]

---

## Repository scripts

### Training + Optuna
- `resnet_train.py`: trains ResNet18 on CIFAR‑10 using *best Optuna hyperparameters* (hard-coded `best_params`), logs train/test curves, saves metrics to pickle, saves checkpoint. 
- `optuna_study.py`: reconstructs an Optuna study from saved trial data, saves Optuna visualization figures (optimization history, hyperparameter importance, parallel coordinates, contour). 

### Quantization
- `half_quantization.py`: FP16-related quantization workflow (half precision). 
- `binaryconnect.py`: defines a `BC` wrapper (save full-precision weights, binarize, restore, clip). 
- `bc_quantization.py`: full BinaryConnect training/eval pipeline + plots + rough model-size/throughput analysis, saves `bc.pth` and metrics pickle. 

### Pruning
- `structured_pruning.py`: structured pruning + short finetune + accuracy eval + MAC counting + sparsity plot + compact/dense saving. 
- `unstructured_pruning.py`: unstructured pruning + short finetune + accuracy eval + MAC counting + sparsity plot + simple score.
- `combination_structured.py`: “structured pruning + FP16” combined flow + MAC counting + score + sparsity plot.
- `combination_unstructured.py`: “unstructured pruning + FP16” combined flow + MAC counting + score + sparsity plot.
- Additional variants: `pruning_quantized.py`, `pruning_binarized.py`. 

### Factorization
- `svd.py`: replaces eligible conv layers with an SVD-based two-layer approximation (rank configurable), trains and compares ranks, saves multiple figures in `plotresults/`. 
- `factorization.py`, `factorization_plot.py`, `fac.py`: utilities to load metrics/checkpoints and generate plots (assumes additional factorization modules/paths exist). 

### Plot helpers / misc
- `plot.py`, `plotgr.py`, `graph.py`, `ratio_unstruc.py`: extra plotting/analysis scripts.
- `NAS.py`, `pl.py`: NAS/prototyping utilities.

---

## Setup

### 1) Clone + environment
```bash
git clone https://github.com/Massinaam/resnet.git
cd resnet

python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\activate # Windows

pip install -U pip
pip install torch torchvision numpy matplotlib optuna
```

These packages are used across the scripts (PyTorch/Torchvision, Numpy, Matplotlib, Optuna).

### 2) Required local module
Many scripts do:
```python
from models import resnet
```
So you need something like:
```
models/
  resnet.py   # defines ResNet18 (and possibly others)
```
Otherwise the scripts will fail to import the model. 

---

## Data (CIFAR-10)

Scripts use `torchvision.datasets.CIFAR10(..., download=True)` with normalization:
- mean: `(0.4914, 0.4822, 0.4465)`
- std:  `(0.2023, 0.1994, 0.2010)` 

`resnet_train.py` uses data augmentation for training:
- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()` 
Paths vary by script, e.g. `resnet_train.py` uses `root_dir = "optimizedl-cifar10"`. 

---

## Run: Baseline training (ResNet18)

### Train with the “best Optuna params” (hard-coded)
`resnet_train.py`:
- Builds dataloaders
- Trains for `n_epochs` from `best_params`
- Evaluates after each epoch
- Saves:
  - `pklfiles/finaltrainingmetrics.pkl`
  - `pthfiles/resnet18cifar10best.pth`

```bash
python resnet_train.py
```

Outputs:
- A `.pkl` file with train/test losses/accuracies and the selected hyperparameters. 

---

## Run: BinaryConnect quantization

`bc_quantization.py` implements the BinaryConnect workflow:
1. Save full precision weights.
2. Replace weights by `sign()` (binarize) for the forward pass.
3. Backprop, then restore full precision weights.
4. Optimizer step in full precision.
5. Clip weights to [-1, 1]. 

It trains and evaluates, then saves:
- `bc.pth`
- `pklfiles/lossesaccuraciesquantization.pkl`
- plots in `plotresults/` including `accuracylossperepoch.png` and `throughputvsmodelsize.png`. 

```bash
python bc_quantization.py
```

---

## Run: Pruning

### Structured pruning
`structured_pruning.py`:
- Loads a checkpoint from `pthfiles/comb.pth`
- Converts the model to FP16 (`model.half()`)
- Applies `ln_structured` pruning (removes filters / neurons) and then finetunes for 5 epochs
- Evaluates with CUDA autocast (half)
- Produces `sparsityparcouche.png`
- Saves a “compact” checkpoint `pthfiles/resnet18cifar10prunedcompact.pth` after removing pruning wrappers.

```bash
python structured_pruning.py
```

### Unstructured pruning
`unstructured_pruning.py`:
- Loads `comb.pth`
- Converts to FP16 and prunes with `l1_unstructured` (when `structured=False`)
- Finetunes for 5 epochs
- Evaluates with CUDA autocast (half)
- Computes:
  - global sparsity from nonzero weights
  - approximate MACs via forward hooks on Conv2d/Linear
  - a simple “score” using sparsity + quantization ratio + params + MACs
- Produces a per-layer sparsity bar plot `sparsiteparcouche.png`.

```bash
python unstructured_pruning.py
```

### “Combination” scripts
- `combination_structured.py`: structured pruning + FP16 + scoring + per-layer sparsity plot.
- `combination_unstructured.py`: unstructured pruning + FP16 + scoring + per-layer sparsity plot. 

```bash
python combination_structured.py
python combination_unstructured.py
```

> Important: these scripts reference specific checkpoint names (e.g., `comb.pth`) and some saving paths start with a leading dot like `.resnet18cifar10prunedandquantized.pth`; adjust paths if needed. 
---

## Run: SVD factorization

`svd.py`:
- Trains a baseline ResNet18, then factorizes eligible 3×3 conv layers using SVD into two Conv2d layers.
- Evaluates multiple ranks (including baseline `rank=None`) and saves figures in `plotresults/`:
  - accuracy vs #params
  - compression vs accuracy
  - train/test loss per epoch
  - train/test accuracy per epoch
  - training time per model
  - memory footprint vs parameters 

```bash
python svd.py
```

---

## Output folders (expected)

Depending on the script, outputs include:
- `pthfiles/` for checkpoints (e.g., `resnet18cifar10best.pth`, `resnet18cifar10prunedcompact.pth`). 
- `pklfiles/` for serialized metrics (pickle). 
- `plotresults/`, `figures/` for plots.

Create them if they do not exist (some scripts already call `os.makedirs(..., exist_ok=True)`). 

---

## Notes / Troubleshooting

- GPU is recommended: several scripts use `autocast(device_type="cuda", dtype=torch.half)` and explicitly cast inputs to half. 
- If you get import errors for `models.resnet`, add the missing `models/` package with your ResNet implementation. 
- If a script fails because a checkpoint path does not exist (e.g. `pthfiles/comb.pth`), either:
  - run the script that generates it (if any), or
  - update the path to point to an existing `.pth` file. 

```
