# TORM-G: Task-Oriented Radio Map Generator for Coverage Hole Detection

## Overview
This repository provides a PyTorch-based research prototype for task-oriented radio map learning and coverage hole detection in wireless networks.

![Overall Framework](Overall_Framework.png)

## Motivation
Conventional RM-first pipelines reconstruct a full radio map first and then derive coverage holes. This project explores a direct task-oriented alternative: predicting coverage holes from sparse measurements, base-station information, and environmental priors.

## Problem Setting
### Inputs
- Sparse wireless measurements
- Base station information
- City/building mask or environmental prior

### Output
- Coverage hole probability map over a 2D spatial grid

## Method Overview
The current implementation follows this high-level pipeline:
1. Sparse measurement encoding
2. Base-station-aware spatial encoding
3. Environment-aware grid construction
4. Latent grid fusion / cross-attention
5. Direct coverage hole prediction

## Implemented Components
- Task-oriented latent grid model variants (`mvp_crossattn*`)
- Environment-aware grid fusion (`mvp_crossattn_envgrid`, `mvp_crossattn_envgrid_plus`)
- Direct CHD baselines (`unet_baseline`, `fcn_baseline`, `deeplabv3_lite_baseline`)
- RM-first baseline training/evaluation path (`train_rm_unet.py`)
- Evaluation scripts (e.g., measurement sparsity sweep)
- Result aggregation scripts
- Efficiency profiling scripts

## Repository Structure
- `configs/`: Example configuration for running training with relative paths
- `models/`: Model definitions (task-oriented and baseline variants)
- `scripts/`: Training, evaluation, aggregation, and efficiency benchmarking scripts
- `data_utils/`: Dataset loading utilities for cache-based CHD/RM supervision
- `figures/`: Placeholder folder for public figures used in documentation
- `task_registry.py`: Task-id/task-name registry shared by scripts

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Basic Usage
Example commands (adjust paths for your local setup):

```bash
python scripts/train_mvp.py --config configs/example_envgrid_plus.json
```

```bash
python scripts/train_rm_unet.py --config path/to/your_rm_unet_config.json
```

```bash
python scripts/benchmark_efficiency_single.py \
  --config configs/example_envgrid_plus.json \
  --output outputs/efficiency_single.json \
  --csv_output outputs/efficiency_single.csv
```

Note: Some scripts expect specific training outputs/checkpoints to exist. Please check config fields and output paths before running.

## Dataset Note
The original dataset and cached artifacts are not included in this repository. Prepare data and cache files separately before training/evaluation.

## Experimental Status
This repository currently focuses on the model implementation, training pipeline, and evaluation protocol. Detailed benchmark results will be released after further validation.

## Disclaimer
This codebase is an active research prototype. Interfaces, configurations, and experimental protocols may evolve.
