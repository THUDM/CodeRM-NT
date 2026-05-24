# Reward Model Training

Two-stage CodeRM-NT training pipeline:

1. **Stage 1 – Scalar / MSE** (`train_rm.py`): minimizes $(R_\phi(x, y) − r)^2$ on $D_{scalar} = \{(x, y, r)\}$.
2. **Stage 2 – Pairwise / Contrastive** (`train_rm_contrastive.py`): minimizes $−\log\sigma(R_\phi(x, y_w) − R_\phi(x, y_l))$ on $D_{pair} = {(x, y_w, y_l)}$, initialized from the Stage-1 checkpoint.

## Quick Start

Unzip the reward data:

```bash
cd rm_train && unzip data/data_train_rm.zip -d data/
```

To train with 8 GPUs, launch the consecutive two-stage run:

```bash
bash train_rm_consecutive.sh
```

The checkpoints are saved in `models/<EXPNAME>-{scalar,contrast}/`.
