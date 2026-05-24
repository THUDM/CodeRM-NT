# GRPO with TRL

GRPO training of Qwen2.5-Coder-7B on KodCode-V1 based on TRL with 8 GPUs (7 for training, 1 for rollout). Supports two reward sources:

| Starting script                        | Trainer                            | Reward source |
|----------------------------------------|------------------------------------|---------------|
| `run_train_grpo_kodcode_rm.sh`         | `train_grpo_kodcode_rm.py`         | CodeRM-NT     |
| `run_train_grpo_kodcode_unit_test.sh`  | `train_grpo_kodcode_unit_test.py`  | Unit tests    |

## Quick Start

Before starting, replace the `--wandb_key <wandb_key>` placeholder within the starting script with your real W&B API key.

### Reward 1: CodeRM-NT

We publish the CodeRM-NT checkpoint at [`Rishubi/CodeRM-NT`](https://huggingface.co/Rishubi/CodeRM-NT), and `run_train_grpo_kodcode_rm.sh` points `REWARD_MODEL_PATH` at it by default. Override the variable if you trained your own.

```bash
bash run_train_grpo_kodcode_rm.sh
```

### Reward 2: Unit Tests

Run:

```bash
bash run_train_grpo_kodcode_unit_test.sh
```
