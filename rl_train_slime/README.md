# GRPO with slime

GRPO training of **Qwen3-4B-Thinking** (on [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct)) and **GLM-4-9B-0414** (on [KodCode-V1](https://huggingface.co/datasets/KodCode/KodCode-V1)) with 8 GPUs, using the [slime](https://github.com/THUDM/slime) framework.

## Quick Start

### 1. Prepare the environment

Using the prebuilt slime image is recommended. Mount this `rl_train_slime/` directory into the container so its contents are available at `/root/slime/`:

```bash
docker run --rm --gpus all --shm-size=16g --ulimit stack=67108864 -it \
    -v /path/to/CodeRM-NT/rl_train_slime:/coderm-slime \
    slimerl/slime:latest /bin/bash
cd /root/slime
cp -r /coderm-slime/. .
```

Every step below is performed inside this container, at `/root/slime`.

### 2. Prepare the base checkpoint

Convert the HuggingFace weights of the base model to Megatron format:

```bash
# GLM-4-9B-0414
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint zai-org/GLM-4-9B-0414 \
    --save /root/slime/ckpt/glm4-release/

# Qwen3-4B-Thinking-2507
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint Qwen/Qwen3-4B-Thinking-2507 \
    --save /root/slime/ckpt/qwen3-release/
```

The resulting directories are what the launchers pass via `--hf-checkpoint` (raw HF) and `--ref-load` (converted torch_dist).

### 3. Stage the rollout data

Unzip the preprocessed data files under `data/`:

```bash
unzip data/kodcode.zip -d data/
unzip data/oci.zip -d data/
```

This produces `data/kodcode/{kodcode-train,kodcode-test}.jsonl` and `data/oci/{oci_entry_train,oci_entry_val}.jsonl`.

### 4. Start the reward servers

Run a reward server depending on which reward you want to use for training. 

#### CodeRM-NT reward
We publish the CodeRM-NT checkpoint at [`Rishubi/CodeRM-NT`](https://huggingface.co/Rishubi/CodeRM-NT), and `reward_server_coderm.py` loads it by default via `reward_model_path`. Override the variable if you trained your own.

```bash
CUDA_VISIBLE_DEVICES=7 uvicorn reward_server_coderm:app --host 0.0.0.0 --port 8100
```

#### Unit-test reward

If training with OpenCodeInstruct, run `reward_server_exec.py` with:

```bash
uvicorn reward_server_exec:app --host 0.0.0.0 --port 8200
```

If training with KodCode, no additional server is needed as `code_reward_kodcode.py` runs `pytest` within itself.

If a reward server runs on a different host from the training job, update `reward_server_ip` in `code_reward_kodcode.py` or `code_reward_oci.py` (default is `localhost`).

### 5. Launch training

Run the training launcher based on your base model and reward:

```bash
# Qwen3-4B-Thinking on OCI-5k
bash scripts/run-qwen3.sh rm     # CodeRM-NT reward
bash scripts/run-qwen3.sh test   # Unit-test reward

# GLM-4-9B-0414 on KodCode
bash scripts/run-glm4.sh rm         # CodeRM-NT reward
bash scripts/run-glm4.sh test       # Unit-test reward
```

Inside each launcher, set `--hf-checkpoint` as the path to the original HuggingFace checkpoint and `--ref-load` as the path to the converted Megatron checkpoint in step 2. Also replace the `--wandb-key <wandb_key>` placeholder with your real W&B API key.

Checkpoints are saved in `/root/slime/ckpt/code-rm/$EXPNAME/`.
