#!/bin/bash
#SBATCH --job-name=vlm_sweep_80g
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=/scratch/users/k24116578/vlm_token_retrans
CONDA_ROOT=/users/k24116578/anaconda3
CONDA_ENV=/scratch/users/k24116578/token_env
HF_CACHE=/scratch/prj/nmes_simeone/xty_hf_models
SWEEP=${SWEEP:-layer}
VALUES=${VALUES:-4,8,12,16,20,24}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-300}
VAL_SAMPLES=${VAL_SAMPLES:-100}

cd "$PROJECT_DIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"device {i}: {props.name}, {props.total_memory / 1024**3:.2f} GiB")
PY

echo "SWEEP=${SWEEP}"
echo "VALUES=${VALUES}"
echo "TRAIN_SAMPLES=${TRAIN_SAMPLES}"
echo "VAL_SAMPLES=${VAL_SAMPLES}"

python -u scripts/10_run_sweep.py \
  --base_config configs/stage1_mlp_gqa.yaml \
  --sweep "$SWEEP" \
  --values "$VALUES" \
  --train_samples "$TRAIN_SAMPLES" \
  --val_samples "$VAL_SAMPLES" \
  --train_jsonl data/gqa_train1000.jsonl \
  --val_jsonl data/gqa_val200.jsonl
