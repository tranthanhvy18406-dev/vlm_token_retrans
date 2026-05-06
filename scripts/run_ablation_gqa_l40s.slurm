#!/bin/bash
#SBATCH --job-name=vlm_ablate_l40s
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

PROJECT_DIR=/scratch/users/k24116578/vlm_token_retrans
CONDA_ROOT=/users/k24116578/anaconda3
CONDA_ENV=/scratch/users/k24116578/token_env
HF_CACHE=/scratch/prj/nmes_simeone/xty_hf_models

CONFIG=${CONFIG:?Set CONFIG to an ablation yaml path}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-500}
EVAL_SAMPLES=${EVAL_SAMPLES:-300}
EVAL_JSONL=${EVAL_JSONL:-data/gqa_test300.jsonl}
BUILD_CACHE=${BUILD_CACHE:-0}
TRAIN=${TRAIN:-1}
EVAL=${EVAL:-1}
MMR_LAMBDAS=${MMR_LAMBDAS:-}

cd "$PROJECT_DIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT=$(python - <<PY
from src.utils import load_yaml
cfg = load_yaml("$CONFIG")
print(cfg["train"]["checkpoint_path"])
PY
)

python -u - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"device {i}: {p.name}, {p.total_memory / 1024**3:.2f} GiB")
PY

echo "CONFIG=${CONFIG}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "TRAIN_SAMPLES=${TRAIN_SAMPLES}"
echo "EVAL_SAMPLES=${EVAL_SAMPLES}"
echo "EVAL_JSONL=${EVAL_JSONL}"
echo "BUILD_CACHE=${BUILD_CACHE}"
echo "TRAIN=${TRAIN}"
echo "EVAL=${EVAL}"
echo "MMR_LAMBDAS=${MMR_LAMBDAS}"

if [ "$BUILD_CACHE" = "1" ]; then
  python -u scripts/02_build_oracle_cache.py \
    --config "$CONFIG" \
    --split train \
    --max_samples "$TRAIN_SAMPLES"
fi

if [ "$TRAIN" = "1" ]; then
  python -u scripts/03_train_mlp_scorer.py \
    --config "$CONFIG"
fi

if [ "$EVAL" = "1" ]; then
  EVAL_ARGS=(
    --config "$CONFIG"
    --checkpoint "$CHECKPOINT"
    --jsonl "$EVAL_JSONL"
    --max_samples "$EVAL_SAMPLES"
  )
  if [ -n "$MMR_LAMBDAS" ]; then
    EVAL_ARGS+=(--mlp_mmr_lambdas "$MMR_LAMBDAS")
  fi
  python -u scripts/04_eval_retrans_loss.py "${EVAL_ARGS[@]}"
fi
