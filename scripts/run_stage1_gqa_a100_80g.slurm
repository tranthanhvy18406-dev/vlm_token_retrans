#!/bin/bash
#SBATCH --job-name=vlm_gqa_80g
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=/scratch/users/k24116578/vlm_token_retrans
CONDA_ROOT=/users/k24116578/anaconda3
CONDA_ENV=/scratch/users/k24116578/token_env
HF_CACHE=/scratch/prj/nmes_simeone/xty_hf_models
GQA_ROOT=${GQA_ROOT:-/scratch/prj/nmes_simeone/datasets/gqa}
MAX_SAMPLES=${MAX_SAMPLES:-200}
EVAL_SAMPLES=${EVAL_SAMPLES:-50}

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
    p = torch.cuda.get_device_properties(i)
    print(f"device {i}: {p.name}, {p.total_memory / 1024**3:.2f} GiB")
PY

echo "GQA_ROOT=${GQA_ROOT}"
echo "MAX_SAMPLES=${MAX_SAMPLES}"
echo "EVAL_SAMPLES=${EVAL_SAMPLES}"

python -u scripts/08_prepare_gqa_jsonl.py \
  --questions "$GQA_ROOT/questions1.2.zip" \
  --image_dir data/gqa_images \
  --download_missing_images_from_vg \
  --output data/gqa_gqa200.jsonl \
  --split val \
  --max_samples "$MAX_SAMPLES"

python -u scripts/01_smoke_test_llava.py \
  --config configs/stage1_mlp_gqa.yaml

python -u scripts/02_build_oracle_cache.py \
  --config configs/stage1_mlp_gqa.yaml \
  --split train \
  --max_samples "$MAX_SAMPLES"

python -u scripts/03_train_mlp_scorer.py \
  --config configs/stage1_mlp_gqa.yaml

python -u scripts/04_eval_retrans_loss.py \
  --config configs/stage1_mlp_gqa.yaml \
  --checkpoint outputs/checkpoints/mlp_scorer_gqa_teacher.pt \
  --max_samples "$EVAL_SAMPLES"
