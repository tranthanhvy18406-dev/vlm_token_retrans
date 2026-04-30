#!/bin/bash
#SBATCH --job-name=vlm_gqa_80g
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=/scratch/users/k24116578/vlm_token_retrans
CONDA_ROOT=/users/k24116578/anaconda3
CONDA_ENV=/scratch/users/k24116578/token_env
HF_CACHE=/scratch/prj/nmes_simeone/xty_hf_models
GQA_ROOT=${GQA_ROOT:-/scratch/prj/nmes_simeone/datasets/gqa}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-1000}
VAL_SAMPLES=${VAL_SAMPLES:-200}
TEST_SAMPLES=${TEST_SAMPLES:-300}
PREPARE_GQA_SPLITS=${PREPARE_GQA_SPLITS:-auto}

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
echo "TRAIN_SAMPLES=${TRAIN_SAMPLES}"
echo "VAL_SAMPLES=${VAL_SAMPLES}"
echo "TEST_SAMPLES=${TEST_SAMPLES}"
echo "PREPARE_GQA_SPLITS=${PREPARE_GQA_SPLITS}"

if [ "$PREPARE_GQA_SPLITS" = "1" ] || {
  [ "$PREPARE_GQA_SPLITS" = "auto" ] && {
    [ ! -s data/gqa_train1000.jsonl ] || \
    [ ! -s data/gqa_val200.jsonl ] || \
    [ ! -s data/gqa_test300.jsonl ]
  }
}; then
  python -u scripts/09_prepare_gqa_splits.py \
    --questions "$GQA_ROOT/questions1.2.zip" \
    --image_dir data/gqa_images \
    --train_output data/gqa_train1000.jsonl \
    --val_output data/gqa_val200.jsonl \
    --test_output data/gqa_test300.jsonl \
    --train_count "$TRAIN_SAMPLES" \
    --val_count "$VAL_SAMPLES" \
    --test_count "$TEST_SAMPLES"
else
  echo "Using existing GQA split jsonl files."
fi

python -u scripts/01_smoke_test_llava.py \
  --config configs/stage1_mlp_gqa.yaml

python -u scripts/02_build_oracle_cache.py \
  --config configs/stage1_mlp_gqa.yaml \
  --split train \
  --max_samples "$TRAIN_SAMPLES"

python -u scripts/03_train_mlp_scorer.py \
  --config configs/stage1_mlp_gqa.yaml

python -u scripts/04_eval_retrans_loss.py \
  --config configs/stage1_mlp_gqa.yaml \
  --checkpoint outputs/checkpoints/mlp_scorer_gqa_teacher_train1000.pt \
  --jsonl data/gqa_val200.jsonl \
  --max_samples "$VAL_SAMPLES"

python -u scripts/04_eval_retrans_loss.py \
  --config configs/stage1_mlp_gqa.yaml \
  --checkpoint outputs/checkpoints/mlp_scorer_gqa_teacher_train1000.pt \
  --jsonl data/gqa_test300.jsonl \
  --max_samples "$TEST_SAMPLES"
