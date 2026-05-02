#!/bin/bash
#SBATCH --job-name=vlm_test_eval
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100_80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

PROJECT_DIR=/scratch/users/k24116578/vlm_token_retrans
CONDA_ROOT=/users/k24116578/anaconda3
CONDA_ENV=/scratch/users/k24116578/token_env
HF_CACHE=/scratch/prj/nmes_simeone/xty_hf_models
EVAL_JSONL=${EVAL_JSONL:-data/gqa_test300.jsonl}
CHECKPOINT=${CHECKPOINT:-outputs/checkpoints/mlp_scorer_gqa_teacher_train1000.pt}
EVAL_SAMPLES=${EVAL_SAMPLES:-300}

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

python -u scripts/04_eval_retrans_loss.py \
  --config configs/stage1_mlp_gqa.yaml \
  --checkpoint "$CHECKPOINT" \
  --jsonl "$EVAL_JSONL" \
  --max_samples "$EVAL_SAMPLES"
