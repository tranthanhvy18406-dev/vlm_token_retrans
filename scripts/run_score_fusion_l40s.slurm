#!/bin/bash
#SBATCH --job-name=vlm_fusion_l40s
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:30:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --requeue

set -euo pipefail

PROJECT_DIR=/scratch/users/k24116578/vlm_token_retrans
CONDA_ROOT=/users/k24116578/anaconda3
CONDA_ENV=/scratch/users/k24116578/token_env
HF_CACHE=/scratch/prj/nmes_simeone/xty_hf_models

PRECISION_CONFIG=${PRECISION_CONFIG:-configs/s5_fullcand_weak_listwise_005_gqa.yaml}
PRECISION_CHECKPOINT=${PRECISION_CHECKPOINT:-outputs/checkpoints/s5_fullcand_weak_listwise_005_gqa.pt}
PRECISION_LABEL=${PRECISION_LABEL:-s5}
COVERAGE_CONFIG=${COVERAGE_CONFIG:-configs/s7c_attention_robust_gap010_gqa.yaml}
COVERAGE_CHECKPOINT=${COVERAGE_CHECKPOINT:-outputs/checkpoints/s7c_attention_robust_gap010_gqa.pt}
COVERAGE_LABEL=${COVERAGE_LABEL:-s7c}
EVAL_JSONL=${EVAL_JSONL:-data/gqa_val_from_train_500_700.jsonl}
EVAL_SAMPLES=${EVAL_SAMPLES:-200}
ALPHAS=${ALPHAS:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}
COMPOSE_RS=${COMPOSE_RS:-16:0,4,8,12,16;32:0,8,16,24,32;64:0,8,16,24,32,48,64}
METRIC_BATCH_SIZE=${METRIC_BATCH_SIZE:-8}
OUTPUT_JSON=${OUTPUT_JSON:-outputs/fusion/h0_val200_s5_s7c.json}

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

echo "PRECISION_CONFIG=${PRECISION_CONFIG}"
echo "PRECISION_CHECKPOINT=${PRECISION_CHECKPOINT}"
echo "PRECISION_LABEL=${PRECISION_LABEL}"
echo "COVERAGE_CONFIG=${COVERAGE_CONFIG}"
echo "COVERAGE_CHECKPOINT=${COVERAGE_CHECKPOINT}"
echo "COVERAGE_LABEL=${COVERAGE_LABEL}"
echo "EVAL_JSONL=${EVAL_JSONL}"
echo "EVAL_SAMPLES=${EVAL_SAMPLES}"
echo "ALPHAS=${ALPHAS}"
echo "COMPOSE_RS=${COMPOSE_RS}"
echo "METRIC_BATCH_SIZE=${METRIC_BATCH_SIZE}"
echo "OUTPUT_JSON=${OUTPUT_JSON}"

python -u scripts/14_eval_score_fusion.py \
  --precision_config "$PRECISION_CONFIG" \
  --precision_checkpoint "$PRECISION_CHECKPOINT" \
  --precision_label "$PRECISION_LABEL" \
  --coverage_config "$COVERAGE_CONFIG" \
  --coverage_checkpoint "$COVERAGE_CHECKPOINT" \
  --coverage_label "$COVERAGE_LABEL" \
  --jsonl "$EVAL_JSONL" \
  --max_samples "$EVAL_SAMPLES" \
  --alphas "$ALPHAS" \
  --compose_rs "$COMPOSE_RS" \
  --metric_batch_size "$METRIC_BATCH_SIZE" \
  --output_json "$OUTPUT_JSON"
