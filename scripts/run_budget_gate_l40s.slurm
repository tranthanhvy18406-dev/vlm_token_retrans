#!/bin/bash
#SBATCH --job-name=vlm_gate_l40s
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
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

CONFIG=${CONFIG:-configs/g1_frozen_budget_gate_gqa.yaml}
GATE_CHECKPOINT=${GATE_CHECKPOINT:-outputs/checkpoints/g1_frozen_budget_gate_gqa.pt}
TRAIN=${TRAIN:-1}
EVAL=${EVAL:-1}
EVAL_JSONL=${EVAL_JSONL:-data/gqa_test300.jsonl}
EVAL_SAMPLES=${EVAL_SAMPLES:-300}
METRIC_BATCH_SIZE=${METRIC_BATCH_SIZE:-8}
ORACLE_CHUNK_SIZE=${ORACLE_CHUNK_SIZE:-16}
BOOTSTRAP=${BOOTSTRAP:-1000}
DELTA_BASELINES=${DELTA_BASELINES:-g1_gate,h1a,s7c,s5}
OUTPUT_JSON=${OUTPUT_JSON:-outputs/budget_gate/g1_official_paired_test300.json}

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

echo "CONFIG=${CONFIG}"
echo "GATE_CHECKPOINT=${GATE_CHECKPOINT}"
echo "TRAIN=${TRAIN}"
echo "EVAL=${EVAL}"
echo "EVAL_JSONL=${EVAL_JSONL}"
echo "EVAL_SAMPLES=${EVAL_SAMPLES}"
echo "OUTPUT_JSON=${OUTPUT_JSON}"

if [[ "$TRAIN" == "1" ]]; then
  python -u scripts/16_train_budget_gate.py --config "$CONFIG"
fi

if [[ "$EVAL" == "1" ]]; then
  python -u scripts/17_eval_budget_gate.py \
    --gate_label g1_gate \
    --gate_config "$CONFIG" \
    --gate_checkpoint "$GATE_CHECKPOINT" \
    --scorer s5,configs/s5_fullcand_weak_listwise_005_gqa.yaml,outputs/checkpoints/s5_fullcand_weak_listwise_005_gqa.pt \
    --scorer s7c,configs/s7c_attention_robust_gap010_gqa.yaml,outputs/checkpoints/s7c_attention_robust_gap010_gqa.pt \
    --scorer h1a,configs/h1a_fullcand_query_pairwise_gqa.yaml,outputs/checkpoints/h1a_fullcand_query_pairwise_gqa.pt \
    --jsonl "$EVAL_JSONL" \
    --max_samples "$EVAL_SAMPLES" \
    --metric_batch_size "$METRIC_BATCH_SIZE" \
    --oracle_chunk_size "$ORACLE_CHUNK_SIZE" \
    --bootstrap "$BOOTSTRAP" \
    --delta_baselines "$DELTA_BASELINES" \
    --output_json "$OUTPUT_JSON"
fi
