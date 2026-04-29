#!/bin/bash
#SBATCH --job-name=vlm_token_stage1
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
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
MAX_SAMPLES=${MAX_SAMPLES:-20}

cd "$PROJECT_DIR"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export HF_HOME="$HF_CACHE"
export HF_HUB_CACHE="$HF_CACHE/hub"
export TRANSFORMERS_CACHE="$HF_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER=0
export TOKENIZERS_PARALLELISM=false

python -u - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

nvidia-smi || true

python -u scripts/00_create_toy_data.py --output-dir data --num-samples "$MAX_SAMPLES"
python -u scripts/05_download_model.py --config configs/stage1_mlp.yaml
python -u scripts/01_smoke_test_llava.py --config configs/stage1_mlp.yaml
python -u scripts/02_build_oracle_cache.py --config configs/stage1_mlp.yaml --split train --max_samples "$MAX_SAMPLES"
python -u scripts/03_train_mlp_scorer.py --config configs/stage1_mlp.yaml
python -u scripts/04_eval_retrans_loss.py --config configs/stage1_mlp.yaml --checkpoint outputs/checkpoints/mlp_scorer.pt --max_samples "$MAX_SAMPLES"
