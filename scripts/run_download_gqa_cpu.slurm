#!/bin/bash
#SBATCH --job-name=gqa_download
#SBATCH --partition=interruptible_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_DIR=/scratch/users/k24116578/vlm_token_retrans
GQA_ROOT=${GQA_ROOT:-/scratch/prj/nmes_simeone/datasets/gqa}
DOWNLOAD_IMAGES=${DOWNLOAD_IMAGES:-1}

cd "$PROJECT_DIR"

args=(--root "$GQA_ROOT" --questions)
if [ "$DOWNLOAD_IMAGES" = "1" ]; then
  args+=(--images)
fi

python -u scripts/07_download_gqa_assets.py "${args[@]}"
