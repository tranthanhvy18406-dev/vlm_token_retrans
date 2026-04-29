#!/bin/bash
#SBATCH --job-name=vlm_token_debug
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -x

PROJECT_DIR=/scratch/users/k24116578/vlm_token_retrans
CONDA_ROOT=/users/k24116578/anaconda3
CONDA_ENV=/scratch/users/k24116578/token_env

echo "hostname=$(hostname)"
echo "pwd before=$(pwd)"
echo "PROJECT_DIR real=$(readlink -f "$PROJECT_DIR" || true)"
echo "CONDA_ENV real=$(readlink -f "$CONDA_ENV" || true)"
ls -ld "$PROJECT_DIR" "$CONDA_ENV" "$CONDA_ENV/bin/python" || true
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
which python
python - <<'PY'
import os
import site
import sys

print("executable:", sys.executable)
print("prefix:", sys.prefix)
print("base_prefix:", sys.base_prefix)
print("path:", sys.path)
print("site:", site.getsitepackages())
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("torch dir exists:", os.path.exists("/scratch/users/k24116578/token_env/lib/python3.12/site-packages/torch"))
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
except Exception as exc:
    print("torch import failed:", repr(exc))
PY
