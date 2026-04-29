# VLM Token Retransmission Stage 1

This prototype freezes `llava-hf/llava-1.5-7b-hf`, randomly corrupts projected visual tokens, extracts per-token hidden states at a selected LLM layer, and trains an MLP scorer to rank damaged tokens by one-token retransmission value.

## Environment

```bash
source /users/k24116578/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/users/k24116578/token_env
cd /scratch/users/k24116578/vlm_token_retrans
```

Model files are configured to load from:

```text
/scratch/prj/nmes_simeone/xty_hf_models/llava-1.5-7b-hf-local
```

## Quick Run

```bash
python scripts/00_create_toy_data.py --output-dir data --num-samples 200
python scripts/01_smoke_test_llava.py --config configs/stage1_mlp.yaml
python scripts/02_build_oracle_cache.py --config configs/stage1_mlp.yaml --split train --max_samples 200
python scripts/03_train_mlp_scorer.py --config configs/stage1_mlp.yaml
python scripts/04_eval_retrans_loss.py --config configs/stage1_mlp.yaml --checkpoint outputs/checkpoints/mlp_scorer.pt --max_samples 50
```

`01_smoke_test_llava.py` runs label-mask, image-replacement, and native-vs-manual
LLaVA forward checks by default. Use `--skip_debug_checks` only when you want the
old minimal smoke test.

For Slurm:

```bash
sbatch --export=ALL,MAX_SAMPLES=200,EVAL_SAMPLES=50 scripts/run_stage1_a100_80g.slurm
```

Generated toy images, oracle cache files, checkpoints, logs, model weights, and conda
environments are intentionally ignored by git.
