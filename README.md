# VLM Token Retransmission Stage 1

This prototype freezes `llava-hf/llava-1.5-7b-hf`, randomly corrupts projected visual tokens, extracts per-token hidden states at a selected LLM layer, and trains an MLP scorer to rank damaged tokens by one-token retransmission value.

The default oracle target is clean-teacher consistency:

```text
gain_i = KL(p_full || p_damaged) - KL(p_full || p_restore_i)
```

This avoids using exact ground-truth answer CE as the ranking label.

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
python scripts/04_eval_retrans_loss.py --config configs/stage1_mlp.yaml --checkpoint outputs/checkpoints/mlp_scorer_teacher.pt --max_samples 50
```

`01_smoke_test_llava.py` runs label-mask, image-replacement, and native-vs-manual
LLaVA forward checks by default. Use `--skip_debug_checks` only when you want the
old minimal smoke test.

For Slurm:

```bash
sbatch --export=ALL,MAX_SAMPLES=200,EVAL_SAMPLES=50 scripts/run_stage1_a100_80g.slurm
```

To rerun only the grouped evaluation with an existing checkpoint:

```bash
sbatch --export=ALL,EVAL_SAMPLES=50 scripts/run_eval_a100_80g.slurm
```

For real data already present in `data/gqa_mini.jsonl`, disable toy data regeneration:

```bash
sbatch --export=ALL,EVAL_SAMPLES=50,CREATE_TOY_DATA=0 scripts/run_eval_a100_80g.slurm
```

## GQA Run

Download the official GQA questions once. Questions are about 1.4 GB. The default
pipeline does not download the full 20 GB image zip; it downloads only the selected
Visual Genome images into `data/gqa_images`.

```bash
python scripts/07_download_gqa_assets.py --root /scratch/prj/nmes_simeone/datasets/gqa --questions
```

Or as a CPU Slurm job:

```bash
sbatch --export=ALL,GQA_ROOT=/scratch/prj/nmes_simeone/datasets/gqa scripts/run_download_gqa_cpu.slurm
```

Then run the GQA stage-1 experiment. The preparation script filters to balanced
questions, excludes yes/no answers by default, downloads only the selected images,
and writes disjoint train/test jsonl files:

```text
data/gqa_train1000.jsonl
data/gqa_test300.jsonl
```

```bash
sbatch --export=ALL,GQA_ROOT=/scratch/prj/nmes_simeone/datasets/gqa,TRAIN_SAMPLES=500,TEST_SAMPLES=300 scripts/run_stage1_gqa_a100_80g.slurm
```

This uses `configs/stage1_mlp_gqa.yaml`, builds the teacher-KL cache under
`outputs/oracle_cache/gqa_teacher_train1000`, trains
`outputs/checkpoints/mlp_scorer_gqa_teacher_train1000.pt`, then evaluates on the
GQA val-balanced held-out test jsonl with no image overlap from the train cache.

## Greedy Set Oracle Check

The normal oracle ranks tokens by single-token gain. For top-K retransmission,
token interactions mean that this is not the true set oracle. To check that case,
run the cumulative greedy oracle on a small test subset:

```bash
sbatch --export=ALL,GREEDY_SAMPLES=30,GREEDY_CANDIDATE_LIMIT=96,GREEDY_BUDGETS=64 scripts/run_greedy_oracle_check_a100_80g.slurm
```

With `GREEDY_CANDIDATE_LIMIT=96`, the greedy oracle is computed over the top 96
single-token candidates to keep cost bounded. Increase this limit for a more exact
but slower check.

## Sweeps

Layer sweep:

```bash
env SWEEP=layer VALUES=8,12,16,20 TRAIN_SAMPLES=200 EVAL_SAMPLES=50 \
  sbatch --export=ALL,SWEEP,VALUES,TRAIN_SAMPLES,EVAL_SAMPLES scripts/run_sweep_gqa_a100_80g.slurm
```

Drop-ratio sweep:

```bash
env SWEEP=drop VALUES=0.25,0.5,0.75 TRAIN_SAMPLES=200 EVAL_SAMPLES=50 \
  sbatch --export=ALL,SWEEP,VALUES,TRAIN_SAMPLES,EVAL_SAMPLES scripts/run_sweep_gqa_a100_80g.slurm
```

Experiment notes and current results are in `docs/experiment_log.md`.

Generated toy images, oracle cache files, checkpoints, logs, model weights, and conda
environments are intentionally ignored by git.
