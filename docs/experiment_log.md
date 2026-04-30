# VLM Token Retransmission Experiment Log

Last updated: 2026-04-30

## Current Setup

- Model: local `llava-1.5-7b-hf-local`
- Model path: `/scratch/prj/nmes_simeone/xty_hf_models/llava-1.5-7b-hf-local`
- Main config: `configs/stage1_mlp_gqa.yaml`
- Current hidden layer: `model.layer_idx = 12`
- Layer meaning: zero-indexed 13th LLaVA language-model decoder layer, after projected visual tokens are inserted
- Corruption: random visual-token erasure
- Default drop ratio: `0.5`
- Scorer input: `h_i^l` plus `[x_i, y_i, r_i, m_i]`
- Scorer target: rank damaged tokens by retransmission value

## Target Definition

The original ground-truth CE oracle was:

```text
gain_i = CE(y_gt, V_damaged) - CE(y_gt, V_restore_i)
```

This was replaced by clean-teacher consistency:

```text
gain_i = KL(p_full || p_damaged) - KL(p_full || p_restore_i)
```

Reason: GQA/toy answer strings do not always match LLaVA's preferred full-image answer distribution. Ground-truth CE can therefore produce cases where the full image has higher teacher-forcing loss than the damaged image.

## Sanity Checks Added

- Prompt prefix equality check between prompt-only tokenization and full prompt-answer prefix.
- First supervised label token printout.
- Image-token replacement max-diff check.
- Native LLaVA forward vs manual `inputs_embeds` forward loss comparison.
- Grouped eval reporting for `vision_sensitive` samples when using GT CE.

## Previous GT-CE GQA 200 Run

Config summary:

```text
dataset: GQA subset, 200 train cache samples
metric: ground-truth CE
layer_idx: 12
drop_ratio: 0.5
```

Key finding:

```text
full_loss < no_loss on only 13 / 50 eval samples
```

Interpretation:

```text
GT answer CE is noisy for this use case. The model may prefer a different answer spelling or distribution under the full image, so GT CE is not a reliable retransmission oracle.
```

Training metrics:

```text
final recall@32 = 0.3728
final ndcg@32   = 0.7257
```

## Previous Teacher-KL GQA 200 Run

Config summary:

```text
dataset: GQA subset, 200 train cache samples
metric: clean-teacher KL
layer_idx: 12
drop_ratio: 0.5
checkpoint: outputs/checkpoints/mlp_scorer_gqa_teacher.pt
cache: outputs/oracle_cache/gqa_teacher_train
```

Training:

```text
epoch=0 loss=0.696696 recall@32=0.2897 ndcg@32=0.6873
epoch=1 loss=0.692486 recall@32=0.2966 ndcg@32=0.6975
epoch=2 loss=0.691401 recall@32=0.3116 ndcg@32=0.7067
epoch=3 loss=0.687926 recall@32=0.3419 ndcg@32=0.7281
epoch=4 loss=0.683660 recall@32=0.3528 ndcg@32=0.7296
```

Eval recovery, 50 samples:

```text
K=16
random      19.01%
hidden_norm 21.39%
mlp         23.95%
oracle      41.45%

K=32
random      37.70%
hidden_norm 37.35%
mlp         44.51%
oracle      53.84%

K=64
random      61.54%
hidden_norm 62.88%
mlp         72.77%
oracle      70.62%
```

Interpretation:

```text
Teacher-KL resolves the full-vs-damaged GT-CE pathology. MLP beats random and hidden-norm, especially at larger K.
```

The `oracle` above is a single-token gain oracle, not a true set oracle. MLP exceeding it at `K=64` is therefore possible because token utilities are not additive.

## Oracle Naming

Current eval now distinguishes:

```text
oracle_single: rank tokens by one-token gain_i
oracle_greedy: cumulative greedy set oracle
```

The greedy oracle selects:

```text
i_t = argmax_i D(V + S_{t-1}) - D(V + S_{t-1} + i)
```

Because this is expensive, the implemented check defaults to a capped pool of top single-token candidates:

```text
GREEDY_CANDIDATE_LIMIT=96
GREEDY_SAMPLES=30
GREEDY_BUDGETS=64
```

## Formal GQA Split Plan

The current formal run uses non-overlapping split files:

```text
train: data/gqa_train1000.jsonl
val:   data/gqa_val200.jsonl
test:  data/gqa_test300.jsonl
```

The split script:

```text
scripts/09_prepare_gqa_splits.py
```

Filters balanced GQA questions, excludes yes/no answers by default, keeps short answers, enforces disjoint image ids across splits, and downloads only selected Visual Genome images into:

```text
data/gqa_images/
```

## Formal GQA 1000/200/300 Run

Command:

```bash
sbatch --export=ALL,GQA_ROOT=/scratch/prj/nmes_simeone/datasets/gqa,TRAIN_SAMPLES=1000,VAL_SAMPLES=200,TEST_SAMPLES=300 scripts/run_stage1_gqa_a100_80g.slurm
```

Outputs:

```text
cache:      outputs/oracle_cache/gqa_teacher_train1000
checkpoint: outputs/checkpoints/mlp_scorer_gqa_teacher_train1000.pt
```

Status:

```text
submitted
Slurm job: 33619174
```

Dependent checks:

```text
greedy set oracle check: job 33619177, afterok:33619174
layer sweep:             job 33619178, afterok:33619177
drop-ratio sweep:        job 33619179, afterok:33619178
```

## C. Layer Sweep

Goal:

```text
Find the best layer for early retransmission decisions.
```

Sweep:

```text
layer_idx in {4, 8, 12, 16, 20, 24}
```

Command:

```bash
sbatch --export=ALL,SWEEP=layer,VALUES=4,8,12,16,20,24,TRAIN_SAMPLES=300,VAL_SAMPLES=100 scripts/run_sweep_gqa_a100_80g.slurm
```

Primary plot:

```text
KL recovery vs layer index
```

Expected story:

```text
Very shallow layers may lack answer-relevant semantics. Very deep layers are later and more expensive for early retransmission. Middle layers may provide the best accuracy/latency tradeoff.
```

## D. Drop Ratio Sweep

Goal:

```text
Check whether selective retransmission becomes more valuable as channel damage increases.
```

Sweep:

```text
drop_ratio in {0.25, 0.5, 0.75}
```

Command:

```bash
sbatch --export=ALL,SWEEP=drop,VALUES=0.25,0.5,0.75,TRAIN_SAMPLES=300,VAL_SAMPLES=100 scripts/run_sweep_gqa_a100_80g.slurm
```

Primary plot:

```text
KL recovery vs drop ratio
```

Expected story:

```text
As drop ratio increases, random retransmission should degrade faster. A learned scorer should preserve more recovery by concentrating budget on high-impact damaged tokens.
```
