# VLM Token Retransmission Experiment Log

Last updated: 2026-05-07

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

## Scorer Ablation Plan Added on 2026-05-04

Implementation changes:

```text
Exp-1: old MLP + fixed listwise KL reduction
Exp-2: Exp-1 + hidden norm/mean/std aux features
Exp-3: Exp-2 checkpoint + MMR lambda sweep at eval only
Exp-4: Exp-2 + query-conditioned scorer using last prompt-token hidden
Exp-5: Exp-4 checkpoint + MMR lambda sweep at eval only
```

Configs and runner:

```text
configs/exp1_fixed_listwise_gqa.yaml
configs/exp2_norm_aux_gqa.yaml
configs/exp4_query_gqa.yaml
scripts/run_ablation_gqa_a100_80g.slurm
```

The query cache is stored separately under:

```text
outputs/oracle_cache/gqa_teacher_train1000_query
```

## Scorer Ablation Results on 2026-05-05

All runs below evaluate `data/gqa_test300.jsonl` with 300 samples and teacher-KL
recovery. L40S and A40 replicated the same trend; L40S numbers are listed first.

Previous baseline from the main POC:

```text
Old MLP: K=16 23.37%, K=32 41.15%, K=64 65.49%
```

L40S ablation:

```text
Exp-1 fixed listwise:
K=16 12.11%, K=32 27.64%, K=64 54.44%

Exp-2 fixed listwise + norm aux:
K=16 13.22%, K=32 30.28%, K=64 56.77%

Exp-3 Exp-2 + best MMR:
K=16 13.46% at lambda=0.2
K=32 30.64% at lambda=0.2
K=64 57.04% at lambda=0.15

Exp-4 query-conditioned:
K=16 19.00%, K=32 35.92%, K=64 61.51%

Exp-5 query-conditioned + best MMR:
K=16 19.43% at lambda=0.15
K=32 36.19% at lambda=0.05
K=64 61.70% at lambda=0.2
```

A40 replication:

```text
Exp-1 fixed listwise:
K=16 12.10%, K=32 27.71%, K=64 54.55%

Exp-2 fixed listwise + norm aux:
K=16 12.90%, K=32 29.55%, K=64 56.61%

Exp-3 Exp-2 + best MMR:
K=16 13.37% at lambda=0.2
K=32 30.25% at lambda=0.2
K=64 57.03% at lambda=0.2

Exp-4 query-conditioned:
K=16 19.81%, K=32 35.91%, K=64 62.15%

Exp-5 query-conditioned + best MMR:
K=16 20.10% at lambda=0.1
K=32 35.98% at lambda=0.15
K=64 62.09% at lambda=0.05
```

Interpretation:

```text
1. Fixed listwise KL alone hurt badly versus Old MLP.
2. Hidden norm aux recovers a small amount but remains well below Old MLP.
3. MMR gives small positive deltas on top of Exp-2/Exp-4, but not enough to beat Old MLP.
4. Query-conditioned scorer is the strongest new component, especially over Exp-2,
   but this implementation still does not beat the Old MLP baseline.
5. Best current ablation is query-conditioned with or without MMR depending on K,
   but Old MLP remains better at K=16/32/64.
```

## Listwise Scale Follow-up on 2026-05-06

Motivation:

```text
The fixed listwise KL used z-scored gains, tau=0.1, reduction=sum, and
list_loss_weight=0.2. On 100 cached samples this made the target very peaked:
average target top-1 probability was 0.813 at tau=0.1. With tau=1.0 it dropped
to 0.213. Normalizing KL by candidate count also changes the auxiliary loss
from a large sample-level objective into a small regularizer.
```

Configs:

```text
S0: old checkpoint + current eval
S1: legacy aux + pairwise only
S2a: legacy aux + pairwise + weak listwise, tau=1.0, weight=0.005, KL/M
S2b: legacy aux + pairwise + weak listwise, tau=1.0, weight=0.01, KL/M
S3: legacy aux + weighted_pairwise + topK CE, no listwise
S4: query-conditioned + norm_stats aux + weighted_pairwise + topK CE, no listwise
S4b: query-conditioned + norm_stats aux + pairwise only
S4c: query-conditioned + norm_stats aux + weak listwise, tau=1.0, weight=0.005, KL/M
```

All runs below evaluate `data/gqa_test300.jsonl` with 300 samples and teacher-KL
recovery on L40S. `All samples` and `Vision-sensitive samples` are identical for
this teacher-KL run: `full_metric=0` and all 300 samples have `full_metric < no_metric`.

Same-run baselines:

| K | random | hidden_norm | oracle_single |
| - | -: | -: | -: |
| 16 | 20.64% | 21.15% | 46.89% |
| 32 | 38.71% | 37.98% | 58.45% |
| 64 | 63.75% | 62.04% | 71.28% |

MLP recovery:

| Experiment | K=16 | K=32 | K=64 |
| - | -: | -: | -: |
| S0 old checkpoint, current eval | 23.51% | 41.09% | 65.68% |
| S1 pairwise only | 23.62% | 41.04% | 65.25% |
| S2a weak listwise 0.005 | 23.99% | 41.51% | 65.64% |
| S2b weak listwise 0.01 | 23.68% | 41.08% | 65.55% |
| S3 weighted pairwise + topK CE | 21.03% | 35.23% | 59.89% |
| S4 query + weighted pairwise + topK CE | 24.79% | 38.96% | 63.62% |
| S4b query + pairwise only | 25.14% | 40.89% | 65.14% |
| S4c query + weak listwise 0.005 | 25.22% | 40.78% | 64.89% |
| S5 full-candidate cache + weak listwise 0.005 | 25.17% | 42.81% | 65.85% |
| S6 full-candidate + attention aux + weak listwise 0.005 | 23.85% | 41.16% | 69.01% |
| S7a S6 + robust pairwise gap 0.02 | 24.77% | 41.99% | 68.61% |
| S7b S6 + robust pairwise gap 0.05 | 24.13% | 41.65% | 68.87% |
| S7c S6 + robust pairwise gap 0.10 | 24.04% | 41.52% | 69.11% |

Training-side recall@32/ndcg@32 after epoch 4:

```text
S1: 0.3328 / 0.7185
S2a: 0.3350 / 0.7196
S2b: 0.3284 / 0.7167
S3: 0.3675 / 0.7237
S4: 0.3950 / 0.7667
S4b: 0.3575 / 0.7398
S4c: 0.3591 / 0.7405
S5: 0.1669 / 0.6629
S6: 0.2097 / 0.6795
S7a: 0.1969 / 0.6759
S7b: 0.2044 / 0.6783
S7c: 0.2075 / 0.6783
```

Interpretation:

```text
1. Pairwise-only recovers the old baseline. This confirms that the previous
   fixed listwise setup was not a harmless implementation fix; it changed the
   optimization target too strongly.
2. Weak listwise with tau=1.0, KL/M, and weight=0.005 is the best legacy-MLP
   result in this round, but the gain is small: +0.48 at K=16 and +0.42 at K=32
   versus S0, with K=64 essentially tied.
3. Weighted pairwise + topK CE improves cache ranking metrics but hurts final
   teacher-KL retransmission recovery. Do not use S3 as the default objective.
4. Query-conditioned scorer is real for small budgets: S4b/S4c reach the best
   K=16 recovery in this round, about +1.2 to +1.7 points over S0/S2a.
   However, it does not beat legacy weak-listwise at K=32 or K=64.
5. Weak listwise does not help the query scorer in this setting: S4c is almost
   tied with S4b at K=16 and slightly worse at K=32/64.
6. For the next default, use S2a if the target is balanced K=16/32/64 recovery.
   Use S4b only if the paper or system emphasizes very small retransmission
   budgets such as K=16.
7. Full-candidate oracle labels are the strongest improvement so far. S5
   matches the query scorer at K=16, clearly beats S2a/S4b/S4c at K=32, and
   gives the best K=64 in this round. The low S5 cache recall@32 is expected
   because the candidate set is much larger than the 128-candidate cache; final
   retransmission recovery is the relevant metric.
8. Attention aux changes the budget tradeoff. It hurts K=16 and K=32 versus
   S5, but substantially improves K=64. S7c reaches the best K=64 recovery
   in this round, close to the same-run oracle_single. This suggests attention
   features are useful for broad coverage at larger retransmission budgets, but
   need a budget-conditioned or hybrid head before they are safe as the default.
9. Robust pairwise gives only small changes on top of attention aux. gap=0.02
   is the best of the robust variants for K=16/K=32, while gap=0.10 is best
   for K=64. None of the robust variants dominates S5 across all budgets.
```

## H0/H1 Follow-up

H0 tests whether the S5 precision scorer and S7c attention/coverage scorer can
be fused without retraining. The sweep uses held-out rows 501-700 from
`data/gqa_train1000.jsonl` as a 200-sample validation split, then applies the
validation-selected settings to `data/gqa_test300.jsonl`.

```text
H0 validation, S5 + S7c, fast no-oracle eval:
K=16 best: fusion_a0p9 = 31.67%
K=32 best: S7c / fusion_a0 = 48.47%
K=64 best: S7c / fusion_a0 = 67.79%

H0 test300, selected alphas:
S5         : 23.64 / 44.83 / 64.70
S7c        : 21.63 / 44.25 / 67.51
fusion_a0p9: 23.82 / 44.79 / 65.27
```

Interpretation: simple score fusion and rank composition do not unlock much
extra complementarity. The only positive result is a small K=16 gain from
`alpha=0.9`; K=32 stays with S5 and K=64 stays with the attention/coverage
scorer. This argues for budget-aware training/gating rather than a fixed
post-hoc blend.

H1 retrains the query-conditioned scorer on the S5 full-candidate cache. The
S5 cache already contains `query_hidden` and full damaged-candidate oracle gains,
so no new oracle cache build is needed.

```text
H1 configs:
configs/h1a_fullcand_query_pairwise_gqa.yaml
configs/h1b_fullcand_query_weak_listwise_005_gqa.yaml

H1 test300, fast no-oracle eval:
H1a query + pairwise only     : 25.59 / 45.97 / 67.59
H1b query + weak listwise 0.005: 25.12 / 46.06 / 67.56
```

H1a was also compared directly against S7c in the same fast-eval run:

```text
H1a query : 25.59 / 45.97 / 67.63
S7c attn  : 21.63 / 44.25 / 67.51
fusion 0.5: 23.05 / 45.33 / 67.98
```

Interpretation:

```text
1. Full-candidate supervision makes query-conditioned scoring substantially
   stronger. It is now the best small-budget scorer in these runs.
2. Weak listwise still does not help query K=16; H1a pairwise-only is cleaner.
   H1b has a tiny K=32 edge but not enough to justify using listwise by default.
3. H1a also matches or slightly beats S7c at K=64 in the fast-eval protocol.
   The only useful H1a/S7c fusion is K=64 alpha=0.5, +0.35 over H1a; K=16/K=32
   should stay pure H1a.
4. These H0/H1 numbers use the fast no-oracle evaluator, so compare them within
   the same run/protocol. The official `04_eval_retrans_loss.py` table includes
   random and oracle_single, which changes the RNG sequence for corruption masks.
```

## P0 Official Paired H1 Evaluation on 2026-05-07

P0 reruns S5, S7c, H1a, and H1b in one official paired evaluator so every method
sees the same corruption mask per sample. It also reports random, hidden_norm,
and oracle_single in the same loop. The paired script uses the same mean-loss
recovery aggregation as `04_eval_retrans_loss.py`, and adds 1000-sample paired
bootstrap CIs. S7c requires attention features, so this run uses
`attn_implementation=eager` for all methods; compare methods within this run.

```text
script: scripts/15_eval_paired_scorers.py
runner: scripts/run_paired_eval_l40s.slurm
job: 33764351, L40S interruptible, completed in 01:51:53
output: outputs/paired_eval/p0_h1_official_paired_test300_interruptible.json
eval: data/gqa_test300.jsonl, 300 samples, teacher-KL, official aggregate recovery
```

All samples are vision-sensitive under teacher-KL in this run, so the All and
Vision-sensitive tables are identical.

| Method | K=16 | K=32 | K=64 |
| - | -: | -: | -: |
| random | 20.54% | 38.73% | 63.71% |
| hidden_norm | 21.00% | 37.85% | 61.70% |
| oracle_single | 47.19% | 58.29% | 70.98% |
| S5 full-candidate legacy | 25.03% | 42.74% | 65.57% |
| S7c attention coverage | 24.01% | 41.47% | 69.15% |
| H1a full-candidate query pairwise | 26.07% | 41.28% | 65.46% |
| H1b full-candidate query weak listwise | 26.12% | 41.44% | 65.24% |

Key paired deltas:

```text
K=16:
H1a - S5 = +1.04, CI95 [-0.10, +2.23]
H1b - S5 = +1.09, CI95 [-0.05, +2.27]
S7c - H1a = -2.06, CI95 [-3.65, -0.54]

K=32:
S5 - H1a = +1.46, CI95 [-0.33, +3.20]
S7c - S5 = -1.27, CI95 [-3.46, +1.00]
H1b - H1a = +0.15, CI95 [-0.07, +0.40]

K=64:
S7c - H1a = +3.69, CI95 [+1.21, +6.38]
S7c - S5 = +3.58, CI95 [+1.56, +5.62]
H1a - S5 = -0.11, CI95 [-2.14, +2.03]
```

Interpretation:

```text
1. Query-conditioned scoring is the best small-budget receiver-side scorer.
   H1a/H1b are best at K=16, but the H1b weak-listwise edge over H1a is only
   +0.05 and not meaningful. Use H1a as the clean pairwise-only query baseline.
2. The official paired run does not support saying H1a dominates all budgets.
   At K=32, S5 is numerically best, although its +1.46 over H1a still has a CI
   crossing zero.
3. S7c is the clear K=64 coverage scorer. Its +3.6 point gain over both H1a
   and S5 has a positive paired CI.
4. The next main direction should be budget-aware mixture/gating of query
   precision and attention coverage, not another single global ranking head.
```

## H2 FiLM Query Follow-up

H2a tests a FiLM-style query modulation scorer on the same full-candidate cache
and pairwise-only objective as H1a.

```text
config: configs/h2a_film_query_pairwise_gqa.yaml
checkpoint: outputs/checkpoints/h2a_film_query_pairwise_gqa.pt
training job: 33764279

epoch=0 loss=0.689101 recall@32=0.1244 ndcg@32=0.6514
epoch=1 loss=0.685186 recall@32=0.1481 ndcg@32=0.6626
epoch=2 loss=0.683019 recall@32=0.1462 ndcg@32=0.6646
epoch=3 loss=0.680737 recall@32=0.1572 ndcg@32=0.6704
epoch=4 loss=0.677548 recall@32=0.1963 ndcg@32=0.6850
```

Fast no-oracle eval against H1a:

```text
H1a query pairwise: 25.59 / 45.97 / 67.59
H2a FiLM query   : 21.11 / 38.54 / 61.91
fusion alpha=0.5 : 24.00 / 43.80 / 66.03
```

Interpretation:

```text
H2a is a negative result. This FiLM scorer is a full replacement trained from
scratch, and it does not preserve H1a behavior. Do not run it in official eval.
If FiLM is revisited, use it as a residual delta on top of an initialized H1a
scorer rather than as a randomly initialized replacement.
```

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

## Formal GQA Train/Test Split Plan

The current formal run uses non-overlapping train/test split files:

```text
train: data/gqa_train1000.jsonl
test:  data/gqa_test300.jsonl
```

The split script:

```text
scripts/09_prepare_gqa_splits.py
```

Filters balanced GQA questions, excludes yes/no answers by default, keeps short answers, enforces disjoint image ids across train/test, and downloads only selected Visual Genome images into:

```text
data/gqa_images/
```

## Lightweight GQA 500/300 Train/Test Run

Command:

```bash
sbatch --export=ALL,GQA_ROOT=/scratch/prj/nmes_simeone/datasets/gqa,TRAIN_SAMPLES=500,TEST_SAMPLES=300 scripts/run_stage1_gqa_a100_80g.slurm
```

Outputs:

```text
cache:      outputs/oracle_cache/gqa_teacher_train1000
checkpoint: outputs/checkpoints/mlp_scorer_gqa_teacher_train1000.pt
```

Status:

```text
train/test-only rerun submitted
previous Slurm job: 33619174, PREEMPTED after 439 / 1000 cache files
completed intermediate train/val/test Slurm job: 33638747, COMPLETED in 01:04:16
current train/test-only Slurm job: 33640992
cache built: 500 / 500
train/test-only Slurm job 33640992 was PREEMPTED during model loading
eval-only test300 Slurm job: 33650649
eval-only test300 Slurm job 33650649 completed in 01:19:16
```

Sweep jobs:

```text
layer sweep, values {8, 12, 16, 20}, train=200, test=50: job 33640993
drop sweep, values {0.25, 0.5, 0.75}, train=200, test=50: job 33651626
greedy set oracle deferred until the main POC result is stable
```

Main POC validation result, `data/gqa_val200.jsonl`, first 100 samples:

```text
K=16 recovery: random 20.08%, hidden_norm 19.08%, mlp 25.11%, oracle_single 45.17%
K=32 recovery: random 39.69%, hidden_norm 41.62%, mlp 44.65%, oracle_single 55.60%
K=64 recovery: random 63.81%, hidden_norm 64.35%, mlp 65.41%, oracle_single 72.47%
```

Main POC test result, `data/gqa_test300.jsonl`, first 100 samples:

```text
K=16 recovery: random 23.82%, hidden_norm 24.99%, mlp 28.63%, oracle_single 52.73%
K=32 recovery: random 48.37%, hidden_norm 44.51%, mlp 48.58%, oracle_single 61.14%
K=64 recovery: random 64.68%, hidden_norm 62.33%, mlp 64.50%, oracle_single 71.83%
```

Main POC test result, `data/gqa_test300.jsonl`, all 300 samples:

```text
K=16 recovery: random 20.44%, hidden_norm 21.44%, mlp 23.37%, oracle_single 47.16%
K=32 recovery: random 38.60%, hidden_norm 37.89%, mlp 41.15%, oracle_single 58.42%
K=64 recovery: random 63.66%, hidden_norm 61.66%, mlp 65.49%, oracle_single 71.03%
```

Interpretation:

```text
The POC passes the main trend on validation: MLP beats random and hidden_norm at all K, but it is still well below single-token oracle.
On all 300 held-out test samples, MLP beats random and hidden_norm at K=16/32/64, but remains below the single-token oracle.
```

Layer sweep result, `train=200`, `test=50`:

```text
layer=8:  K16 mlp 22.66%, K32 mlp 41.58%, K64 mlp 61.67%
layer=12: K16 mlp 22.18%, K32 mlp 43.59%, K64 mlp 65.09%
layer=16: K16 mlp 21.80%, K32 mlp 40.96%, K64 mlp 65.75%
layer=20: K16 mlp 21.93%, K32 mlp 42.87%, K64 mlp 63.63%
```

Preliminary completed drop point:

```text
drop=0.25, train=300, val=100
K=16: mlp recovery 18.82%, oracle_single 48.23%
K=32: mlp recovery 26.35%, oracle_single 53.63%
K=64: mlp recovery 41.96%, oracle_single 57.69%
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
