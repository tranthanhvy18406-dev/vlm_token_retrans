import argparse
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corruption import corrupt_image_features
from src.dataset import load_jsonl
from src.llava_wrapper import LlavaRetransWrapper
from src.metrics import recovery_ratio
from src.scorer import MLPRetransScorer
from src.utils import get_torch_dtype, load_yaml, make_2d_positions, mean_or_nan, set_seed


METHOD_NAMES = ["full", "no", "random", "hidden_norm", "mlp", "oracle"]


def normalize_metric_name(name: str) -> str:
    name = name.lower()
    if name in {"ce", "gt_ce", "ground_truth_ce"}:
        return "gt_ce"
    if name in {"teacher_kl", "clean_teacher_kl", "consistency_kl"}:
        return "teacher_kl"
    raise ValueError(f"Unsupported eval/oracle metric: {name}")


def restore_by_indices(corrupted, full, indices):
    restored = corrupted.clone()
    restored[:, indices, :] = full[:, indices, :]
    return restored


def choose_random(candidates, k):
    if candidates.numel() <= k:
        return candidates
    perm = torch.randperm(candidates.numel(), device=candidates.device)
    return candidates[perm[:k]]


def choose_hidden_norm(hidden, candidates, k):
    scores = hidden[candidates].float().norm(dim=-1)
    k = min(k, candidates.numel())
    return candidates[torch.topk(scores, k=k).indices]


def choose_mlp(scorer, hidden, pos, reliability, damaged_float, candidates, k):
    h = hidden[candidates].float()
    aux = torch.cat(
        [
            pos[candidates].float(),
            reliability[candidates].float(),
            damaged_float[candidates].float(),
        ],
        dim=-1,
    )
    scores = scorer(h, aux)
    k = min(k, candidates.numel())
    return candidates[torch.topk(scores, k=k).indices]


def compute_metric_from_image_features(
    wrapper,
    prepared,
    image_features,
    metric_name,
    teacher_log_probs=None,
):
    if metric_name == "gt_ce":
        return wrapper.compute_loss_from_image_features(prepared, image_features)
    return wrapper.compute_teacher_kl_from_image_features(
        prepared=prepared,
        teacher_log_probs=teacher_log_probs,
        image_features=image_features,
    )


def compute_metric_batch_from_embeds(
    wrapper,
    prepared,
    embeds,
    count,
    metric_name,
    teacher_log_probs=None,
):
    attention_mask = prepared.attention_mask.repeat(count, 1)
    labels = prepared.labels.repeat(count, 1)
    if metric_name == "gt_ce":
        return wrapper.compute_loss_batch_from_embeds(
            embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
    return wrapper.compute_teacher_kl_batch_from_embeds(
        embeds=embeds,
        attention_mask=attention_mask,
        labels=labels,
        teacher_log_probs=teacher_log_probs,
    )


def rank_oracle(
    wrapper,
    prepared,
    corrupted,
    full,
    candidates,
    metric_name,
    teacher_log_probs=None,
    chunk_size=16,
):
    base_metric = compute_metric_from_image_features(
        wrapper=wrapper,
        prepared=prepared,
        image_features=corrupted,
        metric_name=metric_name,
        teacher_log_probs=teacher_log_probs,
    )

    gains = []
    all_cands = []

    for start in range(0, candidates.numel(), chunk_size):
        cand = candidates[start : start + chunk_size]
        count = cand.numel()
        if count == 0:
            continue

        embeds = wrapper.build_candidate_restored_embeds(
            prepared=prepared,
            corrupted_features=corrupted,
            full_features=full,
            candidate_indices=cand,
        )

        candidate_metrics = compute_metric_batch_from_embeds(
            wrapper=wrapper,
            prepared=prepared,
            embeds=embeds,
            count=count,
            metric_name=metric_name,
            teacher_log_probs=teacher_log_probs,
        )

        gains.append(base_metric - candidate_metrics)
        all_cands.append(cand)

    gains = torch.cat(gains, dim=0)
    all_cands = torch.cat(all_cands, dim=0)
    order = torch.argsort(gains, descending=True)
    return all_cands[order]


def make_stats(budgets):
    return {k: {name: [] for name in METHOD_NAMES} for k in budgets}


def append_losses(stats, k, loss_full, loss_no, losses):
    stats[k]["full"].append(float(loss_full.item()))
    stats[k]["no"].append(float(loss_no.item()))
    for name, value in losses.items():
        stats[k][name].append(float(value.item()))


def print_stats_block(title, stats, budgets, metric_name):
    sample_count = len(stats[budgets[0]]["full"]) if budgets else 0
    metric_label = "teacher KL" if metric_name == "teacher_kl" else "GT CE loss"
    print(f"\n===== {title} (n={sample_count}), lower {metric_label} is better =====")
    if sample_count == 0:
        print("No samples.")
        return

    for k in budgets:
        print(f"\nK={k}")
        full_mean = mean_or_nan(stats[k]["full"])
        no_mean = mean_or_nan(stats[k]["no"])
        for name, values in stats[k].items():
            if values:
                mean_val = mean_or_nan(values)
                print(f"{name:12s}: {mean_val:.6f}")
        print("Recovery ratios:")
        for name in ["random", "hidden_norm", "mlp", "oracle"]:
            method_mean = mean_or_nan(stats[k][name])
            recovery = recovery_ratio(no_mean, method_mean, full_mean)
            print(f"{name:12s}: {100.0 * recovery:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["corruption"]["seed"])

    device = cfg["model"]["device"]

    wrapper = LlavaRetransWrapper(
        model_name=cfg["model"]["name"],
        device=device,
        dtype=get_torch_dtype(cfg["model"]["dtype"]),
        cache_dir=cfg["model"].get("cache_dir"),
        local_files_only=bool(cfg["model"].get("local_files_only", False)),
        device_map=cfg["model"].get("device_map"),
    )

    scorer = MLPRetransScorer(
        hidden_dim=cfg["scorer"]["hidden_dim"],
        aux_dim=cfg["scorer"]["aux_dim"],
        hidden1=cfg["scorer"]["hidden1"],
        hidden2=cfg["scorer"]["hidden2"],
        dropout=0.0,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    scorer.load_state_dict(ckpt["model"])
    scorer.eval()

    samples = load_jsonl(
        cfg["data"]["val_jsonl"],
        image_root=cfg["data"]["image_root"],
        max_samples=args.max_samples,
    )

    budgets = cfg["eval"]["budgets"]
    eval_metric = normalize_metric_name(
        cfg.get("eval", {}).get("metric", cfg["oracle"].get("target", "gt_ce"))
    )
    layer_idx = cfg["model"]["layer_idx"]
    print(f"eval metric: {eval_metric}")

    stats_all = make_stats(budgets)
    stats_vs = make_stats(budgets)
    stats_other = make_stats(budgets)

    for sample_idx, sample in enumerate(tqdm(samples, desc="eval retrans loss")):
        try:
            prepared = wrapper.prepare_inputs(sample.image, sample.question, sample.answer)
            full = prepared.image_features

            corrupted, damaged_mask, reliability = corrupt_image_features(
                full,
                drop_ratio=cfg["corruption"]["drop_ratio"],
                mode=cfg["corruption"]["mode"],
                mask_value=cfg["corruption"]["mask_value"],
            )

            candidates = damaged_mask.nonzero(as_tuple=False).squeeze(-1)
            if candidates.numel() == 0:
                continue

            hidden = wrapper.get_layer_visual_hidden(
                prepared=prepared,
                image_features=corrupted,
                layer_idx=layer_idx,
            )

            num_tokens = full.shape[1]
            pos = make_2d_positions(num_tokens, device=device)
            damaged_float = damaged_mask.float().unsqueeze(-1)

            teacher_log_probs = None
            if eval_metric == "teacher_kl":
                teacher_log_probs = wrapper.compute_teacher_log_probs_from_image_features(
                    prepared,
                    full,
                )

            loss_full = compute_metric_from_image_features(
                wrapper=wrapper,
                prepared=prepared,
                image_features=full,
                metric_name=eval_metric,
                teacher_log_probs=teacher_log_probs,
            )
            loss_no = compute_metric_from_image_features(
                wrapper=wrapper,
                prepared=prepared,
                image_features=corrupted,
                metric_name=eval_metric,
                teacher_log_probs=teacher_log_probs,
            )
            is_vision_sensitive = float(loss_full.item()) < float(loss_no.item())
            print(
                f"sample={sample_idx} "
                f"metric={eval_metric}, "
                f"full={loss_full.item():.6f}, "
                f"no={loss_no.item():.6f}, "
                f"vision_sensitive={is_vision_sensitive}"
            )
            oracle_ranked = rank_oracle(
                wrapper=wrapper,
                prepared=prepared,
                corrupted=corrupted,
                full=full,
                candidates=candidates,
                metric_name=eval_metric,
                teacher_log_probs=teacher_log_probs,
                chunk_size=cfg["oracle"]["candidate_chunk_size"],
            )

            for k in budgets:
                idx_random = choose_random(candidates, k)
                idx_hidden = choose_hidden_norm(hidden, candidates, k)
                idx_mlp = choose_mlp(
                    scorer=scorer,
                    hidden=hidden,
                    pos=pos,
                    reliability=reliability,
                    damaged_float=damaged_float,
                    candidates=candidates,
                    k=k,
                )
                idx_oracle = oracle_ranked[: min(k, oracle_ranked.numel())]

                losses = {
                    "random": compute_metric_from_image_features(
                        wrapper,
                        prepared,
                        restore_by_indices(corrupted, full, idx_random),
                        eval_metric,
                        teacher_log_probs,
                    ),
                    "hidden_norm": compute_metric_from_image_features(
                        wrapper,
                        prepared,
                        restore_by_indices(corrupted, full, idx_hidden),
                        eval_metric,
                        teacher_log_probs,
                    ),
                    "mlp": compute_metric_from_image_features(
                        wrapper,
                        prepared,
                        restore_by_indices(corrupted, full, idx_mlp),
                        eval_metric,
                        teacher_log_probs,
                    ),
                    "oracle": compute_metric_from_image_features(
                        wrapper,
                        prepared,
                        restore_by_indices(corrupted, full, idx_oracle),
                        eval_metric,
                        teacher_log_probs,
                    ),
                }

                append_losses(stats_all, k, loss_full, loss_no, losses)
                if is_vision_sensitive:
                    append_losses(stats_vs, k, loss_full, loss_no, losses)
                else:
                    append_losses(stats_other, k, loss_full, loss_no, losses)

        except Exception as exc:
            print(f"[WARN] eval sample failed: {exc}")
            continue

    print_stats_block("All samples", stats_all, budgets, eval_metric)
    print_stats_block("Vision-sensitive samples: full_metric < no_metric", stats_vs, budgets, eval_metric)
    print_stats_block("Other samples: full_metric >= no_metric", stats_other, budgets, eval_metric)


if __name__ == "__main__":
    main()
