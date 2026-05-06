import argparse
import os
import sys

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corruption import corrupt_image_features
from src.dataset import load_jsonl
from src.features import aux_dim_for_mode, build_aux_features
from src.llava_wrapper import LlavaRetransWrapper
from src.metrics import recovery_ratio
from src.scorer import build_scorer_from_config, scorer_requires_query
from src.utils import get_torch_dtype, load_yaml, make_2d_positions, mean_or_nan, set_seed


METHOD_NAMES = [
    "full",
    "no",
    "random",
    "hidden_norm",
    "mlp",
    "oracle_single",
    "oracle_greedy",
]

RETRANSMISSION_METHODS = [
    "random",
    "hidden_norm",
    "mlp",
    "oracle_single",
    "oracle_greedy",
]


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


def build_scorer_aux(
    hidden,
    pos,
    reliability,
    damaged_float,
    candidates,
    scorer_cfg,
    attn_q_to_vis=None,
    attn_text_to_vis_mean=None,
):
    aux_mode = scorer_cfg.get("aux_mode", "legacy")
    pos_fourier_bands = int(scorer_cfg.get("pos_fourier_bands", 0))
    attention_layers = scorer_cfg.get("attention_layers", [])
    aux = build_aux_features(
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        selection=candidates,
        mode=aux_mode,
        pos_fourier_bands=pos_fourier_bands,
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    expected_aux_dim = int(scorer_cfg["aux_dim"])
    actual_aux_dim = aux.size(-1)
    inferred_aux_dim = aux_dim_for_mode(aux_mode, pos_fourier_bands, attention_layers)
    if actual_aux_dim != expected_aux_dim or inferred_aux_dim != expected_aux_dim:
        raise RuntimeError(
            "Aux dimension mismatch: "
            f"config={expected_aux_dim}, actual={actual_aux_dim}, inferred={inferred_aux_dim}. "
            f"mode={aux_mode}, pos_fourier_bands={pos_fourier_bands}"
        )
    return aux


def score_mlp_candidates(
    scorer,
    hidden,
    pos,
    reliability,
    damaged_float,
    candidates,
    scorer_cfg,
    query_hidden=None,
    attn_q_to_vis=None,
    attn_text_to_vis_mean=None,
):
    h = hidden[candidates].float()
    aux = build_scorer_aux(
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        candidates=candidates,
        scorer_cfg=scorer_cfg,
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    with torch.no_grad():
        scores = scorer(h, aux, q=query_hidden)
    return scores


def choose_mlp(
    scorer,
    hidden,
    pos,
    reliability,
    damaged_float,
    candidates,
    k,
    scorer_cfg,
    query_hidden=None,
    attn_q_to_vis=None,
    attn_text_to_vis_mean=None,
):
    scores = score_mlp_candidates(
        scorer=scorer,
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        candidates=candidates,
        scorer_cfg=scorer_cfg,
        query_hidden=query_hidden,
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    k = min(k, candidates.numel())
    return candidates[torch.topk(scores, k=k).indices]


def choose_mlp_mmr(
    scorer,
    hidden,
    pos,
    reliability,
    damaged_float,
    candidates,
    k,
    scorer_cfg,
    query_hidden=None,
    attn_q_to_vis=None,
    attn_text_to_vis_mean=None,
    lam=0.15,
):
    scores = score_mlp_candidates(
        scorer=scorer,
        hidden=hidden,
        pos=pos,
        reliability=reliability,
        damaged_float=damaged_float,
        candidates=candidates,
        scorer_cfg=scorer_cfg,
        query_hidden=query_hidden,
        attn_q_to_vis=attn_q_to_vis,
        attn_text_to_vis_mean=attn_text_to_vis_mean,
    )
    h_norm = F.normalize(hidden[candidates].float(), dim=-1)

    selected = []
    remaining = torch.arange(candidates.numel(), device=candidates.device)

    for _ in range(min(k, candidates.numel())):
        if not selected:
            chosen = remaining[torch.argmax(scores[remaining])]
        else:
            selected_tensor = torch.stack(selected)
            sim = h_norm[remaining] @ h_norm[selected_tensor].T
            redundancy = sim.max(dim=1).values
            mmr_score = scores[remaining] - lam * redundancy
            chosen = remaining[torch.argmax(mmr_score)]

        selected.append(chosen)
        remaining = remaining[remaining != chosen]

        if remaining.numel() == 0:
            break

    if not selected:
        return candidates[:0]
    selected_tensor = torch.stack(selected)
    return candidates[selected_tensor]


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


def choose_greedy_set_oracle(
    wrapper,
    prepared,
    corrupted,
    full,
    candidate_pool,
    budgets,
    metric_name,
    teacher_log_probs=None,
    chunk_size=16,
):
    """
    Cumulative greedy set oracle.

    At step t, recompute marginal gain conditioned on already-restored tokens.
    This is much more expensive than single-token oracle, so callers should
    pass a capped candidate_pool and a small sample count.
    """
    max_budget = min(max(budgets), candidate_pool.numel())
    selected = []
    remaining = candidate_pool.clone()
    current = corrupted.clone()
    selected_by_budget = {}

    for step in range(max_budget):
        base_metric = compute_metric_from_image_features(
            wrapper=wrapper,
            prepared=prepared,
            image_features=current,
            metric_name=metric_name,
            teacher_log_probs=teacher_log_probs,
        )

        gains = []
        all_cands = []
        for start in range(0, remaining.numel(), chunk_size):
            cand = remaining[start : start + chunk_size]
            count = cand.numel()
            if count == 0:
                continue

            embeds = wrapper.build_candidate_restored_embeds(
                prepared=prepared,
                corrupted_features=current,
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
        best_pos = int(torch.argmax(gains).item())
        best_idx = all_cands[best_pos]
        selected.append(best_idx)

        current[:, best_idx, :] = full[:, best_idx, :]
        remaining = remaining[remaining != best_idx]

        budget = step + 1
        if budget in budgets:
            selected_by_budget[budget] = torch.stack(selected)

    return selected_by_budget


def make_stats(budgets, method_names, retransmission_method_names):
    stats = {}
    for k in budgets:
        stats[k] = {name: [] for name in method_names}
        for name in retransmission_method_names:
            stats[k][f"{name}__full"] = []
            stats[k][f"{name}__no"] = []
    return stats


def append_losses(stats, k, loss_full, loss_no, losses):
    stats[k]["full"].append(float(loss_full.item()))
    stats[k]["no"].append(float(loss_no.item()))
    for name, value in losses.items():
        stats[k][name].append(float(value.item()))
        stats[k][f"{name}__full"].append(float(loss_full.item()))
        stats[k][f"{name}__no"].append(float(loss_no.item()))


def print_stats_block(
    title,
    stats,
    budgets,
    metric_name,
    method_names,
    retransmission_method_names,
):
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
        for name in method_names:
            values = stats[k][name]
            if values:
                mean_val = mean_or_nan(values)
                print(f"{name:14s}: {mean_val:.6f} (n={len(values)})")
        print("Recovery ratios:")
        for name in retransmission_method_names:
            if not stats[k][name]:
                continue
            method_mean = mean_or_nan(stats[k][name])
            method_full_mean = mean_or_nan(stats[k][f"{name}__full"])
            method_no_mean = mean_or_nan(stats[k][f"{name}__no"])
            recovery = recovery_ratio(method_no_mean, method_mean, method_full_mean)
            print(f"{name:14s}: {100.0 * recovery:.2f}%")


def parse_int_set(value: str) -> set[int]:
    if not value:
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_float_list(value: str | None) -> list[float]:
    if not value:
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_mmr_method_name(lam: float) -> str:
    value = f"{lam:g}".replace(".", "p").replace("-", "m")
    return f"mlp_mmr_{value}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", default=None)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--enable_greedy_oracle", action="store_true")
    parser.add_argument("--greedy_max_samples", type=int, default=30)
    parser.add_argument("--greedy_candidate_limit", type=int, default=96)
    parser.add_argument("--greedy_budgets", default="64")
    parser.add_argument(
        "--mlp_mmr_lambdas",
        default=None,
        help="Comma-separated lambda values for MLP+MMR reranking, e.g. 0.05,0.1,0.15,0.2.",
    )
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
        attn_implementation=cfg["model"].get("attn_implementation"),
    )

    scorer = build_scorer_from_config(cfg, dropout=0.0).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    scorer.load_state_dict(ckpt["model"])
    scorer.eval()

    eval_jsonl = args.jsonl or cfg["data"].get("test_jsonl") or cfg["data"]["val_jsonl"]
    print(f"eval jsonl: {eval_jsonl}")
    samples = load_jsonl(
        eval_jsonl,
        image_root=cfg["data"]["image_root"],
        max_samples=args.max_samples,
    )

    budgets = cfg["eval"]["budgets"]
    greedy_budgets = parse_int_set(args.greedy_budgets) & set(budgets)
    if args.mlp_mmr_lambdas is None:
        mmr_lambdas = [float(value) for value in cfg.get("eval", {}).get("mlp_mmr_lambdas", [])]
    else:
        mmr_lambdas = parse_float_list(args.mlp_mmr_lambdas)
    mmr_method_names = [format_mmr_method_name(lam) for lam in mmr_lambdas]
    method_names = METHOD_NAMES + mmr_method_names
    retransmission_method_names = RETRANSMISSION_METHODS + mmr_method_names
    eval_metric = normalize_metric_name(
        cfg.get("eval", {}).get("metric", cfg["oracle"].get("target", "gt_ce"))
    )
    layer_idx = cfg["model"]["layer_idx"]
    print(f"eval metric: {eval_metric}")
    if mmr_lambdas:
        print(f"MLP+MMR lambdas: {mmr_lambdas}")
    if args.enable_greedy_oracle:
        print(
            "greedy oracle enabled: "
            f"max_samples={args.greedy_max_samples}, "
            f"candidate_limit={args.greedy_candidate_limit}, "
            f"budgets={sorted(greedy_budgets)}"
        )

    stats_all = make_stats(budgets, method_names, retransmission_method_names)
    stats_vs = make_stats(budgets, method_names, retransmission_method_names)
    stats_other = make_stats(budgets, method_names, retransmission_method_names)

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

            query_hidden = None
            if scorer_requires_query(cfg["scorer"]):
                hidden, query_hidden = wrapper.get_layer_visual_and_query_hidden(
                    prepared=prepared,
                    image_features=corrupted,
                    layer_idx=layer_idx,
                )
            else:
                hidden = wrapper.get_layer_visual_hidden(
                    prepared=prepared,
                    image_features=corrupted,
                    layer_idx=layer_idx,
                )

            attn_q_to_vis = None
            attn_text_to_vis_mean = None
            if cfg["scorer"].get("aux_mode", "legacy") == "norm_attn":
                attn_q_to_vis, attn_text_to_vis_mean = wrapper.get_prompt_to_visual_attention(
                    prepared=prepared,
                    image_features=corrupted,
                    layer_indices=cfg["scorer"].get("attention_layers", [2, 4, 8, 12]),
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
            greedy_by_budget = {}
            if (
                args.enable_greedy_oracle
                and greedy_budgets
                and sample_idx < args.greedy_max_samples
            ):
                candidate_pool = oracle_ranked[
                    : min(args.greedy_candidate_limit, oracle_ranked.numel())
                ]
                greedy_by_budget = choose_greedy_set_oracle(
                    wrapper=wrapper,
                    prepared=prepared,
                    corrupted=corrupted,
                    full=full,
                    candidate_pool=candidate_pool,
                    budgets=greedy_budgets,
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
                    scorer_cfg=cfg["scorer"],
                    query_hidden=query_hidden,
                    attn_q_to_vis=attn_q_to_vis,
                    attn_text_to_vis_mean=attn_text_to_vis_mean,
                )
                idx_oracle_single = oracle_ranked[: min(k, oracle_ranked.numel())]

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
                    "oracle_single": compute_metric_from_image_features(
                        wrapper,
                        prepared,
                        restore_by_indices(corrupted, full, idx_oracle_single),
                        eval_metric,
                        teacher_log_probs,
                    ),
                }
                for lam, method_name in zip(mmr_lambdas, mmr_method_names):
                    idx_mlp_mmr = choose_mlp_mmr(
                        scorer=scorer,
                        hidden=hidden,
                        pos=pos,
                        reliability=reliability,
                        damaged_float=damaged_float,
                        candidates=candidates,
                        k=k,
                        scorer_cfg=cfg["scorer"],
                        query_hidden=query_hidden,
                        attn_q_to_vis=attn_q_to_vis,
                        attn_text_to_vis_mean=attn_text_to_vis_mean,
                        lam=lam,
                    )
                    losses[method_name] = compute_metric_from_image_features(
                        wrapper,
                        prepared,
                        restore_by_indices(corrupted, full, idx_mlp_mmr),
                        eval_metric,
                        teacher_log_probs,
                    )
                if k in greedy_by_budget:
                    losses["oracle_greedy"] = compute_metric_from_image_features(
                        wrapper,
                        prepared,
                        restore_by_indices(corrupted, full, greedy_by_budget[k]),
                        eval_metric,
                        teacher_log_probs,
                    )

                append_losses(stats_all, k, loss_full, loss_no, losses)
                if is_vision_sensitive:
                    append_losses(stats_vs, k, loss_full, loss_no, losses)
                else:
                    append_losses(stats_other, k, loss_full, loss_no, losses)

        except Exception as exc:
            print(f"[WARN] eval sample failed: {exc}")
            continue

    print_stats_block(
        "All samples",
        stats_all,
        budgets,
        eval_metric,
        method_names,
        retransmission_method_names,
    )
    print_stats_block(
        "Vision-sensitive samples: full_metric < no_metric",
        stats_vs,
        budgets,
        eval_metric,
        method_names,
        retransmission_method_names,
    )
    print_stats_block(
        "Other samples: full_metric >= no_metric",
        stats_other,
        budgets,
        eval_metric,
        method_names,
        retransmission_method_names,
    )


if __name__ == "__main__":
    main()
