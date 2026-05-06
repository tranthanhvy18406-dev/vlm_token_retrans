import argparse
import json
import os
import sys

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.corruption import corrupt_image_features
from src.dataset import load_jsonl
from src.features import aux_dim_for_mode, build_aux_features
from src.llava_wrapper import LlavaRetransWrapper
from src.metrics import recovery_ratio
from src.scorer import build_scorer_from_config, scorer_requires_query
from src.utils import get_torch_dtype, load_yaml, make_2d_positions, mean_or_nan, set_seed


def normalize_metric_name(name: str) -> str:
    name = name.lower()
    if name in {"ce", "gt_ce", "ground_truth_ce"}:
        return "gt_ce"
    if name in {"teacher_kl", "clean_teacher_kl", "consistency_kl"}:
        return "teacher_kl"
    raise ValueError(f"Unsupported eval/oracle metric: {name}")


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def default_alphas() -> list[float]:
    return [round(0.1 * idx, 1) for idx in range(11)]


def parse_budget_values(value: str | None, defaults: dict[int, list[int]]) -> dict[int, list[int]]:
    if not value:
        return defaults

    parsed: dict[int, list[int]] = {}
    for block in value.split(";"):
        if not block.strip():
            continue
        budget_str, values_str = block.split(":", maxsplit=1)
        parsed[int(budget_str.strip())] = [
            int(item.strip()) for item in values_str.split(",") if item.strip()
        ]
    return parsed


def format_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def make_stats(budgets: list[int], method_names: list[str]) -> dict[int, dict[str, list[float]]]:
    stats = {}
    for k in budgets:
        stats[k] = {"full": [], "no": []}
        for name in method_names:
            stats[k][name] = []
            stats[k][f"{name}__full"] = []
            stats[k][f"{name}__no"] = []
    return stats


def append_losses(
    stats: dict[int, dict[str, list[float]]],
    k: int,
    loss_full: torch.Tensor,
    loss_no: torch.Tensor,
    losses: dict[str, torch.Tensor],
) -> None:
    stats[k]["full"].append(float(loss_full.item()))
    stats[k]["no"].append(float(loss_no.item()))
    for name, value in losses.items():
        stats[k][name].append(float(value.item()))
        stats[k][f"{name}__full"].append(float(loss_full.item()))
        stats[k][f"{name}__no"].append(float(loss_no.item()))


def stats_to_summary(
    stats: dict[int, dict[str, list[float]]],
    budgets: list[int],
    method_names: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    summary = {}
    for k in budgets:
        full_mean = mean_or_nan(stats[k]["full"])
        no_mean = mean_or_nan(stats[k]["no"])
        budget_summary = {}
        for name in method_names:
            values = stats[k][name]
            if not values:
                continue
            method_mean = mean_or_nan(values)
            method_full_mean = mean_or_nan(stats[k][f"{name}__full"])
            method_no_mean = mean_or_nan(stats[k][f"{name}__no"])
            budget_summary[name] = {
                "loss": method_mean,
                "recovery": recovery_ratio(method_no_mean, method_mean, method_full_mean),
                "n": len(values),
            }
        summary[str(k)] = {
            "full": {"loss": full_mean, "n": len(stats[k]["full"])},
            "no": {"loss": no_mean, "n": len(stats[k]["no"])},
            "methods": budget_summary,
        }
    return summary


def print_stats_block(
    title: str,
    stats: dict[int, dict[str, list[float]]],
    budgets: list[int],
    method_names: list[str],
    metric_name: str,
) -> None:
    sample_count = len(stats[budgets[0]]["full"]) if budgets else 0
    metric_label = "teacher KL" if metric_name == "teacher_kl" else "GT CE loss"
    print(f"\n===== {title} (n={sample_count}), lower {metric_label} is better =====")
    if sample_count == 0:
        print("No samples.")
        return

    summary = stats_to_summary(stats, budgets, method_names)
    for k in budgets:
        print(f"\nK={k}")
        print(f"{'full':18s}: {summary[str(k)]['full']['loss']:.6f}")
        print(f"{'no':18s}: {summary[str(k)]['no']['loss']:.6f}")
        for name in method_names:
            method = summary[str(k)]["methods"].get(name)
            if method is None:
                continue
            print(f"{name:18s}: {method['loss']:.6f} (n={method['n']})")

        print("Recovery ratios:")
        for name in method_names:
            method = summary[str(k)]["methods"].get(name)
            if method is None:
                continue
            print(f"{name:18s}: {100.0 * method['recovery']:.2f}%")

        ranked = sorted(
            summary[str(k)]["methods"].items(),
            key=lambda item: item[1]["recovery"],
            reverse=True,
        )
        best = ", ".join(f"{name}={100.0 * value['recovery']:.2f}%" for name, value in ranked[:5])
        print(f"Top recovery: {best}")


def compute_metric_from_image_features(
    wrapper: LlavaRetransWrapper,
    prepared,
    image_features: torch.Tensor,
    metric_name: str,
    teacher_log_probs: torch.Tensor | None,
) -> torch.Tensor:
    if metric_name == "gt_ce":
        return wrapper.compute_loss_from_image_features(prepared, image_features)
    return wrapper.compute_teacher_kl_from_image_features(
        prepared=prepared,
        teacher_log_probs=teacher_log_probs,
        image_features=image_features,
    )


def compute_metric_batch_from_embeds(
    wrapper: LlavaRetransWrapper,
    prepared,
    embeds: torch.Tensor,
    metric_name: str,
    teacher_log_probs: torch.Tensor | None,
) -> torch.Tensor:
    count = embeds.size(0)
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


def build_restored_set_embeds(
    prepared,
    corrupted_features: torch.Tensor,
    full_features: torch.Tensor,
    selected_sets: list[torch.Tensor],
) -> torch.Tensor:
    count = len(selected_sets)
    base_features = corrupted_features.repeat(count, 1, 1)
    for row, indices in enumerate(selected_sets):
        if indices.numel() == 0:
            continue
        indices = indices.to(base_features.device)
        base_features[row, indices, :] = full_features[0, indices, :]

    base_embeds = prepared.inputs_embeds_full.repeat(count, 1, 1)
    base_embeds[:, prepared.image_positions, :] = base_features
    return base_embeds


def compute_selection_losses(
    wrapper: LlavaRetransWrapper,
    prepared,
    corrupted: torch.Tensor,
    full: torch.Tensor,
    selections: dict[str, torch.Tensor],
    metric_name: str,
    teacher_log_probs: torch.Tensor | None,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    names = list(selections.keys())
    losses: dict[str, torch.Tensor] = {}

    for start in range(0, len(names), batch_size):
        chunk_names = names[start : start + batch_size]
        embeds = build_restored_set_embeds(
            prepared=prepared,
            corrupted_features=corrupted,
            full_features=full,
            selected_sets=[selections[name] for name in chunk_names],
        )
        chunk_losses = compute_metric_batch_from_embeds(
            wrapper=wrapper,
            prepared=prepared,
            embeds=embeds,
            metric_name=metric_name,
            teacher_log_probs=teacher_log_probs,
        )
        for name, value in zip(chunk_names, chunk_losses):
            losses[name] = value.detach()
    return losses


def build_scorer_aux(
    hidden: torch.Tensor,
    pos: torch.Tensor,
    reliability: torch.Tensor,
    damaged_float: torch.Tensor,
    candidates: torch.Tensor,
    scorer_cfg: dict,
    attn_q_to_vis: torch.Tensor | None = None,
    attn_text_to_vis_mean: torch.Tensor | None = None,
) -> torch.Tensor:
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
            f"config={expected_aux_dim}, actual={actual_aux_dim}, inferred={inferred_aux_dim}."
        )
    return aux


def score_candidates(
    scorer,
    scorer_cfg: dict,
    hidden: torch.Tensor,
    pos: torch.Tensor,
    reliability: torch.Tensor,
    damaged_float: torch.Tensor,
    candidates: torch.Tensor,
    query_hidden: torch.Tensor | None = None,
    attn_q_to_vis: torch.Tensor | None = None,
    attn_text_to_vis_mean: torch.Tensor | None = None,
) -> torch.Tensor:
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
        return scorer(hidden[candidates].float(), aux, q=query_hidden).float()


def zscore(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return (values - values.mean()) / (values.std(unbiased=False) + 1.0e-6)


def topk_local(scores: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), scores.numel())
    if k <= 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    return torch.topk(scores, k=k).indices


def choose_by_scores(candidates: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    return candidates[topk_local(scores, k)]


def choose_composed(
    candidates: torch.Tensor,
    precision_scores: torch.Tensor,
    coverage_scores: torch.Tensor,
    k: int,
    r: int,
) -> torch.Tensor:
    n = candidates.numel()
    k = min(int(k), n)
    r = min(max(int(r), 0), k)

    selected_mask = torch.zeros(n, dtype=torch.bool, device=candidates.device)
    first = topk_local(precision_scores, r)
    if first.numel() > 0:
        selected_mask[first] = True

    remaining = (~selected_mask).nonzero(as_tuple=False).squeeze(-1)
    fill = min(k - first.numel(), remaining.numel())
    if fill > 0:
        second = remaining[topk_local(coverage_scores[remaining], fill)]
        selected = torch.cat([first, second], dim=0)
    else:
        selected = first

    return candidates[selected]


def load_scorer(cfg: dict, checkpoint_path: str, device: str):
    scorer = build_scorer_from_config(cfg, dropout=0.0).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    scorer.load_state_dict(ckpt["model"])
    scorer.eval()
    return scorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision_config", required=True)
    parser.add_argument("--precision_checkpoint", required=True)
    parser.add_argument("--coverage_config", required=True)
    parser.add_argument("--coverage_checkpoint", required=True)
    parser.add_argument("--precision_label", default="s5")
    parser.add_argument("--coverage_label", default="s7c")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--alphas", default=",".join(str(value) for value in default_alphas()))
    parser.add_argument(
        "--compose_rs",
        default=None,
        help="Format: '16:0,4,8,12,16;32:0,8,16,24,32;64:0,8,16,24,32,48,64'.",
    )
    parser.add_argument("--metric_batch_size", type=int, default=8)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    precision_cfg = load_yaml(args.precision_config)
    coverage_cfg = load_yaml(args.coverage_config)
    set_seed(coverage_cfg["corruption"]["seed"])

    if precision_cfg["eval"]["budgets"] != coverage_cfg["eval"]["budgets"]:
        raise RuntimeError("Precision and coverage configs must use the same eval budgets.")
    budgets = [int(k) for k in coverage_cfg["eval"]["budgets"]]
    alphas = parse_float_list(args.alphas)
    compose_rs = parse_budget_values(
        args.compose_rs,
        defaults={
            16: [0, 4, 8, 12, 16],
            32: [0, 8, 16, 24, 32],
            64: [0, 8, 16, 24, 32, 48, 64],
        },
    )

    device = coverage_cfg["model"]["device"]
    wrapper = LlavaRetransWrapper(
        model_name=coverage_cfg["model"]["name"],
        device=device,
        dtype=get_torch_dtype(coverage_cfg["model"]["dtype"]),
        cache_dir=coverage_cfg["model"].get("cache_dir"),
        local_files_only=bool(coverage_cfg["model"].get("local_files_only", False)),
        device_map=coverage_cfg["model"].get("device_map"),
        attn_implementation=coverage_cfg["model"].get("attn_implementation"),
    )

    precision_scorer = load_scorer(precision_cfg, args.precision_checkpoint, device)
    coverage_scorer = load_scorer(coverage_cfg, args.coverage_checkpoint, device)

    eval_metric = normalize_metric_name(
        coverage_cfg.get("eval", {}).get("metric", coverage_cfg["oracle"].get("target", "gt_ce"))
    )
    layer_idx = coverage_cfg["model"]["layer_idx"]
    if precision_cfg["model"]["layer_idx"] != layer_idx:
        raise RuntimeError("Fusion currently expects both scorers to use the same layer_idx.")

    alpha_methods = [f"fusion_a{format_float(alpha)}" for alpha in alphas]
    compose_methods = []
    for k in budgets:
        for r in compose_rs.get(k, []):
            compose_methods.append(f"compose_r{r}")
    base_methods = [args.precision_label, args.coverage_label, "hidden_norm"]
    method_names = base_methods + alpha_methods + sorted(
        set(compose_methods),
        key=compose_methods.index,
    )

    print(f"eval jsonl: {args.jsonl}")
    print(f"max_samples: {args.max_samples}")
    print(f"eval metric: {eval_metric}")
    print(f"budgets: {budgets}")
    print(f"alphas: {alphas}")
    print(f"compose_rs: {compose_rs}")
    print("Using coverage wrapper for all forwards; this keeps S5/S7c comparable in one run.")

    samples = load_jsonl(
        args.jsonl,
        image_root=coverage_cfg["data"]["image_root"],
        max_samples=args.max_samples,
    )

    stats_all = make_stats(budgets, method_names)
    stats_vs = make_stats(budgets, method_names)
    stats_other = make_stats(budgets, method_names)

    for sample_idx, sample in enumerate(tqdm(samples, desc="eval score fusion")):
        try:
            prepared = wrapper.prepare_inputs(sample.image, sample.question, sample.answer)
            full = prepared.image_features
            corrupted, damaged_mask, reliability = corrupt_image_features(
                full,
                drop_ratio=coverage_cfg["corruption"]["drop_ratio"],
                mode=coverage_cfg["corruption"]["mode"],
                mask_value=coverage_cfg["corruption"]["mask_value"],
            )
            candidates = damaged_mask.nonzero(as_tuple=False).squeeze(-1)
            if candidates.numel() == 0:
                continue

            query_hidden = None
            if scorer_requires_query(precision_cfg["scorer"]) or scorer_requires_query(
                coverage_cfg["scorer"]
            ):
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
            if (
                precision_cfg["scorer"].get("aux_mode", "legacy") == "norm_attn"
                or coverage_cfg["scorer"].get("aux_mode", "legacy") == "norm_attn"
            ):
                attn_q_to_vis, attn_text_to_vis_mean = wrapper.get_prompt_to_visual_attention(
                    prepared=prepared,
                    image_features=corrupted,
                    layer_indices=coverage_cfg["scorer"].get(
                        "attention_layers",
                        precision_cfg["scorer"].get("attention_layers", [2, 4, 8, 12]),
                    ),
                )

            num_tokens = full.shape[1]
            pos = make_2d_positions(num_tokens, device=device)
            damaged_float = damaged_mask.float().unsqueeze(-1)

            precision_scores = score_candidates(
                scorer=precision_scorer,
                scorer_cfg=precision_cfg["scorer"],
                hidden=hidden,
                pos=pos,
                reliability=reliability,
                damaged_float=damaged_float,
                candidates=candidates,
                query_hidden=query_hidden,
                attn_q_to_vis=attn_q_to_vis,
                attn_text_to_vis_mean=attn_text_to_vis_mean,
            )
            coverage_scores = score_candidates(
                scorer=coverage_scorer,
                scorer_cfg=coverage_cfg["scorer"],
                hidden=hidden,
                pos=pos,
                reliability=reliability,
                damaged_float=damaged_float,
                candidates=candidates,
                query_hidden=query_hidden,
                attn_q_to_vis=attn_q_to_vis,
                attn_text_to_vis_mean=attn_text_to_vis_mean,
            )
            precision_z = zscore(precision_scores)
            coverage_z = zscore(coverage_scores)
            hidden_norm_scores = hidden[candidates].float().norm(dim=-1)

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

            if sample_idx % 10 == 0:
                print(
                    f"sample={sample_idx} metric={eval_metric} "
                    f"full={loss_full.item():.6f} no={loss_no.item():.6f} "
                    f"vision_sensitive={is_vision_sensitive}"
                )

            for k in budgets:
                selections = {
                    args.precision_label: choose_by_scores(candidates, precision_scores, k),
                    args.coverage_label: choose_by_scores(candidates, coverage_scores, k),
                    "hidden_norm": choose_by_scores(candidates, hidden_norm_scores, k),
                }
                for alpha, method_name in zip(alphas, alpha_methods):
                    fused_scores = alpha * precision_z + (1.0 - alpha) * coverage_z
                    selections[method_name] = choose_by_scores(candidates, fused_scores, k)

                for r in compose_rs.get(k, []):
                    selections[f"compose_r{r}"] = choose_composed(
                        candidates=candidates,
                        precision_scores=precision_scores,
                        coverage_scores=coverage_scores,
                        k=k,
                        r=r,
                    )

                losses = compute_selection_losses(
                    wrapper=wrapper,
                    prepared=prepared,
                    corrupted=corrupted,
                    full=full,
                    selections=selections,
                    metric_name=eval_metric,
                    teacher_log_probs=teacher_log_probs,
                    batch_size=args.metric_batch_size,
                )

                append_losses(stats_all, k, loss_full, loss_no, losses)
                if is_vision_sensitive:
                    append_losses(stats_vs, k, loss_full, loss_no, losses)
                else:
                    append_losses(stats_other, k, loss_full, loss_no, losses)

        except Exception as exc:
            print(f"[WARN] fusion eval sample failed: {exc}")
            continue

    print_stats_block("All samples", stats_all, budgets, method_names, eval_metric)
    print_stats_block(
        "Vision-sensitive samples: full_metric < no_metric",
        stats_vs,
        budgets,
        method_names,
        eval_metric,
    )
    print_stats_block(
        "Other samples: full_metric >= no_metric",
        stats_other,
        budgets,
        method_names,
        eval_metric,
    )

    if args.output_json:
        output = {
            "jsonl": args.jsonl,
            "max_samples": args.max_samples,
            "metric": eval_metric,
            "budgets": budgets,
            "alphas": alphas,
            "compose_rs": compose_rs,
            "all": stats_to_summary(stats_all, budgets, method_names),
            "vision_sensitive": stats_to_summary(stats_vs, budgets, method_names),
            "other": stats_to_summary(stats_other, budgets, method_names),
        }
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
